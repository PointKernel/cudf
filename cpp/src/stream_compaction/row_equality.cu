/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "row_equality.hpp"

#include <cudf/column/column_child_offsets.hpp>
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include <cstdint>
#include <limits>
#include <vector>

namespace cudf::detail {
namespace {

constexpr int block_size = 256;

struct pair_indices {
  size_type lhs;
  size_type rhs;
  size_type owner;
};

// A null pointer denotes the implicit adjacent pairs (i, i-1), avoiding maps for flat columns.
struct pair_view {
  pair_indices const* data;
  size_type size;
};

struct flat_view : column_device_view_core {
  explicit flat_view(column_view const& column)
    : column_device_view_core{column.type(),
                              column.size(),
                              column.head(),
                              column.null_count(),
                              column.null_mask(),
                              column.offset(),
                              nullptr,
                              0}
  {
  }
};

unsigned int blocks_for(size_type size)
{
  return static_cast<unsigned int>((static_cast<std::size_t>(size) + block_size - 1) / block_size);
}

template <bool Adjacent, typename Operation>
__global__ void apply_pairs(pair_view pairs, bitmask_type* mismatches, Operation operation)
{
  auto const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool different   = false;
  pair_indices pair{-1, -1, 0};
  if (index < static_cast<std::size_t>(pairs.size)) {
    auto const i = static_cast<size_type>(index);
    if constexpr (Adjacent) {
      pair = i == 0 ? pair_indices{-1, -1, 0} : pair_indices{i, i - 1, i};
    } else {
      pair = pairs.data[i];
    }
    // Operations also initialize output coordinates for inactive pairs.
    different = operation(pair, i);
  }
  if constexpr (Adjacent) {
    // Each warp owns one complete mask word. All lanes participate, including the last partial
    // warp. Column launches are ordered on one stream, so combining their words needs no atomics.
    auto const bits = __ballot_sync(0xffffffffu, different);
    if (threadIdx.x % 32 == 0 && index < static_cast<std::size_t>(pairs.size)) {
      mismatches[index / 32] |= bits;
    }
  } else {
    // Nested elements can share a root owner. Never read this mask while updating it here.
    if (different) { atomicOr(mismatches + pair.owner / 32, bitmask_type{1} << (pair.owner % 32)); }
  }
}

template <typename Operation>
void launch_pairs(pair_view pairs,
                  bitmask_type* mismatches,
                  Operation operation,
                  cuda::stream_ref stream)
{
  if (pairs.size == 0) { return; }
  if (pairs.data == nullptr) {
    apply_pairs<true>
      <<<blocks_for(pairs.size), block_size, 0, stream.get()>>>(pairs, mismatches, operation);
  } else {
    apply_pairs<false>
      <<<blocks_for(pairs.size), block_size, 0, stream.get()>>>(pairs, mismatches, operation);
  }
  CUDF_CUDA_TRY(cudaGetLastError());
}

struct null_check {
  bool stop;
  bool different;
};

__device__ null_check check_nulls(flat_view column, pair_indices pair, bool nulls_equal)
{
  if (pair.lhs < 0) { return {true, false}; }
  if (!column.nullable()) { return {false, false}; }
  auto const left  = column.is_null(pair.lhs);
  auto const right = column.is_null(pair.rhs);
  return {left || right, (left || right) && !(nulls_equal && left && right)};
}

template <typename T, bool Boolean = false>
struct compare_fixed_width {
  flat_view column;
  bool nulls_equal;

  __device__ bool operator()(pair_indices pair, size_type) const
  {
    auto const nulls = check_nulls(column, pair, nulls_equal);
    if (nulls.stop) { return nulls.different; }
    auto const lhs = column.data<T>()[pair.lhs];
    auto const rhs = column.data<T>()[pair.rhs];
    if constexpr (Boolean) {
      return (lhs != 0) != (rhs != 0);
    } else if constexpr (cuda::std::is_floating_point_v<T>) {
      return lhs != rhs && !(cuda::std::isnan(lhs) && cuda::std::isnan(rhs));
    } else {
      return lhs != rhs;
    }
  }
};

template <typename Offset>
struct compare_strings {
  flat_view column;
  Offset const* offsets;
  bool nulls_equal;

  __device__ string_view element(size_type row) const
  {
    auto const i     = row + column.offset();
    auto const begin = offsets[i];
    return {column.head<char>() + begin, static_cast<size_type>(offsets[i + 1] - begin)};
  }

  __device__ bool operator()(pair_indices pair, size_type) const
  {
    auto const nulls = check_nulls(column, pair, nulls_equal);
    return nulls.stop ? nulls.different : element(pair.lhs) != element(pair.rhs);
  }
};

// Equality of integral, chrono and decimal values within one column needs only their physical
// width. Sharing those instantiations avoids compiling a separate kernel for each logical type.
template <std::size_t Width>
using word_type = cuda::std::conditional_t<
  Width == 1,
  uint8_t,
  cuda::std::conditional_t<
    Width == 2,
    uint16_t,
    cuda::std::conditional_t<Width == 4,
                             uint32_t,
                             cuda::std::conditional_t<Width == 8, uint64_t, __uint128_t>>>>;

struct dispatch_leaf {
  template <typename T>
  void operator()(column_view const& column,
                  pair_view pairs,
                  bitmask_type* mismatches,
                  bool nulls_equal,
                  cuda::stream_ref stream) const
  {
    if constexpr (is_fixed_width<T>()) {
      using storage_type  = device_storage_type_t<T>;
      using physical_type = cuda::std::
        conditional_t<cuda::std::is_floating_point_v<T>, T, word_type<sizeof(storage_type)>>;
      launch_pairs(pairs,
                   mismatches,
                   compare_fixed_width<physical_type, cuda::std::is_same_v<T, bool>>{
                     flat_view{column}, nulls_equal},
                   stream);
    } else {
      CUDF_FAIL("Unsupported leaf in adjacent row equality");
    }
  }
};

struct map_struct {
  flat_view column;
  pair_indices* output;
  bool nulls_equal;

  __device__ bool operator()(pair_indices pair, size_type i) const
  {
    auto const nulls = check_nulls(column, pair, nulls_equal);
    output[i] =
      nulls.stop ? pair_indices{-1, -1, pair.owner}
                 : pair_indices{pair.lhs + column.offset(), pair.rhs + column.offset(), pair.owner};
    return nulls.different;
  }
};

template <typename Index>
struct map_dictionary {
  flat_view column;
  Index const* indices;
  pair_indices* output;
  bool nulls_equal;

  __device__ bool operator()(pair_indices pair, size_type i) const
  {
    auto const nulls = check_nulls(column, pair, nulls_equal);
    output[i]        = nulls.stop
                         ? pair_indices{-1, -1, pair.owner}
                         : pair_indices{static_cast<size_type>(indices[pair.lhs + column.offset()]),
                                 static_cast<size_type>(indices[pair.rhs + column.offset()]),
                                 pair.owner};
    return nulls.different;
  }
};

struct dispatch_dictionary {
  template <typename Index>
  void operator()(column_view const& column,
                  pair_view pairs,
                  bitmask_type* mismatches,
                  pair_indices* output,
                  bool nulls_equal,
                  cuda::stream_ref stream) const
  {
    if constexpr (is_index_type<Index>()) {
      launch_pairs(
        pairs,
        mismatches,
        map_dictionary<Index>{flat_view{column},
                              column.child(dictionary_indices_column_index).data<Index>(),
                              output,
                              nulls_equal},
        stream);
    } else {
      CUDF_FAIL("Invalid dictionary index type");
    }
  }
};

struct list_lengths {
  flat_view column;
  size_type const* offsets;
  uint64_t* lengths;
  pair_indices* starts;
  bool nulls_equal;

  __device__ bool operator()(pair_indices pair, size_type i) const
  {
    lengths[i]       = 0;
    starts[i]        = {-1, -1, pair.owner};
    auto const nulls = check_nulls(column, pair, nulls_equal);
    if (nulls.stop) { return nulls.different; }
    auto const lhs = pair.lhs + column.offset();
    auto const rhs = pair.rhs + column.offset();
    auto const n   = offsets[lhs + 1] - offsets[lhs];
    if (n != offsets[rhs + 1] - offsets[rhs]) { return true; }
    lengths[i] = static_cast<uint64_t>(n);
    starts[i]  = {offsets[lhs], offsets[rhs], pair.owner};
    return false;
  }
};

__global__ void expand_lists(pair_indices const* starts,
                             uint64_t const* prefix,
                             size_type size,
                             pair_indices* output)
{
  auto const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::size_t>(size)) { return; }
  auto const pair = starts[index];
  auto const end  = prefix[index + 1];
  for (auto j = prefix[index]; j < end; ++j) {
    auto const offset = static_cast<size_type>(j - prefix[index]);
    output[j]         = {pair.lhs + offset, pair.rhs + offset, pair.owner};
  }
}

void compare_column(column_view const& column,
                    pair_view pairs,
                    bitmask_type* mismatches,
                    bool nulls_equal,
                    bool normalize_struct_children,
                    cuda::stream_ref stream,
                    memory_resources resources)
{
  if (pairs.size == 0) { return; }
  auto const temp_mr = resources.get_temporary_mr();
  switch (column.type().id()) {
    case type_id::STRUCT: {
      auto mapped = rmm::device_uvector<pair_indices>(pairs.size, stream, temp_mr);
      launch_pairs(
        pairs, mismatches, map_struct{flat_view{column}, mapped.data(), nulls_equal}, stream);
      for (size_type child_index = 0; child_index < column.num_children(); ++child_index) {
        auto child = column.child(child_index);
        if (normalize_struct_children) {
          // Match row equality preprocessing: a struct's offset replaces its children's offsets.
          // Stop normalizing when traversing below a list or dictionary.
          child = column_view{child.type(),
                              child.size() + child.offset(),
                              child.head(),
                              child.null_mask(),
                              child.null_count(),
                              0,
                              std::vector<column_view>{child.child_begin(), child.child_end()}};
        }
        compare_column(child,
                       {mapped.data(), pairs.size},
                       mismatches,
                       nulls_equal,
                       normalize_struct_children,
                       stream,
                       resources);
      }
      return;
    }
    case type_id::LIST: {
      auto starts = rmm::device_uvector<pair_indices>(pairs.size, stream, temp_mr);
      auto lengths =
        rmm::device_uvector<uint64_t>(static_cast<std::size_t>(pairs.size) + 1, stream, temp_mr);
      auto prefix = rmm::device_uvector<uint64_t>(lengths.size(), stream, temp_mr);
      CUDF_CUDA_TRY(
        cudaMemsetAsync(lengths.data() + pairs.size, 0, sizeof(uint64_t), stream.get()));
      launch_pairs(pairs,
                   mismatches,
                   list_lengths{flat_view{column},
                                column.child(offsets_column_index).data<size_type>(),
                                lengths.data(),
                                starts.data(),
                                nulls_equal},
                   stream);
      std::size_t scratch_bytes{};
      CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
        nullptr, scratch_bytes, lengths.data(), prefix.data(), lengths.size(), stream.get()));
      auto scratch = rmm::device_buffer(scratch_bytes, stream, temp_mr);
      CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(scratch.data(),
                                                  scratch_bytes,
                                                  lengths.data(),
                                                  prefix.data(),
                                                  lengths.size(),
                                                  stream.get()));
      uint64_t count{};
      cudf::detail::cuda_memcpy_async(host_span<uint64_t>{&count, 1},
                                      device_span<uint64_t const>{prefix.data() + pairs.size, 1},
                                      stream);
      stream.sync();
      CUDF_EXPECTS(count <= static_cast<uint64_t>(std::numeric_limits<size_type>::max()),
                   "Too many nested comparison pairs",
                   std::overflow_error);
      if (count == 0) { return; }
      auto children = rmm::device_uvector<pair_indices>(count, stream, temp_mr);
      expand_lists<<<blocks_for(pairs.size), block_size, 0, stream.get()>>>(
        starts.data(), prefix.data(), pairs.size, children.data());
      CUDF_CUDA_TRY(cudaGetLastError());
      compare_column(column.child(lists_column_view::child_column_index),
                     {children.data(), static_cast<size_type>(count)},
                     mismatches,
                     nulls_equal,
                     false,
                     stream,
                     resources);
      return;
    }
    case type_id::DICTIONARY32: {
      auto mapped = rmm::device_uvector<pair_indices>(pairs.size, stream, temp_mr);
      type_dispatcher(column.child(dictionary_indices_column_index).type(),
                      dispatch_dictionary{},
                      column,
                      pairs,
                      mismatches,
                      mapped.data(),
                      nulls_equal,
                      stream);
      compare_column(column.child(dictionary_keys_column_index),
                     {mapped.data(), pairs.size},
                     mismatches,
                     nulls_equal,
                     false,
                     stream,
                     resources);
      return;
    }
    case type_id::STRING: {
      auto const offsets = column.child(offsets_column_index);
      if (offsets.type().id() == type_id::INT32) {
        launch_pairs(
          pairs,
          mismatches,
          compare_strings<int32_t>{flat_view{column}, offsets.head<int32_t>(), nulls_equal},
          stream);
      } else {
        CUDF_EXPECTS(offsets.type().id() == type_id::INT64, "Invalid string offset type");
        launch_pairs(
          pairs,
          mismatches,
          compare_strings<int64_t>{flat_view{column}, offsets.head<int64_t>(), nulls_equal},
          stream);
      }
      return;
    }
    default:
      type_dispatcher(
        column.type(), dispatch_leaf{}, column, pairs, mismatches, nulls_equal, stream);
  }
}

}  // namespace

rmm::device_uvector<bitmask_type> adjacent_row_mismatches(table_view const& input,
                                                          null_equality nulls_equal,
                                                          cuda::stream_ref stream,
                                                          memory_resources resources)
{
  auto const words = (static_cast<std::size_t>(input.num_rows()) + 31) / 32;
  auto result      = rmm::device_uvector<bitmask_type>(words, stream, resources.get_output_mr());
  if (words == 0) { return result; }
  CUDF_CUDA_TRY(
    cudaMemsetAsync(result.data(), 0, result.size() * sizeof(bitmask_type), stream.get()));
  for (auto const& column : input) {
    compare_column(column,
                   {nullptr, input.num_rows()},
                   result.data(),
                   nulls_equal == null_equality::EQUAL,
                   true,
                   stream,
                   resources);
  }
  return result;
}

}  // namespace cudf::detail
