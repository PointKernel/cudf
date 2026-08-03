/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cudf {
namespace detail {

namespace {

struct alignas(16) fixed_width_16_byte_storage {
  uint64_t low;
  uint64_t high;
};

template <std::size_t ElementSize>
struct fixed_width_storage;

template <>
struct fixed_width_storage<1> {
  using type = uint8_t;
};

template <>
struct fixed_width_storage<2> {
  using type = uint16_t;
};

template <>
struct fixed_width_storage<4> {
  using type = uint32_t;
};

template <>
struct fixed_width_storage<8> {
  using type = uint64_t;
};

template <>
struct fixed_width_storage<16> {
  using type = fixed_width_16_byte_storage;
};

struct fixed_width_gather_column {
  void const* source;
  void* destination;
};

/**
 * @brief Gathers a homogeneous table of fixed-width columns.
 *
 * Each thread loads one gather-map entry and reuses it for a tile of columns. The element size and
 * bounds policy are dispatched before launch, keeping the device loop free of type dispatch and
 * runtime element-size branches.
 */
template <std::size_t ElementSize,
          bool NullifyOutOfBounds,
          int32_t ColumnsPerTile,
          typename MapIterator>
CUDF_KERNEL void gather_fixed_width_columns_kernel(fixed_width_gather_column const* columns,
                                                   size_type num_columns,
                                                   MapIterator gather_map,
                                                   size_type num_rows,
                                                   size_type source_size)
{
  using storage_type = typename fixed_width_storage<ElementSize>::type;

  auto const row = static_cast<size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) { return; }

  auto const source_index = gather_map[row];
  if constexpr (NullifyOutOfBounds) {
    auto const out_of_bounds = is_signed_iterator<MapIterator>()
                                 ? source_index < 0 || source_index >= source_size
                                 : source_index >= source_size;
    if (out_of_bounds) { return; }
  }

  for (auto first_column = static_cast<size_type>(blockIdx.y * ColumnsPerTile);
       first_column < num_columns;
       first_column += static_cast<size_type>(gridDim.y * ColumnsPerTile)) {
#pragma unroll
    for (int32_t tile_offset = 0; tile_offset < ColumnsPerTile; ++tile_offset) {
      auto const column_index = first_column + tile_offset;
      if (column_index < num_columns) {
        auto const source = static_cast<storage_type const*>(columns[column_index].source);
        auto destination  = static_cast<storage_type*>(columns[column_index].destination);
        destination[row]  = source[source_index];
      }
    }
  }
}

template <std::size_t ElementSize, bool NullifyOutOfBounds, typename MapIterator>
void gather_fixed_width_columns(std::vector<fixed_width_gather_column> const& columns,
                                MapIterator gather_map,
                                size_type num_rows,
                                size_type source_size,
                                rmm::cuda_stream_view stream)
{
  if (num_rows == 0) { return; }

  auto device_columns =
    make_device_uvector_async(columns, stream, cudf::get_current_device_resource_ref());

  constexpr int32_t block_size       = 256;
  constexpr int32_t columns_per_tile = 4;
  auto const row_grid                = cudf::detail::grid_1d{num_rows, block_size};
  auto const column_tiles =
    cudf::util::div_rounding_up_safe(columns.size(), std::size_t{columns_per_tile});
  auto const grid_y = std::min<std::size_t>(column_tiles, 65535);

  gather_fixed_width_columns_kernel<ElementSize, NullifyOutOfBounds, columns_per_tile>
    <<<dim3{static_cast<uint32_t>(row_grid.num_blocks), static_cast<uint32_t>(grid_y), 1},
       block_size,
       0,
       stream.value()>>>(device_columns.data(),
                         static_cast<size_type>(columns.size()),
                         gather_map,
                         num_rows,
                         source_size);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <bool NullifyOutOfBounds, typename MapIterator>
void dispatch_gather_fixed_width_columns(std::size_t element_size,
                                         std::vector<fixed_width_gather_column> const& columns,
                                         MapIterator gather_map,
                                         size_type num_rows,
                                         size_type source_size,
                                         rmm::cuda_stream_view stream)
{
  switch (element_size) {
    case 1:
      return gather_fixed_width_columns<1, NullifyOutOfBounds>(
        columns, gather_map, num_rows, source_size, stream);
    case 2:
      return gather_fixed_width_columns<2, NullifyOutOfBounds>(
        columns, gather_map, num_rows, source_size, stream);
    case 4:
      return gather_fixed_width_columns<4, NullifyOutOfBounds>(
        columns, gather_map, num_rows, source_size, stream);
    case 8:
      return gather_fixed_width_columns<8, NullifyOutOfBounds>(
        columns, gather_map, num_rows, source_size, stream);
    case 16:
      return gather_fixed_width_columns<16, NullifyOutOfBounds>(
        columns, gather_map, num_rows, source_size, stream);
    default: CUDF_FAIL("Unsupported fixed-width element size");
  }
}

template <typename MapIterator>
std::unique_ptr<table> try_gather_homogeneous_fixed_width(table_view const& source_table,
                                                          MapIterator gather_map_begin,
                                                          MapIterator gather_map_end,
                                                          out_of_bounds_policy bounds_policy,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{
  constexpr size_type min_fused_columns = 64;
  if (source_table.num_columns() < min_fused_columns) { return {}; }

  auto const first_type = source_table.column(0).type();
  if (!cudf::is_fixed_width(first_type)) { return {}; }

  auto const element_size = cudf::size_of(first_type);
  if (std::any_of(source_table.begin(), source_table.end(), [&](auto const& column) {
        return !cudf::is_fixed_width(column.type()) || cudf::size_of(column.type()) != element_size;
      })) {
    return {};
  }

  auto const num_rows = static_cast<size_type>(cudf::distance(gather_map_begin, gather_map_end));
  std::vector<std::unique_ptr<column>> destination_columns;
  std::vector<fixed_width_gather_column> gather_columns;
  destination_columns.reserve(source_table.num_columns());
  gather_columns.reserve(source_table.num_columns());

  for (auto const& source_column : source_table) {
    auto destination_column =
      cudf::allocate_like(source_column, num_rows, cudf::mask_allocation_policy::NEVER, stream, mr);
    auto const source_data =
      static_cast<std::byte const*>(source_column.head()) + source_column.offset() * element_size;
    gather_columns.push_back({source_data, destination_column->mutable_view().head()});
    destination_columns.push_back(std::move(destination_column));
  }

  if (bounds_policy == out_of_bounds_policy::NULLIFY) {
    dispatch_gather_fixed_width_columns<true>(
      element_size, gather_columns, gather_map_begin, num_rows, source_table.num_rows(), stream);
  } else {
    dispatch_gather_fixed_width_columns<false>(
      element_size, gather_columns, gather_map_begin, num_rows, source_table.num_rows(), stream);
  }

  auto const needs_new_bitmask = bounds_policy == out_of_bounds_policy::NULLIFY ||
                                 cudf::has_nested_nullable_columns(source_table);
  if (needs_new_bitmask) {
    auto const has_possible_nulls =
      bounds_policy == out_of_bounds_policy::NULLIFY || cudf::has_nested_nulls(source_table);
    if (has_possible_nulls) {
      auto const op = bounds_policy == out_of_bounds_policy::NULLIFY
                        ? gather_bitmask_op::NULLIFY
                        : gather_bitmask_op::DONT_CHECK;
      gather_bitmask(source_table, gather_map_begin, destination_columns, op, stream, mr);
    } else {
      for (size_type i = 0; i < source_table.num_columns(); ++i) {
        set_all_valid_null_masks(source_table.column(i), *destination_columns[i], stream, mr);
      }
    }
  }

  return std::make_unique<table>(std::move(destination_columns), num_rows);
}

}  // namespace

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds_policy,
                              negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not gather_map.has_nulls(), "gather_map contains nulls", std::invalid_argument);

  // create index type normalizing iterator for the gather_map
  auto map_begin = indexalator_factory::make_input_iterator(gather_map);
  auto map_end   = map_begin + gather_map.size();

  if (neg_indices == negative_index_policy::ALLOWED) {
    cudf::size_type n_rows = source_table.num_rows();
    auto idx_converter     = cuda::proclaim_return_type<size_type>(
      [n_rows] __device__(size_type in) { return in < 0 ? in + n_rows : in; });
    auto transformed_begin = thrust::make_transform_iterator(map_begin, idx_converter);
    auto transformed_end   = thrust::make_transform_iterator(map_end, idx_converter);
    if (auto result = try_gather_homogeneous_fixed_width(
          source_table, transformed_begin, transformed_end, bounds_policy, stream, mr)) {
      return result;
    }
    return gather(source_table, transformed_begin, transformed_end, bounds_policy, stream, mr);
  }
  if (auto result = try_gather_homogeneous_fixed_width(
        source_table, map_begin, map_end, bounds_policy, stream, mr)) {
    return result;
  }
  return gather(source_table, map_begin, map_end, bounds_policy, stream, mr);
}

std::unique_ptr<table> gather(table_view const& source_table,
                              device_span<size_type const> const gather_map,
                              out_of_bounds_policy bounds_policy,
                              negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(gather_map.size() <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "gather map size exceeds the column size limit",
               std::overflow_error);
  auto map_col = column_view(data_type{type_to_id<size_type>()},
                             static_cast<size_type>(gather_map.size()),
                             gather_map.data(),
                             nullptr,
                             0);
  return detail::gather(source_table, map_col, bounds_policy, neg_indices, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds_policy,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto const index_policy = is_unsigned(gather_map.type()) ? negative_index_policy::NOT_ALLOWED
                                                           : negative_index_policy::ALLOWED;

  return detail::gather(source_table, gather_map, bounds_policy, index_policy, stream, mr);
}

std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds_policy,
                              negative_index_policy neg_indices,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::gather(source_table, gather_map, bounds_policy, neg_indices, stream, mr);
}

}  // namespace cudf
