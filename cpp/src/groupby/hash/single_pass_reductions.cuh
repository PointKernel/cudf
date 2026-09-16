/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "single_pass_reductions.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/element_argminmax.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction/detail/sum_overflow.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {

/// Reads a fixed-width element, going through the keys when the column is a dictionary.
template <typename T>
struct value_accessor {
  column_device_view col;
  bool is_dictionary;

  __device__ T operator()(size_type row) const
  {
    if (is_dictionary) {
      auto const keys = col.child(dictionary_column_view::keys_column_index);
      return keys.element<T>(static_cast<size_type>(col.element<dictionary32>(row)));
    }
    return col.element<T>(row);
  }
};

template <typename T>
value_accessor<T> reduction_context::accessor() const
{
  return {d_values, is_dictionary(values.type())};
}

/// Maps a grouped position to the value of the input row at that position, substituting
/// `null_value` for null rows and optionally squaring the value.
template <typename Source, typename Target, bool Square>
struct grouped_value_fn {
  size_type const* grouped_rows;
  value_accessor<Source> value;
  Target null_value;
  bool has_nulls;

  __device__ bool col_is_null(size_type row) const
  {
    return has_nulls && value.col.is_null_nocheck(row);
  }

  __device__ Target compute(size_type row) const
  {
    auto const result = static_cast<Target>(value(row));
    if constexpr (Square) { return result * result; }
    return result;
  }

  __device__ Target operator()(size_type position) const
  {
    auto const row = grouped_rows[position];
    return col_is_null(row) ? null_value : compute(row);
  }
};

/// A reduced value together with whether any of the reduced rows was valid, so that a nullable
/// aggregation and its null mask come out of one pass.
template <typename Result>
struct valid_value {
  Result value;
  bool valid;
};

template <typename Op, typename Result>
struct valid_value_op {
  __device__ valid_value<Result> operator()(valid_value<Result> const& lhs,
                                            valid_value<Result> const& rhs) const
  {
    return {Op{}(lhs.value, rhs.value), lhs.valid || rhs.valid};
  }
};

/// Maps a grouped position to the value of the input row at that position and its validity; null
/// rows contribute the identity.
template <typename Source, typename Target, bool Square>
struct grouped_valid_value_fn {
  grouped_value_fn<Source, Target, Square> value;

  __device__ valid_value<Target> operator()(size_type position) const
  {
    auto const row = value.grouped_rows[position];
    if (value.col_is_null(row)) { return {value.null_value, false}; }
    return {value.compute(row), true};
  }
};

template <typename Result>
struct split_valid_value_fn {
  __device__ cuda::std::tuple<Result, bool> operator()(valid_value<Result> const& v) const
  {
    return {v.value, v.valid};
  }
};

/// Maps a grouped position to a SUM_OVERFLOW accumulator, treating nulls as a zero contribution.
template <typename DeviceType>
struct grouped_sum_overflow_fn {
  size_type const* grouped_rows;
  value_accessor<DeviceType> value;
  bool has_nulls;

  __device__ cudf::reduction::detail::sum_overflow_result<DeviceType> operator()(
    size_type position) const
  {
    auto const row = grouped_rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return {DeviceType{0}, 0}; }
    return {value(row), 0};
  }
};

/// Splits a reduced accumulator into the sum and overflow-flag children of the output struct.
template <typename DeviceType>
struct split_sum_overflow_fn {
  __device__ cuda::std::tuple<DeviceType, bool> operator()(
    cudf::reduction::detail::sum_overflow_result<DeviceType> const& accumulator) const
  {
    return {accumulator.sum, accumulator.wraps != 0};
  }
};

/// Sums accumulated together when several additive aggregations are requested on one column.
template <typename Result>
struct fused_sums {
  Result sum;
  Result sum_of_squares;
  size_type count;
};

template <typename Result>
struct fused_sums_plus {
  __device__ fused_sums<Result> operator()(fused_sums<Result> const& lhs,
                                           fused_sums<Result> const& rhs) const
  {
    return {lhs.sum + rhs.sum, lhs.sum_of_squares + rhs.sum_of_squares, lhs.count + rhs.count};
  }
};

/// Maps a grouped position to the sums contributed by the input row at that position.
template <typename Source, typename Result>
struct grouped_fused_sums_fn {
  size_type const* grouped_rows;
  value_accessor<Source> value;
  bool has_nulls;

  __device__ fused_sums<Result> operator()(size_type position) const
  {
    auto const row = grouped_rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return {Result{0}, Result{0}, 0}; }
    auto const result = static_cast<Result>(value(row));
    return {result, result * result, 1};
  }
};

/// Splits the reduced sums into the SUM, SUM_OF_SQUARES and COUNT_VALID outputs.
template <typename Result>
struct split_fused_sums_fn {
  __device__ cuda::std::tuple<Result, Result, size_type> operator()(
    fused_sums<Result> const& sums) const
  {
    return {sums.sum, sums.sum_of_squares, sums.count};
  }
};

constexpr bool is_fusable_sum(aggregation::Kind kind)
{
  return kind == aggregation::SUM || kind == aggregation::SUM_OF_SQUARES ||
         kind == aggregation::COUNT_VALID;
}

/// Extrema retain the input representation while SUM uses its promoted result type.
template <typename Source, typename Result>
struct fused_minmax_sum {
  Source minimum;
  Source maximum;
  Result sum;
  bool valid;
};

template <typename Source, typename Result>
struct fused_minmax_sum_op {
  __device__ fused_minmax_sum<Source, Result> operator()(
    fused_minmax_sum<Source, Result> const& lhs, fused_minmax_sum<Source, Result> const& rhs) const
  {
    return {cudf::detail::corresponding_operator_t<aggregation::MIN>{}(lhs.minimum, rhs.minimum),
            cudf::detail::corresponding_operator_t<aggregation::MAX>{}(lhs.maximum, rhs.maximum),
            cudf::detail::corresponding_operator_t<aggregation::SUM>{}(lhs.sum, rhs.sum),
            lhs.valid || rhs.valid};
  }
};

template <typename Source, typename Result>
struct grouped_fused_minmax_sum_fn {
  size_type const* grouped_rows;
  value_accessor<Source> value;
  bool has_nulls;
  fused_minmax_sum<Source, Result> identity;

  __device__ fused_minmax_sum<Source, Result> operator()(size_type position) const
  {
    auto const row = grouped_rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return identity; }
    auto const result = value(row);
    return {result, result, static_cast<Result>(result), true};
  }
};

template <typename Source, typename Result>
struct split_fused_minmax_sum_fn {
  __device__ cuda::std::tuple<Source, Source, Result, bool> operator()(
    fused_minmax_sum<Source, Result> const& value) const
  {
    return {value.minimum, value.maximum, value.sum, value.valid};
  }
};

constexpr bool is_fusable_minmax_sum(aggregation::Kind kind)
{
  return kind == aggregation::MIN || kind == aggregation::MAX || kind == aggregation::SUM;
}

/// Maximum rows per chunk, independent of the distribution of group sizes.
constexpr size_type rows_per_chunk               = 1 << 10;
constexpr thread_index_type reduction_block_size = 256;

/// Fold groups no wider than a warp in one thread, using only their actual values.
template <typename ValueIterator, typename OutputIterator, typename Op, typename T>
CUDF_KERNEL void reduce_small_groups_kernel(
  device_span<size_type const> offsets, ValueIterator values, OutputIterator output, Op op, T init)
{
  auto const num_groups = static_cast<thread_index_type>(offsets.size() - 1);
  for (auto group = cudf::detail::grid_1d::global_thread_id(); group < num_groups;
       group += cudf::detail::grid_1d::grid_stride()) {
    auto const begin = static_cast<cuda::std::int64_t>(offsets[group]);
    auto const end   = static_cast<cuda::std::int64_t>(offsets[group + 1]);
    if (end - begin <= cudf::detail::warp_size) {
      T partial = values[begin];
      for (auto position = begin + 1; position < end; ++position) {
        partial = op(partial, values[position]);
      }
      output[group] = op(init, partial);
    }
  }
}

/// Reduces each range with one warp or block and writes its original output index.
template <int threads_per_segment,
          typename BeginIterator,
          typename EndIterator,
          typename ValueIterator,
          typename OutputIterator,
          typename OutputIndexIterator,
          typename Op,
          typename T>
CUDF_KERNEL void reduce_segments_kernel(size_type num_segments,
                                        BeginIterator begins,
                                        EndIterator ends,
                                        ValueIterator values,
                                        OutputIterator output,
                                        OutputIndexIterator output_indices,
                                        Op op,
                                        T init)
{
  static_assert(threads_per_segment == cudf::detail::warp_size ||
                threads_per_segment == reduction_block_size);
  using segment_reduce = cuda::std::conditional_t<threads_per_segment == cudf::detail::warp_size,
                                                  cub::WarpReduce<T, cudf::detail::warp_size>,
                                                  cub::BlockReduce<T, reduction_block_size>>;
  __shared__
    typename segment_reduce::TempStorage storage[reduction_block_size / threads_per_segment];
  auto const segment    = cudf::detail::grid_1d::global_thread_id() / threads_per_segment;
  auto const lane       = static_cast<size_type>(threadIdx.x % threads_per_segment);
  auto const collective = threadIdx.x / threads_per_segment;
  if (segment >= num_segments) { return; }
  auto const begin = static_cast<cuda::std::int64_t>(begins[segment]);
  auto const end   = static_cast<cuda::std::int64_t>(ends[segment]);
  if (begin == end) {
    if (lane == 0) { output[output_indices[segment]] = init; }
    return;
  }
  T partial     = init;
  auto position = begin + lane;
  if (position < end) {
    partial = values[position];
    for (position += threads_per_segment; position < end; position += threads_per_segment) {
      partial = op(partial, values[position]);
    }
  }
  auto const valid_lanes =
    static_cast<int>(cuda::std::min<cuda::std::int64_t>(end - begin, threads_per_segment));
  auto const result = segment_reduce{storage[collective]}.Reduce(partial, op, valid_lanes);
  if (lane == 0) { output[output_indices[segment]] = op(init, result); }
}

/// Launches ranges without allocating storage; iterators encode selection and scheduling.
template <int threads_per_segment,
          typename BeginIterator,
          typename EndIterator,
          typename ValueIterator,
          typename OutputIterator,
          typename OutputIndexIterator,
          typename Op,
          typename T>
void reduce_segments(size_type num_segments,
                     BeginIterator begins,
                     EndIterator ends,
                     ValueIterator values,
                     OutputIterator output,
                     OutputIndexIterator output_indices,
                     Op op,
                     T init,
                     cuda::stream_ref stream)
{
  if (num_segments == 0) { return; }
  auto const config = cudf::detail::grid_1d{
    static_cast<thread_index_type>(num_segments) * threads_per_segment, reduction_block_size};
  reduce_segments_kernel<threads_per_segment>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      num_segments, begins, ends, values, output, output_indices, op, init);
  CUDF_CUDA_TRY(cudaGetLastError());
}

/// Small and bounded groups write final outputs; only long groups need chunk partials.
template <typename ValueIterator, typename OutputIterator, typename Op, typename T>
void reduce_groups(grouped_rows const& grouped,
                   ValueIterator values,
                   OutputIterator output,
                   Op op,
                   T init,
                   cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr)
{
  auto const num_groups      = static_cast<size_type>(grouped.offsets.size() - 1);
  auto const num_warp_groups = static_cast<size_type>(grouped.warp_groups.size());
  if (num_warp_groups < num_groups) {
    auto const config = cudf::detail::grid_1d{num_groups, reduction_block_size};
    reduce_small_groups_kernel<<<config.num_blocks,
                                 config.num_threads_per_block,
                                 0,
                                 stream.get()>>>(grouped.offsets, values, output, op, init);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  auto const num_long_groups   = grouped.group_chunks.is_empty()
                                   ? size_type{0}
                                   : static_cast<size_type>(grouped.group_chunks.size() - 1);
  auto const num_direct_groups = num_warp_groups - num_long_groups;
  auto const group_ids         = grouped.warp_groups.begin();
  if (num_direct_groups > 0) {
    reduce_segments<cudf::detail::warp_size>(
      num_direct_groups,
      cuda::make_permutation_iterator(grouped.offsets.begin(), group_ids),
      cuda::make_permutation_iterator(grouped.offsets.begin() + 1, group_ids),
      values,
      output,
      group_ids,
      op,
      init,
      stream);
  }
  if (num_long_groups == 0) { return; }
  auto const num_chunks = static_cast<size_type>(grouped.chunk_ranges.size());
  rmm::device_uvector<T> partials(num_chunks, stream, mr);
  auto const begins = cuda::transform_iterator{
    grouped.chunk_ranges.begin(),
    [] __device__(cuda::std::array<size_type, 2> const& range) -> size_type { return range[0]; }};
  auto const ends = cuda::transform_iterator{
    grouped.chunk_ranges.begin(),
    [] __device__(cuda::std::array<size_type, 2> const& range) -> size_type { return range[1]; }};
  reduce_segments<reduction_block_size>(
    num_chunks,
    cuda::make_permutation_iterator(begins, grouped.chunk_order.begin()),
    cuda::make_permutation_iterator(ends, grouped.chunk_order.begin()),
    values,
    partials.begin(),
    grouped.chunk_order.begin(),
    op,
    init,
    stream);
  reduce_segments<reduction_block_size>(num_long_groups,
                                        grouped.group_chunks.begin(),
                                        grouped.group_chunks.begin() + 1,
                                        partials.begin(),
                                        output,
                                        group_ids + num_direct_groups,
                                        op,
                                        init,
                                        stream);
}

/// Representation used to share reduction instantiations across column types.
/// `device_storage_type_t` unwraps decimals but leaves chrono wrappers intact, so chrono columns
/// additionally use `T::rep` to reduce as their underlying integers.
template <typename T>
struct rep_type {
  using type = device_storage_type_t<T>;
};

template <typename T>
  requires(cudf::is_chrono<T>())
struct rep_type<T> {
  using type = typename T::rep;
};

template <typename T>
using rep_type_t = typename rep_type<T>::type;

template <aggregation::Kind K, typename T>
constexpr bool is_reduction_supported()
{
  switch (K) {
    case aggregation::SUM:
    case aggregation::PRODUCT:
    case aggregation::SUM_OF_SQUARES:
    case aggregation::SUM_OVERFLOW: return cudf::detail::is_valid_aggregation<T, K>();
    // Target-type validity alone does not constrain extrema's storage or comparisons.
    case aggregation::MIN:
    case aggregation::MAX: return cudf::is_fixed_width<T>() && is_relationally_comparable<T, T>();
    case aggregation::ARGMIN:
    case aggregation::ARGMAX: return is_relationally_comparable<T, T>();
    default: return false;
  }
}

template <aggregation::Kind K>
struct grouped_reduction_fn {
  template <typename T>
    requires(is_reduction_supported<K, T>() &&
             (K == aggregation::SUM || K == aggregation::PRODUCT ||
              K == aggregation::SUM_OF_SQUARES || K == aggregation::MIN || K == aggregation::MAX))
  std::unique_ptr<column> operator()(reduction_context const& ctx,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
  {
    using Source = rep_type_t<T>;
    using Result = rep_type_t<cudf::detail::target_type_t<T, K>>;
    using Op     = cudf::detail::corresponding_operator_t<K>;

    auto result = make_fixed_width_column(cudf::detail::target_type(ctx.values_type, K),
                                          ctx.num_groups,
                                          mask_state::UNALLOCATED,
                                          stream,
                                          mr.get_output_mr());
    if (ctx.num_groups == 0) { return result; }

    using value_fn      = grouped_value_fn<Source, Result, K == aggregation::SUM_OF_SQUARES>;
    auto const identity = Op::template identity<Result>();
    auto const value =
      value_fn{ctx.grouped.rows.data(), ctx.accessor<Source>(), identity, ctx.values.has_nulls()};
    auto const output = result->mutable_view().begin<Result>();
    if (!ctx.nullable) {
      reduce_groups(ctx.grouped,
                    cudf::detail::make_counting_transform_iterator(0, value),
                    output,
                    Op{},
                    identity,
                    stream,
                    mr.get_temporary_mr());
      return result;
    }

    // The validity of a group (any valid row) rides along with its value in one pass.
    rmm::device_uvector<bool> group_valid(ctx.num_groups, stream, mr.get_temporary_mr());
    auto const values = cudf::detail::make_counting_transform_iterator(
      0, grouped_valid_value_fn < Source, Result, K == aggregation::SUM_OF_SQUARES > {value});
    auto const outputs = cuda::transform_output_iterator{
      cuda::make_zip_iterator(output, group_valid.begin()), split_valid_value_fn<Result>{}};
    reduce_groups(ctx.grouped,
                  values,
                  outputs,
                  valid_value_op<Op, Result>{},
                  valid_value<Result>{identity, false},
                  stream,
                  mr.get_temporary_mr());
    auto [null_mask, null_count] = cudf::detail::valid_if(
      group_valid.begin(), group_valid.end(), cuda::std::identity{}, stream, mr);
    result->set_null_mask(std::move(null_mask), null_count);
    return result;
  }

  template <typename T>
    requires(is_reduction_supported<K, T>() &&
             (K == aggregation::ARGMIN || K == aggregation::ARGMAX))
  std::unique_ptr<column> operator()(reduction_context const& ctx,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
  {
    auto result = make_size_type_column(ctx, stream, mr);
    if (ctx.num_groups == 0) { return result; }

    // The grouped rows are the input row indices themselves, so reducing them with the
    // element comparator yields the input index of each group's extremum. The sentinel identity
    // loses against every valid row and is left in place for all-null groups.
    constexpr auto is_argmin = K == aggregation::ARGMIN;
    reduce_groups(ctx.grouped,
                  ctx.grouped.rows.begin(),
                  result->mutable_view().begin<size_type>(),
                  cudf::detail::element_argminmax_fn<rep_type_t<T>>{
                    ctx.d_values, ctx.values.has_nulls(), is_argmin},
                  is_argmin ? cudf::detail::ARGMIN_SENTINEL : cudf::detail::ARGMAX_SENTINEL,
                  stream,
                  mr.get_temporary_mr());
    set_group_null_mask(*result, ctx, stream, mr);
    return result;
  }

  template <typename T>
    requires(is_reduction_supported<K, T>() && K == aggregation::SUM_OVERFLOW)
  std::unique_ptr<column> operator()(reduction_context const& ctx,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
  {
    using Source      = rep_type_t<T>;
    using accumulator = cudf::reduction::detail::sum_overflow_result<Source>;

    auto sum_child = make_fixed_width_column(
      ctx.values_type, ctx.num_groups, mask_state::UNALLOCATED, stream, mr.get_output_mr());
    auto overflow_child = make_fixed_width_column(data_type{type_id::BOOL8},
                                                  ctx.num_groups,
                                                  mask_state::UNALLOCATED,
                                                  stream,
                                                  mr.get_output_mr());
    if (ctx.num_groups > 0) {
      auto const values = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_sum_overflow_fn<Source>{
          ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls()});
      auto const children = cuda::transform_output_iterator{
        cuda::make_zip_iterator(sum_child->mutable_view().begin<Source>(),
                                overflow_child->mutable_view().begin<bool>()),
        split_sum_overflow_fn<Source>{}};
      reduce_groups(ctx.grouped,
                    values,
                    children,
                    cudf::reduction::detail::overflow_sum_op<Source>{},
                    accumulator{},
                    stream,
                    mr.get_temporary_mr());
    }

    auto [null_mask, null_count] = ctx.nullable && ctx.num_groups > 0
                                     ? reduce_group_validity(ctx, stream, mr)
                                     : std::pair{rmm::device_buffer{}, size_type{0}};
    std::vector<std::unique_ptr<column>> children;
    children.push_back(std::move(sum_child));
    children.push_back(std::move(overflow_child));
    return create_structs_hierarchy(ctx.num_groups,
                                    std::move(children),
                                    null_count,
                                    std::move(null_mask),
                                    stream,
                                    mr.get_output_mr());
  }

  template <typename T>
    requires(!is_reduction_supported<K, T>())
  std::unique_ptr<column> operator()(reduction_context const&,
                                     cuda::stream_ref,
                                     cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for hash groupby aggregation");
  }
};

/// Computes the SUM, SUM_OF_SQUARES and COUNT_VALID aggregations requested on one column, as
/// extracted for MEAN, M2, VARIANCE and STD, with a single grouped reduction.
struct fused_sums_fn {
  template <typename T>
    requires(cudf::detail::is_product_supported<T>())
  std::vector<std::unique_ptr<column>> operator()(reduction_context const& ctx,
                                                  host_span<aggregation::Kind const> kinds,
                                                  std::span<int8_t const> is_intermediate,
                                                  cuda::stream_ref stream,
                                                  cudf::memory_resources mr) const
  {
    using Source = rep_type_t<T>;
    using Result = rep_type_t<cudf::detail::target_type_t<T, aggregation::SUM>>;
    static_assert(
      cuda::std::
        is_same_v<Result, rep_type_t<cudf::detail::target_type_t<T, aggregation::SUM_OF_SQUARES>>>);

    // Every sum is reduced, but only explicitly requested results use the output resource.
    auto const make_output = [&](aggregation::Kind kind) {
      auto const it        = std::find(kinds.begin(), kinds.end(), kind);
      auto const requested = it != kinds.end() && !is_intermediate[it - kinds.begin()];
      return make_fixed_width_column(cudf::detail::target_type(ctx.values_type, kind),
                                     ctx.num_groups,
                                     mask_state::UNALLOCATED,
                                     stream,
                                     requested ? mr.get_output_mr() : mr.get_temporary_mr());
    };
    auto sum            = make_output(aggregation::SUM);
    auto sum_of_squares = make_output(aggregation::SUM_OF_SQUARES);
    auto count          = make_output(aggregation::COUNT_VALID);
    auto const counts   = count->view().template begin<size_type>();
    if (ctx.num_groups > 0) {
      auto const values = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_fused_sums_fn<Source, Result>{
          ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls()});
      auto const outputs = cuda::transform_output_iterator{
        cuda::make_zip_iterator(sum->mutable_view().template begin<Result>(),
                                sum_of_squares->mutable_view().template begin<Result>(),
                                count->mutable_view().template begin<size_type>()),
        split_fused_sums_fn<Result>{}};
      reduce_groups(ctx.grouped,
                    values,
                    outputs,
                    fused_sums_plus<Result>{},
                    fused_sums<Result>{Result{0}, Result{0}, 0},
                    stream,
                    mr.get_temporary_mr());
    }

    std::vector<std::unique_ptr<column>> results;
    for (std::size_t i = 0; i < kinds.size(); ++i) {
      auto result = kinds[i] == aggregation::SUM              ? std::move(sum)
                    : kinds[i] == aggregation::SUM_OF_SQUARES ? std::move(sum_of_squares)
                                                              : std::move(count);
      // A sum is null when its group has no valid row, which the valid count already tells.
      auto const nullable =
        !is_intermediate[i] && kinds[i] != aggregation::COUNT_VALID && ctx.values.has_nulls();
      if (nullable && ctx.num_groups > 0) {
        auto [null_mask, null_count] = cudf::detail::valid_if(
          counts,
          counts + ctx.num_groups,
          [] __device__(size_type count) { return count > 0; },
          stream,
          mr);
        result->set_null_mask(std::move(null_mask), null_count);
      }
      results.push_back(std::move(result));
    }
    return results;
  }

  template <typename T>
    requires(!cudf::detail::is_product_supported<T>())
  std::vector<std::unique_ptr<column>> operator()(reduction_context const&,
                                                  host_span<aggregation::Kind const>,
                                                  std::span<int8_t const>,
                                                  cuda::stream_ref,
                                                  cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for fused hash groupby sums");
  }
};

/// Reduces consecutive MIN/MAX/SUM requests on one input with one value load per row.
struct fused_minmax_sum_fn {
  template <typename T>
    requires(is_reduction_supported<aggregation::SUM, T>() &&
             is_reduction_supported<aggregation::MIN, T>() &&
             is_reduction_supported<aggregation::MAX, T>())
  std::vector<std::unique_ptr<column>> operator()(reduction_context const& ctx,
                                                  host_span<aggregation::Kind const> kinds,
                                                  std::span<int8_t const> is_intermediate,
                                                  cuda::stream_ref stream,
                                                  cudf::memory_resources mr) const
  {
    using Source           = rep_type_t<T>;
    using Result           = rep_type_t<cudf::detail::target_type_t<T, aggregation::SUM>>;
    using Min              = cudf::detail::corresponding_operator_t<aggregation::MIN>;
    using Max              = cudf::detail::corresponding_operator_t<aggregation::MAX>;
    using Sum              = cudf::detail::corresponding_operator_t<aggregation::SUM>;
    auto const make_output = [&](aggregation::Kind kind) {
      auto const it        = std::find(kinds.begin(), kinds.end(), kind);
      auto const requested = it != kinds.end() && !is_intermediate[it - kinds.begin()];
      return make_fixed_width_column(cudf::detail::target_type(ctx.values_type, kind),
                                     ctx.num_groups,
                                     mask_state::UNALLOCATED,
                                     stream,
                                     requested ? mr.get_output_mr() : mr.get_temporary_mr());
    };
    auto minimum = make_output(aggregation::MIN);
    auto maximum = make_output(aggregation::MAX);
    auto sum     = make_output(aggregation::SUM);
    rmm::device_uvector<bool> group_valid(ctx.num_groups, stream, mr.get_temporary_mr());
    if (ctx.num_groups > 0) {
      auto const identity = fused_minmax_sum<Source, Result>{Min::template identity<Source>(),
                                                             Max::template identity<Source>(),
                                                             Sum::template identity<Result>(),
                                                             false};
      auto const values   = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_fused_minmax_sum_fn<Source, Result>{
          ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls(), identity});
      auto const outputs = cuda::transform_output_iterator{
        cuda::make_zip_iterator(minimum->mutable_view().template begin<Source>(),
                                maximum->mutable_view().template begin<Source>(),
                                sum->mutable_view().template begin<Result>(),
                                group_valid.begin()),
        split_fused_minmax_sum_fn<Source, Result>{}};
      reduce_groups(ctx.grouped,
                    values,
                    outputs,
                    fused_minmax_sum_op<Source, Result>{},
                    identity,
                    stream,
                    mr.get_temporary_mr());
    }
    std::vector<std::unique_ptr<column>> results;
    for (std::size_t i = 0; i < kinds.size(); ++i) {
      auto result = kinds[i] == aggregation::MIN   ? std::move(minimum)
                    : kinds[i] == aggregation::MAX ? std::move(maximum)
                                                   : std::move(sum);
      if (!is_intermediate[i] && ctx.values.has_nulls() && ctx.num_groups > 0) {
        auto [null_mask, null_count] = cudf::detail::valid_if(
          group_valid.begin(), group_valid.end(), cuda::std::identity{}, stream, mr);
        result->set_null_mask(std::move(null_mask), null_count);
      }
      results.push_back(std::move(result));
    }
    return results;
  }

  template <typename T>
    requires(!(is_reduction_supported<aggregation::SUM, T>() &&
               is_reduction_supported<aggregation::MIN, T>() &&
               is_reduction_supported<aggregation::MAX, T>()))
  std::vector<std::unique_ptr<column>> operator()(reduction_context const&,
                                                  host_span<aggregation::Kind const>,
                                                  std::span<int8_t const>,
                                                  cuda::stream_ref,
                                                  cudf::memory_resources) const
  {
    CUDF_FAIL("Unsupported type for fused hash groupby extrema and sum");
  }
};

template <aggregation::Kind K>
std::unique_ptr<column> compute_reduction(reduction_context const& ctx,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  return type_dispatcher(ctx.values_type, grouped_reduction_fn<K>{}, ctx, stream, mr);
}

}  // namespace cudf::groupby::detail::hash
