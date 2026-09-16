/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_single_pass_aggs.hpp"
#include "single_pass_reductions.cuh"
#include "single_pass_reductions.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/stream>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/partition.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {

/// A group is valid when any of its rows is valid.
std::pair<rmm::device_buffer, size_type> reduce_group_validity(reduction_context const& ctx,
                                                               cuda::stream_ref stream,
                                                               cudf::memory_resources mr)
{
  rmm::device_uvector<bool> group_valid(ctx.num_groups, stream, mr.get_temporary_mr());
  reduce_groups(ctx.grouped,
                cuda::make_permutation_iterator(cudf::detail::make_validity_iterator(ctx.d_values),
                                                ctx.grouped.rows.begin()),
                group_valid.begin(),
                cuda::std::logical_or<bool>{},
                false,
                stream,
                mr);
  return cudf::detail::valid_if(
    group_valid.begin(), group_valid.end(), cuda::std::identity{}, stream, mr);
}

void set_group_null_mask(column& result,
                         reduction_context const& ctx,
                         cuda::stream_ref stream,
                         cudf::memory_resources mr)
{
  if (!ctx.nullable || ctx.num_groups == 0) { return; }
  auto [null_mask, null_count] = reduce_group_validity(ctx, stream, mr);
  result.set_null_mask(std::move(null_mask), null_count);
}

std::unique_ptr<column> make_size_type_column(reduction_context const& ctx,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources mr)
{
  return make_fixed_width_column(data_type{type_to_id<size_type>()},
                                 ctx.num_groups,
                                 mask_state::UNALLOCATED,
                                 stream,
                                 mr.get_output_mr());
}

std::unique_ptr<column> count_groups(reduction_context const& ctx,
                                     bool valid_only,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  auto result = make_size_type_column(ctx, stream, mr);
  if (ctx.num_groups == 0) { return result; }

  if (valid_only && ctx.values.has_nulls()) {
    auto const valid_counts = cuda::transform_iterator{
      cuda::make_permutation_iterator(cudf::detail::make_validity_iterator(ctx.d_values),
                                      ctx.grouped.rows.begin()),
      [] __device__(bool valid) -> size_type { return static_cast<size_type>(valid); }};
    reduce_groups(ctx.grouped,
                  valid_counts,
                  result->mutable_view().begin<size_type>(),
                  cuda::std::plus<size_type>{},
                  size_type{0},
                  stream,
                  mr);
  } else {
    thrust::adjacent_difference(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                                ctx.grouped.offsets.begin() + 1,
                                ctx.grouped.offsets.end(),
                                result->mutable_view().begin<size_type>());
  }
  return result;
}

/// Calls `f.template operator()<K>(args...)` for the reduction kind `kind`.
template <typename F, typename... Args>
auto dispatch_reduction_kind(aggregation::Kind kind, F&& f, Args&&... args)
{
  switch (kind) {
    case aggregation::SUM:
      return f.template operator()<aggregation::SUM>(std::forward<Args>(args)...);
    case aggregation::PRODUCT:
      return f.template operator()<aggregation::PRODUCT>(std::forward<Args>(args)...);
    case aggregation::SUM_OF_SQUARES:
      return f.template operator()<aggregation::SUM_OF_SQUARES>(std::forward<Args>(args)...);
    case aggregation::MIN:
      return f.template operator()<aggregation::MIN>(std::forward<Args>(args)...);
    case aggregation::MAX:
      return f.template operator()<aggregation::MAX>(std::forward<Args>(args)...);
    case aggregation::ARGMIN:
      return f.template operator()<aggregation::ARGMIN>(std::forward<Args>(args)...);
    case aggregation::ARGMAX:
      return f.template operator()<aggregation::ARGMAX>(std::forward<Args>(args)...);
    case aggregation::SUM_OVERFLOW:
      return f.template operator()<aggregation::SUM_OVERFLOW>(std::forward<Args>(args)...);
    default: CUDF_FAIL("Unsupported hash groupby aggregation");
  }
}

struct compute_reduction_fn {
  reduction_context const& ctx;

  template <aggregation::Kind K>
  std::unique_ptr<column> operator()(cuda::stream_ref stream, cudf::memory_resources mr) const
  {
    return compute_reduction<K>(ctx, stream, mr);
  }
};

struct compute_reductions_fn {
  host_span<reduction_context const> contexts;
  std::span<int8_t const> is_intermediate;

  template <aggregation::Kind K>
  std::vector<std::unique_ptr<column>> operator()(cuda::stream_ref stream,
                                                  cudf::memory_resources mr) const
  {
    if constexpr (K == aggregation::SUM || K == aggregation::SUM_OF_SQUARES ||
                  K == aggregation::PRODUCT || K == aggregation::MIN || K == aggregation::MAX) {
      return compute_reductions<K>(contexts, is_intermediate, stream, mr);
    } else {
      CUDF_FAIL("Unsupported batched hash groupby aggregation");
    }
  }
};

template <aggregation::Kind K>
struct is_reduction_supported_fn {
  template <typename T>
  bool operator()() const
  {
    return is_reduction_supported<K, T>();
  }
};

struct is_reduction_kind_supported_fn {
  data_type values_type;

  template <aggregation::Kind K>
  bool operator()() const
  {
    return type_dispatcher(values_type, is_reduction_supported_fn<K>{});
  }
};

std::unique_ptr<column> compute_aggregation(aggregation::Kind kind,
                                            reduction_context const& ctx,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  switch (kind) {
    case aggregation::COUNT_VALID: return count_groups(ctx, true, stream, mr);
    case aggregation::COUNT_ALL: return count_groups(ctx, false, stream, mr);
    default: return dispatch_reduction_kind(kind, compute_reduction_fn{ctx}, stream, mr);
  }
}

bool is_single_pass_agg_supported(data_type values_type, aggregation::Kind kind)
{
  // Values of STRUCT and LIST types are not aggregated by the hash groupby.
  if (cudf::is_nested(values_type)) { return false; }
  switch (kind) {
    case aggregation::COUNT_VALID:
    case aggregation::COUNT_ALL: return true;
    case aggregation::SUM:
    case aggregation::PRODUCT:
    case aggregation::SUM_OF_SQUARES:
    case aggregation::MIN:
    case aggregation::MAX:
    case aggregation::ARGMIN:
    case aggregation::ARGMAX:
    case aggregation::SUM_OVERFLOW:
      return dispatch_reduction_kind(kind, is_reduction_kind_supported_fn{values_type});
    default: return false;
  }
}

grouped_rows make_grouped_rows(device_span<size_type const> rows,
                               device_span<size_type const> offsets,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  auto const temp_mr    = mr.get_temporary_mr();
  auto const num_rows   = static_cast<size_type>(rows.size());
  auto const num_groups = static_cast<size_type>(offsets.size() - 1);
  grouped_rows grouped{
    rows,
    offsets,
    rmm::device_uvector<size_type>{0, stream, mr.get_output_mr()},
    rmm::device_uvector<size_type>{0, stream, mr.get_output_mr()},
    rmm::device_uvector<cuda::std::array<size_type, 2>>{0, stream, mr.get_output_mr()},
    rmm::device_uvector<size_type>{0, stream, mr.get_output_mr()}};
  // Included groups are nonempty, so equality proves every included group is a singleton.
  if (num_groups == 0 || num_groups == num_rows) { return grouped; }

  auto const policy     = rmm::exec_policy_nosync(stream, temp_mr);
  auto const group_ids  = cuda::counting_iterator<size_type>{0};
  auto const needs_warp = [offsets = offsets.begin()] __device__(size_type group) {
    return offsets[group + 1] - offsets[group] > cudf::detail::warp_size;
  };
  // Only groups wider than a warp need compact IDs and warp work.
  auto const num_warp_groups =
    static_cast<size_type>(thrust::count_if(policy, group_ids, group_ids + num_groups, needs_warp));
  if (num_warp_groups == 0) { return grouped; }
  grouped.warp_groups.resize(num_warp_groups, stream);
  thrust::copy_if(
    policy, group_ids, group_ids + num_groups, grouped.warp_groups.begin(), needs_warp);
  auto const long_groups =
    thrust::partition(policy,
                      grouped.warp_groups.begin(),
                      grouped.warp_groups.end(),
                      [offsets = offsets.begin()] __device__(size_type group) {
                        return offsets[group + 1] - offsets[group] <= rows_per_chunk;
                      });
  auto const num_long_groups = static_cast<size_type>(grouped.warp_groups.end() - long_groups);
  if (num_long_groups == 0) { return grouped; }

  auto const chunk_counts = cudf::detail::make_counting_transform_iterator(
    0, [offsets = offsets.begin(), long_groups] __device__(size_type index) -> size_type {
      auto const group = long_groups[index];
      return cudf::util::div_rounding_up_safe(offsets[group + 1] - offsets[group], rows_per_chunk);
    });
  grouped.group_chunks.resize(static_cast<std::size_t>(num_long_groups) + 1, stream);
  grouped.group_chunks.set_element_to_zero_async(0, stream);
  thrust::inclusive_scan(
    policy, chunk_counts, chunk_counts + num_long_groups, grouped.group_chunks.begin() + 1);
  auto const num_chunks = grouped.group_chunks.back_element(stream);

  // Short groups leave gaps in CSR positions; store both endpoints of each long chunk.
  grouped.chunk_ranges.resize(num_chunks, stream);
  thrust::tabulate(
    policy,
    grouped.chunk_ranges.begin(),
    grouped.chunk_ranges.end(),
    [offsets = offsets.begin(),
     long_groups,
     group_chunks = grouped.group_chunks.begin(),
     group_chunks_end =
       grouped.group_chunks.end()] __device__(size_type chunk) -> cuda::std::array<size_type, 2> {
      auto const index = static_cast<size_type>(
        cuda::std::upper_bound(group_chunks, group_chunks_end, chunk) - group_chunks - 1);
      auto const group = long_groups[index];
      auto const begin =
        static_cast<cuda::std::int64_t>(offsets[group]) +
        static_cast<cuda::std::int64_t>(chunk - group_chunks[index]) * rows_per_chunk;
      auto const end =
        cuda::std::min(begin + rows_per_chunk, static_cast<cuda::std::int64_t>(offsets[group + 1]));
      return {static_cast<size_type>(begin), static_cast<size_type>(end)};
    });

  // Reuse a first-stored-row scheduling hint for long chunks across all value columns.
  rmm::device_uvector<size_type> first_rows(num_chunks, stream, temp_mr);
  auto const chunk_begins = cuda::transform_iterator{
    grouped.chunk_ranges.begin(),
    [] __device__(cuda::std::array<size_type, 2> const& range) -> size_type { return range[0]; }};
  thrust::gather(policy, chunk_begins, chunk_begins + num_chunks, rows.begin(), first_rows.begin());
  grouped.chunk_order.resize(num_chunks, stream);
  thrust::sequence(policy, grouped.chunk_order.begin(), grouped.chunk_order.end());
  thrust::sort_by_key(policy, first_rows.begin(), first_rows.end(), grouped.chunk_order.begin());
  return grouped;
}

std::vector<std::unique_ptr<column>> compute_single_pass_aggs(
  table_view const& values,
  host_span<aggregation::Kind const> agg_kinds,
  std::span<int8_t const> is_agg_intermediate,
  grouped_rows const& grouped,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  CUDF_EXPECTS(values.num_columns() == static_cast<size_type>(agg_kinds.size()),
               "The number of values columns and aggregation kinds must be the same.");
  CUDF_EXPECTS(values.num_columns() == static_cast<size_type>(is_agg_intermediate.size()),
               "The number of values columns and intermediate flags must be the same.");

  auto const num_groups = static_cast<size_type>(grouped.offsets.size() - 1);
  auto const num_aggs   = agg_kinds.size();

  // Returns one past the last of the consecutive additive aggregations on the column of `begin`
  // that can be computed together with the aggregation at `begin`.
  auto const fused_end = [&](std::size_t begin, data_type values_type) {
    auto const& col = values.column(begin);
    if (!is_fusable_sum(agg_kinds[begin]) ||
        !is_single_pass_agg_supported(values_type, aggregation::SUM_OF_SQUARES)) {
      return begin + 1;
    }
    auto end = begin + 1;
    while (end < num_aggs && is_fusable_sum(agg_kinds[end]) &&
           cudf::detail::is_shallow_equivalent(col, values.column(end)) &&
           std::find(agg_kinds.begin() + begin, agg_kinds.begin() + end, agg_kinds[end]) ==
             agg_kinds.begin() + end) {
      ++end;
    }
    return end;
  };

  auto const minmax_sum_end = [&](std::size_t begin, data_type values_type) {
    if (!is_fusable_minmax_sum(agg_kinds[begin]) ||
        !is_single_pass_agg_supported(values_type, aggregation::SUM) ||
        !is_single_pass_agg_supported(values_type, aggregation::MIN) ||
        !is_single_pass_agg_supported(values_type, aggregation::MAX)) {
      return begin + 1;
    }
    auto end = begin + 1;
    while (end < num_aggs && is_fusable_minmax_sum(agg_kinds[end]) &&
           cudf::detail::is_shallow_equivalent(values.column(begin), values.column(end)) &&
           std::find(agg_kinds.begin() + begin, agg_kinds.begin() + end, agg_kinds[end]) ==
             agg_kinds.begin() + end) {
      ++end;
    }
    auto const sum =
      std::find(agg_kinds.begin() + begin, agg_kinds.begin() + end, aggregation::SUM);
    auto const sum_index = static_cast<std::size_t>(sum - agg_kinds.begin());
    // Do not consume a SUM that already participates in the existing additive fusion.
    if (sum == agg_kinds.begin() + end || fused_end(sum_index, values_type) > sum_index + 1) {
      return begin + 1;
    }
    return end;
  };

  // Keep each same-input fused run intact; otherwise batch adjacent compatible reductions.
  auto const batch_end = [&](std::size_t begin, data_type values_type, bool nullable) {
    auto const kind = agg_kinds[begin];
    if (kind != aggregation::SUM && kind != aggregation::SUM_OF_SQUARES &&
        kind != aggregation::PRODUCT && kind != aggregation::MIN && kind != aggregation::MAX) {
      return begin + 1;
    }
    auto end = begin + 1;
    while (end < num_aggs && agg_kinds[end] == kind) {
      auto const& col = values.column(end);
      auto const type =
        is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();
      if (type != values_type || (!is_agg_intermediate[end] && col.has_nulls()) != nullable ||
          fused_end(end, type) > end + 1 || minmax_sum_end(end, type) > end + 1) {
        break;
      }
      ++end;
    }
    return end;
  };

  std::vector<std::unique_ptr<column>> results;
  results.reserve(num_aggs);
  for (std::size_t i = 0; i < num_aggs;) {
    auto const& col = values.column(i);
    auto d_col      = column_device_view::create(col, stream, mr.get_temporary_mr());
    auto const values_type =
      is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();
    auto const kind = agg_kinds[i];
    // Counts are never null, and intermediate results skip the null mask to avoid the extra work.
    auto const nullable = !is_agg_intermediate[i] && kind != aggregation::COUNT_VALID &&
                          kind != aggregation::COUNT_ALL && col.has_nulls();
    auto const ctx = reduction_context{col, *d_col, values_type, grouped, num_groups, nullable};

    auto const sums_end    = fused_end(i, values_type);
    auto const extrema_end = minmax_sum_end(i, values_type);
    auto end               = std::max(sums_end, extrema_end);
    if (end > i + 1) {
      auto const compute_fused =
        extrema_end > sums_end ? compute_fused_minmax_sum : compute_fused_sums;
      auto fused = compute_fused(ctx,
                                 host_span<aggregation::Kind const>{agg_kinds}.subspan(i, end - i),
                                 is_agg_intermediate.subspan(i, end - i),
                                 stream,
                                 mr);
      std::move(fused.begin(), fused.end(), std::back_inserter(results));
    } else if (end = batch_end(i, values_type, nullable); end > i + 1) {
      std::vector<decltype(d_col)> device_views;
      std::vector<reduction_context> contexts;
      device_views.reserve(end - i);
      contexts.reserve(end - i);
      device_views.push_back(std::move(d_col));
      contexts.push_back(ctx);
      for (auto j = i + 1; j < end; ++j) {
        auto const& next = values.column(j);
        device_views.push_back(column_device_view::create(next, stream, mr.get_temporary_mr()));
        contexts.push_back(
          {next, *device_views.back(), values_type, grouped, num_groups, nullable});
      }
      auto batch = dispatch_reduction_kind(
        kind, compute_reductions_fn{contexts, is_agg_intermediate.subspan(i, end - i)}, stream, mr);
      std::move(batch.begin(), batch.end(), std::back_inserter(results));
    } else {
      auto const resources = is_agg_intermediate[i] ? cudf::memory_resources{mr.get_temporary_mr(),
                                                                             mr.get_temporary_mr()}
                                                    : mr;
      results.push_back(compute_aggregation(kind, ctx, stream, resources));
    }
    i = end;
  }
  return results;
}

}  // namespace cudf::groupby::detail::hash
