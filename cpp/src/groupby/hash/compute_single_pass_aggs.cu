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
#include <cuda/std/functional>
#include <cuda/stream>
#include <thrust/adjacent_difference.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>

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
                mr.get_temporary_mr());
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
                  mr.get_temporary_mr());
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
  auto const num_groups = static_cast<size_type>(offsets.size() - 1);
  grouped_rows grouped{
    rows, offsets, rmm::device_uvector<size_type>{rows.size(), stream, mr.get_output_mr()}};
  if (num_groups == 0) { return grouped; }

  // Every group is nonempty, so the interior offsets identify distinct run starts.
  auto const policy = rmm::exec_policy_nosync(stream, mr.get_temporary_mr());
  thrust::fill(policy, grouped.labels.begin(), grouped.labels.end(), size_type{0});
  thrust::for_each(
    policy,
    offsets.begin() + 1,
    offsets.end() - 1,
    [labels = grouped.labels.data()] __device__(size_type offset) { labels[offset] = 1; });
  thrust::inclusive_scan(
    policy, grouped.labels.begin(), grouped.labels.end(), grouped.labels.begin());
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

  std::vector<std::unique_ptr<column>> results;
  results.reserve(num_aggs);
  for (std::size_t i = 0; i < num_aggs;) {
    auto const& col  = values.column(i);
    auto const d_col = column_device_view::create(col, stream, mr.get_temporary_mr());
    auto const values_type =
      is_dictionary(col.type()) ? dictionary_column_view(col).keys().type() : col.type();
    auto const kind = agg_kinds[i];
    // Counts are never null, and intermediate results skip the null mask to avoid the extra work.
    auto const nullable = !is_agg_intermediate[i] && kind != aggregation::COUNT_VALID &&
                          kind != aggregation::COUNT_ALL && col.has_nulls();
    auto const ctx = reduction_context{col, *d_col, values_type, grouped, num_groups, nullable};

    auto const end = fused_end(i, values_type);
    if (end > i + 1) {
      auto fused =
        compute_fused_sums(ctx,
                           host_span<aggregation::Kind const>{agg_kinds}.subspan(i, end - i),
                           is_agg_intermediate.subspan(i, end - i),
                           stream,
                           mr);
      std::move(fused.begin(), fused.end(), std::back_inserter(results));
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
