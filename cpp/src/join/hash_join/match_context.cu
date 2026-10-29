/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "common.cuh"
#include "dispatch.cuh"
#include "hash_csr_kernels.cuh"
#include "join/join_match_state.hpp"

#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/types.hpp>

#include <cuda/iterator>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>

namespace cudf::detail {
namespace {
struct padded_match_counts {
  size_type const* counts;
  size_type rows;
  CUDF_HOST_DEVICE size_type operator()(std::size_t i) const
  {
    return i < static_cast<std::size_t>(rows) ? counts[i] : size_type{0};
  }
};

struct hash_join_match_state final : join_match_state {
  hash_join_match_state(void const* owner,
                        join_kind kind,
                        std::size_t size,
                        cuda::device_buffer<size_type> groups)
    : join_match_state{owner, kind, size}, groups{std::move(groups)}
  {
  }
  cuda::device_buffer<size_type> groups;
};
}  // namespace

template <typename Hasher>
cudf::join_match_context hash_join<Hasher>::make_match_context(
  join_kind kind,
  table_view const& left,
  cuda::stream_ref stream,
  cudf::memory_resources resources) const
{
  CUDF_FUNC_RANGE();
  validate_hash_join_probe(_right, left, _has_nulls);
  auto const output_mr = resources.get_output_mr();
  auto const temp_mr   = resources.get_temporary_mr();
  rmm::device_uvector<size_type> counts(left.num_rows(), stream, output_mr);
  cuda::device_buffer<size_type> groups(
    stream, output_mr, _is_empty ? 0 : left.num_rows(), cuda::no_init);
  auto const outer = kind != join_kind::INNER_JOIN;
  std::size_t total{};
  if (_is_empty) {
    thrust::fill(rmm::exec_policy_nosync(stream, temp_mr),
                 counts.begin(),
                 counts.end(),
                 outer ? size_type{1} : size_type{0});
    total = outer ? static_cast<std::size_t>(left.num_rows()) : 0;
    // Match counts must also be ready for retrieval on another stream.
    stream.sync();
  } else {
    auto const preprocessed_left =
      cudf::detail::row::equality::preprocessed_table::create(left, stream, temp_mr);
    auto const row_bitmask =
      cudf::detail::bitmask_and(left, stream, resources.as_temporary()).first;
    auto const valid_rows = _nulls_equal == null_equality::UNEQUAL ? row_bitmask.data() : nullptr;
    cudf::detail::device_scalar<cuda::std::uint64_t> total_matches{0, stream, temp_mr};
    auto count_matches = [&](auto equality, auto hasher) {
      dispatch_bool(outer, [&](auto is_outer) {
        launch_hash_csr_probe_count_kernel<decltype(is_outer)::value>(left.num_rows(),
                                                                      valid_rows,
                                                                      groups.data(),
                                                                      counts.data(),
                                                                      total_matches.data(),
                                                                      nullptr,
                                                                      nullptr,
                                                                      _impl->hash_table(),
                                                                      _impl->csr(),
                                                                      equality,
                                                                      hasher,
                                                                      stream);
      });
    };
    dispatch_join_comparator(_right,
                             left,
                             _preprocessed_right,
                             preprocessed_left,
                             _has_nulls,
                             _nulls_equal,
                             count_matches);
    total = total_matches.value(stream);
  }
  return cudf::join_match_context{
    left,
    std::make_unique<rmm::device_uvector<size_type>>(std::move(counts)),
    std::make_unique<hash_join_match_state>(this, kind, total, std::move(groups))};
}

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::partitioned_join_retrieve(join_kind kind,
                                             cudf::join_match_context const& matches,
                                             size_type left_start_idx,
                                             size_type left_end_idx,
                                             cuda::stream_ref stream,
                                             cudf::memory_resources resources) const
{
  CUDF_FUNC_RANGE();
  auto const counts_view = matches.match_counts();
  auto const left_rows   = matches.left_table().num_rows();
  CUDF_EXPECTS(counts_view.size() == static_cast<std::size_t>(left_rows),
               "Match count size does not match left table",
               std::invalid_argument);
  CUDF_EXPECTS(left_start_idx >= 0 && left_end_idx >= left_start_idx && left_end_idx <= left_rows,
               "Invalid partition bounds",
               std::invalid_argument);
  if (matches._state == nullptr) {
    return retrieve_uncached_join(kind, matches, left_start_idx, left_end_idx, stream, resources);
  }
  matches._state->validate(this, kind);
  auto const& state    = static_cast<hash_join_match_state const&>(*matches._state);
  auto const rows      = left_end_idx - left_start_idx;
  auto const output_mr = resources.get_output_mr();
  auto const temp_mr   = resources.get_temporary_mr();
  if (rows == 0 || (_is_empty && kind == join_kind::INNER_JOIN)) {
    return {std::make_unique<rmm::device_uvector<size_type>>(0, stream, output_mr),
            std::make_unique<rmm::device_uvector<size_type>>(0, stream, output_mr)};
  }
  if (_is_empty) {
    auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(rows, stream, output_mr);
    auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(rows, stream, output_mr);
    thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                       cuda::counting_iterator<size_type>{0},
                       rows,
                       [left  = left_indices->data(),
                        right = right_indices->data(),
                        left_start_idx] __device__(size_type i) -> void {
                         left[i]  = left_start_idx + i;
                         right[i] = cudf::JoinNoMatch;
                       });
    return {std::move(left_indices), std::move(right_indices)};
  }

  cuda::device_buffer<cuda::std::int64_t> offsets(
    stream, temp_mr, static_cast<std::size_t>(rows) + 1, cuda::no_init);
  // Append a virtual zero instead of copying the counts into a padded buffer.
  auto const counts    = counts_view.data() + left_start_idx;
  auto const input     = cuda::transform_iterator(cuda::counting_iterator<std::size_t>{0},
                                              padded_match_counts{counts, rows});
  auto const input_end = input + static_cast<std::size_t>(rows) + 1;
  auto const size      = [&] {
    if (left_start_idx == 0 && left_end_idx == left_rows) {
      // Preparation already synchronized to obtain this total. Only build the offsets here.
      thrust::exclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                             input,
                             input_end,
                             offsets.begin(),
                             cuda::std::int64_t{0});
      return state.output_size;
    }
    return static_cast<std::size_t>(cudf::detail::sizes_to_offsets(
      input, input_end, offsets.begin(), 0, stream, resources.as_temporary()));
  }();
  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(size, stream, output_mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(size, stream, output_mr);
  auto const groups  = state.groups.data() + left_start_idx;
  if (kind == join_kind::INNER_JOIN) {
    launch_hash_csr_inner_retrieve_kernel(size,
                                          rows,
                                          offsets.data(),
                                          groups,
                                          _impl->csr(),
                                          left_start_idx,
                                          left_indices->data(),
                                          right_indices->data(),
                                          stream);
  } else {
    launch_hash_csr_outer_retrieve_kernel(size,
                                          rows,
                                          offsets.data(),
                                          groups,
                                          _impl->csr(),
                                          left_start_idx,
                                          left_indices->data(),
                                          right_indices->data(),
                                          stream);
  }
  return {std::move(left_indices), std::move(right_indices)};
}

template cudf::join_match_context hash_join<hash_join_hasher>::make_match_context(
  join_kind, table_view const&, cuda::stream_ref, cudf::memory_resources) const;
template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::partitioned_join_retrieve(join_kind,
                                                       cudf::join_match_context const&,
                                                       size_type,
                                                       size_type,
                                                       cuda::stream_ref,
                                                       cudf::memory_resources) const;
}  // namespace cudf::detail
