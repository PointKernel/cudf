/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "stream_compaction_common.cuh"
#include "unique_helpers.cuh"

#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/device/dispatch/dispatch_select_if.cuh>

namespace cudf::detail {

// Limit copies of the state-heavy row comparator in each CUB agent while retaining stable,
// single-pass selection.
struct unique_flat_select_policy {
  struct Policy900 : cub::ChainedPolicy<900, Policy900, Policy900> {
    using SelectIfPolicyT = cub::AgentSelectIfPolicy<128,
                                                     5,
                                                     cub::BLOCK_LOAD_DIRECT,
                                                     cub::LOAD_DEFAULT,
                                                     cub::BLOCK_SCAN_WARP_SCANS,
                                                     cub::detail::no_delay_constructor_t<0>>;
  };

  using MaxPolicy = Policy900;
};

template <typename Predicate>
struct unique_flat_predicate {
  mutable Predicate predicate;

  __device__ bool operator()(size_type row) const { return predicate(row); }
};

size_type unique_flat(table_view const& keys,
                      mutable_column_device_view& output,
                      duplicate_keep_option keep,
                      null_equality nulls_equal,
                      rmm::cuda_stream_view stream)
{
  auto const comp = cudf::detail::row::equality::self_comparator(keys, stream);
  auto const row_equal =
    comp.equal_to<false>(nullate::DYNAMIC{has_nested_nulls(keys)}, nulls_equal);
  auto const begin        = cuda::counting_iterator<size_type>{0};
  auto const output_begin = output.begin<size_type>();
  auto const predicate = unique_flat_predicate{unique_copy_fn<decltype(begin), decltype(row_equal)>{
    begin, keep, row_equal, keys.num_rows() - 1}};
  cudf::detail::device_scalar<size_type> output_count(
    0, stream, cudf::get_current_device_resource_ref());

  using dispatch_t = cub::DispatchSelectIf<decltype(begin),
                                           cub::NullType*,
                                           decltype(output_begin),
                                           size_type*,
                                           decltype(predicate),
                                           cub::NullType,
                                           size_type,
                                           cub::SelectImpl::Select,
                                           unique_flat_select_policy>;
  std::size_t temporary_storage_bytes{};
  CUDF_CUDA_TRY(dispatch_t::Dispatch(nullptr,
                                     temporary_storage_bytes,
                                     begin,
                                     nullptr,
                                     output_begin,
                                     output_count.data(),
                                     predicate,
                                     cub::NullType{},
                                     keys.num_rows(),
                                     stream.value()));
  rmm::device_buffer temporary_storage(
    temporary_storage_bytes, stream, cudf::get_current_device_resource_ref());
  CUDF_CUDA_TRY(dispatch_t::Dispatch(temporary_storage.data(),
                                     temporary_storage_bytes,
                                     begin,
                                     nullptr,
                                     output_begin,
                                     output_count.data(),
                                     predicate,
                                     cub::NullType{},
                                     keys.num_rows(),
                                     stream.value()));
  return output_count.value(stream);
}

}  // namespace cudf::detail
