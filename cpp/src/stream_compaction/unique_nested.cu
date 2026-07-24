/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "stream_compaction_common.cuh"
#include "unique_helpers.cuh"

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace cudf::detail {

size_type unique_nested(table_view const& keys,
                        mutable_column_device_view& output,
                        duplicate_keep_option keep,
                        null_equality nulls_equal,
                        rmm::cuda_stream_view stream)
{
  auto const num_rows  = keys.num_rows();
  auto const comp      = cudf::detail::row::equality::self_comparator(keys, stream);
  auto const row_equal = comp.equal_to<true>(nullate::DYNAMIC{has_nested_nulls(keys)}, nulls_equal);
  auto d_results       = rmm::device_uvector<bool>(num_rows, stream);
  auto const begin     = cuda::counting_iterator<size_type>{0};
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    begin,
    begin + num_rows,
    d_results.begin(),
    unique_copy_fn<decltype(begin), decltype(row_equal)>{begin, keep, row_equal, num_rows - 1});
  auto const result_end = cudf::detail::copy_if(begin,
                                                begin + num_rows,
                                                d_results.begin(),
                                                output.begin<size_type>(),
                                                cuda::std::identity{},
                                                stream);
  return static_cast<size_type>(cuda::std::distance(output.begin<size_type>(), result_end));
}

}  // namespace cudf::detail
