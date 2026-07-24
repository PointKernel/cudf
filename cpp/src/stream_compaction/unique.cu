/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "unique_helpers.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <utility>
#include <vector>

namespace cudf {
namespace detail {
std::unique_ptr<table> unique(table_view const& input,
                              std::vector<size_type> const& keys,
                              duplicate_keep_option keep,
                              null_equality nulls_equal,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  // If keep is KEEP_ANY, just alias it to KEEP_FIRST.
  if (keep == duplicate_keep_option::KEEP_ANY) { keep = duplicate_keep_option::KEEP_FIRST; }

  auto const num_rows = input.num_rows();
  if (num_rows == 0 or input.num_columns() == 0 or keys.empty()) { return empty_like(input); }

  auto unique_indices = make_numeric_column(
    data_type{type_to_id<size_type>()}, num_rows, mask_state::UNALLOCATED, stream, mr);
  auto mutable_view = mutable_column_device_view::create(*unique_indices, stream);
  auto keys_view    = input.select(keys);

  size_type const unique_size =
    cudf::detail::has_nested_columns(keys_view)
      ? unique_nested(keys_view, *mutable_view, keep, nulls_equal, stream)
      : unique_flat(keys_view, *mutable_view, keep, nulls_equal, stream);
  auto indices_view = cudf::detail::slice(column_view(*unique_indices), 0, unique_size, stream);

  // gather unique rows and return
  return detail::gather(input,
                        indices_view,
                        out_of_bounds_policy::DONT_CHECK,
                        negative_index_policy::NOT_ALLOWED,
                        stream,
                        mr);
}
}  // namespace detail

std::unique_ptr<table> unique(table_view const& input,
                              std::vector<size_type> const& keys,
                              duplicate_keep_option const keep,
                              null_equality nulls_equal,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::unique(input, keys, keep, nulls_equal, stream, mr);
}

}  // namespace cudf
