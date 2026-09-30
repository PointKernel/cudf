/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "row_equality.hpp"

#include <cudf/detail/copy.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_select.cuh>
#include <cuda/iterator>
#include <cuda/stream>

#include <vector>

namespace cudf {
namespace detail {
namespace {
struct keep_row {
  bitmask_type const* mismatches;
  size_type num_rows;
  duplicate_keep_option keep;

  __device__ bool operator()(size_type i) const
  {
    auto const first =
      keep == duplicate_keep_option::KEEP_LAST || i == 0 || bit_is_set(mismatches, i);
    auto const last = keep == duplicate_keep_option::KEEP_FIRST || i == num_rows - 1 ||
                      bit_is_set(mismatches, i + 1);
    return first && last;
  }
};
}  // namespace

std::unique_ptr<table> unique(table_view const& input,
                              std::vector<size_type> const& keys,
                              duplicate_keep_option keep,
                              null_equality nulls_equal,
                              cuda::stream_ref stream,
                              memory_resources mr)
{
  if (keep == duplicate_keep_option::KEEP_ANY) { keep = duplicate_keep_option::KEEP_FIRST; }
  auto const num_rows = input.num_rows();
  if (num_rows == 0 || input.num_columns() == 0 || keys.empty()) { return empty_like(input); }

  auto const temp_mr = mr.get_temporary_mr();
  auto indices       = rmm::device_uvector<size_type>(num_rows, stream, temp_mr);
  size_type output_size{};
  {
    auto const mismatches = adjacent_row_mismatches(
      input.select(keys), nulls_equal, stream, memory_resources{temp_mr, temp_mr});
    auto count           = device_scalar<size_type>(stream, temp_mr);
    auto rows            = cuda::counting_iterator<size_type>{0};
    auto const predicate = keep_row{mismatches.data(), num_rows, keep};
    std::size_t scratch_bytes{};
    CUDF_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                        scratch_bytes,
                                        rows,
                                        indices.data(),
                                        count.data(),
                                        num_rows,
                                        predicate,
                                        stream.get()));
    auto scratch = rmm::device_buffer(scratch_bytes, stream, temp_mr);
    CUDF_CUDA_TRY(cub::DeviceSelect::If(scratch.data(),
                                        scratch_bytes,
                                        rows,
                                        indices.data(),
                                        count.data(),
                                        num_rows,
                                        predicate,
                                        stream.get()));
    output_size = count.value(stream);
  }
  return detail::gather(
    input,
    device_span<size_type const>{indices.data(), static_cast<std::size_t>(output_size)},
    out_of_bounds_policy::DONT_CHECK,
    negative_index_policy::NOT_ALLOWED,
    stream,
    mr);
}
}  // namespace detail

std::unique_ptr<table> unique(table_view const& input,
                              std::vector<size_type> const& keys,
                              duplicate_keep_option keep,
                              null_equality nulls_equal,
                              cuda::stream_ref stream,
                              memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::unique(input, keys, keep, nulls_equal, stream, mr);
}
}  // namespace cudf
