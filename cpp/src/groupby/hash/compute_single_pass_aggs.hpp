/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf::groupby::detail::hash {
/**
 * @brief Computes all aggregations from `requests` that require a single pass
 * over the data and stores the results in `sparse_results`
 */
template <typename SetType>
rmm::device_uvector<cudf::size_type> compute_single_pass_aggs(
  int64_t num_rows,
  bitmask_type const* row_bitmask,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  cudf::detail::result_cache* sparse_results,
  SetType& global_set,
  bool skip_rows_with_nulls,
  rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
