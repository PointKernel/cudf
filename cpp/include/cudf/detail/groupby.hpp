/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <memory>
#include <utility>

namespace cudf {
namespace groupby::detail::hash {
/**
 * @brief Indicates if a set of aggregation requests can be satisfied with a
 * direct HashCSR reductions.
 *
 * @param requests The set of columns to aggregate and the aggregations to
 * perform
 * @return Whether every request supports the direct reductions
 */
bool can_use_single_pass_aggregations(std::span<aggregation_request const> requests);

// Hash-based groupby
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> groupby(
  table_view const& keys,
  std::span<aggregation_request const> requests,
  null_policy include_null_keys,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);
}  // namespace groupby::detail::hash
}  // namespace cudf
