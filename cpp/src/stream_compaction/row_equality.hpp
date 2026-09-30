/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

namespace cudf::detail {

/**
 * @brief Compare adjacent rows using host-dispatched, ahead-of-time column kernels.
 *
 * Bit i is set when rows i and i-1 differ. Bit zero is unused. NaNs compare equal.
 * Schema traversal and type selection happen on the host; device kernels only see typed
 * accessors and row coordinates. Nested columns expand coordinate pairs as necessary.
 * The returned mask uses the output resource and scratch uses the temporary resource.
 */
rmm::device_uvector<bitmask_type> adjacent_row_mismatches(table_view const& input,
                                                          null_equality nulls_equal,
                                                          cuda::stream_ref stream,
                                                          memory_resources resources);

}  // namespace cudf::detail
