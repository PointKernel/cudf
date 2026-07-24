/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::detail {

size_type unique_flat(table_view const& keys,
                      mutable_column_device_view& output,
                      duplicate_keep_option keep,
                      null_equality nulls_equal,
                      rmm::cuda_stream_view stream);

size_type unique_nested(table_view const& keys,
                        mutable_column_device_view& output,
                        duplicate_keep_option keep,
                        null_equality nulls_equal,
                        rmm::cuda_stream_view stream);

}  // namespace cudf::detail
