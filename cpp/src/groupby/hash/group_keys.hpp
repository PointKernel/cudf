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

#include <cstddef>
#include <optional>

namespace cudf::groupby::detail::hash {

/// The keys grouped by the HashCSR build.
struct grouped_keys {
  size_type num_groups;
  size_type num_grouped_rows;
  rmm::device_uvector<size_type> key_rows;       ///< One representative input row per group
  rmm::device_uvector<size_type> group_offsets;  ///< `num_groups + 1` offsets into `grouped_rows`
  rmm::device_uvector<size_type> grouped_rows;   ///< Input rows reordered so groups are contiguous
};

/**
 * @brief Groups input rows with HashCSR using preprocessed row operators.
 *
 * Group offsets are built only when requested, and grouped rows additionally require the offsets.
 * When grouped rows are requested, `stable_rows` retains their original order within each group.
 * `domain_capacity` bounds the table capacity for keys drawn from a small finite domain.
 */
template <typename Equal, typename Hash>
grouped_keys group_keys(size_type num_rows,
                        bitmask_type const* row_bitmask,
                        Equal const& d_row_equal,
                        Hash const& d_row_hash,
                        bool need_group_offsets,
                        bool need_grouped_rows,
                        std::optional<std::size_t> domain_capacity,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr,
                        bool stable_rows = false);

/**
 * @brief Groups input keys with HashCSR, optionally materializing group offsets and grouped rows.
 *
 * When grouped rows are requested, `stable_rows` retains their original order within each group.
 */
grouped_keys group_keys(table_view const& keys,
                        null_policy include_null_keys,
                        bool need_group_offsets,
                        bool need_grouped_rows,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr,
                        bool stable_rows = false);

}  // namespace cudf::groupby::detail::hash
