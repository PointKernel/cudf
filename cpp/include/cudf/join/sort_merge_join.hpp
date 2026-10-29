/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <concepts>
#include <memory>
#include <type_traits>

/**
 * @file
 * @brief Class implementing the sort-merge algorithm for table joins
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 */

// Forward declaration
namespace detail {
class sort_merge_join;
}

/**
 * @brief Class that implements sort-merge algorithm for table joins
 *
 * This class enables the sort-merge join scheme that builds a preprocessed right table once,
 * and probes as many times as needed. All join methods (`inner_join()`, `left_join()`,
 * `inner_join_match_context()`, and `partitioned_inner_join()`) are thread-safe and can be
 * called concurrently from multiple threads on the same instance.
 */
class sort_merge_join {
 public:
  sort_merge_join() = delete;
  ~sort_merge_join();
  sort_merge_join(sort_merge_join const&)            = delete;
  sort_merge_join(sort_merge_join&&)                 = delete;
  sort_merge_join& operator=(sort_merge_join const&) = delete;
  sort_merge_join& operator=(sort_merge_join&&)      = delete;

  /**
   * @brief Construct a sort-merge join object that pre-processes the right table
   * on creation, and can be used on subsequent join operations with multiple
   * left tables.
   *
   * @note The `sort_merge_join` object must not outlive the table viewed by `right`,
   * else behavior is undefined.
   *
   * @param right The right table
   * @param is_right_sorted Enum to indicate if right table is pre-sorted
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  sort_merge_join(table_view const& right,
                  sorted is_right_sorted,
                  null_equality compare_nulls = null_equality::EQUAL,
                  cuda::stream_ref stream     = cudf::get_default_stream());

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * an inner join between the right table passed while creating the sort_merge_join object, and the
   * left table.
   * @see cudf::inner_join().
   *
   * This method is thread-safe and can be called concurrently from multiple threads.
   *
   * @param left The left table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device
   * memory.
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] that can be used to
   * construct the result of performing an inner join between two tables
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(table_view const& left,
             cuda::stream_ref stream           = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns the row indices that can be used to construct the result of performing
   * a left join between the right table passed while creating the sort_merge_join object, and the
   * left table.
   * @see cudf::left_join().
   *
   * This method is thread-safe and can be called concurrently from multiple threads.
   *
   * @param left The left table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the join indices' device memory
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] that can be used to
   * construct the result of performing a left join between two tables
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(table_view const& left,
            cuda::stream_ref stream           = cudf::get_default_stream(),
            rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Creates a join_match_context for an inner join.
   *
   * See join_match_context for lifetime and stream requirements.
   *
   * @param left The left join-key table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate retained match state
   *
   * @return Match context for partitioned_inner_join()
   */
  std::unique_ptr<join_match_context> inner_join_match_context(
    table_view const& left,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns inner-join indices for a partition of the left table.
   *
   * Left indices refer to the original left table. The context can be reused for other
   * partitions.
   *
   * @throw std::invalid_argument If match counts are missing or have the wrong size,
   * the bounds are invalid, or the context was not created by this join object.
   * Also thrown if `context.left_table_context` is null.
   *
   * @param context Match context and left-row bounds
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned indices
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`]
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_inner_join(
    cudf::join_partition_context const& context,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns inner-join indices for a partition of the left table.
   *
   * @tparam Resources cudf::memory_resources
   *
   * @param context Match context and left-row bounds
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param resources Memory resources for output and temporary allocations
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`]
   */
  template <typename Resources>
    requires std::same_as<std::remove_cvref_t<Resources>, cudf::memory_resources>
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_inner_join(join_partition_context const& context,
                         cuda::stream_ref stream,
                         Resources&& resources) const
  {
    return partitioned_join(context, stream, resources);
  }

  /**
   * @brief Creates a join_match_context for an inner join.
   *
   * @tparam Resources cudf::memory_resources
   *
   * @param left The left join-key table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param resources Memory resources for output and temporary allocations
   *
   * @return Match context for partitioned_inner_join()
   */
  template <typename Resources>
    requires std::same_as<std::remove_cvref_t<Resources>, cudf::memory_resources>
  [[nodiscard]] std::unique_ptr<join_match_context> inner_join_match_context(
    table_view const& left, cuda::stream_ref stream, Resources&& resources) const
  {
    return make_match_context(left, stream, resources);
  }

  /**
   * @brief Returns inner-join indices for a partition of the left table.
   *
   * Left indices refer to the original left table. The context can be reused for other
   * partitions.
   *
   * @throw std::invalid_argument If match counts are missing or have the wrong size,
   * the bounds are invalid, or the context was not created by this join object.
   *
   * @param matches Match context for an inner join
   * @param left_start_idx First left row, inclusive
   * @param left_end_idx Last left row, exclusive
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param resources Memory resources for output and temporary allocations
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`]
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_inner_join(
    join_match_context const& matches,
    size_type left_start_idx,
    size_type left_end_idx,
    cuda::stream_ref stream          = cudf::get_default_stream(),
    cudf::memory_resources resources = cudf::get_current_device_resource_ref()) const;

 private:
  std::unique_ptr<join_match_context> make_match_context(table_view const& left,
                                                         cuda::stream_ref stream,
                                                         cudf::memory_resources resources) const;
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_join(join_partition_context const& context,
                   cuda::stream_ref stream,
                   cudf::memory_resources resources) const;

  using impl_type = cudf::detail::sort_merge_join;  ///< Implementation type
  std::unique_ptr<impl_type const> _impl;           ///< Pointer to implementation
};

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
