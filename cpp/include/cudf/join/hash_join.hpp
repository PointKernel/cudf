/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/hashing.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

/**
 * @file
 * @brief Class definition for hash join, which builds a hash table with the right table and
 * probes it with left tables.
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 */

// forward declaration
namespace hashing::detail {
/**
 * @brief Forward declaration for our Murmur Hash 3 implementation
 */
template <typename T>
class MurmurHash3_x86_32;
}  // namespace hashing::detail

namespace detail {
/**
 * @brief Forward declaration for our hash join
 */
template <typename T>
class hash_join;
}  // namespace detail

/**
 * @brief Hash join that builds a hash table with the right table on construction and probes
 * results in subsequent `*_join` member functions.
 *
 * This class enables the hash join scheme that builds with the right table once and probes
 * with many left tables (possibly in parallel).
 */
class hash_join {
 public:
  using impl_type = typename cudf::detail::hash_join<
    cudf::hashing::detail::MurmurHash3_x86_32<cudf::hash_value_type>>;  ///< Implementation type

  hash_join() = delete;
  ~hash_join();
  hash_join(hash_join const&)            = delete;
  hash_join(hash_join&&)                 = delete;
  hash_join& operator=(hash_join const&) = delete;
  hash_join& operator=(hash_join&&)      = delete;

  /**
   * @brief Construct a hash join object for subsequent probe calls.
   *
   * @note The `hash_join` object must not outlive the table viewed by `right`, else behavior is
   * undefined.
   *
   * @throws std::invalid_argument if the right table has no columns
   *
   * @param right The right table, from which the hash table is built
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the internal hash table
   */
  hash_join(cudf::table_view const& right,
            null_equality compare_nulls,
            cuda::stream_ref stream = cudf::get_default_stream(),
            cuda::mr::any_resource<cuda::mr::device_accessible> mr =
              cudf::get_current_device_resource_ref());

  /**
   * @brief Construct a hash join object for subsequent probe calls.
   *
   * @note The `hash_join` object must not outlive the table viewed by `right`, else behavior is
   * undefined.
   *
   * @throws std::invalid_argument if the right table has no columns
   * @throws std::invalid_argument if load_factor is not greater than 0 and less than or equal to 1
   *
   * @param right The right table, from which the hash table is built
   * @param has_nulls Flag to indicate if there exists any nulls in the `right` table or
   *                  any `left` table that will be used later for join
   * @param compare_nulls Controls whether null join-key values should match or not
   * @param load_factor The hash table occupancy ratio in (0,1]. A value of 0.5 means 50% desired
   * occupancy.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the internal hash table
   */
  hash_join(cudf::table_view const& right,
            nullable_join has_nulls,
            null_equality compare_nulls,
            double load_factor,
            cuda::stream_ref stream = cudf::get_default_stream(),
            cuda::mr::any_resource<cuda::mr::device_accessible> mr =
              cudf::get_current_device_resource_ref());

  /**
   * Returns the row indices that can be used to construct the result of performing
   * an inner join between two tables. @see cudf::inner_join().
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   * @throw cudf::logic_error If `output_size` is provided and does not equal the actual output
   * size.
   *
   * @param left The left table, from which the tuples are probed
   * @param output_size Optional value which allows users to specify the exact output size.
   * Supplying it no longer avoids any work: the output size is always computed internally and
   * the supplied value is only validated against it.  Prefer omitting it; it will be deprecated
   * and removed in a future release.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing an inner join between two tables with `left` and `right`
   * as the join keys .
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(cudf::table_view const& left,
             std::optional<std::size_t> output_size = {},
             cuda::stream_ref stream                = cudf::get_default_stream(),
             rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * Returns the row indices that can be used to construct the result of performing
   * a left join between two tables. @see cudf::left_join().
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   * @throw cudf::logic_error If `output_size` is provided and does not equal the actual output
   * size.
   *
   * @param left The left table, from which the tuples are probed
   * @param output_size Optional value which allows users to specify the exact output size.
   * Supplying it no longer avoids any work: the output size is always computed internally and
   * the supplied value is only validated against it.  Prefer omitting it; it will be deprecated
   * and removed in a future release.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing a left join between two tables with `left` and `right`
   * as the join keys.
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(cudf::table_view const& left,
            std::optional<std::size_t> output_size = {},
            cuda::stream_ref stream                = cudf::get_default_stream(),
            rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref()) const;

  /**
   * Returns the row indices that can be used to construct the result of performing
   * a full join between two tables. @see cudf::full_join().
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   * @throw cudf::logic_error If `output_size` is provided and does not equal the actual output
   * size.
   *
   * @param left The left table, from which the tuples are probed
   * @param output_size Optional value which allows users to specify the exact output size.
   * Supplying it no longer avoids any work: the output size is always computed internally and
   * the supplied value is only validated against it.  Prefer omitting it; it will be deprecated
   * and removed in a future release.
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned table and columns' device
   * memory.
   *
   * @return A pair of columns [`left_indices`, `right_indices`] that can be used to construct
   * the result of performing a full join between two tables with `left` and `right`
   * as the join keys .
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  full_join(cudf::table_view const& left,
            std::optional<std::size_t> output_size = {},
            cuda::stream_ref stream                = cudf::get_default_stream(),
            rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref()) const;

  /**
   * Returns the exact number of matches (rows) when performing an inner join with the specified
   * left table.
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param left The left table, from which the tuples are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return The exact number of output when performing an inner join between two tables with
   * `left` and `right` as the join keys .
   */
  [[nodiscard]] std::size_t inner_join_size(
    cudf::table_view const& left, cuda::stream_ref stream = cudf::get_default_stream()) const;

  /**
   * Returns the exact number of matches (rows) when performing a left join with the specified left
   * table.
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param left The left table, from which the tuples are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @return The exact number of output when performing a left join between two tables with `left`
   * and `right` as the join keys .
   */
  [[nodiscard]] std::size_t left_join_size(
    cudf::table_view const& left, cuda::stream_ref stream = cudf::get_default_stream()) const;

  /**
   * Returns the exact number of matches (rows) when performing a full join with the specified left
   * table.
   *
   * @throw std::invalid_argument If the input left table has nulls while this hash_join object was
   * not constructed with null check.
   *
   * @param left The left table, from which the tuples are probed
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the intermediate table and columns' device
   * memory.
   *
   * @return The exact number of output when performing a full join between two tables with `left`
   * and `right` as the join keys .
   */
  [[nodiscard]] std::size_t full_join_size(
    cudf::table_view const& left,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Creates a join_match_context for an inner join.
   *
   * See join_match_context for lifetime and stream requirements.
   *
   * @throw std::invalid_argument If `left` has nulls but this join object was
   * constructed without null checks.
   *
   * @param left The left join-key table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate retained match state
   *
   * @return Match context for partitioned_inner_join()
   */
  [[nodiscard]] cudf::join_match_context inner_join_match_context(
    cudf::table_view const& left,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Creates a join_match_context for a left join.
   *
   * See join_match_context for lifetime and stream requirements.
   *
   * @throw std::invalid_argument If `left` has nulls but this join object was
   * constructed without null checks.
   *
   * @param left The left join-key table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate retained match state
   *
   * @return Match context for partitioned_left_join()
   */
  [[nodiscard]] cudf::join_match_context left_join_match_context(
    cudf::table_view const& left,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Creates a join_match_context for a full join.
   *
   * See join_match_context for lifetime and stream requirements.
   *
   * @throw std::invalid_argument If `left` has nulls but this join object was
   * constructed without null checks.
   *
   * @param left The left join-key table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate retained match state
   *
   * @return Match context for partitioned_full_join()
   */
  [[nodiscard]] cudf::join_match_context full_join_match_context(
    cudf::table_view const& left,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns inner-join indices for a partition of the left table.
   *
   * Left indices refer to the original left table. The context can be reused for other
   * partitions.
   *
   * @throw std::invalid_argument If match counts are missing or have the wrong size,
   * the bounds are invalid, or a cached context belongs to another join object or kind.
   * Also thrown if `context.left_table_context` is null.
   *
   * @param context Match context and left-row bounds
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned indices
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`]
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_inner_join(
    cudf::join_partition_context const& context,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns left-join indices for a partition of the left table.
   *
   * Left indices refer to the original left table. The context can be reused for other
   * partitions.
   *
   * @throw std::invalid_argument If match counts are missing or have the wrong size,
   * the bounds are invalid, or a cached context belongs to another join object or kind.
   * Also thrown if `context.left_table_context` is null.
   *
   * @param context Match context and left-row bounds
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned indices
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`]
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_left_join(
    cudf::join_partition_context const& context,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Returns full-join indices for a partition of the left table.
   *
   * Left indices refer to the original left table. The context can be reused for other
   * partitions.
   * Call finalize_partitioned_full_join() to append unmatched right rows.
   *
   * @throw std::invalid_argument If match counts are missing or have the wrong size,
   * the bounds are invalid, or a cached context belongs to another join object or kind.
   * Also thrown if `context.left_table_context` is null.
   *
   * @param context Match context and left-row bounds
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned indices
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`]
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_full_join(
    cudf::join_partition_context const& context,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Finalizes a partitioned full join by concatenating all per-partition results
   * and appending the unmatched right rows (the complement).
   *
   * Call this method after calling `partitioned_full_join()` for every partition.  It combines
   * the per-partition indices with the unmatched right row indices (a global property
   * across all partitions) and returns a single `(left_indices, right_indices)` pair equivalent
   * to the output of `full_join()`.
   *
   * @param left_partials Per-partition `left_indices` views produced by `partitioned_full_join()`
   * @param right_partials Per-partition `right_indices` views produced by `partitioned_full_join()`
   * @param left_table_num_rows Total number of rows in the original left table
   * @param right_table_num_rows Total number of rows in the right table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the result device memory
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`] representing the full
   *         join output.
   */
  [[nodiscard]] static std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                                 std::unique_ptr<rmm::device_uvector<size_type>>>
  finalize_partitioned_full_join(
    cudf::host_span<cudf::device_span<size_type const> const> left_partials,
    cudf::host_span<cudf::device_span<size_type const> const> right_partials,
    size_type left_table_num_rows,
    size_type right_table_num_rows,
    cuda::stream_ref stream           = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

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
    return partitioned_join(join_kind::INNER_JOIN, context, stream, resources);
  }

  /**
   * @brief Returns left-join indices for a partition of the left table.
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
  partitioned_left_join(join_partition_context const& context,
                        cuda::stream_ref stream,
                        Resources&& resources) const
  {
    return partitioned_join(join_kind::LEFT_JOIN, context, stream, resources);
  }

  /**
   * @brief Returns full-join indices for a partition of the left table.
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
  partitioned_full_join(join_partition_context const& context,
                        cuda::stream_ref stream,
                        Resources&& resources) const
  {
    return partitioned_join(join_kind::FULL_JOIN, context, stream, resources);
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
  [[nodiscard]] join_match_context inner_join_match_context(table_view const& left,
                                                            cuda::stream_ref stream,
                                                            Resources&& resources) const
  {
    return make_match_context(join_kind::INNER_JOIN, left, stream, resources);
  }

  /**
   * @brief Returns inner-join indices for a partition of the left table.
   *
   * Left indices refer to the original left table. The context can be reused for other
   * partitions.
   *
   * @throw std::invalid_argument If match counts are missing or have the wrong size,
   * the bounds are invalid, or a cached context belongs to another join object or kind.
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

  /**
   * @brief Creates a join_match_context for a left join.
   *
   * @tparam Resources cudf::memory_resources
   *
   * @param left The left join-key table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param resources Memory resources for output and temporary allocations
   *
   * @return Match context for partitioned_left_join()
   */
  template <typename Resources>
    requires std::same_as<std::remove_cvref_t<Resources>, cudf::memory_resources>
  [[nodiscard]] join_match_context left_join_match_context(table_view const& left,
                                                           cuda::stream_ref stream,
                                                           Resources&& resources) const
  {
    return make_match_context(join_kind::LEFT_JOIN, left, stream, resources);
  }

  /**
   * @brief Returns left-join indices for a partition of the left table.
   *
   * Left indices refer to the original left table. The context can be reused for other
   * partitions.
   *
   * @throw std::invalid_argument If match counts are missing or have the wrong size,
   * the bounds are invalid, or a cached context belongs to another join object or kind.
   *
   * @param matches Match context for a left join
   * @param left_start_idx First left row, inclusive
   * @param left_end_idx Last left row, exclusive
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param resources Memory resources for output and temporary allocations
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`]
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_left_join(
    join_match_context const& matches,
    size_type left_start_idx,
    size_type left_end_idx,
    cuda::stream_ref stream          = cudf::get_default_stream(),
    cudf::memory_resources resources = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Creates a join_match_context for a full join.
   *
   * @tparam Resources cudf::memory_resources
   *
   * @param left The left join-key table
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param resources Memory resources for output and temporary allocations
   *
   * @return Match context for partitioned_full_join()
   */
  template <typename Resources>
    requires std::same_as<std::remove_cvref_t<Resources>, cudf::memory_resources>
  [[nodiscard]] join_match_context full_join_match_context(table_view const& left,
                                                           cuda::stream_ref stream,
                                                           Resources&& resources) const
  {
    return make_match_context(join_kind::FULL_JOIN, left, stream, resources);
  }

  /**
   * @brief Returns full-join indices for a partition of the left table.
   *
   * Left indices refer to the original left table. The context can be reused for other
   * partitions.
   * Call finalize_partitioned_full_join() to append unmatched right rows.
   *
   * @throw std::invalid_argument If match counts are missing or have the wrong size,
   * the bounds are invalid, or a cached context belongs to another join object or kind.
   *
   * @param matches Match context for a full join
   * @param left_start_idx First left row, inclusive
   * @param left_end_idx Last left row, exclusive
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param resources Memory resources for output and temporary allocations
   *
   * @return A pair of device vectors [`left_indices`, `right_indices`]
   */
  [[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                          std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_full_join(
    join_match_context const& matches,
    size_type left_start_idx,
    size_type left_end_idx,
    cuda::stream_ref stream          = cudf::get_default_stream(),
    cudf::memory_resources resources = cudf::get_current_device_resource_ref()) const;

 private:
  join_match_context make_match_context(join_kind kind,
                                        table_view const& left,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources resources) const;
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  partitioned_join(join_kind kind,
                   join_partition_context const& context,
                   cuda::stream_ref stream,
                   cudf::memory_resources resources) const;

  std::unique_ptr<impl_type const> _impl;
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
