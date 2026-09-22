/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <cuda/stream>

#include <cstddef>

namespace cudf::detail {

/**
 * @file
 * @brief Host launchers for the conditional join kernels.
 *
 * The kernels are defined in conditional_join_kernels.cuh. Each `has_nulls` flavor of the AST
 * expression evaluator is instantiated in its own translation unit (conditional_join_kernels.cu
 * for `has_nulls == false`, conditional_join_kernels_nulls.cu for `has_nulls == true`) to keep
 * the compile time of any single translation unit bounded. Callers must include only this
 * header so that the kernel templates are not instantiated again in the calling translation
 * unit.
 */

/**
 * @brief Launches the kernel that computes the output size of a conditional join.
 *
 * @tparam has_nulls Whether the expression may evaluate to null
 *
 * @param left_table The left table
 * @param right_table The right table
 * @param join_type The type of join to be performed
 * @param device_expression_data Container of device data required to evaluate the expression
 * @param swap_tables Whether the left and right tables have been swapped
 * @param output_size Device pointer to the output size counter
 * @param config Launch configuration of the kernel
 * @param shmem_size_per_block Dynamic shared memory required per block
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <bool has_nulls>
void launch_compute_conditional_join_output_size(
  table_device_view const& left_table,
  table_device_view const& right_table,
  join_kind join_type,
  ast::detail::expression_device_view device_expression_data,
  bool swap_tables,
  std::size_t* output_size,
  grid_1d const& config,
  std::size_t shmem_size_per_block,
  cuda::stream_ref stream);

/**
 * @brief Launches the kernel that performs an inner, left, or full conditional join.
 *
 * @tparam has_nulls Whether the expression may evaluate to null
 *
 * @param left_table The left table
 * @param right_table The right table
 * @param join_type The type of join to be performed
 * @param join_output_l The left result of the join operation
 * @param join_output_r The right result of the join operation
 * @param current_idx Device pointer to the running count of output rows
 * @param device_expression_data Container of device data required to evaluate the expression
 * @param max_size The maximum number of pairs that will be produced
 * @param swap_tables Whether the left and right tables have been swapped
 * @param config Launch configuration of the kernel
 * @param shmem_size_per_block Dynamic shared memory required per block
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <bool has_nulls>
void launch_conditional_join(table_device_view const& left_table,
                             table_device_view const& right_table,
                             join_kind join_type,
                             size_type* join_output_l,
                             size_type* join_output_r,
                             std::size_t* current_idx,
                             ast::detail::expression_device_view device_expression_data,
                             std::size_t max_size,
                             bool swap_tables,
                             grid_1d const& config,
                             std::size_t shmem_size_per_block,
                             cuda::stream_ref stream);

/**
 * @brief Launches the kernel that performs a left semi or left anti conditional join.
 *
 * @tparam has_nulls Whether the expression may evaluate to null
 *
 * @param left_table The left table
 * @param right_table The right table
 * @param join_type The type of join to be performed
 * @param join_output_l The left result of the join operation
 * @param current_idx Device pointer to the running count of output rows
 * @param device_expression_data Container of device data required to evaluate the expression
 * @param max_size The maximum number of rows that will be produced
 * @param config Launch configuration of the kernel
 * @param shmem_size_per_block Dynamic shared memory required per block
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <bool has_nulls>
void launch_conditional_join_anti_semi(table_device_view const& left_table,
                                       table_device_view const& right_table,
                                       join_kind join_type,
                                       size_type* join_output_l,
                                       std::size_t* current_idx,
                                       ast::detail::expression_device_view device_expression_data,
                                       std::size_t max_size,
                                       grid_1d const& config,
                                       std::size_t shmem_size_per_block,
                                       cuda::stream_ref stream);

}  // namespace cudf::detail
