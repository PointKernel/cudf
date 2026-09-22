/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Instantiates the conditional join kernels for expressions that cannot evaluate to null. The
// nullable flavor lives in conditional_join_kernels_nulls.cu; keeping the two apart halves the
// AST expression evaluator instantiations per translation unit.

#include "join/conditional_join_kernels.cuh"
#include "join/conditional_join_kernels.hpp"

namespace cudf::detail {

template void launch_compute_conditional_join_output_size<false>(
  table_device_view const& left_table,
  table_device_view const& right_table,
  join_kind join_type,
  ast::detail::expression_device_view device_expression_data,
  bool swap_tables,
  std::size_t* output_size,
  grid_1d const& config,
  std::size_t shmem_size_per_block,
  cuda::stream_ref stream);

template void launch_conditional_join<false>(
  table_device_view const& left_table,
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

template void launch_conditional_join_anti_semi<false>(
  table_device_view const& left_table,
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
