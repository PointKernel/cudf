/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "hash_join_impl.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <cooperative_groups.h>
#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/stream>

namespace cudf::detail {

/// Warps per retrieval block.
constexpr thread_index_type hash_csr_warps_per_block =
  hash_csr_block_size / cudf::detail::warp_size;
/// Maximum output pairs written by each retrieval lane.
constexpr thread_index_type hash_csr_outputs_per_lane = 32;

/**
 * @brief Device implementation of launch_hash_csr_probe_count_kernel.
 *
 * Uses the launcher's buffer contract with a one-dimensional launch.
 */
template <bool IsOuter, typename Equal, typename Hasher>
CUDF_KERNEL void hash_csr_probe_count_kernel(size_type num_rows,
                                             bitmask_type const* valid_rows,
                                             size_type* probe_slots,
                                             size_type* match_counts,
                                             cuda::std::uint32_t* matched_slots,
                                             cuda::std::uint64_t* matched_build_rows,
                                             hash_set_ref<hash_set_key_type> hash_set,
                                             csr_ref csr,
                                             Equal equal,
                                             Hasher hasher)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const index = static_cast<size_type>(row);
    auto slot        = hash_set.capacity;
    if (valid_rows == nullptr || cudf::bit_is_set(valid_rows, index)) {
      auto const hash_value = hasher(index);
      slot = hash_set.find(hash_set_key_type{hash_value, index}, hash_value, equal);
    }

    auto const found = slot != hash_set.capacity;
    auto const count = found ? csr.size(static_cast<size_type>(slot)) : size_type{0};
    if (probe_slots != nullptr) {
      probe_slots[index] = found ? static_cast<size_type>(slot) : CUDF_SIZE_TYPE_SENTINEL;
    }
    if (match_counts != nullptr) {
      match_counts[index] = IsOuter ? cuda::std::max(count, size_type{1}) : count;
    }

    // Tally each matched build slot once, regardless of how many probe rows match it.
    if constexpr (IsOuter) {
      if (found && matched_slots != nullptr) {
        auto matched_slot_ref =
          cuda::atomic_ref<cuda::std::uint32_t, cuda::thread_scope_device>{matched_slots[slot]};
        auto expected = cuda::std::uint32_t{0};
        if (matched_slot_ref.compare_exchange_strong(
              expected, cuda::std::uint32_t{1}, cuda::memory_order_relaxed)) {
          cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device>{*matched_build_rows}
            .fetch_add(static_cast<cuda::std::uint64_t>(count), cuda::memory_order_relaxed);
        }
      }
    }
  }
}

/**
 * @brief Enqueues probe slots, output counts, and optional matched-build tracking.
 *
 * Invalid or unmatched rows count once in outer mode and zero times in inner mode. The set and
 * CSR must describe the same completed build. Zero flags and counter before a fresh tally;
 * each matched build row is counted once. Does nothing when num_rows is zero.
 *
 * @tparam IsOuter Whether unmatched probe rows produce an output pair
 * @tparam Equal Device callable comparing a probe key with a stored build key
 * @tparam Hasher Device callable hashing a probe row index
 *
 * @param num_rows Number of probe rows
 * @param valid_rows Probe validity mask, or nullptr to include every row
 * @param probe_slots Optional num_rows slots; unmatched rows receive CUDF_SIZE_TYPE_SENTINEL
 * @param match_counts Optional num_rows output counts
 * @param matched_slots Optional capacity-sized match flags; ignored in inner mode
 * @param matched_build_rows Counter required with matched_slots in outer mode; otherwise optional
 * @param hash_set Completed set of build keys
 * @param csr Build row indices grouped by set slot
 * @param equal Probe-to-build key equality predicate
 * @param hasher Probe row hash function, consistent with the build hashes
 * @param stream CUDA stream used for the kernel launch
 */
template <bool IsOuter, typename Equal, typename Hasher>
void launch_hash_csr_probe_count_kernel(size_type num_rows,
                                        bitmask_type const* valid_rows,
                                        size_type* probe_slots,
                                        size_type* match_counts,
                                        cuda::std::uint32_t* matched_slots,
                                        cuda::std::uint64_t* matched_build_rows,
                                        hash_set_ref<hash_set_key_type> hash_set,
                                        csr_ref csr,
                                        Equal equal,
                                        Hasher hasher,
                                        cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  hash_csr_probe_count_kernel<IsOuter>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(num_rows,
                                                                           valid_rows,
                                                                           probe_slots,
                                                                           match_counts,
                                                                           matched_slots,
                                                                           matched_build_rows,
                                                                           hash_set,
                                                                           csr,
                                                                           equal,
                                                                           hasher);
  CUDF_CUDA_TRY(cudaGetLastError());
}

/**
 * @brief Device implementation of launch_hash_csr_retrieve_kernel.
 *
 * Uses the launcher's buffer contract. Launch one-dimensional blocks of hash_csr_block_size
 * threads with enough warps to cover output_size.
 *
 * `outputs_per_warp` ranges from 1 through `warp_size * hash_csr_outputs_per_lane`.
 */
template <bool IsOuter>
CUDF_KERNEL void hash_csr_retrieve_kernel(cuda::std::int64_t output_size,
                                          size_type num_probe_rows,
                                          cuda::std::int64_t outputs_per_warp,
                                          cuda::std::int64_t const* offsets,
                                          size_type const* probe_slots,
                                          csr_ref csr,
                                          size_type left_index_offset,
                                          size_type* left_indices,
                                          size_type* right_indices)
{
  auto const warp = cooperative_groups::tiled_partition<cudf::detail::warp_size>(
    cooperative_groups::this_thread_block());
  auto const lane_id       = static_cast<thread_index_type>(warp.thread_rank());
  auto const warp_in_block = static_cast<thread_index_type>(threadIdx.x) / cudf::detail::warp_size;
  auto const global_warp =
    static_cast<cuda::std::int64_t>(blockIdx.x) * hash_csr_warps_per_block + warp_in_block;
  auto const range_begin = outputs_per_warp * global_warp;
  if (range_begin >= output_size) { return; }
  auto const range_end = cuda::std::min(range_begin + outputs_per_warp, output_size);

  size_type endpoint_probe{};
  if (lane_id < 2) {
    auto const endpoint = lane_id == 0 ? range_begin : range_end - 1;
    endpoint_probe      = static_cast<size_type>(
      cuda::std::upper_bound(offsets, offsets + num_probe_rows + 1, endpoint) - offsets - 1);
  }
  auto const first_probe = warp.shfl(endpoint_probe, 0);
  auto const last_probe  = warp.shfl(endpoint_probe, 1);

#pragma unroll
  for (thread_index_type item = 0; item < hash_csr_outputs_per_lane; ++item) {
    auto const output_index = range_begin + lane_id + item * cudf::detail::warp_size;
    if (output_index < range_end) {
      auto const probe_row =
        first_probe == last_probe
          ? first_probe
          : static_cast<size_type>(cuda::std::upper_bound(offsets + first_probe,
                                                          offsets + last_probe + 2,
                                                          output_index) -
                                   offsets - 1);
      auto const slot            = probe_slots[probe_row];
      left_indices[output_index] = probe_row + left_index_offset;
      if constexpr (IsOuter) {
        if (slot == CUDF_SIZE_TYPE_SENTINEL) {
          right_indices[output_index] = JoinNoMatch;
          continue;
        }
      }
      auto const local_match      = static_cast<size_type>(output_index - offsets[probe_row]);
      right_indices[output_index] = csr.values[csr.begin(slot) + local_match];
    }
  }
}

/**
 * @brief Enqueues probe/build index pairs from CSR matches.
 *
 * Offsets must prefix-sum probe counts for the same IsOuter mode, from zero to output_size.
 * Outer mode writes JoinNoMatch for unmatched probes; full-join callers append unmatched build rows
 * separately. Does nothing when output_size is zero.
 *
 * @tparam IsOuter Whether unmatched probe rows produce a JoinNoMatch right index
 *
 * @param output_size Number of output pairs
 * @param num_probe_rows Number of probe rows
 * @param offsets Nondecreasing num_probe_rows + 1 exclusive output offsets
 * @param probe_slots Build slot per probe row, or CUDF_SIZE_TYPE_SENTINEL for an unmatched row
 * @param csr Completed CSR of build row indices
 * @param left_index_offset Added to each partition-local probe row index
 * @param left_indices Output buffer with space for output_size probe indices
 * @param right_indices Output buffer with space for output_size build indices
 * @param stream CUDA stream used for the kernel launch
 */
template <bool IsOuter>
void launch_hash_csr_retrieve_kernel(cuda::std::int64_t output_size,
                                     size_type num_probe_rows,
                                     cuda::std::int64_t const* offsets,
                                     size_type const* probe_slots,
                                     csr_ref csr,
                                     size_type left_index_offset,
                                     size_type* left_indices,
                                     size_type* right_indices,
                                     cuda::stream_ref stream)
{
  if (output_size == 0) { return; }
  auto const min_blocks = size_type{2} * cudf::detail::num_multiprocessors();
  constexpr auto outputs_per_block =
    hash_csr_warps_per_block * cudf::detail::warp_size * hash_csr_outputs_per_lane;
  auto const requested_blocks = cudf::util::div_rounding_up_safe(
    output_size, static_cast<cuda::std::int64_t>(outputs_per_block));
  auto const num_blocks = static_cast<cuda::std::uint32_t>(
    cuda::std::max<cuda::std::int64_t>(requested_blocks, min_blocks));
  auto const num_warps = static_cast<cuda::std::int64_t>(num_blocks) * hash_csr_warps_per_block;
  auto const outputs_per_warp = cudf::util::div_rounding_up_safe(output_size, num_warps);

  hash_csr_retrieve_kernel<IsOuter>
    <<<num_blocks, hash_csr_block_size, 0, stream.get()>>>(output_size,
                                                           num_probe_rows,
                                                           outputs_per_warp,
                                                           offsets,
                                                           probe_slots,
                                                           csr,
                                                           left_index_offset,
                                                           left_indices,
                                                           right_indices);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::detail
