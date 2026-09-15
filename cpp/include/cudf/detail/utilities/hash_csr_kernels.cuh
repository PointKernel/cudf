/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/hash_csr.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/std/bit>
#include <cuda/stream>

namespace cudf::detail {

/// How input rows reserve ranks within their CSR segments.
enum class hash_csr_count_mode {
  per_row,  ///< One atomic increment per row; segments are indexed by hash-set slot
  per_warp  ///< One increment per equal-index peer group in a warp; supports representative indices
};

/**
 * @brief Inserts row keys, counts rows per segment, and records their ranks for CSR construction.
 *
 * Positions receives `{segment_index, rank}` per row, or sentinels for excluded rows.
 * The caller scans counts into segment starts, then uses fill_hash_csr() to write the row indices.
 * Row order within a segment is unspecified.
 *
 * Per-row mode counts by slot. Per-warp mode combines equal-index increments within a warp
 * and supports representative-row indexing. Null positions selects a per-warp keys-only build.
 *
 * @pre Set storage is initialized as required by hash_set_ref. Counts are zeroed and cover
 * `set.capacity` slots or `num_rows` representatives; positions covers `num_rows` rows.
 * A per-warp keys-only build needs neither buffer.
 * @pre For per-warp builds, a non-null overflow flag starts at zero. Failure sets it to one
 * and may leave positions unwritten; discard all partial results before retrying. All other
 * builds require every insertion to fit the probe limit.
 * @pre Launch a one-dimensional grid with whole warps per block.
 *
 * @tparam Mode How rows reserve ranks within a segment
 * @tparam Set hash_set_ref specialization for the stored row-key representation
 * @tparam Equal Device-callable row-key equality predicate
 * @tparam Hasher Device-callable function mapping a row index to its hash value
 *
 * @param num_rows Number of input rows
 * @param valid_rows Inclusion bitmask, or nullptr to include every row
 * @param[out] positions Per-row segment indices and ranks; nullptr for a per-warp keys-only build
 * @param[in,out] slot_counts Segment counts; may be nullptr for a keys-only build
 * @param count_by_representative Whether per-warp counts use representative rows instead of slots;
 * ignored in per-row mode
 * @param[in,out] set Initialized set in which to insert row keys
 * @param equal Equality predicate invoked with an input key and a stored key
 * @param hasher Hash function for input row indices
 * @param[in,out] overflow Optional per-warp failure flag; ignored in per-row mode
 */
template <hash_csr_count_mode Mode, typename Set, typename Equal, typename Hasher>
CUDF_KERNEL void build_hash_csr_kernel(size_type num_rows,
                                       bitmask_type const* valid_rows,
                                       hash_csr_build_position* positions,
                                       size_type* slot_counts,
                                       bool count_by_representative,
                                       Set set,
                                       Equal equal,
                                       Hasher hasher,
                                       int* overflow)
{
  constexpr bool warp_count = Mode == hash_csr_count_mode::per_warp;
  auto const lane   = warp_count ? static_cast<cuda::std::uint32_t>(threadIdx.x % warp_size) : 0u;
  auto const stride = grid_1d::grid_stride();
  if constexpr (warp_count) {
    // A block already in flight finishes its bounded probes; the caller discards the partial
    // build after any overflow. All threads participate in the block-wide cancellation check.
    if (overflow != nullptr) {
      auto const stop =
        threadIdx.x == 0 && cuda::atomic_ref<int, cuda::thread_scope_device>{*overflow}.load(
                              cuda::memory_order_relaxed) != 0;
      if (__syncthreads_or(stop)) { return; }
    }
  }
  // Warp counting keeps every lane in the loop, including lanes past the last input row.
  // Per-row counting retains the ordinary scalar grid-stride loop.
  for (auto first_row = grid_1d::global_thread_id() - lane; first_row < num_rows;
       first_row += stride) {
    auto const row = first_row + lane;
    auto const active =
      row < num_rows &&
      (valid_rows == nullptr || cudf::bit_is_set(valid_rows, static_cast<size_type>(row)));
    auto slot = static_cast<cuda::std::uint32_t>(CUDF_SIZE_TYPE_SENTINEL);
    if (active) {
      auto const index      = static_cast<size_type>(row);
      auto const hash_value = hasher(index);
      auto const key        = [&] {
        if constexpr (cuda::std::is_same_v<typename Set::key_type, size_type>) {
          return index;
        } else {
          return typename Set::key_type{hash_value, index};
        }
      }();
      auto const position = set.insert(key, hash_value, equal).first;
      if (position.slot != set.capacity) {
        slot = position.slot;
        if constexpr (warp_count) {
          if (count_by_representative) {
            if constexpr (cuda::std::is_same_v<typename Set::key_type, size_type>) {
              slot = static_cast<cuda::std::uint32_t>(position.key);
            } else {
              slot = static_cast<cuda::std::uint32_t>(position.key.second);
            }
          }
        }
      } else if constexpr (warp_count) {
        if (overflow != nullptr) {
          cuda::atomic_ref<int, cuda::thread_scope_device>{*overflow}.store(
            1, cuda::memory_order_relaxed);
        }
      }
    }
    if constexpr (warp_count) {
      // Keys-only builds need the set, but neither positions nor counts.
      if (positions == nullptr) { continue; }
    }
    auto const has_slot = slot != static_cast<cuda::std::uint32_t>(CUDF_SIZE_TYPE_SENTINEL);
    unsigned int active_mask{};
    if constexpr (warp_count) { active_mask = __ballot_sync(0xffff'ffffu, has_slot); }
    if (has_slot) {
      auto count = cuda::atomic_ref<size_type, cuda::thread_scope_device>{slot_counts[slot]};
      size_type rank;
      if constexpr (warp_count) {
        auto const peers  = __match_any_sync(active_mask, slot);
        auto const leader = cuda::std::countr_zero(peers);
        size_type first_rank{};
        if (lane == static_cast<cuda::std::uint32_t>(leader)) {
          first_rank = count.fetch_add(static_cast<size_type>(cuda::std::popcount(peers)),
                                       cuda::memory_order_relaxed);
        }
        first_rank = __shfl_sync(peers, first_rank, leader);
        rank =
          first_rank + static_cast<size_type>(cuda::std::popcount(peers & ((1u << lane) - 1u)));
      } else {
        rank = count.fetch_add(size_type{1}, cuda::memory_order_relaxed);
      }
      positions[row] = {slot, rank};
    } else if (row < num_rows) {
      positions[row] = {static_cast<cuda::std::uint32_t>(CUDF_SIZE_TYPE_SENTINEL),
                        size_type{CUDF_SIZE_TYPE_SENTINEL}};
    }
  }
}

/**
 * @copydoc build_hash_csr_kernel
 *
 * @note Enqueues work without allocating or synchronizing. Buffers and functor data must remain
 * alive until completion. Zero rows enqueue no work.
 * @param stream CUDA stream on which to enqueue the build
 */
template <hash_csr_count_mode Mode, typename Set, typename Equal, typename Hasher>
void build_hash_csr(size_type num_rows,
                    bitmask_type const* valid_rows,
                    hash_csr_build_position* positions,
                    size_type* slot_counts,
                    bool count_by_representative,
                    Set set,
                    Equal equal,
                    Hasher hasher,
                    int* overflow,
                    cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  build_hash_csr_kernel<Mode>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(num_rows,
                                                                           valid_rows,
                                                                           positions,
                                                                           slot_counts,
                                                                           count_by_representative,
                                                                           set,
                                                                           equal,
                                                                           hasher,
                                                                           overflow);
  CUDF_CUDA_TRY(cudaGetLastError());
}

/**
 * @brief Fills CSR segments with row indices using their recorded segment indices and ranks.
 *
 * Writes row `i` to `values[starts[positions[i].first] + positions[i].second]`.
 * Sentinel segment indices are skipped.
 *
 * @pre Build and scan finish before this kernel executes. Starts uses the same segment-index
 * domain as positions; segments are disjoint and have room for every recorded rank.
 *
 * @tparam PositionIterator Device-accessible random-access iterator over hash_csr_build_position
 * @tparam OffsetIterator Device-accessible random-access iterator over segment starts
 *
 * @param num_rows Number of input rows
 * @param positions Per-row segment indices and ranks, covering `num_rows` rows
 * @param starts Starting offsets indexed by the segment indices in positions
 * @param[out] values Storage for all included row indices
 */
template <typename PositionIterator, typename OffsetIterator>
CUDF_KERNEL void fill_hash_csr_kernel(size_type num_rows,
                                      PositionIterator positions,
                                      OffsetIterator starts,
                                      size_type* values)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const position = positions[row];
    if (position.first == static_cast<cuda::std::uint32_t>(CUDF_SIZE_TYPE_SENTINEL)) { continue; }
    values[starts[position.first] + position.second] = static_cast<size_type>(row);
  }
}

/**
 * @copydoc fill_hash_csr_kernel
 *
 * @note Enqueues work without allocating or synchronizing. Iterator storage and values must
 * remain alive until completion. Zero rows enqueue no work.
 * @param stream CUDA stream on which to enqueue the fill
 */
template <typename PositionIterator, typename OffsetIterator>
void fill_hash_csr(size_type num_rows,
                   PositionIterator positions,
                   OffsetIterator starts,
                   size_type* values,
                   cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, hash_csr_block_size};
  fill_hash_csr_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows, positions, starts, values);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::detail
