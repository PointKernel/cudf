/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/hash_csr.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/std/bit>
#include <cuda/stream>

namespace cudf::detail::hash_csr {

inline constexpr thread_index_type block_size = 256;

/// Allocate ranks individually, or combine increments for equal count indices within a warp.
enum class count_mode { per_row, per_warp };

template <count_mode Mode, typename Table, typename Equal, typename Hasher>
CUDF_KERNEL void build_count_kernel(size_type num_rows,
                                    bitmask_type const* valid_rows,
                                    build_position_type* positions,
                                    size_type* slot_counts,
                                    bool count_by_representative,
                                    Table table,
                                    Equal equal,
                                    Hasher hasher,
                                    int* overflow)
{
  constexpr bool warp_count = Mode == count_mode::per_warp;
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
    auto slot = no_slot;
    if (active) {
      auto const index  = static_cast<size_type>(row);
      auto const result = table.insert_or_find(index, hasher(index), equal);
      if (result.slot != table.capacity) {
        slot = result.slot;
        if constexpr (warp_count) {
          if (count_by_representative) {
            slot = static_cast<cuda::std::uint32_t>(result.representative);
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
      // Keys-only builds need the table, but neither positions nor counts.
      if (positions == nullptr) { continue; }
    }
    auto const has_slot = slot != no_slot;
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
      positions[row] = {cuda::std::uint32_t{no_slot}, size_type{CUDF_SIZE_TYPE_SENTINEL}};
    }
  }
}

/**
 * @brief Inserts valid input rows and records each row's count index and rank within that count.
 *
 * Counts must initially be zero. Positions has `num_rows` entries and counts covers the table
 * capacity, or `num_rows` when counting by representative. A null validity mask includes all rows.
 * Per-row mode requires positions/counts and indexes counts by slot. Per-warp mode also supports
 * keys-only builds (null positions/counts), representative indexing, and bounded-build cancellation
 * through an optional, initially zero overflow flag. On overflow the caller must discard the
 * incomplete positions/counts. Neither mode guarantees input order within a key.
 */
template <count_mode Mode, typename Table, typename Equal, typename Hasher>
void build_count(size_type num_rows,
                 bitmask_type const* valid_rows,
                 build_position_type* positions,
                 size_type* slot_counts,
                 bool count_by_representative,
                 Table table,
                 Equal equal,
                 Hasher hasher,
                 int* overflow,
                 cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, block_size};
  build_count_kernel<Mode>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(num_rows,
                                                                           valid_rows,
                                                                           positions,
                                                                           slot_counts,
                                                                           count_by_representative,
                                                                           table,
                                                                           equal,
                                                                           hasher,
                                                                           overflow);
  CUDF_CUDA_TRY(cudaGetLastError());
}

template <typename PositionIterator, typename OffsetIterator>
CUDF_KERNEL void fill_kernel(size_type num_rows,
                             PositionIterator positions,
                             OffsetIterator starts,
                             size_type* values)
{
  auto const stride = grid_1d::grid_stride();
  for (auto row = grid_1d::global_thread_id(); row < num_rows; row += stride) {
    auto const position = positions[row];
    if (position.first == no_slot) { continue; }
    values[starts[position.first] + position.second] = static_cast<size_type>(row);
  }
}

/**
 * @brief Scatters valid rows into CSR segments using their recorded count index and rank.
 *
 * `starts` is indexed by the same count index recorded in positions. Its disjoint segments must
 * have room for every rank assigned by build_count. Iterators allow consumers to supply implicit
 * starts or cache-modified position loads without allocating another buffer.
 */
template <typename PositionIterator, typename OffsetIterator>
void fill(size_type num_rows,
          PositionIterator positions,
          OffsetIterator starts,
          size_type* values,
          cuda::stream_ref stream)
{
  if (num_rows == 0) { return; }
  auto const config = grid_1d{num_rows, block_size};
  fill_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    num_rows, positions, starts, values);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::detail::hash_csr
