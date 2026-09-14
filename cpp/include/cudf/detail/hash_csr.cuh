/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/hashing.hpp>
#include <cudf/types.hpp>

#include <cuco/pair.cuh>
#include <cuda/atomic>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

namespace cudf::detail::hash_csr {

/// Slot marker for an excluded row or an unsuccessful insertion.
inline constexpr cuda::std::uint32_t no_slot = static_cast<cuda::std::uint32_t>(-1);

/// Count index and rank within that count, recorded for each input row.
using build_position_type = cuco::pair<cuda::std::uint32_t, size_type>;

/**
 * @brief Storage and probing choices for a row-key table.
 *
 * Cached hashes use power-of-two capacities and CAS-first insertion. Compact row entries use
 * arbitrary capacities and read before CAS, avoiding writes when repeated keys are found.
 */
enum class key_storage { hash_and_row, row };

/// Result of inserting a row key or finding its existing representative.
struct insertion_result {
  cuda::std::uint32_t slot;  ///< Table capacity when the probe limit is exhausted
  size_type representative;  ///< Stored input row, or CUDF_SIZE_TYPE_SENTINEL on failure
  bool inserted;             ///< Whether this insertion claimed an empty slot
};

/**
 * @brief Non-owning, linearly probed table mapping equal row keys to a representative row.
 *
 * The caller initializes every byte of the entries to 0xff and keeps the referenced row keys
 * alive. Capacity must be nonzero, less than `no_slot`, and a power of two for cached hashes.
 * Compact tables probe at most `max_probes` slots; cached tables probe their entire capacity.
 * Equality compares `{hash, row}` entries for cached storage and row indices for compact storage.
 * Concurrent insertions require a device-scope atomic load or CAS for every accessed entry.
 */
template <key_storage Storage>
struct table_ref {
  using entry_type = cuda::std::conditional_t<Storage == key_storage::hash_and_row,
                                              cuco::pair<hash_value_type, size_type>,
                                              size_type>;

  entry_type* entries;
  cuda::std::uint32_t capacity;
  cuda::std::uint32_t max_probes{};  ///< Used only by compact row storage

  /**
   * @brief Inserts a row key or returns the slot and representative of an equal key.
   *
   * An unsuccessful bounded insertion leaves any earlier successful insertions intact. The
   * caller must discard that partial build or otherwise handle the missing row.
   */
  template <typename Equal>
  __device__ insertion_result insert_or_find(size_type row,
                                             hash_value_type hash,
                                             Equal const& equal) const
  {
    constexpr bool cached = Storage == key_storage::hash_and_row;
    auto const key        = [&] {
      if constexpr (cached) {
        return entry_type{hash, row};
      } else {
        return row;
      }
    }();
    auto const empty = [] {
      if constexpr (cached) {
        return entry_type{static_cast<hash_value_type>(-1), size_type{CUDF_SIZE_TYPE_SENTINEL}};
      } else {
        return size_type{CUDF_SIZE_TYPE_SENTINEL};
      }
    }();
    auto slot        = cached ? static_cast<cuda::std::uint32_t>(hash) & (capacity - 1)
                              : static_cast<cuda::std::uint32_t>(hash % capacity);
    auto const limit = cached ? capacity : max_probes;
    for (cuda::std::uint32_t step = 0; step < limit; ++step) {
      auto entry   = cuda::atomic_ref<entry_type, cuda::thread_scope_device>{entries[slot]};
      auto current = empty;
      if constexpr (!cached) { current = entry.load(cuda::memory_order_relaxed); }
      if ((cached || current == empty) &&
          entry.compare_exchange_strong(current, key, cuda::memory_order_relaxed)) {
        return {slot, row, true};
      }
      if (equal(key, current)) {
        if constexpr (cached) {
          return {slot, current.second, false};
        } else {
          return {slot, current, false};
        }
      }
      if constexpr (cached) {
        slot = (static_cast<cuda::std::uint32_t>(hash) + step + 1) & (capacity - 1);
      } else {
        slot = slot + 1 == capacity ? 0 : slot + 1;
      }
    }
    return {capacity, CUDF_SIZE_TYPE_SENTINEL, false};
  }

  /**
   * @brief Finds a key in a completed table, returning capacity if absent.
   *
   * Uses ordinary loads and must not run concurrently with insertion or initialization.
   */
  template <typename Equal>
  __device__ cuda::std::uint32_t find(entry_type key, Equal equal) const
  {
    static_assert(Storage == key_storage::hash_and_row);
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      auto const slot    = (static_cast<cuda::std::uint32_t>(key.first) + step) & (capacity - 1);
      auto const current = entries[slot];
      if (current.second == CUDF_SIZE_TYPE_SENTINEL) { return capacity; }
      if (equal(key, current)) { return slot; }
    }
    return capacity;
  }
};

/// Non-owning CSR whose offsets are cumulative ends (the first segment starts at zero).
struct csr_ref {
  size_type const* cumulative_ends;
  size_type const* values;

  __device__ size_type begin(cuda::std::uint32_t slot) const
  {
    return slot == 0 ? size_type{0} : cumulative_ends[slot - 1];
  }

  __device__ size_type size(cuda::std::uint32_t slot) const
  {
    return cumulative_ends[slot] - begin(slot);
  }
};

}  // namespace cudf::detail::hash_csr
