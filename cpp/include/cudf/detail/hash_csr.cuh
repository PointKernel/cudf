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
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

namespace cudf::detail::hash_csr {

/// Count index and rank within that count, recorded for each input row.
using build_position_type = cuco::pair<cuda::std::uint32_t, size_type>;

/**
 * @brief Non-owning, linearly probed set of unique row keys.
 *
 * Key is either `size_type` (a row index) or `cuco::pair<hash_value_type, size_type>` (a cached
 * hash and row index). Row indices use arbitrary capacities and bounded, load-before-CAS
 * probing. Cached keys retain power-of-two capacities and CAS-first probing of the full set.
 *
 * The caller initializes every entry byte to 0xff and keeps the referenced rows alive. Capacity
 * must be nonzero and less than the maximum `cuda::std::uint32_t` value. Equality compares keys
 * of type Key, and equivalent keys must have equal hashes. Concurrent insertion uses
 * device-scope atomic access to entries.
 */
template <typename Key>
struct hash_set_ref {
  static_assert(cuda::std::is_same_v<Key, size_type> ||
                cuda::std::is_same_v<Key, cuco::pair<hash_value_type, size_type>>);

  using key_type = Key;

  /// Position and stored key captured by insertion, without another load of the set.
  struct insertion_position {
    cuda::std::uint32_t slot;  ///< Capacity when the probe limit is exhausted
    key_type key;              ///< Atomic key snapshot, or the empty key on failure
  };
  using insert_result = cuda::std::pair<insertion_position, bool>;

  key_type* entries;
  cuda::std::uint32_t capacity;
  cuda::std::uint32_t max_probes{};  ///< Used only by compact row storage

  /**
   * @brief Inserts a key if absent and returns its position and whether it was inserted.
   *
   * For an existing equivalent key, the position contains that stored key and the flag is false.
   * Exhausting the probe limit returns `{capacity, empty_key}` and false, leaving earlier
   * insertions intact. The caller must discard that partial build or handle the missing key.
   * The position is a snapshot, not an iterator into storage that other threads may access.
   * `hash_value` is the precomputed hash of `key`.
   */
  template <typename Equal>
  __device__ insert_result insert(key_type key,
                                  hash_value_type hash_value,
                                  Equal const& equal) const
  {
    constexpr bool cached = !cuda::std::is_same_v<key_type, size_type>;
    auto const empty_key  = [] {
      if constexpr (cached) {
        return key_type{cuda::std::numeric_limits<hash_value_type>::max(),
                        size_type{CUDF_SIZE_TYPE_SENTINEL}};
      } else {
        return size_type{CUDF_SIZE_TYPE_SENTINEL};
      }
    }();
    auto slot        = cached ? static_cast<cuda::std::uint32_t>(hash_value) & (capacity - 1)
                              : static_cast<cuda::std::uint32_t>(hash_value % capacity);
    auto const limit = cached ? capacity : max_probes;
    for (cuda::std::uint32_t step = 0; step < limit; ++step) {
      auto entry   = cuda::atomic_ref<key_type, cuda::thread_scope_device>{entries[slot]};
      auto current = empty_key;
      if constexpr (!cached) { current = entry.load(cuda::memory_order_relaxed); }
      if ((cached || current == empty_key) &&
          entry.compare_exchange_strong(current, key, cuda::memory_order_relaxed)) {
        return {{slot, key}, true};
      }
      if (equal(key, current)) { return {{slot, current}, false}; }
      if constexpr (cached) {
        slot = (static_cast<cuda::std::uint32_t>(hash_value) + step + 1) & (capacity - 1);
      } else {
        slot = slot + 1 == capacity ? 0 : slot + 1;
      }
    }
    return {{capacity, empty_key}, false};
  }

  /**
   * @brief Finds a key in a completed set, returning capacity if absent.
   *
   * Uses ordinary loads and must not run concurrently with insertion or initialization.
   * `hash_value` is the precomputed hash of `key`.
   */
  template <typename Equal>
  __device__ cuda::std::uint32_t find(key_type key, hash_value_type hash_value, Equal equal) const
  {
    constexpr bool cached = !cuda::std::is_same_v<key_type, size_type>;
    auto slot =
      cached ? cuda::std::uint32_t{0} : static_cast<cuda::std::uint32_t>(hash_value % capacity);
    auto const limit = cached ? capacity : max_probes;
    for (cuda::std::uint32_t step = 0; step < limit; ++step) {
      if constexpr (cached) {
        slot = (static_cast<cuda::std::uint32_t>(hash_value) + step) & (capacity - 1);
      }
      auto const current = entries[slot];
      auto const empty   = [&] {
        if constexpr (cached) {
          return current.second == CUDF_SIZE_TYPE_SENTINEL;
        } else {
          return current == CUDF_SIZE_TYPE_SENTINEL;
        }
      }();
      if (empty) { return capacity; }
      if (equal(key, current)) { return slot; }
      if constexpr (!cached) { slot = slot + 1 == capacity ? 0 : slot + 1; }
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
