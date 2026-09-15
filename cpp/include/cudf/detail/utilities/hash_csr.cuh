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
#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

namespace cudf::detail {

inline constexpr thread_index_type hash_csr_block_size = 256;  ///< Threads per HashCSR block

/**
 * @brief CSR segment index and rank within that segment for one input row.
 *
 * The segment index is a hash-set slot or its representative row index, as selected by
 * build_hash_csr(). Excluded rows use `CUDF_SIZE_TYPE_SENTINEL` in both fields, converted to the
 * field's type. fill_hash_csr() skips those rows.
 */
using hash_csr_build_position = cuco::pair<cuda::std::uint32_t, size_type>;

/**
 * @brief Non-owning hash set that stores one representative row per equivalent key.
 *
 * Row-index keys use bounded, load-before-CAS probing. Cached-hash keys use CAS-first probing
 * across a power-of-two capacity.
 *
 * @pre Before use, initialize every slot byte to 0xff. Input row indices must be nonnegative;
 * `CUDF_SIZE_TYPE_SENTINEL` is reserved for empty slots.
 * @pre Capacity is nonzero and below the maximum `cuda::std::uint32_t` value. Cached-hash keys
 * require a power-of-two capacity.
 * @pre Slot storage and unchanged row data outlive all operations. Equivalent keys have equal
 * hashes.
 *
 * @note Insertion may run concurrently with other insertions using device-scope atomics.
 * Lookup requires a completed build; neither operation may overlap slot initialization.
 *
 * @tparam Key `size_type` for a row index, or `cuco::pair<hash_value_type, size_type>` for a
 * cached hash value and row index
 */
template <typename Key>
struct hash_set_ref {
  static_assert(cuda::std::is_same_v<Key, size_type> ||
                cuda::std::is_same_v<Key, cuco::pair<hash_value_type, size_type>>);

  using key_type = Key;  ///< Stored row-key type

  /// Whether each stored key includes its precomputed hash value.
  static constexpr bool has_cached_hash = !cuda::std::is_same_v<key_type, size_type>;

  /// Slot index and atomic key snapshot returned by insert().
  struct insertion_position {
    cuda::std::uint32_t slot;  ///< Capacity when the probe limit is exhausted
    key_type key;              ///< Atomic key snapshot, or the empty key on failure
  };
  /// Insertion position and a flag that is true only when this call inserted the key.
  using insert_result = cuda::std::pair<insertion_position, bool>;

  key_type* slots;                   ///< Device-accessible storage for `capacity` keys
  cuda::std::uint32_t capacity;      ///< Number of slots; also the not-found position
  cuda::std::uint32_t max_probes{};  ///< Probe limit for row-index keys; unused for cached hashes

  /**
   * @brief Inserts a key if no equivalent key is present within the probe limit.
   *
   * Returns the existing key when found. Exhaustion leaves earlier insertions intact; callers
   * must handle the missing key or discard the partial build.
   *
   * @pre `hash_value` is the hash of `key` and equals `key.first` for cached-hash keys.
   *
   * @tparam Equal Device-callable predicate comparing two keys for equivalence
   *
   * @param key Row key to insert
   * @param hash_value Precomputed hash of `key`
   * @param equal Equality predicate invoked as `equal(key, stored_key)` for occupied slots
   *
   * @return The stored key and slot with an insertion flag; `{{capacity, empty_key}, false}`
   * if probing is exhausted
   */
  template <typename Equal>
  __device__ insert_result insert(key_type key,
                                  hash_value_type hash_value,
                                  Equal const& equal) const
  {
    auto const empty_key = [] {
      if constexpr (has_cached_hash) {
        return key_type{cuda::std::numeric_limits<hash_value_type>::max(),
                        size_type{CUDF_SIZE_TYPE_SENTINEL}};
      } else {
        return size_type{CUDF_SIZE_TYPE_SENTINEL};
      }
    }();
    auto slot = has_cached_hash ? static_cast<cuda::std::uint32_t>(hash_value) & (capacity - 1)
                                : static_cast<cuda::std::uint32_t>(hash_value % capacity);
    auto const limit = has_cached_hash ? capacity : max_probes;
    for (cuda::std::uint32_t step = 0; step < limit; ++step) {
      auto slot_ref = cuda::atomic_ref<key_type, cuda::thread_scope_device>{slots[slot]};
      auto current  = empty_key;
      if constexpr (!has_cached_hash) { current = slot_ref.load(cuda::memory_order_relaxed); }
      if ((has_cached_hash || current == empty_key) &&
          slot_ref.compare_exchange_strong(current, key, cuda::memory_order_relaxed)) {
        return {{slot, key}, true};
      }
      if (equal(key, current)) { return {{slot, current}, false}; }
      if constexpr (has_cached_hash) {
        slot = (static_cast<cuda::std::uint32_t>(hash_value) + step + 1) & (capacity - 1);
      } else {
        slot = slot + 1 == capacity ? 0 : slot + 1;
      }
    }
    return {{capacity, empty_key}, false};
  }

  /**
   * @brief Finds an equivalent key in a completed set.
   *
   * @pre The build is complete and no thread modifies the slots during lookup.
   * @pre Hashing, equality, and the probe limit are consistent with insertion.
   *
   * @tparam Equal Device-callable predicate comparing a query key with a stored key
   *
   * @param key Row key to find
   * @param hash_value Precomputed hash of `key`; also stored in `key.first` for cached-hash keys
   * @param equal Equality predicate invoked as `equal(key, stored_key)` for occupied slots
   *
   * @return The matching slot, or `capacity` if no key is found within the probe limit
   */
  template <typename Equal>
  __device__ cuda::std::uint32_t find(key_type key, hash_value_type hash_value, Equal equal) const
  {
    auto slot        = has_cached_hash ? cuda::std::uint32_t{0}
                                       : static_cast<cuda::std::uint32_t>(hash_value % capacity);
    auto const limit = has_cached_hash ? capacity : max_probes;
    for (cuda::std::uint32_t step = 0; step < limit; ++step) {
      if constexpr (has_cached_hash) {
        slot = (static_cast<cuda::std::uint32_t>(hash_value) + step) & (capacity - 1);
      }
      auto const current = slots[slot];
      auto const empty   = [&] {
        if constexpr (has_cached_hash) {
          return current.second == CUDF_SIZE_TYPE_SENTINEL;
        } else {
          return current == CUDF_SIZE_TYPE_SENTINEL;
        }
      }();
      if (empty) { return capacity; }
      if (equal(key, current)) { return slot; }
      if constexpr (!has_cached_hash) { slot = slot + 1 == capacity ? 0 : slot + 1; }
    }
    return capacity;
  }
};

/**
 * @brief Non-owning compressed sparse row (CSR) view of grouped row indices.
 *
 * Each segment contains the row indices for one key. Offsets are cumulative ends, so segment
 * zero starts at zero and equal adjacent ends represent an empty segment.
 *
 * @pre Ends are nonnegative and nondecreasing. Values has room for the final cumulative end.
 * Both arrays are device-accessible and remain alive while the view is used.
 */
struct csr_ref {
  size_type const* cumulative_ends;  ///< One past the last value in each segment
  size_type const* values;           ///< Row indices concatenated in segment order

  /// Maps a segment index to its starting offset.
  struct offset_fn {
    size_type const* ends;  ///< Cumulative segment ends

    /// Returns zero for the first segment, or the previous segment's end.
    __device__ size_type operator()(cuda::std::uint32_t slot) const
    {
      return slot == 0 ? size_type{0} : ends[slot - 1];
    }
  };

  /**
   * @brief Gets the starting offset of a segment in `values`.
   *
   * @param slot Valid segment index in `cumulative_ends`
   * @return Zero for the first segment, or the previous segment's cumulative end
   */
  __device__ size_type begin(cuda::std::uint32_t slot) const
  {
    return offset_fn{cumulative_ends}(slot);
  }

  /**
   * @brief Gets the number of rows in a segment.
   *
   * @param slot Valid segment index in `cumulative_ends`
   * @return The segment's cumulative end minus its starting offset
   */
  __device__ size_type size(cuda::std::uint32_t slot) const
  {
    return cumulative_ends[slot] - begin(slot);
  }

  /**
   * @brief Returns a device iterator over segment starts.
   *
   * @pre Cumulative ends remain alive while the iterator is used.
   * @return A random-access iterator whose element at `slot` is `begin(slot)`
   */
  auto begin() const
  {
    return cuda::transform_iterator(cuda::counting_iterator{size_type{0}},
                                    offset_fn{cumulative_ends});
  }
};

}  // namespace cudf::detail
