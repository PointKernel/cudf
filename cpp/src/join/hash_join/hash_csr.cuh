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
#include <cuda/cmath>
#include <cuda/std/cstdint>

namespace cudf::detail {

/// A row index and the high bits of its hash share one 32-bit table entry.
using hash_table_entry_type = cuda::std::uint32_t;

/// Device-side view of the open-addressed table. The low bits hold the representative build
/// row, and the remaining bits hold a hash fingerprint. Fingerprint matches always undergo
/// a full row comparison; reducing the fingerprint width cannot introduce false matches.
struct hash_table_ref {
  hash_table_entry_type* entries;
  cuda::std::uint32_t capacity;
  cuda::std::uint32_t row_mask;
  cuda::fast_mod_div<cuda::std::uint32_t> modulo;

  template <typename Equal>
  __device__ bool equal(cuco::pair<hash_value_type, size_type> key,
                        hash_table_entry_type entry,
                        Equal check_row_equality) const
  {
    return ((key.first ^ entry) & ~row_mask) == 0 &&
           check_row_equality(key, {key.first, static_cast<size_type>(entry & row_mask)});
  }

  template <typename Equal>
  __device__ size_type insert(cuco::pair<hash_value_type, size_type> key, Equal equal_rows) const
  {
    auto const desired = (key.first & ~row_mask) | static_cast<cuda::std::uint32_t>(key.second);
    auto slot          = key.first % modulo;
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      auto entry_ref =
        cuda::atomic_ref<hash_table_entry_type, cuda::thread_scope_device>{entries[slot]};
      auto old = hash_table_entry_type{-1};
      if (entry_ref.compare_exchange_strong(old, desired, cuda::memory_order_relaxed)) {
        return key.second;
      }
      if (equal(key, old, equal_rows)) { return static_cast<size_type>(old & row_mask); }
      ++slot;
      if (slot == capacity) { slot = 0; }
    }
    return CUDF_SIZE_TYPE_SENTINEL;
  }

  template <bool IsBuild = false, typename Equal>
  __device__ size_type find(cuco::pair<hash_value_type, size_type> key, Equal equal_rows) const
  {
    auto slot = key.first % modulo;
    for (cuda::std::uint32_t step = 0; step < capacity; ++step) {
      auto const current = entries[slot];
      if (current == hash_table_entry_type{-1}) { return CUDF_SIZE_TYPE_SENTINEL; }
      // Under null_equality::UNEQUAL a nested row containing nulls need not equal itself.
      // The fill pass must still find the row that claimed this entry during construction.
      if constexpr (IsBuild) {
        if (static_cast<size_type>(current & row_mask) == key.second) { return key.second; }
      }
      if (equal(key, current, equal_rows)) { return static_cast<size_type>(current & row_mask); }
      ++slot;
      if (slot == capacity) { slot = 0; }
    }
    return CUDF_SIZE_TYPE_SENTINEL;
  }
};

/// CSR segments are indexed by representative build row, including zero-length segments for
/// rows that did not claim a table entry. This avoids an offset for every empty hash slot.
struct csr_ref {
  size_type const* offsets;
  size_type const* values;

  __device__ size_type begin(size_type row) const { return offsets[row]; }

  __device__ size_type size(size_type row) const { return offsets[row + 1] - offsets[row]; }
};

}  // namespace cudf::detail
