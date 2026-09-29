/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

#include <cuda/bit>
#include <cuda/cmath>
#include <cuda/std/bit>

#include <cstdint>

namespace cudf::io::fst::detail {

/**
 * @brief A bit-packed array of items that can be backed by registers yet allows to be dynamically
 * addressed at runtime. The data structure is explained in greater detail in the paper <a
 * href="http://www.vldb.org/pvldb/vol13/p616-stehle.pdf">ParPaRaw: Massively Parallel Parsing of
 * Delimiter-Separated Raw Data</a>.
 *
 * @tparam NUM_ITEMS The maximum number of items this data structure is supposed to store
 * @tparam MAX_ITEM_VALUE The maximum value that one item can represent
 * @tparam BackingFragmentT The data type that is holding the fragments
 */
template <uint32_t NUM_ITEMS, uint32_t MAX_ITEM_VALUE, typename BackingFragmentT = uint32_t>
class MultiFragmentInRegArray {
 private:
  /// Minimum number of bits required to represent all values from [0, MAX_ITEM_VALUE]
  static constexpr uint32_t MIN_BITS_PER_ITEM =
    (MAX_ITEM_VALUE == 0) ? 1 : cuda::std::bit_width(MAX_ITEM_VALUE);

  /// Number of bits that each fragment can store
  static constexpr uint32_t NUM_BITS_PER_FRAGMENT = sizeof(BackingFragmentT) * 8;

  /// The number of bits per fragment per item in the array
  static constexpr uint32_t AVAIL_BITS_PER_FRAG_ITEM = NUM_BITS_PER_FRAGMENT / NUM_ITEMS;

  /// The number of bits per item per fragment to be a power of two to avoid costly integer
  /// multiplication
  static constexpr uint32_t BITS_PER_FRAG_ITEM = cuda::std::bit_floor(AVAIL_BITS_PER_FRAG_ITEM);

  // The total number of fragments required to store all the items
  static constexpr uint32_t FRAGMENTS_PER_ITEM =
    cuda::ceil_div(MIN_BITS_PER_ITEM, BITS_PER_FRAG_ITEM);

  BackingFragmentT data[FRAGMENTS_PER_ITEM];

  //------------------------------------------------------------------------------
  // ACCESSORS
  //------------------------------------------------------------------------------
 public:
  CUDF_HOST_DEVICE [[nodiscard]] uint32_t Get(int32_t index) const
  {
    uint32_t val = 0;

    for (uint32_t i = 0; i < FRAGMENTS_PER_ITEM; ++i) {
      val = val | cuda::bitfield_extract(data[i], index * BITS_PER_FRAG_ITEM, BITS_PER_FRAG_ITEM)
                    << (i * BITS_PER_FRAG_ITEM);
    }
    return val;
  }

  CUDF_HOST_DEVICE void Set(uint32_t index, uint32_t value)
  {
    for (uint32_t i = 0; i < FRAGMENTS_PER_ITEM; ++i) {
      uint32_t frag_bits =
        cuda::bitfield_extract(value, i * BITS_PER_FRAG_ITEM, BITS_PER_FRAG_ITEM);
      data[i] = cuda::bitfield_insert(data[i],
                                      static_cast<BackingFragmentT>(frag_bits),
                                      index * BITS_PER_FRAG_ITEM,
                                      BITS_PER_FRAG_ITEM);
    }
  }

  //------------------------------------------------------------------------------
  // CONSTRUCTORS
  //------------------------------------------------------------------------------
  CUDF_HOST_DEVICE MultiFragmentInRegArray()
  {
    for (uint32_t i = 0; i < FRAGMENTS_PER_ITEM; ++i) {
      data[i] = 0;
    }
  }

  CUDF_HOST_DEVICE MultiFragmentInRegArray(uint32_t const (&array)[NUM_ITEMS])
  {
    for (uint32_t i = 0; i < NUM_ITEMS; ++i) {
      Set(i, array[i]);
    }
  }
};

}  // namespace cudf::io::fst::detail
