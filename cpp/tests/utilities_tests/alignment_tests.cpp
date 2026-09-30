/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/alignment.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

namespace {
template <std::size_t Alignment>
struct alignas(Alignment) AlignedValue {
  char value;
};
template <typename T>
struct AlignmentTest : testing::Test {};
using AlignedTypes = testing::Types<AlignedValue<1>,
                                    AlignedValue<2>,
                                    AlignedValue<4>,
                                    AlignedValue<8>,
                                    AlignedValue<16>,
                                    AlignedValue<32>,
                                    AlignedValue<64>,
                                    AlignedValue<256>>;
TYPED_TEST_SUITE(AlignmentTest, AlignedTypes);

TYPED_TEST(AlignmentTest, EveryOffset)
{
  using T = TypeParam;
  alignas(T) char storage[sizeof(T) + 2 * alignof(T)];
  for (std::size_t offset = 0; offset < 2 * alignof(T); ++offset) {
    char* const input  = storage + offset;
    auto const output  = cudf::detail::align_ptr_for_type<T>(input);
    auto const aligned = reinterpret_cast<char*>(output);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(output) % alignof(T), 0);
    EXPECT_GE(aligned, input);
    EXPECT_LT(static_cast<std::size_t>(aligned - input), alignof(T));
    EXPECT_LE(aligned + sizeof(T), storage + sizeof(storage));
    EXPECT_EQ(cudf::detail::align_ptr_for_type<T>(output), output);
  }
}
}  // namespace
