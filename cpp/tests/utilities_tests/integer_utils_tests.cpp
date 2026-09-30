/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/integer_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {
template <typename T>
struct IntegerUtilsTest : testing::Test {};
using IntegerTypes =
  testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>;
TYPED_TEST_SUITE(IntegerUtilsTest, IntegerTypes);

TYPED_TEST(IntegerUtilsTest, RoundingBoundaries)
{
  using T             = TypeParam;
  constexpr T max     = std::numeric_limits<T>::max();
  constexpr T divisor = 3;
  for (T value : {T{0}, T{1}, T{2}, T{3}, T{4}, T{max - 1}, max}) {
    auto const quotient  = value / divisor;
    auto const remainder = value % divisor;
    EXPECT_EQ(cudf::util::div_rounding_up_safe(value, divisor), quotient + (remainder != 0));
    EXPECT_EQ(cudf::util::round_down_safe(value, divisor), quotient * divisor);
    if (remainder == 0 || max - value >= divisor - remainder) {
      EXPECT_EQ(cudf::util::round_up_unsafe(value, divisor),
                (quotient + (remainder != 0)) * divisor);
    }
  }
  EXPECT_EQ(cudf::util::div_rounding_up_safe(max, max), T{1});
  EXPECT_EQ(cudf::util::div_rounding_up_safe(T{0}, max), T{0});
  EXPECT_EQ(cudf::util::div_rounding_up_safe(T{1}, max), T{1});
  EXPECT_EQ(cudf::util::div_rounding_up_safe(max, T{1}), max);
  EXPECT_EQ(cudf::util::round_up_unsafe(max, T{1}), max);
  EXPECT_EQ(cudf::util::round_down_safe(max, T{1}), max);
}

TYPED_TEST(IntegerUtilsTest, SignedDivisionCompatibility)
{
  using T = TypeParam;
  if constexpr (std::is_signed_v<T>) {
    for (T dividend : {T{-7}, T{-6}, T{0}, T{6}, T{7}, std::numeric_limits<T>::min()}) {
      for (T divisor : {T{-3}, T{3}}) {
        EXPECT_EQ(cudf::util::div_rounding_up_safe(dividend, divisor),
                  dividend / divisor + (dividend % divisor != 0));
      }
    }
  }
}

static_assert(cudf::util::div_rounding_up_safe(uint64_t{0}, uint64_t{3}) == 0);
static_assert(cudf::util::div_rounding_up_safe(std::numeric_limits<uint64_t>::max(), uint64_t{2}) ==
              (uint64_t{1} << 63));
static_assert(cudf::util::round_up_unsafe(7, 3) == 9);
static_assert(cudf::util::round_down_safe(7, 3) == 6);
}  // namespace
