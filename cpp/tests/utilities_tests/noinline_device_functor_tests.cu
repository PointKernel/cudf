/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/noinline_device_functor.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/transform.h>

namespace {

struct scaled_offset {
  scaled_offset() = delete;

  scaled_offset(int scale, int offset) : scale{scale}, offset{offset} {}

  scaled_offset& operator=(scaled_offset const&) = delete;

  __device__ int operator()(int value) const noexcept { return scale * value + offset; }

  int scale;
  int offset;
};

}  // namespace

using NoinlineDeviceFunctorTest = cudf::test::BaseFixture;

TEST_F(NoinlineDeviceFunctorTest, CallsStoredFunctor)
{
  auto const stream  = cudf::get_default_stream();
  auto const functor = cudf::detail::noinline_device_functor{scaled_offset{3, 1}, stream};
  static_assert(sizeof(functor.device_ref()) == sizeof(void*));

  rmm::device_uvector<int> result(4, stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator<int>{0},
                    cuda::counting_iterator<int>{4},
                    result.begin(),
                    functor.device_ref());

  auto const actual = cudf::detail::make_pinned_vector(result, stream);
  ASSERT_EQ(actual.size(), 4);
  EXPECT_EQ(actual[0], 1);
  EXPECT_EQ(actual[1], 4);
  EXPECT_EQ(actual[2], 7);
  EXPECT_EQ(actual[3], 10);
}
