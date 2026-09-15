/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/join/hash_join.hpp>
#include <cudf/detail/utilities/hash_csr.cuh>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/cstdint>

#include <cstdint>
#include <utility>

namespace cudf::detail {

using hash_set_key_type = cuco::pair<hash_value_type, size_type>;

template <typename Hasher>
struct hash_join<Hasher>::impl {
  impl(cuda::std::uint32_t capacity,
       size_type rows,
       cuda::stream_ref stream,
       cuda::mr::any_resource<cuda::mr::device_accessible> mr)
    : _mr(std::move(mr)),
      _slots(capacity, stream, _mr),
      _cumulative_ends(capacity, stream, _mr),
      _values(rows, stream, _mr),
      _capacity(capacity)
  {
  }

  hash_set_ref<hash_set_key_type> hash_set() const
  {
    return {const_cast<hash_set_key_type*>(_slots.data()), _capacity};
  }

  csr_ref csr() const { return {_cumulative_ends.data(), _values.data()}; }

  cuda::mr::any_resource<cuda::mr::device_accessible> _mr;
  rmm::device_uvector<hash_set_key_type> _slots;
  rmm::device_uvector<size_type> _cumulative_ends;
  rmm::device_uvector<size_type> _values;
  cuda::std::uint32_t _capacity;
};

}  // namespace cudf::detail
