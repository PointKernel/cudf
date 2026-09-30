/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/memory>

namespace cudf {
namespace detail {

/**
 * @brief Returns the aligned address for holding array of type T in pre-allocated memory.
 *
 * @tparam T The data type to align upon.
 *
 * @param destination pointer to pre-allocated contiguous storage to store type T.
 * @return Pointer of type T, aligned to alignment of type T.
 */
template <typename T>
T* align_ptr_for_type(void* destination)
{
  return static_cast<T*>(cuda::align_up(destination, alignof(T)));
}

}  // namespace detail
}  // namespace cudf
