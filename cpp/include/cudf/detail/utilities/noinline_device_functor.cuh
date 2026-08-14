/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/std/utility>

#include <cstddef>
#include <cstring>

namespace cudf::detail {

/**
 * @brief Device-callable reference to a functor stored in device memory
 *
 * Invokes the stored functor through a non-inlined device call. This prevents the functor's
 * template graph from being inlined into callers and keeps the kernel argument pointer-sized.
 *
 * @tparam Callable Device-callable functor type
 */
template <typename Callable>
struct noinline_device_functor_ref {
  Callable const* callable;  ///< Non-owning device pointer to the functor

  template <typename... Args>
  __attribute__((noinline)) __device__ decltype(auto) operator()(Args&&... args) const
    noexcept(noexcept((*callable)(cuda::std::forward<Args>(args)...)))
  {
    return (*callable)(cuda::std::forward<Args>(args)...);
  }
};

/**
 * @brief Owns a device-resident functor invoked through a non-inlined device reference
 *
 * Copies the object representation of `callable` through pinned host storage into device memory.
 * The callable and any pointees it contains must remain valid for device use. This object must
 * outlive all work that uses a reference returned by `device_ref()`.
 *
 * @note The caller must synchronize the stream before this object's destruction unless later work
 * in the object's scope already synchronizes it.
 *
 * @tparam Callable Device-callable functor type whose object representation can be copied to device
 */
template <typename Callable>
class noinline_device_functor {
 public:
  /**
   * @brief Constructs a device-resident copy of `callable`
   *
   * @param callable Functor to copy to device memory
   * @param stream Stream on which to allocate memory and enqueue the copy
   * @param mr Device memory resource used to allocate storage for the functor
   */
  explicit noinline_device_functor(
    Callable const& callable,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
    : _host_callable{make_pinned_vector_async<std::byte>(sizeof(Callable), stream)},
      _device_callable{sizeof(Callable), stream, mr}
  {
    std::memcpy(_host_callable.data(), &callable, sizeof(Callable));
    cudf::host_span<std::byte const> const source = _host_callable;
    cuda_memcpy_async(cudf::device_span<std::byte>{static_cast<std::byte*>(_device_callable.data()),
                                                   sizeof(Callable)},
                      source,
                      stream);
  }

  noinline_device_functor(noinline_device_functor&&) noexcept            = default;
  noinline_device_functor& operator=(noinline_device_functor&&) noexcept = default;

  noinline_device_functor(noinline_device_functor const&)            = delete;
  noinline_device_functor& operator=(noinline_device_functor const&) = delete;

  /**
   * @brief Returns a pointer-sized device-callable reference to the stored functor
   *
   * @return A non-owning reference valid while this object and the callable's pointees remain valid
   */
  [[nodiscard]] noinline_device_functor_ref<Callable> device_ref() const noexcept
  {
    return {static_cast<Callable const*>(_device_callable.data())};
  }

 private:
  cudf::detail::host_vector<std::byte> _host_callable;
  rmm::device_buffer _device_callable;
};

}  // namespace cudf::detail
