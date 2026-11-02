/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "grouped_reductions.cuh"
#include "single_pass_reductions.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

namespace cudf::groupby::detail::hash {
namespace {

template <typename T>
concept count_type = cudf::is_integral_not_bool<T>() || cudf::is_floating_point<T>();

// Integers of at most 32 bits have exact raw moments: the sum fits int64, the sum of squares
// fits 96 bits, and n * sum_of_squares - sum^2 fits 127 bits for any size_type count.
template <typename T>
constexpr bool has_exact_moments()
{
  return cuda::std::is_integral_v<T> && sizeof(T) <= 4;
}

template <count_type T>
struct exact_moments {
  unsigned __int128 sum_of_squares;
  cuda::std::int64_t sum;
  T count;
};

template <count_type T>
struct exact_moments_plus {
  __device__ exact_moments<T> operator()(exact_moments<T> const& a, exact_moments<T> const& b) const
  {
    return {a.sum_of_squares + b.sum_of_squares, a.sum + b.sum, a.count + b.count};
  }
};

template <typename Source, count_type T>
struct grouped_exact_moments {
  size_type const* rows;
  value_accessor<Source> value;
  bool has_nulls;

  __device__ exact_moments<T> operator()(size_type position) const
  {
    auto const row = rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return {0, 0, T{0}}; }
    using wide = cuda::std::
      conditional_t<cuda::std::is_signed_v<Source>, cuda::std::int64_t, cuda::std::uint64_t>;
    auto const x = static_cast<wide>(value(row));
    // Below 2^64 for every source of at most 32 bits.
    return {static_cast<cuda::std::uint64_t>(x * x), static_cast<cuda::std::int64_t>(x), T{1}};
  }
};

template <count_type T>
struct split_exact_moments {
  __device__ cuda::std::tuple<double, T> operator()(exact_moments<T> const& state) const
  {
    if (state.count == 0) { return {0.0, state.count}; }
    auto const n = static_cast<__int128>(state.count);
    auto const s = static_cast<__int128>(state.sum);
    // n * M2, exactly, rounded once.
    auto const n_m2 = n * static_cast<__int128>(state.sum_of_squares) - s * s;
    return {static_cast<double>(n_m2) / static_cast<double>(state.count), state.count};
  }
};

// Other sources accumulate moments about a shift: the first value reduced by each thread. Since
// the shift is one of the values, s^2 <= (n - 1) * q, so M2 = q - s^2 / n >= q / n and the
// cancellation is bounded by the group size, while the per-row update stays additive.
template <count_type T>
struct shifted_moments {
  double shift;
  double sum;             // sum of (x - shift)
  double sum_of_squares;  // sum of (x - shift)^2, NaN once any value is non-finite
  T count;
};

struct shifted_value {
  double value;
  bool valid;

  template <count_type T>
  __device__ operator shifted_moments<T>() const
  {
    if (!valid) { return {0.0, 0.0, 0.0, T{0}}; }
    return {value,
            0.0,
            cuda::std::isfinite(value) ? 0.0 : cuda::std::numeric_limits<double>::quiet_NaN(),
            T{1}};
  }
};

template <count_type T>
struct merge_shifted_moments {
  // Moves b onto a's shift d = b.shift - a.shift: sum += n_b * d, and
  // sum_of_squares += 2 * d * sum_b + n_b * d^2. A non-finite d leaves the result non-finite.
  __device__ shifted_moments<T> operator()(shifted_moments<T> const& a,
                                           shifted_moments<T> const& b) const
  {
    if (a.count == 0) { return b; }
    if (b.count == 0) { return a; }
    auto const d     = b.shift - a.shift;
    auto const n_b_d = static_cast<double>(b.count) * d;
    return {a.shift,
            a.sum + b.sum + n_b_d,
            a.sum_of_squares + b.sum_of_squares + d * (2.0 * b.sum + n_b_d),
            a.count + b.count};
  }

  // The per-row fold: b is a single value, so d = b.value - a.shift and b.sum == 0.
  __device__ shifted_moments<T> operator()(shifted_moments<T> const& a,
                                           shifted_value const& b) const
  {
    if (!b.valid) { return a; }
    if (a.count == 0) { return static_cast<shifted_moments<T>>(b); }
    auto const d = b.value - a.shift;
    return {a.shift, a.sum + d, cuda::std::fma(d, d, a.sum_of_squares), a.count + T{1}};
  }
};

template <typename Source>
struct grouped_shifted_value {
  size_type const* rows;
  value_accessor<Source> value;
  bool has_nulls;

  __device__ shifted_value operator()(size_type position) const
  {
    auto const row = rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return {0.0, false}; }
    return {static_cast<double>(value(row)), true};
  }
};

template <count_type T>
struct split_shifted_moments {
  __device__ cuda::std::tuple<double, T> operator()(shifted_moments<T> const& state) const
  {
    if (state.count == 0) { return {0.0, state.count}; }
    // Any non-finite value, or overflow while shifting, yields NaN regardless of merge order.
    if (!cuda::std::isfinite(state.sum_of_squares)) {
      return {cuda::std::numeric_limits<double>::quiet_NaN(), state.count};
    }
    auto const m2 =
      state.sum_of_squares - state.sum * (state.sum / static_cast<double>(state.count));
    return {m2 < 0.0 ? 0.0 : m2, state.count};
  }
};

template <count_type CountT>
struct m2_dispatch {
  template <typename T>
  void operator()(reduction_context const& ctx,
                  double* m2,
                  CountT* count,
                  cuda::stream_ref stream,
                  cudf::memory_resources mr) const
  {
    if constexpr (has_exact_moments<T>()) {
      auto const values = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_exact_moments<T, CountT>{
          ctx.grouped.rows.data(), value_accessor<T>{ctx.d_values}, ctx.values.has_nulls()});
      auto const output = cuda::transform_output_iterator{cuda::make_zip_iterator(m2, count),
                                                          split_exact_moments<CountT>{}};
      reduce_groups(ctx.grouped,
                    values,
                    output,
                    exact_moments_plus<CountT>{},
                    exact_moments<CountT>{0, 0, CountT{0}},
                    stream,
                    mr);
    } else if constexpr (cudf::is_numeric<T>() && !cudf::is_fixed_point<T>()) {
      auto const values = cudf::detail::make_counting_transform_iterator(
        0,
        grouped_shifted_value<T>{
          ctx.grouped.rows.data(), value_accessor<T>{ctx.d_values}, ctx.values.has_nulls()});
      auto const output = cuda::transform_output_iterator{cuda::make_zip_iterator(m2, count),
                                                          split_shifted_moments<CountT>{}};
      reduce_groups(ctx.grouped,
                    values,
                    output,
                    merge_shifted_moments<CountT>{},
                    shifted_moments<CountT>{0.0, 0.0, 0.0, CountT{0}},
                    stream,
                    mr);
    } else {
      CUDF_FAIL("Invalid source type for M2 aggregation.");
    }
  }
};

}  // namespace

std::pair<std::unique_ptr<column>, std::unique_ptr<column>> compute_m2_and_count(
  reduction_context const& ctx,
  bool m2_intermediate,
  bool count_intermediate,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  using count_type = size_type;
  // Both results are non-nullable, including zero M2/count for all-null groups.
  auto m2    = make_numeric_column(data_type{type_id::FLOAT64},
                                ctx.num_groups,
                                mask_state::UNALLOCATED,
                                stream,
                                m2_intermediate ? mr.get_temporary_mr() : mr.get_output_mr());
  auto count = make_numeric_column(data_type{cudf::type_to_id<count_type>()},
                                   ctx.num_groups,
                                   mask_state::UNALLOCATED,
                                   stream,
                                   count_intermediate ? mr.get_temporary_mr() : mr.get_output_mr());
  if (ctx.num_groups > 0) {
    type_dispatcher(ctx.values_type,
                    m2_dispatch<count_type>{},
                    ctx,
                    m2->mutable_view().begin<double>(),
                    count->mutable_view().begin<count_type>(),
                    stream,
                    mr);
  }
  return {std::move(m2), std::move(count)};
}

}  // namespace cudf::groupby::detail::hash
