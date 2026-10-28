/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/join/join_common.hpp>
#include <benchmarks/join/nvbench_helpers.hpp>

#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_inner_join(nvbench::state& state,
                        nvbench::type_list<nvbench::enum_type<Nullable>,
                                           nvbench::enum_type<NullEquality>,
                                           nvbench::enum_type<DataType>>)
{
  auto const num_keys = state.get_int64("num_keys");
  auto const mode     = state.get_string("mode");
  auto dtypes         = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  if (mode == "normal") {
    auto join = [](cudf::table_view const& left_input,
                   cudf::table_view const& right_input,
                   cudf::null_equality compare_nulls) {
      return cudf::inner_join(left_input, right_input, compare_nulls);
    };
    BM_join<Nullable, join_t::HASH, NullEquality>(state, dtypes, join);
  } else if (mode == "partitioned") {
    // Partitioned code path: build hash join, compute match context, then retrieve the
    // entire probe table as a single partition.  This exercises the two-phase
    // count-then-retrieve flow used for chunked probing.
    auto join = [](cudf::table_view const& left_input,
                   cudf::table_view const& right_input,
                   cudf::null_equality compare_nulls) {
      auto hash_joiner = cudf::hash_join(right_input, compare_nulls);
      auto matches     = hash_joiner.inner_join_match_context(left_input);
      return hash_joiner.partitioned_inner_join(matches, 0, left_input.num_rows());
    };
    BM_join<Nullable, join_t::HASH, NullEquality>(state, dtypes, join);
  } else {
    CUDF_FAIL("unrecognized mode: " + mode);
  }
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_left_join(nvbench::state& state,
                       nvbench::type_list<nvbench::enum_type<Nullable>,
                                          nvbench::enum_type<NullEquality>,
                                          nvbench::enum_type<DataType>>)
{
  auto const num_keys = state.get_int64("num_keys");
  auto dtypes         = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    return cudf::left_join(left_input, right_input, compare_nulls);
  };
  BM_join<Nullable, join_t::HASH, NullEquality>(state, dtypes, join);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_full_join(nvbench::state& state,
                       nvbench::type_list<nvbench::enum_type<Nullable>,
                                          nvbench::enum_type<NullEquality>,
                                          nvbench::enum_type<DataType>>)
{
  auto const num_keys = state.get_int64("num_keys");
  auto dtypes         = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    return cudf::full_join(left_input, right_input, compare_nulls);
  };
  BM_join<Nullable, join_t::HASH, NullEquality>(state, dtypes, join);
}

template <cudf::null_equality NullEquality, data_type DataType>
void nvbench_inner_join_selectivity(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<NullEquality>, nvbench::enum_type<DataType>>)
{
  auto constexpr num_keys = 1;
  auto const num_probes   = static_cast<cudf::size_type>(state.get_int64("num_probes"));
  auto const selectivity  = state.get_float64("selectivity");
  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto join = [num_probes](cudf::table_view const& left_input,
                           cudf::table_view const& right_input,
                           cudf::null_equality compare_nulls) {
    auto hash_join = cudf::hash_join(right_input, compare_nulls);
    for (auto i = 0; i < num_probes - 1; i++) {
      [[maybe_unused]] auto result = hash_join.inner_join(left_input);
    }
    return hash_join.inner_join(left_input);
  };

  BM_join<false, join_t::HASH, NullEquality>(state, dtypes, join, 1, selectivity);
}

// Keep the build hash table outside the timed region to measure the repeated-batch API used
// by callers that reserve gather-map memory before retrieving a batch.
void nvbench_inner_join_memory_planning(nvbench::state& state)
{
  auto const rows = static_cast<cudf::size_type>(state.get_int64("left_size"));
  auto const mode = state.get_string("mode");
  auto [right, left] =
    generate_input_tables<false>({cudf::type_id::INT32}, 100'000, rows, 0, 1, 0.5);
  auto const left_view = left->view();
  cudf::hash_join join{right->view(), cudf::null_equality::UNEQUAL};
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    if (mode == "count_then_retrieve") {
      auto const size = join.inner_join_size(left_view);
      auto result     = join.inner_join(left_view, size);
    } else if (mode == "cached_context") {
      auto matches = join.inner_join_match_context(left_view);
      auto result  = join.partitioned_inner_join(matches, 0, rows);
      CUDF_EXPECTS(result.first->size() == matches.output_size(), "Incorrect cached join size");
    } else {
      auto result = join.inner_join(left_view);
    }
  });
}

NVBENCH_BENCH(nvbench_inner_join_memory_planning)
  .set_name("inner_join_memory_planning")
  .add_int64_axis("left_size", {1'000, 100'000, 1'000'000})
  .add_string_axis("mode", {"count_then_retrieve", "cached_context", "direct"});

NVBENCH_BENCH_TYPES(nvbench_inner_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      JOIN_DATATYPES))
  .set_name("inner_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("num_keys", nvbench::range(1, 5, 1))
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("skip_large_sizes", {1})
  .add_string_axis("mode", {"normal", "partitioned"});

NVBENCH_BENCH_TYPES(nvbench_left_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      JOIN_DATATYPES))
  .set_name("left_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("num_keys", nvbench::range(1, 5, 1))
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("skip_large_sizes", {1});

NVBENCH_BENCH_TYPES(nvbench_full_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      JOIN_DATATYPES))
  .set_name("full_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("num_keys", nvbench::range(1, 5, 1))
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("skip_large_sizes", {1});

NVBENCH_BENCH_TYPES(nvbench_inner_join_selectivity,
                    NVBENCH_TYPE_AXES(DEFAULT_JOIN_NULL_EQUALITY, SELECTIVITY_JOIN_DATATYPES))
  .set_name("inner_join_selectivity")
  .set_type_axes_names({"NullEquality", "DataType"})
  .add_int64_axis("left_size", {100'000'000})
  .add_int64_axis("right_size", {100'000})
  .add_int64_axis("num_probes", {4})
  .add_float64_axis("selectivity", JOIN_SELECTIVITY_RANGE)
  .add_int64_axis("skip_large_sizes", {1});
