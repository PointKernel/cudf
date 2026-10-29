/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/sort_merge_join.hpp>

#include <rmm/cuda_stream.hpp>

#include <algorithm>
#include <future>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {
using index_type = cudf::size_type;
using column     = cudf::test::fixed_width_column_wrapper<index_type>;
using pairs      = std::vector<std::pair<index_type, index_type>>;

template <typename Result>
pairs to_host_pairs(Result const& result, cuda::stream_ref stream)
{
  auto left  = cudf::detail::make_host_vector_async(*result.first, stream);
  auto right = cudf::detail::make_host_vector_async(*result.second, stream);
  stream.sync();
  pairs output;
  for (std::size_t i = 0; i < left.size(); ++i) {
    output.emplace_back(left[i], right[i]);
  }
  std::sort(output.begin(), output.end());
  return output;
}

template <typename Join>
cudf::join_match_context make_inner_context(Join const& join,
                                            cudf::table_view left,
                                            cuda::stream_ref stream)
{
  if constexpr (std::is_same_v<Join, cudf::hash_join>) {
    return join.inner_join_match_context(left, stream);
  } else {
    return std::move(*join.inner_join_match_context(left, stream));
  }
}

template <typename Join>
void check_inner(Join const& join, cudf::table_view left, cuda::stream_ref stream)
{
  auto matches = make_inner_context(join, left, stream);
  auto counts  = cudf::detail::make_host_vector_async(matches.match_counts(), stream);
  stream.sync();
  auto const expected = [&] {
    if constexpr (std::is_same_v<Join, cudf::hash_join>) {
      return to_host_pairs(join.inner_join(left, std::nullopt, stream), stream);
    } else {
      return to_host_pairs(join.inner_join(left, stream), stream);
    }
  }();
  EXPECT_EQ(expected.size(), matches.output_size());
  EXPECT_EQ(matches.output_size(), std::accumulate(counts.begin(), counts.end(), std::size_t{0}));
  EXPECT_EQ(left.num_rows(), matches.left_table().num_rows());
  EXPECT_EQ(left.column(0).head<void>(), matches.left_table().column(0).template head<void>());

  // Moving preserves borrowed device views and retrieval state.
  auto const count_data = matches.match_counts().data();
  auto moved            = std::move(matches);
  EXPECT_EQ(count_data, moved.match_counts().data());
  EXPECT_THROW(std::ignore = join.partitioned_inner_join(matches, 0, 0, stream),
               std::invalid_argument);
  pairs combined;
  for (index_type start = 0; start < left.num_rows(); start += 2) {
    auto result =
      join.partitioned_inner_join(moved, start, std::min(start + 2, left.num_rows()), stream);
    auto part = to_host_pairs(result, stream);
    combined.insert(combined.end(), part.begin(), part.end());
  }
  std::sort(combined.begin(), combined.end());
  EXPECT_EQ(expected, combined);
  EXPECT_EQ(expected,
            to_host_pairs(join.partitioned_inner_join(moved, 0, left.num_rows(), stream), stream));
  EXPECT_TRUE(
    to_host_pairs(join.partitioned_inner_join(moved, left.num_rows(), left.num_rows(), stream),
                  stream)
      .empty());
  EXPECT_THROW(std::ignore = join.partitioned_inner_join(moved, -1, 0, stream),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = join.partitioned_inner_join(moved, 1, 0, stream),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = join.partitioned_inner_join(moved, 0, left.num_rows() + 1, stream),
               std::invalid_argument);
}
}  // namespace

class JoinMatchContextTest : public cudf::test::BaseFixture {};

TEST_F(JoinMatchContextTest, InnerPartitionsAndNulls)
{
  auto const stream = cudf::test::get_default_stream();
  column left_storage{{99, 3, 1, 2, 0, 2, 5, 99}, {1, 1, 0, 1, 1, 1, 0, 1}};
  column right_keys{{2, 2, 0, 4, 3, 8}, {1, 1, 1, 1, 1, 0}};
  auto left = cudf::slice(cudf::table_view{{left_storage}}, {1, 7}, stream)[0];
  cudf::table_view right{{right_keys}};
  for (auto nulls : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
    cudf::hash_join hash{right, nulls, stream};
    cudf::sort_merge_join merge{right, cudf::sorted::NO, nulls, stream};
    check_inner(hash, left, stream);
    check_inner(merge, left, stream);
  }
}

TEST_F(JoinMatchContextTest, NestedAndStringKeys)
{
  auto const stream = cudf::test::get_default_stream();
  column left_child{{1, 2, 2, 7}, {1, 0, 1, 1}};
  column right_child{{2, 2, 1}, {1, 0, 1}};
  cudf::test::structs_column_wrapper left_struct{{left_child}};
  cudf::test::structs_column_wrapper right_struct{{right_child}};
  cudf::test::strings_column_wrapper left_string{{"a", "b", "b", "c"}};
  cudf::test::strings_column_wrapper right_string{{"b", "b", "a"}};
  cudf::table_view left{{left_struct, left_string}};
  cudf::table_view right{{right_struct, right_string}};
  for (auto nulls : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
    cudf::hash_join hash{right, nulls, stream};
    cudf::sort_merge_join merge{right, cudf::sorted::NO, nulls, stream};
    check_inner(hash, left, stream);
    check_inner(merge, left, stream);
  }
}

TEST_F(JoinMatchContextTest, EmptyInputs)
{
  auto const stream = cudf::test::get_default_stream();
  column empty{};
  column nonempty{{1, 1, 2}};
  for (auto left : {cudf::table_view{{empty}}, cudf::table_view{{nonempty}}}) {
    for (auto right : {cudf::table_view{{empty}}, cudf::table_view{{nonempty}}}) {
      cudf::hash_join hash{right, cudf::null_equality::EQUAL, stream};
      cudf::sort_merge_join merge{right, cudf::sorted::NO, cudf::null_equality::EQUAL, stream};
      check_inner(hash, left, stream);
      check_inner(merge, left, stream);
      auto matches  = hash.left_join_match_context(left, stream);
      auto expected = to_host_pairs(hash.left_join(left, std::nullopt, stream), stream);
      EXPECT_EQ(
        expected,
        to_host_pairs(hash.partitioned_left_join(matches, 0, left.num_rows(), stream), stream));
      if (left.num_rows() > 1) {
        std::erase_if(expected, [](auto const& pair) { return pair.first < 1; });
        EXPECT_EQ(
          expected,
          to_host_pairs(hash.partitioned_left_join(matches, 1, left.num_rows(), stream), stream));
      }
    }
  }
}

TEST_F(JoinMatchContextTest, LeftAndFullPartitions)
{
  auto const stream = cudf::test::get_default_stream();
  column left_keys{{1, 2, 7, 0}, {1, 1, 1, 0}};
  column right_keys{{2, 2, 3, 0}, {1, 1, 1, 0}};
  cudf::table_view left{{left_keys}}, right{{right_keys}};
  for (auto nulls : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
    cudf::hash_join hash{right, nulls, stream};
    auto matches = hash.left_join_match_context(left, stream);
    EXPECT_EQ(to_host_pairs(hash.left_join(left, std::nullopt, stream), stream),
              to_host_pairs(hash.partitioned_left_join(matches, 0, 4, stream), stream));
    EXPECT_THROW(std::ignore = hash.partitioned_inner_join(matches, 0, 4, stream),
                 std::invalid_argument);

    auto full_matches = hash.full_join_match_context(left, stream);
    EXPECT_EQ(matches.output_size(), full_matches.output_size());
    auto first  = hash.partitioned_full_join(full_matches, 0, 2, stream);
    auto second = hash.partitioned_full_join(full_matches, 2, 4, stream);
    std::vector<cudf::device_span<index_type const>> left_parts{*first.first, *second.first};
    std::vector<cudf::device_span<index_type const>> right_parts{*first.second, *second.second};
    auto full =
      cudf::hash_join::finalize_partitioned_full_join(left_parts, right_parts, 4, 4, stream);
    EXPECT_EQ(to_host_pairs(hash.full_join(left, std::nullopt, stream), stream),
              to_host_pairs(full, stream));
  }
}

TEST_F(JoinMatchContextTest, WrongOwnerAndMoveAssignment)
{
  auto const stream = cudf::test::get_default_stream();
  column keys{{1, 2, 2}};
  cudf::table_view table{{keys}};
  cudf::hash_join first{table, cudf::null_equality::EQUAL, stream};
  cudf::hash_join second{table, cudf::null_equality::EQUAL, stream};
  cudf::sort_merge_join merge{table, cudf::sorted::NO, cudf::null_equality::EQUAL, stream};
  auto matches = first.inner_join_match_context(table, stream);
  auto other   = second.inner_join_match_context(table, stream);
  EXPECT_THROW(std::ignore = second.partitioned_inner_join(matches, 0, 3, stream),
               std::invalid_argument);
  EXPECT_THROW(std::ignore = merge.partitioned_inner_join(matches, 0, 3, stream),
               std::invalid_argument);
  other = std::move(matches);
  EXPECT_EQ(5, first.partitioned_inner_join(other, 0, 3, stream).first->size());
  auto merge_matches = merge.inner_join_match_context(table, stream);
  EXPECT_THROW(std::ignore = first.partitioned_inner_join(*merge_matches, 0, 3, stream),
               std::invalid_argument);
}

TEST_F(JoinMatchContextTest, ConcurrentRetrievalAndResources)
{
  auto const stream = cudf::test::get_default_stream();
  column keys{{1, 2, 2, 3}};
  cudf::table_view table{{keys}};
  cudf::hash_join hash{table, cudf::null_equality::EQUAL, stream};
  cudf::test::memory_resource_test_harness memory;
  auto matches = hash.inner_join_match_context(table, stream, memory.resources());
  memory.expect_output_allocations_live(stream);
  memory.expect_temporary_allocations_released(stream);
  auto retrieve = [&] {
    rmm::cuda_stream independent;
    return to_host_pairs(hash.partitioned_inner_join(matches, 0, 4, independent), independent);
  };
  auto first  = std::async(std::launch::async, retrieve);
  auto second = std::async(std::launch::async, retrieve);
  EXPECT_EQ(first.get(), second.get());
  {
    // Once prepared, retrieval needs neither row preprocessing nor the current resource.
    auto fail_current = memory.fail_on_current_device_resource_use();
    auto output       = hash.partitioned_inner_join(matches, 1, 3, stream, memory.resources());
    EXPECT_EQ(output.first->size(), 4);
    stream.sync();
  }
  memory.expect_temporary_allocations_released(stream);
}

TEST_F(JoinMatchContextTest, OutputSizeExceedsInt32)
{
  auto const stream         = cudf::test::get_default_stream();
  constexpr index_type rows = 50'000;
  std::vector<index_type> values(rows, 1);
  column keys(values.begin(), values.end());
  cudf::table_view table{{keys}};
  cudf::hash_join hash{table, cudf::null_equality::EQUAL, stream};
  cudf::sort_merge_join merge{table, cudf::sorted::NO, cudf::null_equality::EQUAL, stream};
  auto check = [&](auto const& join) {
    auto matches = make_inner_context(join, table, stream);
    EXPECT_EQ(matches.output_size(), std::size_t{rows} * rows);
    // Retrieve a small partition without allocating the full 2.5 billion output pairs.
    auto output = to_host_pairs(join.partitioned_inner_join(matches, 7, 9, stream), stream);
    ASSERT_EQ(output.size(), std::size_t{2} * rows);
    for (index_type row = 0; row < rows; ++row) {
      EXPECT_EQ(output[row], std::make_pair(index_type{7}, row));
      EXPECT_EQ(output[rows + row], std::make_pair(index_type{8}, row));
    }
  };
  check(hash);
  check(merge);
}

TEST_F(JoinMatchContextTest, ExistingPartitionApiUsesCache)
{
  auto const stream = cudf::test::get_default_stream();
  column keys{{1, 2, 2, 3}};
  cudf::table_view table{{keys}};
  cudf::hash_join hash{table, cudf::null_equality::EQUAL, stream};
  cudf::test::memory_resource_test_harness memory;
  auto context = hash.inner_join_match_context(table, stream, memory.resources());
  cudf::join_partition_context partition{
    std::make_unique<cudf::join_match_context>(std::move(context)), 1, 3};
  auto fail_current = memory.fail_on_current_device_resource_use();
  auto output       = hash.partitioned_inner_join(partition, stream, memory.resources());
  EXPECT_EQ(output.first->size(), 4);
  stream.sync();
}

TEST_F(JoinMatchContextTest, ExternallyConstructedContext)
{
  auto const stream = cudf::test::get_default_stream();
  column keys{{1, 2, 2, 3}};
  cudf::table_view table{{keys}};
  cudf::hash_join hash{table, cudf::null_equality::EQUAL, stream};
  auto cached = hash.inner_join_match_context(table, stream);
  auto counts =
    std::make_unique<rmm::device_uvector<index_type>>(cudf::detail::make_device_uvector_async(
      cached.match_counts(), stream, cudf::get_current_device_resource_ref()));
  cudf::join_match_context external{table, std::move(counts)};
  EXPECT_THROW(std::ignore = external.output_size(), std::invalid_argument);
  EXPECT_EQ(to_host_pairs(hash.partitioned_inner_join(cached, 1, 3, stream), stream),
            to_host_pairs(hash.partitioned_inner_join(external, 1, 3, stream), stream));
}
