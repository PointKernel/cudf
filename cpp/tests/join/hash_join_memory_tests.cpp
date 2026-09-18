/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

using cudf::size_type;
using join_pair   = std::pair<size_type, size_type>;
using join_result = std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                              std::unique_ptr<rmm::device_uvector<size_type>>>;
template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

std::vector<join_pair> sorted_host_pairs(join_result const& result)
{
  auto const stream = cudf::test::get_default_stream();
  auto const left =
    cudf::detail::make_host_vector(cudf::device_span<size_type const>{*result.first}, stream);
  auto const right =
    cudf::detail::make_host_vector(cudf::device_span<size_type const>{*result.second}, stream);
  std::vector<join_pair> pairs;
  pairs.reserve(left.size());
  for (std::size_t i = 0; i < left.size(); ++i) {
    pairs.emplace_back(left[i], right[i]);
  }
  std::sort(pairs.begin(), pairs.end());
  return pairs;
}

struct HashJoinMemoryTest : public cudf::test::BaseFixture {};

TEST_F(HashJoinMemoryTest, EqualFingerprintsStillCompareKeys)
{
  // These different int64 values have the same full Murmur3 hash, and therefore any shorter
  // fingerprint also collides. Repeated rows must remain in separate equality groups.
  constexpr int64_t first  = 2708834937922957298;
  constexpr int64_t second = 9018382015938582086;
  column_wrapper<int64_t> collision_keys{{first, second}};
  auto const hashes      = cudf::hashing::murmurhash3_x86_32(cudf::table_view{{collision_keys}});
  auto const host_hashes = cudf::detail::make_host_vector(
    cudf::device_span<cudf::hash_value_type const>{hashes->view().data<cudf::hash_value_type>(), 2},
    cudf::test::get_default_stream());
  ASSERT_EQ(host_hashes[0], host_hashes[1]);
  // Seven build rows use 14 slots at the default load factor. Both keys start at the final
  // slot, so insertion and lookup of one distinct key must wrap across slot zero.
  ASSERT_EQ(host_hashes[0] % 14, 13);

  column_wrapper<int64_t> right{{first, second, first, second, second, first, second}};
  column_wrapper<int64_t> left{{second, first, int64_t{0}}};
  cudf::hash_join joiner{cudf::table_view{{right}}, cudf::null_equality::EQUAL};
  auto const result = joiner.left_join(cudf::table_view{{left}});
  std::vector<join_pair> const expected{
    {0, 1}, {0, 3}, {0, 4}, {0, 6}, {1, 0}, {1, 2}, {1, 5}, {2, cudf::JoinNoMatch}};
  EXPECT_EQ(sorted_host_pairs(result), expected);

  column_wrapper<int64_t> only_first{{first, first}};
  column_wrapper<int64_t> only_second{{second}};
  cudf::hash_join missing_joiner{cudf::table_view{{only_first}}, cudf::null_equality::EQUAL};
  EXPECT_EQ(missing_joiner.inner_join_size(cudf::table_view{{only_second}}), 0);
}

TEST_F(HashJoinMemoryTest, RowIndexBitBoundaries)
{
  // Include the largest row ID on either side of changes in the number of row-index bits.
  for (size_type const num_rows : {1, 2, 3, 255, 256, 257, 65535, 65536, 65537}) {
    SCOPED_TRACE(num_rows);
    std::vector<int32_t> keys(num_rows);
    std::iota(keys.begin(), keys.end(), 0);
    column_wrapper<int32_t> right(keys.begin(), keys.end());
    auto const table = cudf::table_view{{right}};
    cudf::hash_join joiner{table, cudf::null_equality::EQUAL};
    auto const result = joiner.inner_join(table);
    auto const pairs  = sorted_host_pairs(result);
    ASSERT_EQ(pairs.size(), static_cast<std::size_t>(num_rows));
    for (size_type i = 0; i < num_rows; ++i) {
      ASSERT_EQ(pairs[i], std::make_pair(i, i));
    }
  }
}

TEST_F(HashJoinMemoryTest, UnevenGroupsNullsAndPartitions)
{
  // Groups cross warp and block boundaries, with gaps between their representative row IDs.
  std::vector<int32_t> build_keys;
  for (int32_t key = 0; key < 10; ++key) {
    build_keys.insert(build_keys.end(), key * key + 1, key);
  }
  std::rotate(build_keys.begin(), build_keys.begin() + 37, build_keys.end());
  std::vector<bool> build_valid(build_keys.size(), true);
  for (std::size_t i = 0; i < build_valid.size(); i += 7) {
    build_valid[i] = false;
  }
  std::vector<int32_t> const probe_keys{8, 3, 0, 10, 8, 1, 5};
  std::vector<bool> const probe_valid{true, false, true, true, true, true, true};
  column_wrapper<int32_t> right(build_keys.begin(), build_keys.end(), build_valid.begin());
  column_wrapper<int32_t> left(probe_keys.begin(), probe_keys.end(), probe_valid.begin());
  auto const right_table = cudf::table_view{{right}};
  auto const left_table  = cudf::table_view{{left}};

  for (auto const nulls : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
    SCOPED_TRACE(nulls == cudf::null_equality::EQUAL ? "nulls equal" : "nulls unequal");
    std::vector<join_pair> expected_inner;
    std::vector<join_pair> expected_left;
    std::vector<bool> matched_build(build_keys.size(), false);
    for (size_type probe = 0; probe < static_cast<size_type>(probe_keys.size()); ++probe) {
      bool matched = false;
      for (size_type build = 0; build < static_cast<size_type>(build_keys.size()); ++build) {
        auto const equal =
          probe_valid[probe] && build_valid[build]
            ? probe_keys[probe] == build_keys[build]
            : !probe_valid[probe] && !build_valid[build] && nulls == cudf::null_equality::EQUAL;
        if (equal) {
          expected_inner.emplace_back(probe, build);
          expected_left.emplace_back(probe, build);
          matched_build[build] = true;
          matched              = true;
        }
      }
      if (!matched) { expected_left.emplace_back(probe, cudf::JoinNoMatch); }
    }
    auto expected_full = expected_left;
    for (size_type build = 0; build < static_cast<size_type>(build_keys.size()); ++build) {
      if (!matched_build[build]) { expected_full.emplace_back(cudf::JoinNoMatch, build); }
    }
    std::sort(expected_full.begin(), expected_full.end());

    cudf::hash_join joiner{right_table, nulls};
    EXPECT_EQ(joiner.inner_join_size(left_table), expected_inner.size());
    EXPECT_EQ(joiner.left_join_size(left_table), expected_left.size());
    EXPECT_EQ(joiner.full_join_size(left_table), expected_full.size());
    EXPECT_EQ(sorted_host_pairs(joiner.inner_join(left_table)), expected_inner);
    EXPECT_EQ(sorted_host_pairs(joiner.left_join(left_table)), expected_left);
    EXPECT_EQ(sorted_host_pairs(joiner.full_join(left_table)), expected_full);

    auto matches   = joiner.full_join_match_context(left_table);
    auto partition = cudf::join_partition_context{
      std::make_unique<cudf::join_match_context>(std::move(matches)), 0, 0};
    std::vector<join_result> outputs;
    for (size_type i = 0; i < left_table.num_rows(); ++i) {
      partition.left_start_idx = i;
      partition.left_end_idx   = i + 1;
      outputs.push_back(joiner.partitioned_full_join(partition));
    }
    std::vector<cudf::device_span<size_type const>> left_parts;
    std::vector<cudf::device_span<size_type const>> right_parts;
    for (auto const& output : outputs) {
      left_parts.emplace_back(*output.first);
      right_parts.emplace_back(*output.second);
    }
    auto const finalized = cudf::hash_join::finalize_partitioned_full_join(
      left_parts, right_parts, left_table.num_rows(), right_table.num_rows());
    EXPECT_EQ(sorted_host_pairs(finalized), expected_full);
  }
}

TEST_F(HashJoinMemoryTest, UnequalNestedNullsKeepAdjacentGroupsIntact)
{
  using lists = cudf::test::lists_column_wrapper<int32_t>;
  using cudf::test::iterators::null_at;
  // Rows containing a null child remain valid top-level rows but do not compare equal to
  // themselves. Their CSR segments must still be filled without disturbing neighboring groups.
  // The final top-level null row is excluded and must leave a sentinel in the build-row cache.
  lists right{
    {{{2, 0}, null_at(1)}, {1}, {{2, 0}, null_at(1)}, {1}, {}, {{3, 0}, null_at(1)}, {}, {2}, {}},
    null_at(8)};
  lists left{{{1}, {{2, 0}, null_at(1)}, {}, {2}, {4}, {}}, null_at(5)};
  auto const left_table = cudf::table_view{{left}};
  cudf::hash_join joiner{cudf::table_view{{right}}, cudf::null_equality::UNEQUAL};
  std::vector<join_pair> const expected_inner{{0, 1}, {0, 3}, {2, 4}, {2, 6}, {3, 7}};
  EXPECT_EQ(joiner.inner_join_size(left_table), expected_inner.size());
  EXPECT_EQ(sorted_host_pairs(joiner.inner_join(left_table)), expected_inner);

  auto expected_full = expected_inner;
  expected_full.insert(expected_full.end(),
                       {{1, cudf::JoinNoMatch},
                        {4, cudf::JoinNoMatch},
                        {5, cudf::JoinNoMatch},
                        {cudf::JoinNoMatch, 0},
                        {cudf::JoinNoMatch, 2},
                        {cudf::JoinNoMatch, 5},
                        {cudf::JoinNoMatch, 8}});
  std::sort(expected_full.begin(), expected_full.end());
  EXPECT_EQ(joiner.full_join_size(left_table), expected_full.size());
  EXPECT_EQ(sorted_host_pairs(joiner.full_join(left_table)), expected_full);
}

TEST_F(HashJoinMemoryTest, ConstructorPeakIncludesTemporaryAllocations)
{
  constexpr size_type num_rows = 1 << 20;
  auto const stream            = cudf::test::get_default_stream();
  for (size_type const cardinality : {size_type{1}, num_rows}) {
    SCOPED_TRACE(cardinality);
    std::vector<int32_t> keys(num_rows);
    for (size_type i = 0; i < num_rows; ++i) {
      keys[i] = i % cardinality;
    }
    column_wrapper<int32_t> right(keys.begin(), keys.end());
    auto const measure_peak = [&](cudf::column_view input,
                                  std::string const& key_type,
                                  std::size_t bytes_per_row) {
      // Input storage is outside the tracked resource. Track the persistent and current resources
      // together so moving construction scratch between them cannot hide a memory regression.
      auto tracked = rmm::mr::statistics_resource_adaptor{cudf::get_current_device_resource_ref()};
      cudf::test::scoped_current_device_resource current{tracked};
      {
        cudf::hash_join joiner{cudf::table_view{{input}},
                               cudf::nullable_join::NO,
                               cudf::null_equality::EQUAL,
                               0.5,
                               stream,
                               tracked};
        stream.sync();
        auto const peak = tracked.get_bytes_counter().peak;
        RecordProperty(
          key_type + (cardinality == 1 ? "_all_same_peak_bytes" : "_unique_peak_bytes"),
          std::to_string(peak));
        EXPECT_LE(peak, bytes_per_row * num_rows);
      }
      stream.sync();
      EXPECT_EQ(tracked.get_bytes_counter().value, 0);
    };
    // Leave room for preprocessing and allocator/CUB variation while rejecting the previous
    // 36-byte-per-row constructor peak. LIST and STRING keys retain a temporary representative.
    measure_peak(right, "int32", 24);
    auto const string_right = cudf::strings::from_integers(right, stream);
    measure_peak(string_right->view(), "string", 32);
    std::vector<size_type> offsets(num_rows + 1);
    std::iota(offsets.begin(), offsets.end(), 0);
    column_wrapper<size_type> list_offsets(offsets.begin(), offsets.end());
    auto const list_right =
      cudf::make_lists_column(num_rows, list_offsets.release(), right.release(), 0, {});
    measure_peak(list_right->view(), "list_int32", 32);
  }
}

}  // namespace
