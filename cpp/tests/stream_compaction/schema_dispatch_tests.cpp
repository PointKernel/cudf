/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <bit>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

using ints = cudf::test::fixed_width_column_wrapper<int32_t>;
using keep = cudf::duplicate_keep_option;
using cudf::test::iterators::nulls_at;

// The last column is a strictly increasing row identifier. Comparing all gathered columns checks
// both grouping semantics and which input row supplies the output for each deterministic policy.
void expect_rows(cudf::table_view const& result,
                 cudf::table_view const& input,
                 std::initializer_list<int32_t> expected_rows)
{
  auto const map      = ints(expected_rows.begin(), expected_rows.end());
  auto const expected = cudf::gather(input, map);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected, result);
}

void check_unique(cudf::table_view const& input,
                  std::vector<cudf::size_type> const& keys,
                  keep policy,
                  std::initializer_list<int32_t> expected_rows,
                  cudf::null_equality nulls = cudf::null_equality::EQUAL)
{
  SCOPED_TRACE(static_cast<int>(policy));
  auto const result = cudf::unique(input, keys, policy, nulls);
  expect_rows(*result, input, expected_rows);
}

struct UniqueSchemaTest : cudf::test::BaseFixture {};

template <typename Index>
struct UniqueSchemaDictionaryTest : cudf::test::BaseFixture {};

using DictionaryIndexTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_SUITE(UniqueSchemaDictionaryTest, DictionaryIndexTypes);

TYPED_TEST(UniqueSchemaDictionaryTest, SlicedDictionaryWithTypedIndices)
{
  auto const keys       = cudf::test::strings_column_wrapper{"a", "b", "c", "skip"};
  auto const indices    = cudf::test::fixed_width_column_wrapper<TypeParam>{3, 0, 0, 1, 1, 2, 0, 3};
  auto const dictionary = cudf::make_dictionary_column(keys, indices);
  auto const ids        = ints{0, 1, 2, 3, 4, 5, 6, 7};
  auto const input      = cudf::slice(cudf::table_view{{dictionary->view(), ids}}, {1, 7}).front();

  check_unique(input, {0}, keep::KEEP_FIRST, {0, 2, 4, 5});
  check_unique(input, {0}, keep::KEEP_LAST, {1, 3, 4, 5});
  check_unique(input, {0}, keep::KEEP_NONE, {4, 5});
}

TEST_F(UniqueSchemaTest, SwitchesSchemasInOneProcess)
{
  // Each call must use its own physical widths and key columns, regardless of earlier schemas.
  auto const ids   = ints{0, 1, 2, 3, 4, 5};
  auto const bytes = cudf::test::fixed_width_column_wrapper<int8_t>{1, 1, 2, 2, 3, 1};
  auto const small = cudf::table_view{{bytes, ids}};
  auto const first = cudf::unique(small, {0}, keep::KEEP_FIRST);
  expect_rows(*first, small, {0, 2, 4, 5});

  auto const words = cudf::test::fixed_width_column_wrapper<int64_t>{1, 1, 257, 257, 513, 1};
  auto const wide  = cudf::table_view{{words, ids}};
  auto const next  = cudf::unique(wide, {0}, keep::KEEP_FIRST);
  expect_rows(*next, wide, {0, 2, 4, 5});

  auto const floats = cudf::test::fixed_width_column_wrapper<float>{0.0, -0.0, 0.0, -0.0, 4.0, 4.0};
  auto const labels = cudf::test::fixed_width_column_wrapper<int16_t>{1, 1, 2, 2, 3, 3};
  auto const mixed  = cudf::table_view{{floats, labels, ids}};
  auto const last   = cudf::unique(mixed, {0, 1}, keep::KEEP_FIRST);
  expect_rows(*last, mixed, {0, 2, 4});

  auto const repeated = cudf::unique(small, {0}, keep::KEEP_FIRST);
  expect_rows(*repeated, small, {0, 2, 4, 5});
}

TEST_F(UniqueSchemaTest, KeyOrderAndRepeatedKeysPreserveRunBoundaries)
{
  // Successive runs differ in different columns. Equal later columns must not erase an earlier
  // mismatch, including when selected keys are reordered or the same key is selected twice.
  auto const numbers = ints{1, 1, 2, 2, 2, 2, 2, 2, 2};
  auto const strings =
    cudf::test::strings_column_wrapper{"a", "a", "a", "a", "b", "b", "b", "b", "b"};
  auto const values = cudf::test::fixed_width_column_wrapper<double>{{7, 7, 7, 7, 7, 7, 99, 123, 7},
                                                                     nulls_at({6, 7})};
  auto const ids    = ints{0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto const input  = cudf::table_view{{numbers, strings, values, ids}};

  for (auto const& keys :
       std::vector<std::vector<cudf::size_type>>{{0, 1, 2}, {2, 0, 1}, {1, 2, 0}, {0, 2, 1, 0}}) {
    check_unique(input, keys, keep::KEEP_FIRST, {0, 2, 4, 6, 8});
    check_unique(input, keys, keep::KEEP_LAST, {1, 3, 5, 7, 8});
    check_unique(input, keys, keep::KEEP_NONE, {8});
    check_unique(input, keys, keep::KEEP_FIRST, {0, 2, 4, 6, 7, 8}, cudf::null_equality::UNEQUAL);
    check_unique(input, keys, keep::KEEP_LAST, {1, 3, 5, 6, 7, 8}, cudf::null_equality::UNEQUAL);
    check_unique(input, keys, keep::KEEP_NONE, {6, 7, 8}, cudf::null_equality::UNEQUAL);
  }
}

TEST_F(UniqueSchemaTest, RowCountsAroundWordAndBlockBoundaries)
{
  for (auto const size : {1, 31, 32, 33, 255, 256, 257}) {
    SCOPED_TRACE(size);
    std::vector<int16_t> first(size);
    std::vector<int64_t> second(size);
    std::vector<int32_t> row_ids(size);
    for (int32_t row = 0; row < size; ++row) {
      first[row]   = row / 4;
      second[row]  = (int64_t{1} << 45) + (row + 1) / 8;
      row_ids[row] = row;
    }
    auto const first_key =
      cudf::test::fixed_width_column_wrapper<int16_t>(first.begin(), first.end());
    auto const second_key =
      cudf::test::fixed_width_column_wrapper<int64_t>(second.begin(), second.end());
    auto const ids   = ints(row_ids.begin(), row_ids.end());
    auto const input = cudf::table_view{{first_key, second_key, ids}};

    for (auto const policy : {keep::KEEP_FIRST, keep::KEEP_LAST, keep::KEEP_NONE}) {
      SCOPED_TRACE(static_cast<int>(policy));
      // Form runs directly from the original host values. Each key introduces boundaries that
      // the other key does not, including adjacent boundaries at rows 31/32 and 255/256.
      std::vector<int32_t> expected_ids;
      for (int32_t begin = 0; begin < size;) {
        auto end = begin + 1;
        while (end < size && first[end] == first[begin] && second[end] == second[begin]) {
          ++end;
        }
        if (policy == keep::KEEP_FIRST || (policy == keep::KEEP_NONE && end == begin + 1)) {
          expected_ids.push_back(begin);
        } else if (policy == keep::KEEP_LAST) {
          expected_ids.push_back(end - 1);
        }
        begin = end;
      }
      auto const expected_map = ints(expected_ids.begin(), expected_ids.end());
      auto const expected     = cudf::gather(input, expected_map);
      auto const result       = cudf::unique(input, {0, 1}, policy);
      CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
    }
  }
}

TEST_F(UniqueSchemaTest, NestedListLengthsEmptyRowsAndNullParents)
{
  // Logical rows: [[1], []] x2, [[1]], [[1,2], []] x2, [] x2, [[]] x2, null x2,
  // [[2], [3,4]] x2, [[2,3], [4]] x2. The last two runs have identical flattened values.
  auto inner              = cudf::test::lists_column_wrapper<int32_t>{{1},
                                                                      {},
                                                                      {1},
                                                                      {},
                                                                      {1},
                                                                      {1, 2},
                                                                      {},
                                                                      {1, 2},
                                                                      {},
                                                                      {},
                                                                      {},
                                                                      {2},
                                                                      {3, 4},
                                                                      {2},
                                                                      {3, 4},
                                                                      {2, 3},
                                                                      {4},
                                                                      {2, 3},
                                                                      {4}};
  auto offsets            = ints{0, 2, 4, 5, 7, 9, 9, 9, 10, 11, 11, 11, 13, 15, 17, 19};
  auto const valid        = nulls_at({9, 10});
  auto [mask, null_count] = cudf::test::detail::make_null_mask(valid, valid + 15);
  auto const lists =
    cudf::make_lists_column(15, offsets.release(), inner.release(), null_count, std::move(mask));
  auto const ids   = ints{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  auto const input = cudf::table_view{{lists->view(), ids}};

  check_unique(input, {0}, keep::KEEP_FIRST, {0, 2, 3, 5, 7, 9, 11, 13});
  check_unique(input, {0}, keep::KEEP_LAST, {1, 2, 4, 6, 8, 10, 12, 14});
  check_unique(input, {0}, keep::KEEP_NONE, {2});
  check_unique(
    input, {0}, keep::KEEP_FIRST, {0, 2, 3, 5, 7, 9, 10, 11, 13}, cudf::null_equality::UNEQUAL);
  check_unique(
    input, {0}, keep::KEEP_LAST, {1, 2, 4, 6, 8, 9, 10, 12, 14}, cudf::null_equality::UNEQUAL);
  check_unique(input, {0}, keep::KEEP_NONE, {2, 9, 10}, cudf::null_equality::UNEQUAL);
}

TEST_F(UniqueSchemaTest, SlicedFloatingDictionaryWithNanAndNulls)
{
  auto const nan     = std::bit_cast<double>(uint64_t{0x7ff8000000000001});
  auto const keys    = cudf::test::fixed_width_column_wrapper<double>{-1.0, 0.0, 7.0, nan};
  auto const indices = cudf::test::fixed_width_column_wrapper<int16_t>{
    {2, 1, 1, 3, 3, 0, 2, 2, 3, 2}, nulls_at({5, 6})};
  auto const dictionary = cudf::make_dictionary_column(keys, indices);
  auto const ids        = ints{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto const input      = cudf::slice(cudf::table_view{{dictionary->view(), ids}}, {1, 9}).front();

  check_unique(input, {0}, keep::KEEP_FIRST, {0, 2, 4, 6, 7});
  check_unique(input, {0}, keep::KEEP_LAST, {1, 3, 5, 6, 7});
  check_unique(input, {0}, keep::KEEP_NONE, {6, 7});
  check_unique(input, {0}, keep::KEEP_FIRST, {0, 2, 4, 5, 6, 7}, cudf::null_equality::UNEQUAL);
  check_unique(input, {0}, keep::KEEP_LAST, {1, 3, 4, 5, 6, 7}, cudf::null_equality::UNEQUAL);
  check_unique(input, {0}, keep::KEEP_NONE, {4, 5, 6, 7}, cudf::null_equality::UNEQUAL);
}

TEST_F(UniqueSchemaTest, MixedDecimalAndChronoStorageWidths)
{
  // Each pair changes only one field relative to the first pair. The decimal128 difference is
  // above bit 64, so an accidentally narrowed physical accessor would merge distinct keys.
  auto const wide = __int128_t{1} << 95;
  auto const high = int64_t{1} << 45;
  auto const d32  = cudf::test::fixed_point_column_wrapper<int32_t>{
    {-10, -10, -11, -11, -10, -10, -10, -10, -10, -10, -10, -10, -10}, numeric::scale_type{-3}};
  auto const d64 = cudf::test::fixed_point_column_wrapper<int64_t>{
    {high, high, high, high, high + 1, high + 1, high, high, high, high, high, high, high},
    numeric::scale_type{2}};
  auto const d128 =
    cudf::test::fixed_point_column_wrapper<__int128_t>{{wide,
                                                        wide,
                                                        wide,
                                                        wide,
                                                        wide,
                                                        wide,
                                                        wide + (__int128_t{1} << 64),
                                                        wide + (__int128_t{1} << 64),
                                                        wide,
                                                        wide,
                                                        wide,
                                                        wide,
                                                        wide},
                                                       numeric::scale_type{-12}};
  auto const times = cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, int64_t>{
    high, high, high, high, high, high, high, high, high + 1, high + 1, high, high, high};
  auto const durations = cudf::test::fixed_width_column_wrapper<cudf::duration_ns, int64_t>{
    high, high, high, high, high, high, high, high, high, high, high + 1, high + 1, high};
  auto const ids   = ints{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  auto const input = cudf::table_view{{d32, d64, d128, times, durations, ids}};

  check_unique(input, {0, 1, 2, 3, 4}, keep::KEEP_FIRST, {0, 2, 4, 6, 8, 10, 12});
  check_unique(input, {0, 1, 2, 3, 4}, keep::KEEP_LAST, {1, 3, 5, 7, 9, 11, 12});
  check_unique(input, {0, 1, 2, 3, 4}, keep::KEEP_NONE, {12});
}

TEST_F(UniqueSchemaTest, SignedZeroAndDifferentNanPayloads)
{
  auto const nan1 = std::bit_cast<double>(uint64_t{0x7ff8000000000001});
  auto const nan2 = std::bit_cast<double>(uint64_t{0xfff8000000000042});
  auto const values =
    cudf::test::fixed_width_column_wrapper<double>{-0.0, 0.0, nan1, nan2, 7.0, -0.0};
  auto const ids   = ints{0, 1, 2, 3, 4, 5};
  auto const input = cudf::table_view{{values, ids}};

  check_unique(input, {0}, keep::KEEP_FIRST, {0, 2, 4, 5});
  check_unique(input, {0}, keep::KEEP_LAST, {1, 3, 4, 5});
  check_unique(input, {0}, keep::KEEP_NONE, {4, 5});
}

TEST_F(UniqueSchemaTest, SlicedParentContainingListsOfNullableStructs)
{
  // Valid A rows contain a null struct with different hidden numeric payloads. The owning
  // factories propagate parent nulls; slicing the outer struct must also offset the list child.
  auto numbers = ints{99, 1, 101, 1, 202, 2, 303, 3, 404, 8, 9, 1, 505, 99};
  auto strings = cudf::test::strings_column_wrapper{
    "skip", "a", "x1", "a", "x2", "b", "x3", "q", "x4", "c", "d", "a", "x5", "skip"};
  auto structs = cudf::test::structs_column_wrapper{{numbers, strings}, nulls_at({2, 4, 6, 8, 12})};
  auto offsets = ints{0, 1, 3, 5, 7, 9, 11, 13, 14};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(
    cudf::make_lists_column(8,
                            offsets.release(),
                            structs.release(),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED)));
  auto const parent = cudf::test::structs_column_wrapper{
    std::move(children), {true, true, true, false, false, true, true, true}};
  auto const ids   = ints{0, 1, 2, 3, 4, 5, 6, 7};
  auto const input = cudf::slice(cudf::table_view{{parent, ids}}, {1, 7}).front();

  check_unique(input, {0}, keep::KEEP_FIRST, {0, 2, 4, 5});
  check_unique(input, {0}, keep::KEEP_LAST, {1, 3, 4, 5});
  check_unique(input, {0}, keep::KEEP_NONE, {4, 5});
  check_unique(input, {0}, keep::KEEP_NONE, {0, 1, 2, 3, 4, 5}, cudf::null_equality::UNEQUAL);
}

TEST_F(UniqueSchemaTest, StructParentReplacesSlicedChildOffsets)
{
  // Struct offsets identify rows in every descendant. Child views can already be sliced; their
  // offsets must be replaced by the parent's offset, rather than accumulated with it.
  auto const values = ints{900, 10, 10, 20, 20, 30, 10, 901, 902, 903};
  auto const child  = cudf::slice(static_cast<cudf::column_view>(values), {2, 9}).front();
  auto const inner =
    cudf::column_view{cudf::data_type{cudf::type_id::STRUCT}, 6, nullptr, nullptr, 0, 1, {child}};
  auto const parent =
    cudf::column_view{cudf::data_type{cudf::type_id::STRUCT}, 6, nullptr, nullptr, 0, 0, {inner}};
  auto const ids   = ints{0, 1, 2, 3, 4, 5};
  auto const input = cudf::slice(cudf::table_view{{parent, ids}}, {1, 6}).front();

  check_unique(input, {0}, keep::KEEP_FIRST, {0, 2, 4});
  check_unique(input, {0}, keep::KEEP_LAST, {1, 3, 4});
  check_unique(input, {0}, keep::KEEP_NONE, {4});
}

TEST_F(UniqueSchemaTest, SlicedStringsWithInt64OffsetsAndEmbeddedZeros)
{
  auto const a = std::string{"a\0b", 3};
  auto const b = std::string{"a\0c", 3};
  std::vector<std::string> data{"skip", a, a, b, b, "", a, "skip"};
  auto const strings  = cudf::test::strings_column_wrapper(data.begin(), data.end());
  auto const original = static_cast<cudf::column_view>(strings);
  auto const offsets  = cudf::cast(original.child(0), cudf::data_type{cudf::type_id::INT64});
  auto const keys     = cudf::column_view{
    original.type(), original.size(), original.head(), nullptr, 0, 0, {offsets->view()}};
  auto const ids   = ints{0, 1, 2, 3, 4, 5, 6, 7};
  auto const input = cudf::slice(cudf::table_view{{keys, ids}}, {1, 7}).front();

  check_unique(input, {0}, keep::KEEP_FIRST, {0, 2, 4, 5});
  check_unique(input, {0}, keep::KEEP_LAST, {1, 3, 4, 5});
  check_unique(input, {0}, keep::KEEP_NONE, {4, 5});
}

TEST_F(UniqueSchemaTest, KeepAnyReturnsOneMemberOfEveryRun)
{
  auto const keys          = ints{3, 3, 1, 1, 2, 2, 3};
  auto const ids           = ints{0, 1, 2, 3, 4, 5, 6};
  auto const input         = cudf::table_view{{keys, ids}};
  auto const expected_keys = ints{3, 1, 2, 3};
  auto const result        = cudf::unique(input, {0}, keep::KEEP_ANY);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, result->view().column(0));
  auto const selected = cudf::test::to_host<int32_t>(result->view().column(1)).first;
  ASSERT_EQ(selected.size(), 4);
  for (int32_t run = 0; run < 3; ++run) {
    EXPECT_GE(selected[run], 2 * run);
    EXPECT_LE(selected[run], 2 * run + 1);
  }
  EXPECT_EQ(selected[3], 6);
  // Any selected representative is allowed, but its payload must come from the same input row.
  auto const expected = cudf::gather(input, result->view().column(1));
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
}

TEST_F(UniqueSchemaTest, NonzeroBooleanStorage)
{
  auto const bytes   = cudf::test::fixed_width_column_wrapper<uint8_t>{1, 2, 0, 0, 255, 3};
  auto const storage = static_cast<cudf::column_view>(bytes);
  auto const keys    = cudf::column_view{
    cudf::data_type{cudf::type_id::BOOL8}, storage.size(), storage.head(), nullptr, 0};
  auto const ids   = ints{0, 1, 2, 3, 4, 5};
  auto const input = cudf::table_view{{keys, ids}};

  // Compare payloads directly to avoid asking the general test comparator to interpret a C++ bool
  // object with noncanonical representation. BOOL8's public storage contract is zero/nonzero.
  auto const consecutive = cudf::unique(input, {0}, keep::KEEP_FIRST);
  auto const unique_ids  = ints{0, 2, 4};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(unique_ids, consecutive->view().column(1));
  auto const last     = cudf::unique(input, {0}, keep::KEEP_LAST);
  auto const last_ids = ints{1, 3, 5};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(last_ids, last->view().column(1));
  auto const singles = cudf::unique(input, {0}, keep::KEEP_NONE);
  EXPECT_EQ(singles->num_rows(), 0);
}

}  // namespace
