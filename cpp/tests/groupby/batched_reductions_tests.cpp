/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

struct GroupbyBatchedReductionsTest : cudf::test::BaseFixture {};

TEST_F(GroupbyBatchedReductionsTest, M2ColumnsWithIndependentMasksAndCountOrder)
{
  // Cover small and large groups with independent column masks. Symmetric valid pairs
  // give exact M2 and keep the expected results independent of the reduction algorithm.
  for (bool const has_long_groups : {false, true}) {
    SCOPED_TRACE(has_long_groups);
    std::array<cudf::size_type, 5> const group_sizes =
      has_long_groups ? std::array<cudf::size_type, 5>{4, 64, 2048, 2, 3072}
                      : std::array<cudf::size_type, 5>{4, 64, 128, 2, 256};
    std::array<double, 5> const first_means{16.0, 32.0, 128.0, 256.0, 512.0};
    std::array<double, 5> const second_means{-32.0, -64.0, -256.0, -512.0, -1024.0};
    std::vector<int32_t> host_keys;
    std::vector<double> first_values;
    std::vector<double> second_values;
    std::vector<bool> first_validity;
    std::vector<bool> second_validity;
    for (cudf::size_type row = 0; row < group_sizes.back(); ++row) {
      for (cudf::size_type group = 0; group < static_cast<cudf::size_type>(group_sizes.size());
           ++group) {
        if (row >= group_sizes[group]) { continue; }
        host_keys.push_back(group);
        first_values.push_back(first_means[group] + (row % 2 == 0 ? -1.0 : 1.0));
        second_values.push_back(second_means[group] + (row % 2 == 0 ? -2.0 : 2.0));
        first_validity.push_back(group != 3 && row % 4 < 2);
        second_validity.push_back(group == 3 ? row == 0 : group != 1 && row % 4 >= 2);
      }
    }
    cudf::test::fixed_width_column_wrapper<int32_t> keys(host_keys.begin(), host_keys.end());
    cudf::test::fixed_width_column_wrapper<double> first(
      first_values.begin(), first_values.end(), first_validity.begin());
    cudf::test::fixed_width_column_wrapper<double> second(
      second_values.begin(), second_values.end(), second_validity.begin());
    cudf::test::dictionary_column_wrapper<double> dictionary_second(
      second_values.begin(), second_values.end(), second_validity.begin());
    std::array<cudf::column_view, 3> const columns{first, second, dictionary_second};
    cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3, 4};

    enum class count_order { AFTER_M2, BEFORE_M2, ALL_FIRST };
    for (auto const order :
         {count_order::AFTER_M2, count_order::BEFORE_M2, count_order::ALL_FIRST}) {
      SCOPED_TRACE(static_cast<int>(order));
      auto const adjacent_count_first = order == count_order::BEFORE_M2;
      auto const all_counts_first     = order == count_order::ALL_FIRST;
      std::vector<cudf::groupby::aggregation_request> requests;
      if (all_counts_first) {
        // Keep count-only requests separate from requests containing moments.
        for (auto const& column : columns) {
          requests.emplace_back();
          requests.back().values = column;
          requests.back().aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>());
        }
      }
      auto const first_m2_request = requests.size();
      for (auto const& column : columns) {
        requests.emplace_back();
        requests.back().values = column;
        auto& aggregations     = requests.back().aggregations;
        if (adjacent_count_first) {
          aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
        }
        aggregations.push_back(cudf::make_m2_aggregation<cudf::groupby_aggregation>());
        if (!adjacent_count_first) {
          aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
        }
        aggregations.push_back(cudf::make_variance_aggregation<cudf::groupby_aggregation>(0));
        aggregations.push_back(cudf::make_variance_aggregation<cudf::groupby_aggregation>(1));
        aggregations.push_back(cudf::make_variance_aggregation<cudf::groupby_aggregation>(2));
        aggregations.push_back(cudf::make_std_aggregation<cudf::groupby_aggregation>(1));
      }

      cudf::groupby::groupby gb(cudf::table_view{{keys}});
      auto result             = gb.aggregate(requests);
      auto const sorted_order = cudf::sorted_order(result.first->view());
      auto sorted_keys        = cudf::gather(result.first->view(), sorted_order->view());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, sorted_keys->get_column(0));
      ASSERT_EQ(result.second.size(), requests.size());

      for (std::size_t column = 0; column < columns.size(); ++column) {
        SCOPED_TRACE(column);
        std::vector<cudf::size_type> counts;
        std::vector<double> moments;
        for (std::size_t group = 0; group < group_sizes.size(); ++group) {
          auto const count = column == 0  ? (group == 3 ? 0 : group_sizes[group] / 2)
                             : group == 3 ? 1
                             : group == 1 ? 0
                                          : group_sizes[group] / 2;
          counts.push_back(count);
          moments.push_back(count <= 1 ? 0.0 : count * (column == 0 ? 1.0 : 4.0));
        }
        cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_counts(counts.begin(),
                                                                                counts.end());
        cudf::test::fixed_width_column_wrapper<double> expected_m2(moments.begin(), moments.end());
        auto const& outputs = result.second[first_m2_request + column].results;
        ASSERT_EQ(outputs.size(), 6);
        std::vector<cudf::column_view> views;
        for (auto const& output : outputs) {
          views.push_back(output->view());
        }
        auto sorted = cudf::gather(cudf::table_view{views}, sorted_order->view());
        CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_m2,
                                            sorted->get_column(adjacent_count_first ? 1 : 0),
                                            cudf::test::debug_output_level::FIRST_ERROR,
                                            64);
        CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_counts,
                                       sorted->get_column(adjacent_count_first ? 0 : 1));
        EXPECT_FALSE(sorted->get_column(0).nullable());
        EXPECT_FALSE(sorted->get_column(1).nullable());
        if (all_counts_first) {
          auto sorted_count =
            cudf::gather(cudf::table_view{{result.second[column].results.front()->view()}},
                         sorted_order->view());
          CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_counts, sorted_count->get_column(0));
        }
        for (cudf::size_type ddof : {0, 1, 2}) {
          std::vector<double> variances;
          std::vector<bool> validity;
          for (std::size_t group = 0; group < counts.size(); ++group) {
            validity.push_back(counts[group] > ddof);
            variances.push_back(validity.back() ? moments[group] / (counts[group] - ddof) : 0.0);
          }
          cudf::test::fixed_width_column_wrapper<double> expected_variance(
            variances.begin(), variances.end(), validity.begin());
          CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_variance,
                                              sorted->get_column(2 + ddof),
                                              cudf::test::debug_output_level::FIRST_ERROR,
                                              64);
          if (ddof == 1) {
            for (auto& value : variances) {
              value = std::sqrt(value);
            }
            cudf::test::fixed_width_column_wrapper<double> expected_std(
              variances.begin(), variances.end(), validity.begin());
            CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
              expected_std, sorted->get_column(5), cudf::test::debug_output_level::FIRST_ERROR, 64);
          }
        }
      }
    }
  }
}

TEST_F(GroupbyBatchedReductionsTest, CountBeforeVarianceUsesOutputAndTemporaryResources)
{
  auto const stream = cudf::test::get_default_stream();
  cudf::test::fixed_width_column_wrapper<int32_t> keys{0, 0, 0, 1, 1, 2, 2};
  cudf::test::fixed_width_column_wrapper<double> values{
    {1.0, 3.0, 99.0, 99.0, 99.0, 5.0, 99.0}, {true, true, false, false, false, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected_counts{2, 0, 1};
  cudf::test::fixed_width_column_wrapper<double> expected_variance{{2.0, 0.0, 0.0},
                                                                   {true, false, false}};
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = values;
  requests[0].aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
  requests[0].aggregations.push_back(cudf::make_variance_aggregation<cudf::groupby_aggregation>(1));

  // COUNT is a public output, while the following M2 is only a variance intermediate.
  auto harness = cudf::test::memory_resource_test_harness{this->mr()};
  {
    auto result = [&] {
      cudf::test::scoped_current_device_resource temporary_scope{harness.temporary_mr()};
      cudf::groupby::groupby gb(cudf::table_view{{keys}});
      return gb.aggregate(requests, stream, harness.output_mr());
    }();
    ASSERT_EQ(result.second.size(), 1);
    auto const& outputs = result.second[0].results;
    ASSERT_EQ(outputs.size(), 2);
    auto const output_bytes =
      result.first->alloc_size() + outputs[0]->alloc_size() + outputs[1]->alloc_size();
    harness.expect_resource_usage(output_bytes,
                                  {cudf::test::output_allocation_expectation::AT_LEAST_LIVE,
                                   cudf::test::temporary_allocation_expectation::SOME},
                                  stream);
    auto const sorted = cudf::sort(
      cudf::table_view{{result.first->view().column(0), *outputs[0], *outputs[1]}}, {}, {}, stream);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, sorted->get_column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_counts, sorted->get_column(1));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_variance, sorted->get_column(2));
    EXPECT_FALSE(outputs[0]->nullable());
  }
  harness.expect_no_live_allocations(stream);
}

TEST_F(GroupbyBatchedReductionsTest, NullableSumsShareValidityWithoutRequestedCount)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{2, 1, 2, 1, 3, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> values{{2, 1, 3, 0, 0, 0},
                                                         {true, true, true, false, false, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sum{{1, 5, 0}, {true, true, false}};
  cudf::test::fixed_width_column_wrapper<int64_t> expected_squares{{1, 13, 0}, {true, true, false}};
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests.front().values = values;
  requests.front().aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests.front().aggregations.push_back(
    cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>());
  cudf::groupby::groupby gb(cudf::table_view{{keys}});
  auto result      = gb.aggregate(requests);
  auto const order = cudf::sorted_order(result.first->view());
  auto sorted_keys = cudf::gather(result.first->view(), order->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, sorted_keys->get_column(0));
  auto const& outputs = result.second.front().results;
  ASSERT_EQ(outputs.size(), 2);
  auto sorted =
    cudf::gather(cudf::table_view{{outputs[0]->view(), outputs[1]->view()}}, order->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sum, sorted->get_column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_squares, sorted->get_column(1));
}

TEST_F(GroupbyBatchedReductionsTest, NullableSumWithOneExtremum)
{
  // Exercise scalar, warp and multi-chunk reductions, plus an all-null group. Each call requests
  // only one extremum, and both result orders must retain the shared validity mask.
  std::array<cudf::size_type, 4> const sizes{3, 33, 1025, 2};
  std::vector<int32_t> key_data;
  std::vector<int32_t> value_data;
  std::vector<bool> validity;
  for (cudf::size_type group = 0; group < static_cast<cudf::size_type>(sizes.size()); ++group) {
    for (cudf::size_type row = 0; row < sizes[group]; ++row) {
      key_data.push_back(group);
      value_data.push_back(group * 100 + row - 500);
      validity.push_back(group != 3 && row % 3 != 1);
    }
  }
  cudf::test::fixed_width_column_wrapper<int32_t> keys(key_data.begin(), key_data.end());
  cudf::test::fixed_width_column_wrapper<int32_t> values(
    value_data.begin(), value_data.end(), validity.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{0, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int64_t> expected_sum{{-998, -8448, 144625, 0},
                                                               {true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_min{{-500, -400, -300, 0},
                                                               {true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_max{{-498, -368, 723, 0},
                                                               {true, true, true, false}};

  for (bool const minimum : {false, true}) {
    for (bool const sum_first : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "minimum=" << minimum << " sum_first=" << sum_first);
      std::vector<cudf::groupby::aggregation_request> requests(1);
      requests[0].values = values;
      auto& aggs         = requests[0].aggregations;
      auto sum           = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
      auto extremum      = minimum ? cudf::make_min_aggregation<cudf::groupby_aggregation>()
                                   : cudf::make_max_aggregation<cudf::groupby_aggregation>();
      if (sum_first) {
        aggs.push_back(std::move(sum));
        aggs.push_back(std::move(extremum));
      } else {
        aggs.push_back(std::move(extremum));
        aggs.push_back(std::move(sum));
      }
      cudf::groupby::groupby gb(cudf::table_view{{keys}});
      auto const [result_keys, results] = gb.aggregate(requests);
      ASSERT_EQ(results.size(), 1);
      ASSERT_EQ(results[0].results.size(), 2);
      auto const order  = cudf::sorted_order(result_keys->view());
      auto const sorted = cudf::gather(
        cudf::table_view{
          {result_keys->view().column(0), *results[0].results[0], *results[0].results[1]}},
        order->view());
      auto const expected_extremum =
        minimum ? cudf::column_view(expected_min) : cudf::column_view(expected_max);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, sorted->get_column(0));
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_sum, sorted->get_column(sum_first ? 1 : 2));
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_extremum, sorted->get_column(sum_first ? 2 : 1));
    }
  }
}
