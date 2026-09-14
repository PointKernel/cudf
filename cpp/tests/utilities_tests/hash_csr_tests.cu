/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_list_utilities.hpp>

#include <cudf/detail/hash_csr_kernels.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cuco/pair.cuh>
#include <cuda/iterator>
#include <cuda/std/type_traits>
#include <thrust/transform.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <vector>

namespace {
namespace csr = cudf::detail::hash_csr;
using csr::count_mode;
using cudf::size_type;
using cudf::detail::make_device_uvector;
using cudf::detail::make_pinned_vector;
using cached_key = cuco::pair<cudf::hash_value_type, size_type>;

template <typename Key>
struct key_equal {
  size_type const* keys;

  __device__ bool operator()(Key lhs, Key rhs) const
  {
    if constexpr (cuda::std::is_same_v<Key, cached_key>) {
      return lhs.first == rhs.first && keys[lhs.second] == keys[rhs.second];
    } else {
      return keys[lhs] == keys[rhs];
    }
  }
};

struct constant_hash {
  cudf::hash_value_type value;
  __device__ cudf::hash_value_type operator()(size_type) const { return value; }
};

template <typename Key, count_mode Mode>
struct build_config {
  using key_type             = Key;
  static constexpr auto mode = Mode;
};

}  // namespace

template <typename Config>
struct HashCsrBuildTest : cudf::test::BaseFixture {};

using BuildConfigs = cudf::test::Types<build_config<cached_key, count_mode::per_row>,
                                       build_config<size_type, count_mode::per_row>,
                                       build_config<cached_key, count_mode::per_warp>,
                                       build_config<size_type, count_mode::per_warp>>;
TYPED_TEST_SUITE(HashCsrBuildTest, BuildConfigs);

namespace {
enum class validity { all_valid, mixed, all_null };

template <typename Config>
void check_membership(validity mask_kind, bool by_representative = false, size_type num_rows = 1027)
{
  using key_type        = typename Config::key_type;
  auto const capacity   = cuda::std::is_same_v<key_type, cached_key>
                            ? (by_representative ? 2048u : 16u)
                            : (by_representative ? 1543u : 13u);
  auto const num_counts = by_representative ? num_rows : static_cast<size_type>(capacity);
  auto const stream     = cudf::get_default_stream();
  auto const mr         = cudf::get_current_device_resource_ref();
  std::vector<size_type> keys(num_rows);
  std::vector<cudf::bitmask_type> mask((num_rows + 31) / 32, 0);
  std::map<size_type, std::vector<size_type>> expected;
  for (size_type row = 0; row < num_rows; ++row) {
    keys[row] = row % 7;
    if (mask_kind == validity::all_valid || (mask_kind == validity::mixed && row % 5 != 0)) {
      mask[row / 32] |= cudf::bitmask_type{1} << (row % 32);
      expected[keys[row]].push_back(row);
    }
  }
  auto const d_keys = make_device_uvector(keys, stream, mr);
  auto const d_mask = make_device_uvector(mask, stream, mr);
  rmm::device_uvector<key_type> entries(capacity, stream, mr);
  auto counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(num_counts, stream, mr);
  rmm::device_uvector<csr::build_position_type> positions(num_rows, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(entries.data(), 0xff, entries.size() * sizeof(*entries.data()), stream.get()));
  csr::build_count<Config::mode>(num_rows,
                                 mask_kind == validity::all_valid ? nullptr : d_mask.data(),
                                 positions.data(),
                                 counts.data(),
                                 by_representative,
                                 csr::hash_set_ref<key_type>{entries.data(), capacity, capacity},
                                 key_equal<key_type>{d_keys.data()},
                                 constant_hash{capacity - 1},
                                 nullptr,
                                 stream);
  auto const h_counts    = make_pinned_vector(counts, stream);
  auto const h_positions = make_pinned_vector(positions, stream);
  std::vector<size_type> ends(num_counts);
  for (auto count : h_counts) {
    ASSERT_GE(count, 0);
  }
  std::partial_sum(h_counts.begin(), h_counts.end(), ends.begin());
  auto const included =
    std::accumulate(expected.begin(), expected.end(), size_type{0}, [](auto n, auto const& group) {
      return n + static_cast<size_type>(group.second.size());
    });
  ASSERT_EQ(ends.back(), included);
  for (size_type row = 0; row < num_rows; ++row) {
    if ((mask[row / 32] & (cudf::bitmask_type{1} << (row % 32))) == 0) {
      EXPECT_EQ(h_positions[row].first, csr::no_slot);
      EXPECT_EQ(h_positions[row].second, cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
    } else {
      ASSERT_LT(h_positions[row].first, num_counts);
      ASSERT_GE(h_positions[row].second, 0);
      ASSERT_LT(h_positions[row].second, h_counts[h_positions[row].first]);
      if (by_representative) { EXPECT_EQ(keys[h_positions[row].first], keys[row]); }
    }
  }
  auto const d_ends = make_device_uvector(ends, stream, mr);
  rmm::device_uvector<size_type> values(included, stream, mr);
  if (included != 0) {
    CUDF_CUDA_TRY(
      cudaMemsetAsync(values.data(), 0xff, values.size() * sizeof(size_type), stream.get()));
  }
  if constexpr (Config::mode == count_mode::per_row) {
    auto starts = cudf::detail::make_counting_transform_iterator(
      size_type{0}, [p = d_ends.data()] __device__(auto slot) -> size_type {
        return slot == 0 ? 0 : p[slot - 1];
      });
    csr::fill(num_rows, positions.data(), starts, values.data(), stream);
  } else {
    std::vector<size_type> starts(num_counts, 0);
    std::copy(ends.begin(), ends.end() - 1, starts.begin() + 1);
    auto const d_starts = make_device_uvector(starts, stream, mr);
    csr::fill(num_rows,
              cub::CacheModifiedInputIterator<cub::LOAD_CS, csr::build_position_type const>{
                positions.data()},
              d_starts.data(),
              values.data(),
              stream);
  }
  auto const h_values = make_pinned_vector(values, stream);
  rmm::device_uvector<csr::build_position_type> bounds(num_counts, stream, mr);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator{size_type{0}},
                    cuda::counting_iterator{num_counts},
                    bounds.begin(),
                    [view = csr::csr_ref{d_ends.data(), values.data()}] __device__(
                      auto slot) -> csr::build_position_type {
                      return {static_cast<cuda::std::uint32_t>(view.begin(slot)), view.size(slot)};
                    });
  auto const h_bounds = make_pinned_vector(bounds, stream);
  std::map<size_type, std::vector<size_type>> actual;
  for (size_type slot = 0; slot < num_counts; ++slot) {
    auto const begin = slot == 0 ? 0 : ends[slot - 1];
    EXPECT_EQ(h_bounds[slot].first, begin);
    EXPECT_EQ(h_bounds[slot].second, h_counts[slot]);
    if (h_counts[slot] == 0) { continue; }
    std::vector<size_type> rows(h_values.begin() + begin, h_values.begin() + ends[slot]);
    std::sort(rows.begin(), rows.end());
    for (auto row : rows) {
      ASSERT_GE(row, 0);
      ASSERT_LT(row, num_rows);
    }
    auto const key = keys[rows.front()];
    EXPECT_TRUE(actual.emplace(key, rows).second);
    EXPECT_EQ(rows, expected.at(key));
  }
  EXPECT_EQ(actual, expected);
}

}  // namespace

TYPED_TEST(HashCsrBuildTest, DuplicatesCollisionsAndTailWarp)
{
  // Exercise both warp boundaries and several blocks with a partial final warp.
  for (auto num_rows : {1, 31, 32, 33, 1027}) {
    check_membership<TypeParam>(validity::all_valid, false, num_rows);
  }
}

TYPED_TEST(HashCsrBuildTest, ExcludedRows)
{
  check_membership<TypeParam>(validity::mixed);
  check_membership<TypeParam>(validity::all_null);
}

using HashCsrTest = cudf::test::BaseFixture;

TEST_F(HashCsrTest, RepresentativeCounts)
{
  check_membership<build_config<cached_key, count_mode::per_warp>>(validity::mixed, true);
  check_membership<build_config<size_type, count_mode::per_warp>>(validity::mixed, true);
}

namespace {
template <typename Key>
void check_insertions()
{
  auto const stream            = cudf::get_default_stream();
  auto const mr                = cudf::get_current_device_resource_ref();
  constexpr size_type num_rows = 259;
  constexpr auto capacity      = cuda::std::is_same_v<Key, cached_key> ? 16u : 13u;
  std::vector<size_type> keys(num_rows + 1);
  for (size_type row = 0; row < num_rows; ++row) {
    keys[row] = row % 7;
  }
  keys.back()       = 99;  // A probe key that is never inserted.
  auto const d_keys = make_device_uvector(keys, stream, mr);
  using set_ref     = csr::hash_set_ref<Key>;
  rmm::device_uvector<Key> entries(capacity, stream, mr);
  rmm::device_uvector<typename set_ref::insert_result> results(num_rows, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(entries.data(), 0xff, entries.size() * sizeof(*entries.data()), stream.get()));
  auto const hash_set = set_ref{entries.data(), capacity, capacity};
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator{size_type{0}},
                    cuda::counting_iterator{num_rows},
                    results.begin(),
                    [hash_set, equal = key_equal<Key>{d_keys.data()}] __device__(auto row) ->
                    typename set_ref::insert_result {
                      auto const hash = hash_set.capacity - 1;
                      if constexpr (cuda::std::is_same_v<Key, cached_key>) {
                        return hash_set.insert(Key{hash, row}, hash, equal);
                      } else {
                        return hash_set.insert(row, hash, equal);
                      }
                    });
  auto const h_results     = make_pinned_vector(results, stream);
  auto const inserted_keys = make_pinned_vector(entries, stream);
  std::map<size_type, cuda::std::uint32_t> slots;
  std::map<size_type, int> insertions;
  for (size_type row = 0; row < num_rows; ++row) {
    auto const result = h_results[row];
    ASSERT_LT(result.first.slot, capacity);
    auto const representative = [&] {
      if constexpr (cuda::std::is_same_v<Key, cached_key>) {
        return result.first.key.second;
      } else {
        return result.first.key;
      }
    }();
    if constexpr (cuda::std::is_same_v<Key, cached_key>) {
      EXPECT_EQ(result.first.key.first, capacity - 1);
      EXPECT_EQ(result.first.key.first, inserted_keys[result.first.slot].first);
      EXPECT_EQ(result.first.key.second, inserted_keys[result.first.slot].second);
    } else {
      EXPECT_EQ(result.first.key, inserted_keys[result.first.slot]);
    }
    ASSERT_GE(representative, 0);
    ASSERT_LT(representative, num_rows);
    EXPECT_EQ(keys[representative], keys[row]);
    if (result.second) { EXPECT_EQ(representative, row); }
    auto const it = slots.emplace(keys[row], result.first.slot).first;
    EXPECT_EQ(it->second, result.first.slot);
    insertions[keys[row]] += result.second;
  }
  std::set<cuda::std::uint32_t> occupied;
  for (auto const& [key, slot] : slots) {
    EXPECT_EQ(insertions[key], 1);
    occupied.insert(slot);
  }
  EXPECT_EQ(occupied.size(), 7);
  EXPECT_EQ(occupied.count(capacity - 1), 1);
  EXPECT_EQ(occupied.count(0), 1);  // All hashes start at the last slot, forcing wraparound.
  rmm::device_uvector<cuda::std::uint32_t> found(num_rows + 1, stream, mr);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    cuda::counting_iterator{size_type{0}},
    cuda::counting_iterator{num_rows + 1},
    found.begin(),
    [hash_set, equal = key_equal<Key>{d_keys.data()}] __device__(auto row) -> cuda::std::uint32_t {
      auto const hash = hash_set.capacity - 1;
      if constexpr (cuda::std::is_same_v<Key, cached_key>) {
        return hash_set.find(Key{hash, row}, hash, equal);
      } else {
        return hash_set.find(row, hash, equal);
      }
    });
  auto const h_found = make_pinned_vector(found, stream);
  for (size_type row = 0; row < num_rows; ++row) {
    EXPECT_EQ(h_found[row], slots.at(keys[row]));
  }
  EXPECT_EQ(h_found.back(), capacity);
  // A keys-only build uses the same set without allocating position or count arrays.
  CUDF_CUDA_TRY(
    cudaMemsetAsync(entries.data(), 0xff, entries.size() * sizeof(*entries.data()), stream.get()));
  csr::build_count<count_mode::per_warp>(num_rows,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         false,
                                         hash_set,
                                         key_equal<Key>{d_keys.data()},
                                         constant_hash{capacity - 1},
                                         nullptr,
                                         stream);
  auto const h_entries = make_pinned_vector(entries, stream);
  std::set<size_type> distinct;
  size_type occupied_count{};
  for (auto entry : h_entries) {
    auto const row = [&] {
      if constexpr (cuda::std::is_same_v<Key, cached_key>) {
        return entry.second;
      } else {
        return entry;
      }
    }();
    if (row == cudf::detail::CUDF_SIZE_TYPE_SENTINEL) { continue; }
    ASSERT_GE(row, 0);
    ASSERT_LT(row, num_rows);
    ++occupied_count;
    distinct.insert(keys[row]);
  }
  EXPECT_EQ(occupied_count, 7);
  EXPECT_EQ(distinct, (std::set<size_type>{0, 1, 2, 3, 4, 5, 6}));

  // Two distinct keys compete for one slot; exhaustion must retain the winner and return
  // the complete empty-key snapshot with a false insertion flag for the other key.
  CUDF_CUDA_TRY(cudaMemsetAsync(entries.data(), 0xff, sizeof(Key), stream.get()));
  auto const full_set = set_ref{entries.data(), 1, 1};
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator{size_type{0}},
                    cuda::counting_iterator{size_type{2}},
                    results.begin(),
                    [full_set, equal = key_equal<Key>{d_keys.data()}] __device__(auto row) ->
                    typename set_ref::insert_result {
                      constexpr cudf::hash_value_type hash = 0;
                      if constexpr (cuda::std::is_same_v<Key, cached_key>) {
                        return full_set.insert(Key{hash, row}, hash, equal);
                      } else {
                        return full_set.insert(row, hash, equal);
                      }
                    });
  auto const full_results = make_pinned_vector(results, stream);
  auto const full_entries = make_pinned_vector(entries, stream);
  auto num_inserted       = 0;
  for (size_type row = 0; row < 2; ++row) {
    auto const result = full_results[row];
    if (result.second) {
      ++num_inserted;
      EXPECT_EQ(result.first.slot, 0u);
      if constexpr (cuda::std::is_same_v<Key, cached_key>) {
        EXPECT_EQ(result.first.key.first, 0u);
        EXPECT_EQ(result.first.key.second, row);
        EXPECT_EQ(full_entries.front().first, result.first.key.first);
        EXPECT_EQ(full_entries.front().second, result.first.key.second);
      } else {
        EXPECT_EQ(result.first.key, row);
        EXPECT_EQ(full_entries.front(), result.first.key);
      }
    } else {
      EXPECT_EQ(result.first.slot, full_set.capacity);
      if constexpr (cuda::std::is_same_v<Key, cached_key>) {
        EXPECT_EQ(result.first.key.first, static_cast<cudf::hash_value_type>(-1));
        EXPECT_EQ(result.first.key.second, cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
      } else {
        EXPECT_EQ(result.first.key, cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
      }
    }
  }
  EXPECT_EQ(num_inserted, 1);
}

}  // namespace

TEST_F(HashCsrTest, CachedInsertionAndKeysOnly) { check_insertions<cached_key>(); }
TEST_F(HashCsrTest, CompactInsertionAndKeysOnly) { check_insertions<size_type>(); }

TEST_F(HashCsrTest, BoundedOverflowAndRetry)
{
  auto const stream                      = cudf::get_default_stream();
  auto const mr                          = cudf::get_current_device_resource_ref();
  constexpr size_type num_rows           = 65;
  constexpr cuda::std::uint32_t capacity = 131;
  std::vector<size_type> keys(num_rows);
  std::iota(keys.begin(), keys.end(), 0);
  auto const d_keys = make_device_uvector(keys, stream, mr);
  rmm::device_uvector<size_type> entries(capacity, stream, mr);
  rmm::device_uvector<size_type> counts(capacity, stream, mr);
  rmm::device_uvector<csr::build_position_type> positions(num_rows, stream, mr);
  rmm::device_uvector<int> overflow(1, stream, mr);
  for (auto const limit : {cuda::std::uint32_t{1}, capacity}) {
    CUDF_CUDA_TRY(
      cudaMemsetAsync(entries.data(), 0xff, entries.size() * sizeof(size_type), stream.get()));
    CUDF_CUDA_TRY(
      cudaMemsetAsync(counts.data(), 0, counts.size() * sizeof(size_type), stream.get()));
    CUDF_CUDA_TRY(cudaMemsetAsync(overflow.data(), 0, sizeof(int), stream.get()));
    csr::build_count<count_mode::per_warp>(
      num_rows,
      nullptr,
      positions.data(),
      counts.data(),
      false,
      csr::hash_set_ref<size_type>{entries.data(), capacity, limit},
      key_equal<size_type>{d_keys.data()},
      constant_hash{capacity - 1},
      overflow.data(),
      stream);
    auto const h_overflow = make_pinned_vector(overflow, stream);
    auto const h_counts   = make_pinned_vector(counts, stream);
    EXPECT_EQ(h_overflow.front(), limit == 1 ? 1 : 0);
    EXPECT_EQ(std::accumulate(h_counts.begin(), h_counts.end(), 0), limit == 1 ? 1 : num_rows);
  }
  // The successful retry must replace positions from the failed attempt.
  auto const h_positions = make_pinned_vector(positions, stream);
  std::set<cuda::std::uint32_t> occupied;
  for (auto position : h_positions) {
    EXPECT_LT(position.first, capacity);
    EXPECT_EQ(position.second, 0);
    occupied.insert(position.first);
  }
  EXPECT_EQ(occupied.size(), num_rows);
}

TYPED_TEST(HashCsrBuildTest, EmptyInput)
{
  auto const stream = cudf::get_default_stream();
  csr::build_count<TypeParam::mode>(0,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    false,
                                    csr::hash_set_ref<typename TypeParam::key_type>{nullptr, 0, 0},
                                    key_equal<typename TypeParam::key_type>{nullptr},
                                    constant_hash{0},
                                    nullptr,
                                    stream);
  csr::fill(0,
            static_cast<csr::build_position_type const*>(nullptr),
            static_cast<size_type const*>(nullptr),
            static_cast<size_type*>(nullptr),
            stream);
}
