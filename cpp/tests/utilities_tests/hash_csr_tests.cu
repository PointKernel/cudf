/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_list_utilities.hpp>

#include <cudf/detail/utilities/hash_csr_kernels.cuh>
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
using cudf::size_type;
using cudf::detail::build_hash_csr;
using cudf::detail::csr_ref;
using cudf::detail::fill_hash_csr;
using cudf::detail::hash_csr_build_position;
using cudf::detail::hash_csr_count_mode;
using cudf::detail::hash_set_ref;
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

template <typename Key, hash_csr_count_mode Mode>
struct build_config {
  using key_type             = Key;
  static constexpr auto mode = Mode;
};

}  // namespace

template <typename Config>
struct HashCsrBuildTest : cudf::test::BaseFixture {};

using BuildConfigs = cudf::test::Types<build_config<cached_key, hash_csr_count_mode::per_row>,
                                       build_config<size_type, hash_csr_count_mode::per_row>,
                                       build_config<cached_key, hash_csr_count_mode::per_warp>,
                                       build_config<size_type, hash_csr_count_mode::per_warp>>;
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
  rmm::device_uvector<key_type> slots(capacity, stream, mr);
  auto counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(num_counts, stream, mr);
  rmm::device_uvector<hash_csr_build_position> positions(num_rows, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(slots.data(), 0xff, slots.size() * sizeof(*slots.data()), stream.get()));
  build_hash_csr<Config::mode>(num_rows,
                               mask_kind == validity::all_valid ? nullptr : d_mask.data(),
                               positions.data(),
                               counts.data(),
                               by_representative,
                               hash_set_ref<key_type>{slots.data(), capacity, capacity},
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
      ASSERT_EQ(h_positions[row].first,
                static_cast<cuda::std::uint32_t>(cudf::detail::CUDF_SIZE_TYPE_SENTINEL));
      ASSERT_EQ(h_positions[row].second, cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
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
  if constexpr (Config::mode == hash_csr_count_mode::per_row) {
    auto const starts = csr_ref{d_ends.data(), values.data()}.begin();
    fill_hash_csr(num_rows, positions.data(), starts, values.data(), stream);
  } else {
    std::vector<size_type> starts(num_counts, 0);
    std::copy(ends.begin(), ends.end() - 1, starts.begin() + 1);
    auto const d_starts = make_device_uvector(starts, stream, mr);
    fill_hash_csr(num_rows,
                  cub::CacheModifiedInputIterator<cub::LOAD_CS, hash_csr_build_position const>{
                    positions.data()},
                  d_starts.data(),
                  values.data(),
                  stream);
  }
  auto const h_values = make_pinned_vector(values, stream);
  std::map<size_type, std::vector<size_type>> actual;
  for (size_type slot = 0; slot < num_counts; ++slot) {
    auto const begin = slot == 0 ? 0 : ends[slot - 1];
    if (h_counts[slot] == 0) { continue; }
    std::vector<size_type> rows(h_values.begin() + begin, h_values.begin() + ends[slot]);
    std::sort(rows.begin(), rows.end());
    for (auto row : rows) {
      ASSERT_GE(row, 0);
      ASSERT_LT(row, num_rows);
    }
    auto const key = keys[rows.front()];
    EXPECT_TRUE(actual.emplace(key, rows).second);
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

namespace {
void check_offsets()
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const ends   = make_device_uvector(std::vector<size_type>{0, 2, 2, 5}, stream, mr);
  rmm::device_uvector<size_type> values(5, stream, mr);
  rmm::device_uvector<hash_csr_build_position> bounds(ends.size(), stream, mr);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    cuda::counting_iterator{size_type{0}},
    cuda::counting_iterator{size_type{4}},
    bounds.begin(),
    [view = csr_ref{ends.data(), values.data()}] __device__(auto slot) -> hash_csr_build_position {
      return {static_cast<cuda::std::uint32_t>(view.begin(slot)), view.size(slot)};
    });
  auto const actual = make_pinned_vector(bounds, stream);
  std::vector<size_type> const starts{0, 0, 2, 2};
  std::vector<size_type> const sizes{0, 2, 0, 3};
  for (size_type slot = 0; slot < 4; ++slot) {
    EXPECT_EQ(actual[slot].first, starts[slot]);
    EXPECT_EQ(actual[slot].second, sizes[slot]);
  }
}
}  // namespace

TEST_F(HashCsrTest, SegmentOffsets) { check_offsets(); }

TEST_F(HashCsrTest, RepresentativeCounts)
{
  check_membership<build_config<cached_key, hash_csr_count_mode::per_warp>>(validity::mixed, true);
  check_membership<build_config<size_type, hash_csr_count_mode::per_warp>>(validity::mixed, true);
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
  using set_ref     = hash_set_ref<Key>;
  rmm::device_uvector<Key> slots(capacity, stream, mr);
  rmm::device_uvector<typename set_ref::insert_result> results(num_rows, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(slots.data(), 0xff, slots.size() * sizeof(*slots.data()), stream.get()));
  auto const hash_set = set_ref{slots.data(), capacity, capacity};
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator{size_type{0}},
                    cuda::counting_iterator{num_rows},
                    results.begin(),
                    [hash_set, equal = key_equal<Key>{d_keys.data()}] __device__(auto row) ->
                    typename set_ref::insert_result {
                      auto const hash_value = hash_set.capacity - 1;
                      if constexpr (cuda::std::is_same_v<Key, cached_key>) {
                        return hash_set.insert(Key{hash_value, row}, hash_value, equal);
                      } else {
                        return hash_set.insert(row, hash_value, equal);
                      }
                    });
  auto const h_results     = make_pinned_vector(results, stream);
  auto const inserted_keys = make_pinned_vector(slots, stream);
  std::map<size_type, cuda::std::uint32_t> key_slots;
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
    auto const it = key_slots.emplace(keys[row], result.first.slot).first;
    EXPECT_EQ(it->second, result.first.slot);
    insertions[keys[row]] += result.second;
  }
  std::set<cuda::std::uint32_t> occupied;
  for (auto const& [key, slot] : key_slots) {
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
      auto const hash_value = hash_set.capacity - 1;
      if constexpr (cuda::std::is_same_v<Key, cached_key>) {
        return hash_set.find(Key{hash_value, row}, hash_value, equal);
      } else {
        return hash_set.find(row, hash_value, equal);
      }
    });
  auto const h_found = make_pinned_vector(found, stream);
  for (size_type row = 0; row < num_rows; ++row) {
    EXPECT_EQ(h_found[row], key_slots.at(keys[row]));
  }
  EXPECT_EQ(h_found.back(), capacity);
  // A keys-only build uses the same set without allocating position or count arrays.
  CUDF_CUDA_TRY(
    cudaMemsetAsync(slots.data(), 0xff, slots.size() * sizeof(*slots.data()), stream.get()));
  build_hash_csr<hash_csr_count_mode::per_warp>(num_rows,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                false,
                                                hash_set,
                                                key_equal<Key>{d_keys.data()},
                                                constant_hash{capacity - 1},
                                                nullptr,
                                                stream);
  auto const h_slots = make_pinned_vector(slots, stream);
  std::set<size_type> distinct;
  size_type occupied_count{};
  for (auto key : h_slots) {
    auto const row = [&] {
      if constexpr (cuda::std::is_same_v<Key, cached_key>) {
        return key.second;
      } else {
        return key;
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
  slots.resize(1, stream);
  results.resize(2, stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(slots.data(), 0xff, sizeof(Key), stream.get()));
  auto const full_set = set_ref{slots.data(), 1, 1};
  thrust::transform(rmm::exec_policy_nosync(stream),
                    cuda::counting_iterator{size_type{0}},
                    cuda::counting_iterator{size_type{2}},
                    results.begin(),
                    [full_set, equal = key_equal<Key>{d_keys.data()}] __device__(auto row) ->
                    typename set_ref::insert_result {
                      constexpr cudf::hash_value_type hash_value = 0;
                      if constexpr (cuda::std::is_same_v<Key, cached_key>) {
                        return full_set.insert(Key{hash_value, row}, hash_value, equal);
                      } else {
                        return full_set.insert(row, hash_value, equal);
                      }
                    });
  auto const full_results = make_pinned_vector(results, stream);
  auto const full_slots   = make_pinned_vector(slots, stream);
  auto num_inserted       = 0;
  for (size_type row = 0; row < 2; ++row) {
    auto const result = full_results[row];
    if (result.second) {
      ++num_inserted;
      EXPECT_EQ(result.first.slot, 0u);
      if constexpr (cuda::std::is_same_v<Key, cached_key>) {
        EXPECT_EQ(result.first.key.first, 0u);
        EXPECT_EQ(result.first.key.second, row);
        EXPECT_EQ(full_slots.front().first, result.first.key.first);
        EXPECT_EQ(full_slots.front().second, result.first.key.second);
      } else {
        EXPECT_EQ(result.first.key, row);
        EXPECT_EQ(full_slots.front(), result.first.key);
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
  rmm::device_uvector<size_type> slots(capacity, stream, mr);
  rmm::device_uvector<size_type> counts(capacity, stream, mr);
  rmm::device_uvector<hash_csr_build_position> positions(num_rows, stream, mr);
  rmm::device_uvector<int> overflow(1, stream, mr);
  for (auto const limit : {cuda::std::uint32_t{1}, capacity}) {
    CUDF_CUDA_TRY(
      cudaMemsetAsync(slots.data(), 0xff, slots.size() * sizeof(size_type), stream.get()));
    CUDF_CUDA_TRY(
      cudaMemsetAsync(counts.data(), 0, counts.size() * sizeof(size_type), stream.get()));
    CUDF_CUDA_TRY(cudaMemsetAsync(overflow.data(), 0, sizeof(int), stream.get()));
    build_hash_csr<hash_csr_count_mode::per_warp>(
      num_rows,
      nullptr,
      positions.data(),
      counts.data(),
      false,
      hash_set_ref<size_type>{slots.data(), capacity, limit},
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

TEST_F(HashCsrTest, EmptyInput)
{
  auto const stream = cudf::get_default_stream();
  build_hash_csr<hash_csr_count_mode::per_row>(0,
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               false,
                                               hash_set_ref<size_type>{nullptr, 0, 0},
                                               key_equal<size_type>{nullptr},
                                               constant_hash{0},
                                               nullptr,
                                               stream);
  fill_hash_csr(0,
                static_cast<hash_csr_build_position const*>(nullptr),
                static_cast<size_type const*>(nullptr),
                static_cast<size_type*>(nullptr),
                stream);
}
