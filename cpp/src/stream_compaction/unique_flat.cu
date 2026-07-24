/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "stream_compaction_common.cuh"
#include "unique_helpers.cuh"

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/table/table_device_view.cuh>

namespace cudf::detail {

__device__ bool physical_elements_equal(column_device_view const& column,
                                        size_type lhs,
                                        size_type rhs)
{
  switch (column.type().id()) {
    case type_id::EMPTY: return true;
    case type_id::INT8:
    case type_id::UINT8:
    case type_id::BOOL8: return column.element<uint8_t>(lhs) == column.element<uint8_t>(rhs);
    case type_id::INT16:
    case type_id::UINT16: return column.element<uint16_t>(lhs) == column.element<uint16_t>(rhs);
    case type_id::INT32:
    case type_id::UINT32:
    case type_id::TIMESTAMP_DAYS:
    case type_id::DURATION_DAYS:
    case type_id::DECIMAL32: return column.element<uint32_t>(lhs) == column.element<uint32_t>(rhs);
    case type_id::INT64:
    case type_id::UINT64:
    case type_id::TIMESTAMP_SECONDS:
    case type_id::TIMESTAMP_MILLISECONDS:
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS:
    case type_id::DURATION_SECONDS:
    case type_id::DURATION_MILLISECONDS:
    case type_id::DURATION_MICROSECONDS:
    case type_id::DURATION_NANOSECONDS:
    case type_id::DECIMAL64: return column.element<uint64_t>(lhs) == column.element<uint64_t>(rhs);
    case type_id::FLOAT32: {
      auto const lhs_value = column.element<float>(lhs);
      auto const rhs_value = column.element<float>(rhs);
      return (isnan(lhs_value) && isnan(rhs_value)) || lhs_value == rhs_value;
    }
    case type_id::FLOAT64: {
      auto const lhs_value = column.element<double>(lhs);
      auto const rhs_value = column.element<double>(rhs);
      return (isnan(lhs_value) && isnan(rhs_value)) || lhs_value == rhs_value;
    }
    case type_id::STRING:
      return column.element<string_view>(lhs) == column.element<string_view>(rhs);
    case type_id::DECIMAL128:
      return column.element<__int128_t>(lhs) == column.element<__int128_t>(rhs);
    case type_id::DICTIONARY32:
    case type_id::LIST:
    case type_id::STRUCT:
    case type_id::NUM_TYPE_IDS: CUDF_UNREACHABLE("Unexpected nested key type");
  }
  CUDF_UNREACHABLE("Unexpected key type");
}

struct flat_row_equality {
  table_device_view keys;
  bool check_nulls;
  null_equality nulls_equal;

  __device__ bool operator()(size_type lhs, size_type rhs) const
  {
    for (size_type column_index = 0; column_index < keys.num_columns(); ++column_index) {
      auto const column = keys.column(column_index);
      if (check_nulls) {
        bool const lhs_is_null = column.is_null(lhs);
        bool const rhs_is_null = column.is_null(rhs);
        if (lhs_is_null && rhs_is_null) {
          if (nulls_equal == null_equality::UNEQUAL) { return false; }
          continue;
        }
        if (lhs_is_null != rhs_is_null) { return false; }
      }

      auto const elements_equal = [&] {
        if (column.type().id() != type_id::DICTIONARY32) {
          return physical_elements_equal(column, lhs, rhs);
        }

        auto const lhs_index       = column.element<dictionary32>(lhs).value();
        auto const rhs_index       = column.element<dictionary32>(rhs).value();
        auto const dictionary_keys = column.child(dictionary_column_view::keys_column_index);
        return physical_elements_equal(dictionary_keys, lhs_index, rhs_index);
      }();
      if (!elements_equal) { return false; }
    }
    return true;
  }
};

size_type unique_flat(table_view const& keys,
                      mutable_column_device_view& output,
                      duplicate_keep_option keep,
                      null_equality nulls_equal,
                      rmm::cuda_stream_view stream)
{
  auto const d_keys    = table_device_view::create(keys, stream);
  auto const row_equal = flat_row_equality{*d_keys, has_nested_nulls(keys), nulls_equal};
  auto const begin     = cuda::counting_iterator<size_type>{0};
  auto const result_end =
    unique_copy(begin, begin + keys.num_rows(), output.begin<size_type>(), row_equal, keep, stream);
  return static_cast<size_type>(cuda::std::distance(output.begin<size_type>(), result_end));
}

}  // namespace cudf::detail
