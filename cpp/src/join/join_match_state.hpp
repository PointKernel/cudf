/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/join/join.hpp>
#include <cudf/utilities/error.hpp>

namespace cudf::detail {
struct join_match_state {
  join_match_state(void const* owner, join_kind kind, std::size_t size)
    : owner{owner}, kind{kind}, output_size{size}
  {
  }

  virtual ~join_match_state() = default;

  void validate(void const* expected_owner, join_kind expected_kind) const
  {
    CUDF_EXPECTS(owner == expected_owner && kind == expected_kind,
                 "Match context belongs to a different join object or join kind",
                 std::invalid_argument);
  }

  void const* owner;
  join_kind kind;
  std::size_t output_size;
};
}  // namespace cudf::detail
