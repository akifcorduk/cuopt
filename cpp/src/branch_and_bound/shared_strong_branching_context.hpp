/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <atomic>
#include <cassert>
#include <span>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct shared_strong_branching_context_t {
  std::vector<std::atomic<int>> solved;
  // Set by batch PDLP for subproblems it is not expected to converge within its iteration limit
  std::vector<std::atomic<int>> struggling;
  std::atomic<int> num_struggling{0};

  explicit shared_strong_branching_context_t(size_t num_subproblems)
    : solved(num_subproblems), struggling(num_subproblems)
  {
    for (auto& s : solved)
      s.store(0);
    for (auto& s : struggling)
      s.store(0);
  }
};

template <typename i_t, typename f_t>
struct shared_strong_branching_context_view_t {
  std::span<std::atomic<int>> solved;
  std::span<std::atomic<int>> struggling;
  std::atomic<int>* num_struggling{nullptr};

  shared_strong_branching_context_view_t() = default;

  shared_strong_branching_context_view_t(std::span<std::atomic<int>> s) : solved(s) {}

  shared_strong_branching_context_view_t(std::span<std::atomic<int>> s,
                                         std::span<std::atomic<int>> st,
                                         std::atomic<int>* n)
    : solved(s), struggling(st), num_struggling(n)
  {
    assert(struggling.empty() || (struggling.size() == solved.size() && num_struggling != nullptr));
  }

  bool is_valid() const { return !solved.empty(); }

  bool tracks_struggling() const { return !struggling.empty(); }

  bool is_struggling(i_t local_idx) const
  {
    assert(local_idx >= 0 && static_cast<size_t>(local_idx) < struggling.size() &&
           "local_idx out of bounds");
    return struggling[local_idx].load() != 0;
  }

  void mark_struggling(i_t local_idx) const
  {
    assert(local_idx >= 0 && static_cast<size_t>(local_idx) < struggling.size() &&
           "local_idx out of bounds");
    if (struggling[local_idx].exchange(1) == 0) { num_struggling->fetch_add(1); }
  }

  bool is_solved(i_t local_idx) const
  {
    assert(local_idx >= 0 && static_cast<size_t>(local_idx) < solved.size() &&
           "local_idx out of bounds");
    return solved[local_idx].load() != 0;
  }

  void mark_solved(i_t local_idx) const
  {
    assert(local_idx >= 0 && static_cast<size_t>(local_idx) < solved.size() &&
           "local_idx out of bounds");
    solved[local_idx].store(1);
  }

  shared_strong_branching_context_view_t subview(i_t offset, i_t count) const
  {
    assert(offset >= 0 && count >= 0 && static_cast<size_t>(offset + count) <= solved.size() &&
           "subview out of bounds");
    return {solved.subspan(offset, count),
            struggling.empty() ? struggling : struggling.subspan(offset, count),
            num_struggling};
  }
};

}  // namespace cuopt::mathematical_optimization::mip
