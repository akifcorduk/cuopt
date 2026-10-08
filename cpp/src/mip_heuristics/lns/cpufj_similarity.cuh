/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/feasibility_jump/cpu/state.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Experimental ruin selection. Cache sparse incidence, never variable-pair similarities.
template <typename i_t, typename f_t>
class cpufj_lns_similarity_t {
 public:
  cpufj_lns_similarity_t(const fj_cpu_problem_t<i_t, f_t>& problem,
                         const std::vector<i_t>& eligible)
    : eligible_(problem.n_variables, false),
      chosen_(problem.n_variables, false),
      row_offsets_(problem.n_constraints + 1, 0),
      column_offsets_(problem.n_variables + 1, 0)
  {
    for (const auto v : eligible)
      eligible_[v] = true;
    std::vector<std::pair<i_t, f_t>> row;
    std::vector<f_t> normalized;
    for (i_t r = 0; r < problem.n_constraints; ++r) {
      row.clear();
      f_t scale = 0;
      for (i_t p = problem.offsets[r]; p < problem.offsets[r + 1]; ++p)
        scale = std::max(scale, std::abs(problem.coefficients[p]));
      if (scale > 0) {
        for (i_t p = problem.offsets[r]; p < problem.offsets[r + 1]; ++p)
          row.emplace_back(problem.variables[p], problem.coefficients[p] / scale);
        std::sort(row.begin(), row.end());
        size_t count = 0;
        for (const auto entry : row) {
          if (count && row[count - 1].first == entry.first)
            row[count - 1].second += entry.second;
          else
            row[count++] = entry;
        }
        row.resize(count);
        scale = 0;
        for (const auto& entry : row)
          scale = std::max(scale, std::abs(entry.second));
        for (const auto& [v, coefficient] : row) {
          if (!eligible_[v] || coefficient == 0) continue;
          row_variables_.push_back(v);
          normalized.push_back(coefficient / scale);
          ++column_offsets_[v + 1];
        }
      }
      row_offsets_[r + 1] = row_variables_.size();
    }
    std::partial_sum(column_offsets_.begin(), column_offsets_.end(), column_offsets_.begin());
    columns_.resize(row_variables_.size());
    auto cursor = column_offsets_;
    for (i_t r = 0; r < problem.n_constraints; ++r)
      for (i_t p = row_offsets_[r]; p < row_offsets_[r + 1]; ++p)
        columns_[cursor[row_variables_[p]]++] = {r, normalized[p]};
  }

  f_t score(i_t first, i_t second, const std::vector<f_t>& assignment, f_t alpha) const
  {
    i_t left = column_offsets_[first], right = column_offsets_[second];
    const i_t left_end = column_offsets_[first + 1], right_end = column_offsets_[second + 1];
    i_t shared   = 0;
    f_t distance = 0;
    while (left < left_end && right < right_end) {
      const auto& a = columns_[left];
      const auto& b = columns_[right];
      if (a.row == b.row) {
        const f_t structural = std::abs(a.coefficient - b.coefficient) / 2;
        f_t state            = 0;
        if (alpha < 1) {
          const f_t first_contribution  = a.coefficient * assignment[first];
          const f_t second_contribution = b.coefficient * assignment[second];
          const f_t epsilon             = std::numeric_limits<f_t>::epsilon();
          const f_t scale =
            std::max({std::abs(first_contribution), std::abs(second_contribution), epsilon});
          const f_t u = first_contribution / scale, v = second_contribution / scale;
          state = std::abs(u - v) / (std::abs(u) + std::abs(v) + epsilon / scale);
        }
        distance += alpha * structural + (1 - alpha) * state;
        ++shared;
        ++left;
        ++right;
      } else if (a.row < b.row) {
        ++left;
      } else {
        ++right;
      }
    }
    if (!shared) return 0;
    const size_t union_size = (size_t)(left_end - column_offsets_[first]) +
                              (size_t)(right_end - column_offsets_[second]) - shared;
    return ((f_t)shared / union_size) * std::max(f_t{0}, 1 - distance / shared);
  }

  // Sample bounded candidates across all incident rows; random ties avoid column-index bias.
  // The caller fills any shortage with its ordinary random ruin selection.
  void select(i_t root,
              const std::vector<f_t>& assignment,
              i_t target_count,
              f_t alpha,
              std::mt19937& rng,
              std::vector<i_t>& out)
  {
    out.clear();
    if (target_count <= 0 || !eligible_[root]) return;
    out.push_back(root);
    const i_t begin = column_offsets_[root], end = column_offsets_[root + 1];
    if (target_count == 1 || begin == end) return;
    std::vector<scored_variable_t> candidates;
    candidates.reserve(candidate_limit);
    chosen_[root] = true;
    std::uniform_int_distribution<i_t> choose_row(begin, end - 1);
    for (i_t attempt = 0; attempt < 10 * candidate_limit && candidates.size() < candidate_limit;
         ++attempt) {
      const i_t row = columns_[choose_row(rng)].row;
      std::uniform_int_distribution<i_t> choose_variable(row_offsets_[row],
                                                         row_offsets_[row + 1] - 1);
      const i_t variable = row_variables_[choose_variable(rng)];
      if (chosen_[variable]) continue;
      chosen_[variable] = true;
      candidates.push_back({variable, score(root, variable, assignment, alpha)});
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
      return a.score > b.score;
    });
    for (const auto& candidate : candidates) {
      if (out.size() < (size_t)target_count) out.push_back(candidate.variable);
      chosen_[candidate.variable] = false;
    }
    chosen_[root] = false;
  }

 private:
  struct entry_t {
    i_t row;
    f_t coefficient;
  };
  struct scored_variable_t {
    i_t variable;
    f_t score;
  };
  static constexpr i_t candidate_limit = 100;
  std::vector<bool> eligible_, chosen_;
  std::vector<i_t> row_offsets_, row_variables_, column_offsets_;
  std::vector<entry_t> columns_;
};

}  // namespace cuopt::mathematical_optimization::mip
