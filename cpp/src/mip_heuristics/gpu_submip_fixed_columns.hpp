/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <mip_heuristics/utils.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// A private GPU representation of an equality LP. The CPU LP, basis, and row identities stay
// unchanged. Only columns fixed at construction may be excluded; later node bounds are gathered
// through the same map, and candidates are restored before the full node feasibility check.
template <typename i_t, typename f_t>
struct gpu_submip_fixed_columns_t {
  gpu_submip_fixed_columns_t(const simplex::lp_problem_t<i_t, f_t>& root,
                             const std::vector<simplex::variable_type_t>& types,
                             bool enabled = true)
    : reduced_lp(root.handle_ptr, root.num_rows, 0, 0),
      full_num_cols_(root.num_cols),
      full_num_rows_(root.num_rows)
  {
    const auto n = static_cast<size_t>(root.num_cols);
    if (root.num_cols < 0 || root.num_rows < 0 || root.lower.size() != n ||
        root.upper.size() != n || root.objective.size() != n || types.size() > n ||
        root.rhs.size() != static_cast<size_t>(root.num_rows)) {
      throw std::invalid_argument("GPU fixed-column reduction received inconsistent LP dimensions");
    }
    full_to_free.assign(n, -1);
    fixed_values.assign(n, f_t{0});
    for (i_t j = 0; j < root.num_cols; ++j) {
      const bool integer =
        static_cast<size_t>(j) < types.size() && types[j] != simplex::variable_type_t::CONTINUOUS;
      // Keep a fractionally fixed integer so its integrality restriction is not lost.
      const bool fixed = enabled && std::isfinite(root.lower[j]) &&
                         root.lower[j] == root.upper[j] &&
                         (!integer || root.lower[j] == std::round(root.lower[j]));
      if (fixed) {
        fixed_columns.push_back(j);
        fixed_values[j] = root.lower[j];
      } else {
        full_to_free[j] = static_cast<i_t>(free_to_full.size());
        free_to_full.push_back(j);
      }
    }
    // An all-fixed root is normally handled by CPU B&B. Retain its original representation
    // if it reaches FP, avoiding a zero-column GPU model and preserving normal validation.
    if (free_to_full.empty() && root.num_cols > 0) {
      fixed_columns.clear();
      std::fill(fixed_values.begin(), fixed_values.end(), f_t{0});
      for (i_t j = 0; j < root.num_cols; ++j) {
        full_to_free[j] = j;
        free_to_full.push_back(j);
      }
    }

    i_t nnz = 0;
    for (i_t j : free_to_full) {
      nnz += root.A.col_start[j + 1] - root.A.col_start[j];
    }
    reduced_lp.num_cols = static_cast<i_t>(free_to_full.size());
    reduced_lp.A.resize(root.num_rows, reduced_lp.num_cols, nnz);
    reduced_lp.lower        = gather(root.lower);
    reduced_lp.upper        = gather(root.upper);
    reduced_lp.objective    = gather(root.objective);
    reduced_lp.obj_constant = f_t{0};
    reduced_lp.obj_scale    = f_t{1};
    reduced_types.reserve(free_to_full.size());
    i_t next = 0;
    for (i_t k = 0; k < reduced_lp.num_cols; ++k) {
      const i_t j               = free_to_full[k];
      reduced_lp.A.col_start[k] = next;
      const auto type =
        static_cast<size_t>(j) < types.size() ? types[j] : simplex::variable_type_t::CONTINUOUS;
      reduced_types.push_back(type);
      for (i_t p = root.A.col_start[j]; p < root.A.col_start[j + 1]; ++p) {
        reduced_lp.A.i[next] = root.A.i[p];
        reduced_lp.A.x[next] = root.A.x[p];
        ++next;
      }
    }
    reduced_lp.A.col_start.back() = next;
    reduced_lp.rhs                = root.rhs;
    if (!fixed_columns.empty()) {
      std::vector<f_t> negative_fixed(n);
      for (size_t j = 0; j < n; ++j) {
        negative_fixed[j] = -fixed_values[j];
      }
      csr_matrix_t<i_t, f_t> matrix(root.num_rows, root.num_cols, root.A.nnz());
      root.A.to_compressed_row(matrix);
      for (i_t row = 0; row < root.num_rows; ++row) {
        // Include the original RHS in the compensated sum, so cancellation between a large
        // fixed activity and the RHS does not discard the remaining free-variable activity.
        reduced_lp.rhs[row] = compensated_dot2_csr(matrix.row_start.data(),
                                                   matrix.j.data(),
                                                   matrix.x.data(),
                                                   negative_fixed.data(),
                                                   row,
                                                   root.rhs[row]);
        if (!std::isfinite(reduced_lp.rhs[row])) {
          throw std::invalid_argument("GPU fixed-column reduction produced a nonfinite RHS");
        }
      }
      objective_shift = compensated_dot2_with_correction(
        root.objective.data(), fixed_values.data(), n, objective_shift_correction_);
      if (!std::isfinite(objective_shift) || !std::isfinite(objective_shift_correction_)) {
        throw std::invalid_argument("GPU fixed-column reduction produced a nonfinite objective");
      }
    }
  }

  void validate_node_bounds(const simplex::lp_problem_t<i_t, f_t>& node) const
  {
    if (node.num_cols != full_num_cols_ || node.num_rows != full_num_rows_ ||
        node.lower.size() != static_cast<size_t>(full_num_cols_) ||
        node.upper.size() != static_cast<size_t>(full_num_cols_)) {
      throw std::invalid_argument("GPU fixed-column session must be recreated after model changes");
    }
    for (i_t j : fixed_columns) {
      if (node.lower[j] != fixed_values[j] || node.upper[j] != fixed_values[j]) {
        throw std::invalid_argument(
          "GPU fixed-column session must be recreated after fixing changes");
      }
    }
  }

  std::vector<f_t> gather(const std::vector<f_t>& full) const
  {
    if (full.size() != static_cast<size_t>(full_num_cols_)) {
      throw std::invalid_argument("GPU fixed-column gather received the wrong vector size");
    }
    std::vector<f_t> reduced;
    reduced.reserve(free_to_full.size());
    for (i_t j : free_to_full) {
      reduced.push_back(full[j]);
    }
    return reduced;
  }

  std::vector<f_t> scatter(const std::vector<f_t>& reduced) const
  {
    if (reduced.size() != free_to_full.size()) {
      throw std::invalid_argument("GPU fixed-column scatter received the wrong vector size");
    }
    auto full = fixed_values;
    for (size_t k = 0; k < free_to_full.size(); ++k) {
      full[free_to_full[k]] = reduced[k];
    }
    return full;
  }

  f_t reduced_cutoff(f_t full_cutoff) const
  {
    if (!std::isfinite(full_cutoff)) { return full_cutoff; }
    // Retain the low part of the fixed objective when a large cutoff nearly cancels it.
    const f_t terms[]{-objective_shift, -objective_shift_correction_};
    const f_t ones[]{f_t{1}, f_t{1}};
    return compensated_dot2(terms, ones, size_t{2}, full_cutoff);
  }

  simplex::lp_problem_t<i_t, f_t> reduced_lp;
  std::vector<simplex::variable_type_t> reduced_types;
  std::vector<i_t> free_to_full;
  std::vector<i_t> full_to_free;
  std::vector<i_t> fixed_columns;
  std::vector<f_t> fixed_values;
  f_t objective_shift{0};

 private:
  i_t full_num_cols_;
  i_t full_num_rows_;
  f_t objective_shift_correction_{0};
};

}  // namespace cuopt::mathematical_optimization::mip
