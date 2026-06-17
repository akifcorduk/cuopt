/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::detail::calib {

// Feature vector used by the deterministic work-unit linear model. The model predicts the
// per-iteration runtime (in seconds, since wups == 1) of a leaf algorithm as a linear combination
// of these features. The layout is shared between the offline calibrator (which fits the
// coefficients) and the runtime evaluator (which evaluates the same model inside the solver), so
// the order MUST stay in sync with the generated coefficient header.
//
// Base features describe the static problem structure; every leaf appends its own dynamic,
// per-iteration features after these (e.g. FJ appends frontier work per step).
struct base_features_t {
  double intercept{1.0};      // constant per-iteration overhead
  double n_vars{0.0};         // number of variables
  double n_constraints{0.0};  // number of constraints
  double nnz{0.0};            // nonzeros in the constraint matrix A
  double row_nnz_var{0.0};    // variance of nonzeros per row of A (per constraint)
  double col_nnz_var{0.0};    // variance of nonzeros per row of A^T (per variable)

  static constexpr std::size_t size() { return 6; }

  void append_to(std::vector<double>& out) const
  {
    out.push_back(intercept);
    out.push_back(n_vars);
    out.push_back(n_constraints);
    out.push_back(nnz);
    out.push_back(row_nnz_var);
    out.push_back(col_nnz_var);
  }

  static std::vector<std::string> names()
  {
    return {"intercept", "n_vars", "n_constraints", "nnz", "row_nnz_var", "col_nnz_var"};
  }
};

// Mean / variance of the gaps between consecutive CSR offsets, i.e. of the per-row nonzero counts.
// `offsets` is the CSR offset array of length (n_rows + 1). Returns {mean, variance}.
inline std::pair<double, double> row_nnz_mean_var(const std::vector<int>& offsets)
{
  if (offsets.size() < 2) { return {0.0, 0.0}; }
  const std::size_t n_rows = offsets.size() - 1;
  double mean              = 0.0;
  for (std::size_t r = 0; r < n_rows; ++r) {
    mean += static_cast<double>(offsets[r + 1] - offsets[r]);
  }
  mean /= static_cast<double>(n_rows);
  double var = 0.0;
  for (std::size_t r = 0; r < n_rows; ++r) {
    const double d = static_cast<double>(offsets[r + 1] - offsets[r]) - mean;
    var += d * d;
  }
  var /= static_cast<double>(n_rows);
  return {mean, var};
}

}  // namespace cuopt::linear_programming::detail::calib
