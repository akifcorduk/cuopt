/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>

#include <rmm/device_uvector.hpp>

#include <vector>

namespace cuopt::linear_programming::detail {

/**
 * Input for batched LP solve. All subproblems share the same constraint matrix A
 * (from shared_problem) but may have different objectives, variable bounds, and
 * constraint bounds.
 *
 * Per-subproblem vectors are stored column-major (subproblem-strided):
 *   element(subproblem_k, var_j) = data[k + j * batch_size]
 * This maps directly to dense matrix columns for SPMM: A * X where each column
 * of X is one subproblem's primal vector.
 */
template <typename i_t, typename f_t>
struct batched_lp_input_t {
  batched_lp_input_t(i_t batch_size_,
                     problem_t<i_t, f_t>& shared_problem_,
                     rmm::cuda_stream_view stream)
    : batch_size(batch_size_),
      shared_problem_ptr(&shared_problem_),
      objective_coefficients(static_cast<size_t>(batch_size_) * shared_problem_.n_variables,
                             stream),
      variable_lower_bounds(static_cast<size_t>(batch_size_) * shared_problem_.n_variables, stream),
      variable_upper_bounds(static_cast<size_t>(batch_size_) * shared_problem_.n_variables, stream),
      constraint_lower_bounds(static_cast<size_t>(batch_size_) * shared_problem_.n_constraints,
                              stream),
      constraint_upper_bounds(static_cast<size_t>(batch_size_) * shared_problem_.n_constraints,
                              stream),
      objective_offsets(batch_size_, f_t{0}),
      initial_primal_solutions(static_cast<size_t>(batch_size_) * shared_problem_.n_variables,
                               stream)
  {
  }

  i_t n_variables() const { return shared_problem_ptr->n_variables; }
  i_t n_constraints() const { return shared_problem_ptr->n_constraints; }

  i_t batch_size;
  problem_t<i_t, f_t>* shared_problem_ptr;

  rmm::device_uvector<f_t> objective_coefficients;    // [batch_size x n_variables]
  rmm::device_uvector<f_t> variable_lower_bounds;     // [batch_size x n_variables]
  rmm::device_uvector<f_t> variable_upper_bounds;     // [batch_size x n_variables]
  rmm::device_uvector<f_t> constraint_lower_bounds;   // [batch_size x n_constraints]
  rmm::device_uvector<f_t> constraint_upper_bounds;   // [batch_size x n_constraints]
  std::vector<f_t> objective_offsets;                 // [batch_size] host-side
  rmm::device_uvector<f_t> initial_primal_solutions;  // [batch_size x n_variables]
  bool has_initial_primal{false};
};

/**
 * Output of batched LP solve.
 */
template <typename i_t, typename f_t>
struct batched_lp_output_t {
  batched_lp_output_t(i_t batch_size_, i_t n_variables, rmm::cuda_stream_view stream)
    : batch_size(batch_size_),
      primal_solutions(static_cast<size_t>(batch_size_) * n_variables, stream),
      termination_statuses(batch_size_),
      objective_values(batch_size_, f_t{0})
  {
  }

  i_t batch_size;
  rmm::device_uvector<f_t> primal_solutions;  // [batch_size x n_variables]
  std::vector<pdlp_termination_status_t> termination_statuses;
  std::vector<f_t> objective_values;
};

/**
 * Solve multiple LP subproblems sharing the same constraint matrix A.
 *
 * TODO: Replace sequential fallback with true batched PDLP using SPMM.
 * Current implementation calls get_relaxed_lp_solution N times sequentially,
 * creating a temporary problem_t copy per subproblem with modified objective,
 * bounds, and RHS.
 */
template <typename i_t, typename f_t>
batched_lp_output_t<i_t, f_t> solve_batched_lp(batched_lp_input_t<i_t, f_t>& input,
                                               const relaxed_lp_settings_t& settings,
                                               const timer_t& timer);

}  // namespace cuopt::linear_programming::detail
