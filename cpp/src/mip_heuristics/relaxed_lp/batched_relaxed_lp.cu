/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "batched_relaxed_lp.cuh"

#include <cuopt/error.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <mip_heuristics/utils.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <raft/core/handle.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/tabulate.h>

namespace cuopt::linear_programming::detail {

/**
 * Sequential fallback: solve each subproblem independently via get_relaxed_lp_solution.
 * For each subproblem k, we copy the per-subproblem objective/bounds into a temporary
 * problem_t copy and invoke the single-problem LP solver.
 */
template <typename i_t, typename f_t>
batched_lp_output_t<i_t, f_t> solve_batched_lp(batched_lp_input_t<i_t, f_t>& input,
                                               const relaxed_lp_settings_t& settings,
                                               const timer_t& timer)
{
  raft::common::nvtx::range fun_scope("solve_batched_lp");

  const i_t B      = input.batch_size;
  const i_t n_vars = input.n_variables();
  const i_t n_cstr = input.n_constraints();
  auto stream      = input.shared_problem_ptr->handle_ptr->get_stream();

  batched_lp_output_t<i_t, f_t> output(B, n_vars, stream);

  const f_t per_subproblem_time = settings.time_limit / B;

  for (i_t k = 0; k < B; ++k) {
    if (timer.check_time_limit()) {
      for (i_t j = k; j < B; ++j) {
        output.termination_statuses[j] = pdlp_termination_status_t::TimeLimit;
      }
      break;
    }

    problem_t<i_t, f_t> temp_p(*input.shared_problem_ptr);

    // Row-major copy: src[k * n + j] -> dst[j] for j in [0, n)
    const f_t* obj_src = input.objective_coefficients.data() + static_cast<size_t>(k) * n_vars;
    raft::copy(temp_p.objective_coefficients.data(), obj_src, n_vars, stream);

    // Copy per-subproblem variable bounds
    const f_t* vlb_src = input.variable_lower_bounds.data() + static_cast<size_t>(k) * n_vars;
    const f_t* vub_src = input.variable_upper_bounds.data() + static_cast<size_t>(k) * n_vars;
    using f_t2         = typename type_2<f_t>::type;
    thrust::tabulate(
      rmm::exec_policy(stream),
      temp_p.variable_bounds.data(),
      temp_p.variable_bounds.data() + n_vars,
      [vlb_src, vub_src] __device__(i_t j) -> f_t2 { return f_t2{vlb_src[j], vub_src[j]}; });

    // Copy per-subproblem constraint bounds
    const f_t* clb_src = input.constraint_lower_bounds.data() + static_cast<size_t>(k) * n_cstr;
    const f_t* cub_src = input.constraint_upper_bounds.data() + static_cast<size_t>(k) * n_cstr;
    raft::copy(temp_p.constraint_lower_bounds.data(), clb_src, n_cstr, stream);
    raft::copy(temp_p.constraint_upper_bounds.data(), cub_src, n_cstr, stream);

    temp_p.presolve_data.objective_offset = input.objective_offsets[k];
    temp_p.check_problem_representation(true);

    // Build per-subproblem assignment for warm start
    rmm::device_uvector<f_t> assignment(n_vars, stream);
    if (input.has_initial_primal) {
      const f_t* primal_src =
        input.initial_primal_solutions.data() + static_cast<size_t>(k) * n_vars;
      raft::copy(assignment.data(), primal_src, n_vars, stream);
    } else {
      thrust::fill(rmm::exec_policy(stream), assignment.begin(), assignment.end(), f_t{0});
    }

    relaxed_lp_settings_t sub_settings = settings;
    sub_settings.time_limit            = per_subproblem_time;
    sub_settings.has_initial_primal    = input.has_initial_primal;
    sub_settings.save_state            = false;

    lp_state_t<i_t, f_t> lp_state(temp_p, stream);
    auto solver_response = get_relaxed_lp_solution(temp_p, assignment, lp_state, sub_settings);

    output.termination_statuses[k] = solver_response.get_termination_status();
    output.objective_values[k]     = (solver_response.get_primal_solution().size() > 0)
                                       ? solver_response.get_objective_value()
                                       : std::numeric_limits<f_t>::infinity();

    // Copy solution back into batched output (row-major)
    if (solver_response.get_primal_solution().size() > 0) {
      f_t* out_dst = output.primal_solutions.data() + static_cast<size_t>(k) * n_vars;
      raft::copy(out_dst, assignment.data(), n_vars, stream);
    }
  }

  input.shared_problem_ptr->handle_ptr->sync_stream();
  return output;
}

#define INSTANTIATE(F_TYPE)                                                \
  template batched_lp_output_t<int, F_TYPE> solve_batched_lp<int, F_TYPE>( \
    batched_lp_input_t<int, F_TYPE> & input,                               \
    const relaxed_lp_settings_t& settings,                                 \
    const timer_t& timer);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming::detail
