/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/mip/solver_stats.hpp>

#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/relaxed_lp/lp_state.cuh>
#include <mip_heuristics/utilities/work_estimation.cuh>
#include <utilities/work_budget_policy.hpp>
#include <utilities/work_calibration.hpp>
#include <utilities/work_limit_context.hpp>
#include <utilities/work_unit_scheduler.hpp>

#include <limits>
#include <memory>

#pragma once

// Forward declare
namespace cuopt::linear_programming::dual_simplex {
template <typename i_t, typename f_t>
class branch_and_bound_t;
}

#include <branch_and_bound/symmetry.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class diversity_manager_t;

template <typename i_t, typename f_t>
class early_cpufj_t;

// Aggregate structure containing the global context of the solving process for convenience:
// The current problem, user settings, raft handle and statistics objects
template <typename i_t, typename f_t>
struct mip_solver_context_t {
  explicit mip_solver_context_t(raft::handle_t const* handle_ptr_,
                                problem_t<i_t, f_t>* problem_ptr_,
                                mip_solver_settings_t<i_t, f_t> settings_)
    : handle_ptr(handle_ptr_), problem_ptr(problem_ptr_), settings(settings_)
  {
    cuopt_assert(problem_ptr != nullptr, "problem_ptr is nullptr");
    stats.set_solution_bound(problem_ptr->maximize ? std::numeric_limits<f_t>::infinity()
                                                   : -std::numeric_limits<f_t>::infinity());
    gpu_heur_loop.deterministic = settings.determinism_mode == CUOPT_MODE_DETERMINISTIC;

    work_calibrator.deterministic = settings.determinism_mode == CUOPT_MODE_DETERMINISTIC;
    work_calibrator.default_wups  = settings.heuristic_params.work_unit_default_wups;
    kernel_work_coeffs.nnz_coeff  = settings.heuristic_params.work_unit_kernel_nnz_coeff;
    kernel_work_coeffs.var_coeff  = settings.heuristic_params.work_unit_kernel_var_coeff;
    kernel_work_coeffs.con_coeff  = settings.heuristic_params.work_unit_kernel_con_coeff;

    // Static problem features for the structural budget policy (computed once; cheap host
    // scalars only). Per-row/col nnz std are left as TODO until needed by structural_policy.
    problem_features.n_vars        = (std::size_t)problem_ptr->n_variables;
    problem_features.n_constraints = (std::size_t)problem_ptr->n_constraints;
    problem_features.nnz           = (std::size_t)problem_ptr->nnz;
    problem_features.row_nnz_mean =
      problem_ptr->n_constraints > 0 ? (double)problem_ptr->nnz / problem_ptr->n_constraints : 0.0;
    problem_features.col_nnz_mean =
      problem_ptr->n_variables > 0 ? (double)problem_ptr->nnz / problem_ptr->n_variables : 0.0;
    problem_features.integer_fraction =
      problem_ptr->n_variables > 0
        ? (double)problem_ptr->integer_indices.size() / problem_ptr->n_variables
        : 0.0;
  }

  mip_solver_context_t(const mip_solver_context_t&)            = delete;
  mip_solver_context_t& operator=(const mip_solver_context_t&) = delete;

  // Creates a budget timer for a heuristic ORCHESTRATING loop (one that records work via FJ / LP /
  // bounds-prop). In deterministic mode it ticks on accumulated GPU work units (reproducible)
  // instead of wall-clock time. Do NOT use this for timers passed into pure bounds-prop /
  // constraint-prop (they record no work internally, so a work clock would never advance -> hang).
  timer_t make_heuristic_timer(double time_limit) const
  {
    timer_t t(time_limit);
    if (settings.determinism_mode == CUOPT_MODE_DETERMINISTIC) {
      t.use_work_clock(&gpu_heur_loop.global_work_units_elapsed,
                       settings.heuristic_params.work_unit_default_wups);
    }
    return t;
  }

  raft::handle_t const* const handle_ptr;
  problem_t<i_t, f_t>* problem_ptr;
  dual_simplex::branch_and_bound_t<i_t, f_t>* branch_and_bound_ptr{nullptr};
  diversity_manager_t<i_t, f_t>* diversity_manager_ptr{nullptr};
  std::atomic<bool> preempt_heuristic_solver_ = false;
  const mip_solver_settings_t<i_t, f_t> settings;
  solver_stats_t<i_t, f_t> stats;
  // Work limit context for tracking work units in deterministic mode (shared across all timers in
  // GPU heuristic loop)
  work_limit_context_t gpu_heur_loop{"GPUHeur"};

  // synchronization every 5 seconds for deterministic mode
  work_unit_scheduler_t work_unit_scheduler_{5.0};

  // Time->work seeding (opportunistic measured / deterministic fixed) and the tunable
  // GPU-kernel work-unit cost model. Seeded from heuristic_params; consumed as components
  // move from time budgets to work-unit budgets.
  work_calibrator_t work_calibrator;
  kernel_work_coeffs_t kernel_work_coeffs;

  // Per-sub-algorithm work-unit budgets come from this pluggable policy (never from the
  // calibrator directly). Default is the transitional time-calibrated policy, so behavior
  // is unchanged; switch budget_policy to &structural_policy to use structural budgets.
  problem_features_t problem_features;
  time_calibrated_policy_t time_calibrated_policy{work_calibrator};
  structural_policy_t structural_policy{work_calibrator};
  work_budget_policy_t* budget_policy{&time_calibrated_policy};

  early_cpufj_t<i_t, f_t>* early_cpufj_ptr{nullptr};
  // Best upper bound from early heuristics, in user-space.
  // Must be converted to the target solver-space before use:
  //   - B&B: problem_ptr->get_solver_obj_from_user_obj(initial_upper_bound)
  //   - CPUFJ: papilo_problem.get_solver_obj_from_user_obj(initial_upper_bound)
  f_t initial_upper_bound{std::numeric_limits<f_t>::infinity()};

  // Matching incumbent assignment in original output space from early heuristics.
  std::vector<f_t> initial_incumbent_assignment{};

  // Symmetry information for orbital fixing during B&B. Null if no exploitable symmetry.
  std::unique_ptr<dual_simplex::mip_symmetry_t<i_t, f_t>> symmetry;
};

}  // namespace cuopt::linear_programming::detail
