/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <mip_heuristics/presolve/bounds_presolve.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <utilities/work_limit_context.hpp>
#include "lp_state.cuh"

namespace cuopt::linear_programming::detail {

struct relaxed_lp_settings_t {
  double tolerance  = 1e-4;
  double time_limit = 1.0;
  // Deterministic stop-gap: when > 0, cap PDLP by iteration count (which is deterministic)
  // instead of (or in addition to) wall-clock time. If left 0, get_relaxed_lp_solution will
  // derive a cap automatically when the problem is in deterministic mode (using lp_iters_per_sec).
  int iteration_limit = 0;
  // Rate used to derive the deterministic iteration cap from time_limit when iteration_limit==0.
  double lp_iters_per_sec           = 2000.0;
  bool check_infeasibility          = true;
  bool return_first_feasible        = false;
  bool save_state                   = true;
  bool per_constraint_residual      = true;
  bool has_initial_primal           = true;
  std::atomic<int>* concurrent_halt = nullptr;
  // Deterministic work accounting. When work_context is set (and in deterministic mode) the relaxed
  // LP interprets time_limit as a pseudo-second work budget: PDLP is capped to a reproducible
  // iteration count (work_budget / work_per_iter) and the steps it actually takes are charged onto
  // the shared work clock. This is what keeps relaxed-LP solves from burning uncharged wall time
  // (the GA offspring LP and rounding-fallback LP previously did). work_per_iter is the calibrated
  // pdlp_device_work_per_iter for the parent problem (mirrors the fp_recombiner wiring).
  cuopt::work_limit_context_t* work_context = nullptr;
  double work_per_iter                      = 0.0;
  // Fixed per-call work charged in deterministic mode on top of work_per_iter * steps. The PDLP work
  // model (work_per_iter) only captures the per-step marginal cost; the fixed host/launch overhead
  // of a solve (solver construction, cuSPARSE setup, initial-solution copies, stream syncs, log
  // flush) is invisible to it. Instances that fire hundreds of thousands of tiny warm-started
  // relaxed LPs (e.g. triptim1, cbs-cta) were therefore dominated by uncharged overhead and blew
  // past the work budget. Charging a fixed cost per call bounds the number of relaxed-LP solves.
  double call_overhead = 0.0;
};

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> get_relaxed_lp_solution(
  problem_t<i_t, f_t>& op_problem,
  solution_t<i_t, f_t>& solution,
  const relaxed_lp_settings_t& settings);

template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> get_relaxed_lp_solution(
  problem_t<i_t, f_t>& op_problem,
  rmm::device_uvector<f_t>& assignment,
  lp_state_t<i_t, f_t>& lp_state,
  const relaxed_lp_settings_t& settings);

template <typename i_t, typename f_t>
bool run_lp_with_vars_fixed(problem_t<i_t, f_t>& op_problem,
                            solution_t<i_t, f_t>& solution,
                            const rmm::device_uvector<i_t>& variables_to_fix,
                            relaxed_lp_settings_t& settings,
                            bound_presolve_t<i_t, f_t>* bound_presolve = nullptr,
                            bool check_fixed_assignment_feasibility    = false,
                            bool use_integer_fixed_problem             = false);

}  // namespace cuopt::linear_programming::detail
