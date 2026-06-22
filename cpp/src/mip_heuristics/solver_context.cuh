/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/mip/solver_stats.hpp>

#include <mip_heuristics/deterministic_calibrator/work_model.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/relaxed_lp/lp_state.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/determinism_log.hpp>
#include <utilities/termination_checker.hpp>
#include <utilities/work_limit_context.hpp>
#include <utilities/work_unit_scheduler.hpp>

#include <algorithm>
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
    : handle_ptr(handle_ptr_),
      problem_ptr(problem_ptr_),
      root_termination(settings_.determinism_mode == CUOPT_MODE_DETERMINISTIC
                         ? std::numeric_limits<f_t>::infinity()
                         : settings_.time_limit,
                       cuopt::termination_checker_t::root_tag_t{}),
      settings(settings_)
  {
    // FJ derives child work-budget checkers from this root. In deterministic mode the root is an
    // infinite wall clock so it never injects wall-clock nondeterminism; budgets are work-based.
    termination = &root_termination;
    cuopt_assert(problem_ptr != nullptr, "problem_ptr is nullptr");
    stats.set_solution_bound(problem_ptr->maximize ? std::numeric_limits<f_t>::infinity()
                                                   : -std::numeric_limits<f_t>::infinity());
    gpu_heur_loop.deterministic = settings.determinism_mode == CUOPT_MODE_DETERMINISTIC;
  }

  mip_solver_context_t(const mip_solver_context_t&)            = delete;
  mip_solver_context_t& operator=(const mip_solver_context_t&) = delete;

  // Creates a budget timer for a heuristic loop. In deterministic mode it ticks on accumulated GPU
  // work units (reproducible) instead of wall-clock time; work units are calibrated to seconds
  // (wups == 1), and the absolute --work-limit is the reproducible global stop. The work clock only
  // advances once leaves are wired to record calibrated work (see deterministic_calibrator/).
  timer_t make_heuristic_timer(double time_limit) const
  {
    timer_t t(time_limit);
    if (settings.determinism_mode == CUOPT_MODE_DETERMINISTIC) {
      t.use_work_clock(&gpu_heur_loop.global_work_units_elapsed, 1.0, settings.work_limit);
    }
    return t;
  }

  raft::handle_t const* const handle_ptr;
  problem_t<i_t, f_t>* problem_ptr;
  dual_simplex::branch_and_bound_t<i_t, f_t>* branch_and_bound_ptr{nullptr};
  diversity_manager_t<i_t, f_t>* diversity_manager_ptr{nullptr};
  std::atomic<bool> preempt_heuristic_solver_ = false;
  // Root termination checker (wall-clock cap; infinite in deterministic mode). FJ derives child
  // work-budget checkers from `termination`, which points at root_termination.
  cuopt::termination_checker_t root_termination;
  cuopt::termination_checker_t* termination{nullptr};
  const mip_solver_settings_t<i_t, f_t> settings;
  solver_stats_t<i_t, f_t> stats;
  // Work limit context for tracking work units in deterministic mode (shared across all timers in
  // GPU heuristic loop)
  work_limit_context_t gpu_heur_loop{"GPUHeur"};

  // Calibrated work-unit model: static structural features (computed once) used to convert a
  // pseudo-second work budget into a deterministic per-leaf iteration limit (wups == 1).
  calib::work_features_t work_features;
  bool work_features_ready{false};

  void ensure_work_features()
  {
    if (work_features_ready) { return; }
    auto stream = handle_ptr->get_stream();
    auto ro     = cuopt::host_copy(problem_ptr->offsets, stream);
    auto co     = cuopt::host_copy(problem_ptr->reverse_offsets, stream);
    handle_ptr->sync_stream();
    work_features = calib::compute_work_features(
      ro, co, (double)problem_ptr->n_variables, (double)problem_ptr->nnz);
    work_features_ready = true;
  }

  // Deterministic iteration budget for a leaf given a pseudo-second work budget (wups == 1).
  i_t pdlp_iters_for_budget(f_t work_budget)
  {
    ensure_work_features();
    const double wpi = calib::pdlp_work_per_iter(work_features);
    return wpi > 0.0 ? std::max<i_t>(1, (i_t)(work_budget / wpi)) : 1;
  }
  i_t fj_steps_for_budget(f_t work_budget)
  {
    ensure_work_features();
    const double wps = calib::fj_work_per_step(work_features, work_features.frontier_work_mean);
    return wps > 0.0 ? std::max<i_t>(1, (i_t)(work_budget / wps)) : 1;
  }

  // synchronization every 5 seconds for deterministic mode
  work_unit_scheduler_t work_unit_scheduler_{5.0};

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
