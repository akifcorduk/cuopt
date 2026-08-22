/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "pre_presolve_primal.cuh"

#include <branch_and_bound/branch_and_bound.hpp>
#include <dual_simplex/solve.hpp>
#include <math_optimization/tic_toc.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <pdlp/translate.hpp>
#include <utilities/logger.hpp>
#include <utilities/scope_guard.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace cuopt::mathematical_optimization::mip {

pre_presolve_thread_budget_t::pre_presolve_thread_budget_t(int available_slots)
  : capacity_(std::max(available_slots, 0)), available_(capacity_)
{
}

bool pre_presolve_thread_budget_t::try_reserve(int slots)
{
  if (slots <= 0) { return true; }
  int available = available_.load(std::memory_order_relaxed);
  while (available >= slots) {
    if (available_.compare_exchange_weak(
          available, available - slots, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      CUOPT_LOG_DEBUG("Pre-presolve OMP budget: reserved=%d available=%d capacity=%d",
                      reserved(),
                      this->available(),
                      capacity());
      return true;
    }
  }
  return false;
}

void pre_presolve_thread_budget_t::release(int slots)
{
  if (slots <= 0) { return; }
  const int previous = available_.fetch_add(slots, std::memory_order_acq_rel);
  if (previous + slots > capacity_) {
    available_.fetch_sub(slots, std::memory_order_relaxed);
    throw std::logic_error("pre-presolve OpenMP task budget released too many slots");
  }
  CUOPT_LOG_DEBUG("Pre-presolve OMP budget: reserved=%d available=%d capacity=%d",
                  reserved(),
                  available(),
                  capacity());
}

namespace {

constexpr int pre_bnb_nnz_limit        = 50'000;
constexpr int pre_bnb_node_limit       = 64;
constexpr int pre_bnb_num_threads      = 1;
constexpr double root_lp_time_limit    = 0.10;
constexpr double pre_bnb_time_limit    = 0.15;
constexpr const char* pre_bnb_arm_name = "PRE-BNB";

template <typename i_t, typename f_t>
simplex::user_problem_t<i_t, f_t> snapshot_original_problem(
  const optimization_problem_t<i_t, f_t>& op_problem)
{
  auto user_problem =
    cuopt_problem_to_user_problem<i_t, f_t>(op_problem.get_handle_ptr(), op_problem);

  // Match problem_t's always-minimizing solver space while retaining the user's objective
  // scaling. The generic optimization_problem_t translation only records objective sense.
  user_problem.obj_scale = op_problem.get_objective_scaling_factor();
  if (op_problem.get_sense()) {
    std::transform(user_problem.objective.begin(),
                   user_problem.objective.end(),
                   user_problem.objective.begin(),
                   [](f_t value) { return -value; });
    std::transform(user_problem.Q_values.begin(),
                   user_problem.Q_values.end(),
                   user_problem.Q_values.begin(),
                   [](f_t value) { return -value; });
    user_problem.obj_constant = -op_problem.get_objective_offset();
    user_problem.obj_scale    = -user_problem.obj_scale;
  }
  return user_problem;
}

template <typename i_t, typename f_t>
simplex::simplex_solver_settings_t<i_t, f_t> make_pre_presolve_lp_settings(
  const mip_solver_settings_t<i_t, f_t>& settings, f_t time_limit)
{
  simplex::simplex_solver_settings_t<i_t, f_t> lp_settings;
  lp_settings.time_limit           = time_limit;
  lp_settings.num_threads          = pre_bnb_num_threads;
  lp_settings.print_presolve_stats = false;
  lp_settings.primal_tol           = settings.tolerances.absolute_tolerance;
  lp_settings.dual_tol             = settings.tolerances.absolute_tolerance;
  lp_settings.integer_tol          = settings.tolerances.integrality_tolerance;
  lp_settings.absolute_mip_gap_tol = settings.tolerances.absolute_mip_gap;
  lp_settings.relative_mip_gap_tol = settings.tolerances.relative_mip_gap;
  lp_settings.set_log(false);
  return lp_settings;
}

template <typename i_t, typename f_t>
void disable_pre_bnb_cuts(simplex::simplex_solver_settings_t<i_t, f_t>& settings)
{
  settings.max_cut_passes             = 0;
  settings.mir_cuts                   = 0;
  settings.mixed_integer_gomory_cuts  = 0;
  settings.knapsack_cuts              = 0;
  settings.flow_cover_cuts            = 0;
  settings.implied_bound_cuts         = 0;
  settings.clique_cuts                = 0;
  settings.zero_half_cuts             = 0;
  settings.strong_chvatal_gomory_cuts = 0;
  settings.reduced_cost_strengthening = 0;
}

class task_slot_release_t {
 public:
  task_slot_release_t(pre_presolve_thread_budget_t* budget, int slots)
    : budget_(budget), slots_(slots)
  {
  }
  ~task_slot_release_t()
  {
    try {
      budget_->release(slots_);
    } catch (const std::exception& e) {
      CUOPT_LOG_ERROR("Failed to release pre-presolve OpenMP task slot: %s", e.what());
    }
  }

 private:
  pre_presolve_thread_budget_t* budget_;
  int slots_;
};

}  // namespace

template <typename i_t, typename f_t>
pre_presolve_primal_t<i_t, f_t>::pre_presolve_primal_t(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const mip_solver_settings_t<i_t, f_t>& settings,
  early_incumbent_callback_t<f_t> incumbent_callback,
  pre_presolve_thread_budget_t& thread_budget)
  : mode_(pre_presolve_primal_branch_mode),
    thread_budget_(thread_budget),
    task_state_(std::make_unique<task_state_t>()),
    replaces_early_gpufj_(pre_presolve_primal_mode_replaces_early_gpufj(mode_))
{
  configure_work(op_problem, settings, std::move(incumbent_callback));
}

template <typename i_t, typename f_t>
pre_presolve_primal_t<i_t, f_t>::~pre_presolve_primal_t()
{
  stop_no_throw();
}

template <typename i_t, typename f_t>
void pre_presolve_primal_t<i_t, f_t>::configure_work(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const mip_solver_settings_t<i_t, f_t>& settings,
  early_incumbent_callback_t<f_t> incumbent_callback)
{
  if (mode_ != 1 && mode_ != 2) { return; }

  if (mode_ == 2 && op_problem.get_nnz() > pre_bnb_nnz_limit) {
    CUOPT_LOG_INFO("Skipping pre-presolve B&B: %d nonzeros exceeds the %d-nonzero gate",
                   op_problem.get_nnz(),
                   pre_bnb_nnz_limit);
    return;
  }

  // This host snapshot is complete before the task is launched. PaPILO may subsequently mutate
  // op_problem without racing the CPU simplex/B&B arm.
  auto user_problem =
    std::make_shared<simplex::user_problem_t<i_t, f_t>>(snapshot_original_problem(op_problem));

  if (mode_ == 1) {
    auto lp_settings =
      make_pre_presolve_lp_settings<i_t, f_t>(settings, static_cast<f_t>(root_lp_time_limit));
    work_ = [user_problem, lp_settings](task_state_t& task_state) mutable {
      if (task_state.stop_requested.load(std::memory_order_acquire)) { return; }

      const double solve_start        = tic();
      auto arm_lp_settings            = lp_settings;
      arm_lp_settings.concurrent_halt = &task_state.root_lp_halt;
      simplex::lp_solution_t<i_t, f_t> lp_solution(user_problem->num_rows, user_problem->num_cols);
      const auto status =
        simplex::solve_linear_program(*user_problem, arm_lp_settings, solve_start, lp_solution);
      if (status == simplex::lp_status_t::OPTIMAL) {
        // A fractional relaxation is an arm-local starting point, not an incumbent.
        task_state.root_lp_point = std::move(lp_solution.x);
      }
      CUOPT_LOG_INFO("Pre-presolve root LP: status=%s elapsed=%.3fs point=%s",
                     simplex::lp_status_to_string(status).c_str(),
                     toc(solve_start),
                     task_state.root_lp_point.empty() ? "unavailable" : "retained");
    };
    return;
  }

  auto bnb_settings =
    make_pre_presolve_lp_settings<i_t, f_t>(settings, static_cast<f_t>(pre_bnb_time_limit));
  bnb_settings.node_limit = pre_bnb_node_limit;
  disable_pre_bnb_cuts(bnb_settings);
  bnb_settings.reliability_branching                    = 0;
  bnb_settings.strong_branching_simplex_iteration_limit = 0;
  bnb_settings.mip_batch_pdlp_strong_branching          = 0;
  bnb_settings.mip_batch_pdlp_reliability_branching     = 0;
  bnb_settings.symmetry                                 = 0;
  bnb_settings.submip_settings.rins                     = 0;
  bnb_settings.submip_settings.max_level                = 0;
  bnb_settings.submip_settings.enable_cpufj             = false;
  bnb_settings.diving_settings                          = settings.diving_params;
  bnb_settings.diving_settings.min_node_depth           = 0;

  const f_t objective_scale    = user_problem->obj_scale;
  const f_t objective_constant = user_problem->obj_constant;
  bnb_settings.solution_callback =
    [incumbent_callback = std::move(incumbent_callback), objective_scale, objective_constant](
      std::vector<f_t>& assignment, f_t solver_obj) {
      if (!incumbent_callback) { return; }
      const f_t user_obj = objective_scale * (solver_obj + objective_constant);
      incumbent_callback(solver_obj, user_obj, assignment, pre_bnb_arm_name);
    };

  work_ = [user_problem, bnb_settings](task_state_t& task_state) mutable {
    if (task_state.stop_requested.load(std::memory_order_acquire)) { return; }

    const double solve_start = tic();
    // B&B stores both constructor arguments by reference; keep these named objects alive through
    // solve() and unregister the live B&B pointer before either is destroyed.
    probing_implied_bound_t<i_t, f_t> probing_implied_bound(user_problem->num_cols);
    branch_and_bound_t<i_t, f_t> branch_and_bound(
      *user_problem, bnb_settings, tic(), probing_implied_bound);
    branch_and_bound.set_concurrent_lp_root_solve(false);
    branch_and_bound.set_submip_halt_callback([&task_state](f_t, f_t) {
      return task_state.stop_requested.load(std::memory_order_acquire);
    });

    {
      std::lock_guard<std::mutex> lock(task_state.root_halt_mutex);
      task_state.root_halt_callback = [&branch_and_bound]() {
        branch_and_bound.set_root_concurrent_halt(1);
      };
      if (task_state.stop_requested.load(std::memory_order_acquire)) {
        task_state.root_halt_callback();
      }
    }
    cuopt::scope_guard unregister_root_halt([&task_state]() {
      std::lock_guard<std::mutex> lock(task_state.root_halt_mutex);
      task_state.root_halt_callback = {};
    });

    if (task_state.stop_requested.load(std::memory_order_acquire)) { return; }
    simplex::mip_solution_t<i_t, f_t> solution(user_problem->num_cols);
    const auto status = branch_and_bound.solve(solution);
    CUOPT_LOG_INFO("Pre-presolve B&B: status=%s elapsed=%.3fs nodes=%d",
                   mip_status_to_string(status).c_str(),
                   toc(solve_start),
                   solution.nodes_explored);
  };
}

template <typename i_t, typename f_t>
bool pre_presolve_primal_t<i_t, f_t>::start()
{
  if (mode_ == 0 || started_) { return started_; }
  if (!work_) {
    if (mode_ == 1 || mode_ == 2) { return false; }
    CUOPT_LOG_WARN("Pre-presolve primal mode %d is not implemented on this branch", mode_);
    return false;
  }

  constexpr int task_slots = 1;
  if (!thread_budget_.try_reserve(task_slots)) {
    CUOPT_LOG_DEBUG("Skipping pre-presolve primal mode %d: no OpenMP task slot", mode_);
    return false;
  }

  started_            = true;
  auto* task_state    = task_state_.get();
  auto* thread_budget = &thread_budget_;
  auto* work          = &work_;

#pragma omp task default(none) firstprivate(task_state, thread_budget, work) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY) depend(out : *task_state)
  {
    task_slot_release_t release(thread_budget, 1);
    try {
      (*work)(*task_state);
    } catch (...) {
      task_state->exception = std::current_exception();
    }
  }
  return true;
}

template <typename i_t, typename f_t>
void pre_presolve_primal_t<i_t, f_t>::stop()
{
  if (!started_) { return; }
  task_state_->stop_requested.store(true, std::memory_order_release);
  task_state_->root_lp_halt.store(1, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(task_state_->root_halt_mutex);
    if (task_state_->root_halt_callback) { task_state_->root_halt_callback(); }
  }
  auto* task_state = task_state_.get();
#pragma omp taskwait depend(in : *task_state)
  started_ = false;
  if (task_state_->exception) { std::rethrow_exception(task_state_->exception); }
}

template <typename i_t, typename f_t>
void pre_presolve_primal_t<i_t, f_t>::stop_no_throw() noexcept
{
  try {
    stop();
  } catch (const std::exception& e) {
    CUOPT_LOG_ERROR("Pre-presolve primal task failed during destruction: %s", e.what());
  } catch (...) {
    CUOPT_LOG_ERROR("Pre-presolve primal task failed during destruction");
  }
}

#if MIP_INSTANTIATE_FLOAT
template class pre_presolve_primal_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class pre_presolve_primal_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
