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

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

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
}

namespace {

constexpr int small_nnz_limit   = 5'000;
constexpr int low_row_limit     = 64;
constexpr int low_row_nnz_limit = 50'000;
constexpr int node_limit        = 64;
constexpr double time_limit     = 0.15;
constexpr const char* arm_name  = "PRE-BNB-MULTIDIVE";

template <typename i_t>
bool is_eligible(i_t num_rows, i_t nnz)
{
  return nnz <= small_nnz_limit || (num_rows <= low_row_limit && nnz <= low_row_nnz_limit);
}

template <typename i_t, typename f_t>
simplex::user_problem_t<i_t, f_t> snapshot_original_problem(
  const optimization_problem_t<i_t, f_t>& op_problem)
{
  auto user_problem =
    cuopt_problem_to_user_problem<i_t, f_t>(op_problem.get_handle_ptr(), op_problem);

  // Match problem_t's always-minimizing solver space while retaining the user's objective scale.
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
simplex::simplex_solver_settings_t<i_t, f_t> make_settings(
  const mip_solver_settings_t<i_t, f_t>& settings, int task_slots)
{
  simplex::simplex_solver_settings_t<i_t, f_t> bnb_settings;
  bnb_settings.time_limit                               = static_cast<f_t>(time_limit);
  bnb_settings.num_threads                              = task_slots;
  bnb_settings.num_best_first_workers                   = 1;
  bnb_settings.node_limit                               = node_limit;
  bnb_settings.print_presolve_stats                     = false;
  bnb_settings.preserve_advanced_basis_dimensions       = true;
  bnb_settings.primal_tol                               = settings.tolerances.absolute_tolerance;
  bnb_settings.dual_tol                                 = settings.tolerances.absolute_tolerance;
  bnb_settings.integer_tol                              = settings.tolerances.integrality_tolerance;
  bnb_settings.absolute_mip_gap_tol                     = settings.tolerances.absolute_mip_gap;
  bnb_settings.relative_mip_gap_tol                     = settings.tolerances.relative_mip_gap;
  bnb_settings.max_cut_passes                           = 0;
  bnb_settings.mir_cuts                                 = 0;
  bnb_settings.mixed_integer_gomory_cuts                = 0;
  bnb_settings.knapsack_cuts                            = 0;
  bnb_settings.flow_cover_cuts                          = 0;
  bnb_settings.implied_bound_cuts                       = 0;
  bnb_settings.clique_cuts                              = 0;
  bnb_settings.zero_half_cuts                           = 0;
  bnb_settings.strong_chvatal_gomory_cuts               = 0;
  bnb_settings.reduced_cost_strengthening               = 0;
  bnb_settings.reliability_branching                    = 0;
  bnb_settings.strong_branching_simplex_iteration_limit = 0;
  bnb_settings.mip_batch_pdlp_strong_branching          = 0;
  bnb_settings.mip_batch_pdlp_reliability_branching     = 0;
  bnb_settings.symmetry                                 = 0;
  bnb_settings.submip_settings.rins                     = 0;
  bnb_settings.submip_settings.rens                     = 0;
  bnb_settings.submip_settings.max_level                = 0;
  bnb_settings.submip_settings.enable_cpufj             = false;
  bnb_settings.diving_settings                          = settings.diving_params;
  bnb_settings.diving_settings.min_node_depth           = 0;
  bnb_settings.set_log(false);
  return bnb_settings;
}

class task_slot_release_t {
 public:
  task_slot_release_t(pre_presolve_thread_budget_t& budget, int slots)
    : budget_(budget), slots_(slots)
  {
  }
  ~task_slot_release_t()
  {
    try {
      budget_.release(slots_);
    } catch (const std::exception& e) {
      CUOPT_LOG_ERROR("Failed to release pre-presolve OpenMP task slot: %s", e.what());
    }
  }

 private:
  pre_presolve_thread_budget_t& budget_;
  int slots_;
};

}  // namespace

template <typename i_t, typename f_t>
struct pre_presolve_primal_t<i_t, f_t>::work_t {
  work_t(const optimization_problem_t<i_t, f_t>& op_problem,
         const mip_solver_settings_t<i_t, f_t>& settings,
         early_incumbent_callback_t<f_t> incumbent_callback,
         int task_slots)
    : user_problem(snapshot_original_problem(op_problem)),
      bnb_settings(make_settings(settings, task_slots))
  {
    const f_t objective_scale      = user_problem.obj_scale;
    const f_t objective_constant   = user_problem.obj_constant;
    bnb_settings.solution_callback = [this,
                                      incumbent_callback = std::move(incumbent_callback),
                                      objective_scale,
                                      objective_constant](std::vector<f_t>& assignment,
                                                          f_t solver_obj) {
      if (!incumbent_callback) { return; }
      const f_t user_obj = objective_scale * (solver_obj + objective_constant);
      incumbent_callback(solver_obj, user_obj, assignment, arm_name);
    };
  }

  void run(std::atomic<bool>& stop_requested)
  {
    if (stop_requested.load(std::memory_order_acquire)) { return; }

    probing_implied_bound_t<i_t, f_t> probing_implied_bound(user_problem.num_cols);
    branch_and_bound_t<i_t, f_t> branch_and_bound(
      user_problem, bnb_settings, tic(), probing_implied_bound);
    branch_and_bound.set_concurrent_lp_root_solve(false);
    branch_and_bound.set_submip_halt_callback(
      [&stop_requested](f_t, f_t) { return stop_requested.load(std::memory_order_acquire); });

    // A zero simplex-iteration limit selects an estimate path rather than disabling initial root
    // strong branching. Neutral zero-count pseudocosts skip that root block and learn at nodes.
    pseudo_costs_t<i_t, f_t> neutral_pseudocost(user_problem.num_cols, bnb_settings);
    std::vector<i_t> identity_reduced_to_original(user_problem.num_cols);
    std::iota(identity_reduced_to_original.begin(), identity_reduced_to_original.end(), i_t{0});
    branch_and_bound.set_initial_pseudocost(neutral_pseudocost, identity_reduced_to_original);

    simplex::mip_solution_t<i_t, f_t> solution(user_problem.num_cols);
    branch_and_bound.solve(solution);
  }

  simplex::user_problem_t<i_t, f_t> user_problem;
  simplex::simplex_solver_settings_t<i_t, f_t> bnb_settings;
};

template <typename i_t, typename f_t>
pre_presolve_primal_t<i_t, f_t>::pre_presolve_primal_t(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const mip_solver_settings_t<i_t, f_t>& settings,
  early_incumbent_callback_t<f_t> incumbent_callback,
  pre_presolve_thread_budget_t& thread_budget,
  int task_slots)
  : thread_budget_(thread_budget),
    task_state_(std::make_unique<task_state_t>()),
    task_slots_(task_slots)
{
  const i_t num_rows = op_problem.get_n_constraints();
  const i_t nnz      = op_problem.get_nnz();
  if (!is_eligible(num_rows, nnz)) { return; }

  work_ =
    std::make_unique<work_t>(op_problem, settings, std::move(incumbent_callback), task_slots_);
}

template <typename i_t, typename f_t>
pre_presolve_primal_t<i_t, f_t>::~pre_presolve_primal_t()
{
  stop_no_throw();
}

template <typename i_t, typename f_t>
bool pre_presolve_primal_t<i_t, f_t>::start()
{
  if (started_) { return true; }
  if (!work_) { return false; }
  if (!thread_budget_.try_reserve(task_slots_)) { return false; }

  started_            = true;
  auto* task_state    = task_state_.get();
  auto* thread_budget = &thread_budget_;
  auto* work          = work_.get();

#pragma omp task default(none) firstprivate(task_state, thread_budget, work) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY) depend(out : *task_state)
  {
    task_slot_release_t release(*thread_budget, work->bnb_settings.num_threads);
    try {
      work->run(task_state->stop_requested);
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
