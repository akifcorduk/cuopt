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

constexpr int pre_bnb_node_limit       = 64;
constexpr int pre_bnb_num_threads      = 1;
constexpr double pre_bnb_time_limit    = 0.15;
constexpr const char* pre_bnb_arm_name = "PRE-BNB-BASIS-PRESERVE-SELECTIVE-BEST";

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
simplex::simplex_solver_settings_t<i_t, f_t> make_pre_bnb_settings(
  const mip_solver_settings_t<i_t, f_t>& settings, i_t num_rows, i_t nnz)
{
  simplex::simplex_solver_settings_t<i_t, f_t> bnb_settings;
  bnb_settings.time_limit                         = static_cast<f_t>(pre_bnb_time_limit);
  bnb_settings.num_threads                        = pre_bnb_num_threads;
  bnb_settings.print_presolve_stats               = false;
  bnb_settings.preserve_advanced_basis_dimensions = true;
  bnb_settings.primal_tol                         = settings.tolerances.absolute_tolerance;
  bnb_settings.dual_tol                           = settings.tolerances.absolute_tolerance;
  bnb_settings.integer_tol                        = settings.tolerances.integrality_tolerance;
  bnb_settings.absolute_mip_gap_tol               = settings.tolerances.absolute_mip_gap;
  bnb_settings.relative_mip_gap_tol               = settings.tolerances.relative_mip_gap;
  bnb_settings.set_log(false);

  bnb_settings.node_limit                               = pre_bnb_node_limit;
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
  bnb_settings.submip_settings.max_level                = 0;
  bnb_settings.submip_settings.enable_cpufj             = false;
  bnb_settings.diving_settings                          = settings.diving_params;
  bnb_settings.diving_settings.min_node_depth           = 0;
  CUOPT_LOG_INFO(
    "PRE_BNB_CONFIG variant=basis_preserve_selective_best task_slots=%d bnb_threads=%d "
    "rows=%d nnz=%d small_nnz_limit=%d low_row_limit=%d max_nnz_limit=%d "
    "time_limit=%.3fs node_limit=%d sb_simplex_iteration_limit=%d initial_root_sb=skipped "
    "initial_pseudocost=neutral diving=retained preserve_advanced_basis_dimensions=%d "
    "publication=best",
    pre_presolve_primal_task_slots,
    pre_bnb_num_threads,
    num_rows,
    nnz,
    pre_presolve_primal_small_nnz_limit,
    pre_presolve_primal_low_row_limit,
    pre_presolve_primal_max_nnz_limit,
    pre_bnb_time_limit,
    pre_bnb_node_limit,
    bnb_settings.strong_branching_simplex_iteration_limit,
    static_cast<int>(bnb_settings.preserve_advanced_basis_dimensions));
  return bnb_settings;
}

template <typename i_t>
void log_pre_bnb_summary(const char* status,
                         double elapsed,
                         i_t nodes,
                         std::size_t generated,
                         bool published,
                         bool stop_requested)
{
  CUOPT_LOG_INFO(
    "PRE_BNB_SUMMARY eligible=1 status=%s elapsed=%.3fs nodes=%d candidates_generated=%zu "
    "candidates_published=%d stop_requested=%d",
    status,
    elapsed,
    nodes,
    generated,
    static_cast<int>(published),
    static_cast<int>(stop_requested));
}

template <typename i_t, typename f_t>
void run_pre_bnb_best(simplex::user_problem_t<i_t, f_t>& user_problem,
                      simplex::simplex_solver_settings_t<i_t, f_t>& bnb_settings,
                      pre_presolve_best_candidate_store_t<f_t>& candidate_store,
                      early_incumbent_callback_t<f_t>& incumbent_callback,
                      f_t objective_scale,
                      f_t objective_constant,
                      std::atomic<bool>& stop_requested)
{
  if (stop_requested.load(std::memory_order_acquire)) {
    log_pre_bnb_summary("NOT_RUN", 0.0, i_t{0}, 0, false, true);
    return;
  }

  const double solve_start = tic();
  try {
    probing_implied_bound_t<i_t, f_t> probing_implied_bound(user_problem.num_cols);
    branch_and_bound_t<i_t, f_t> branch_and_bound(
      user_problem, bnb_settings, tic(), probing_implied_bound);

    // An iteration limit of zero selects the single-pivot strong-branching estimate; it does not
    // disable root pseudocost initialization. Zero-count initial pseudocosts mark initialization
    // complete without biasing either direction, keeping this bounded arm off that estimate path.
    pseudo_costs_t<i_t, f_t> neutral_pseudocosts(user_problem.num_cols, bnb_settings);
    std::vector<i_t> identity_reduced_to_original(user_problem.num_cols);
    std::iota(identity_reduced_to_original.begin(), identity_reduced_to_original.end(), i_t{0});
    branch_and_bound.set_initial_pseudocost(neutral_pseudocosts, identity_reduced_to_original);

    branch_and_bound.set_concurrent_lp_root_solve(false);
    branch_and_bound.set_submip_halt_callback(
      [&stop_requested](f_t, f_t) { return stop_requested.load(std::memory_order_acquire); });

    simplex::mip_solution_t<i_t, f_t> solution(user_problem.num_cols);
    const auto status             = branch_and_bound.solve(solution);
    const auto retained_candidate = candidate_store.snapshot();
    bool published                = false;
    if (retained_candidate.has_candidate && incumbent_callback) {
      const f_t user_objective =
        objective_scale * (retained_candidate.solver_objective + objective_constant);
      incumbent_callback(retained_candidate.solver_objective,
                         user_objective,
                         retained_candidate.assignment,
                         pre_bnb_arm_name);
      published = true;
    }

    const auto status_name = mip_status_to_string(status);
    log_pre_bnb_summary(status_name.c_str(),
                        toc(solve_start),
                        solution.nodes_explored,
                        retained_candidate.generated,
                        published,
                        stop_requested.load(std::memory_order_acquire));
  } catch (...) {
    const auto retained_candidate = candidate_store.snapshot();
    log_pre_bnb_summary("EXCEPTION",
                        toc(solve_start),
                        i_t{0},
                        retained_candidate.generated,
                        false,
                        stop_requested.load(std::memory_order_acquire));
    throw;
  }
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
  if (mode_ != 2) { return; }

  const i_t num_rows = op_problem.get_n_constraints();
  const i_t nnz      = op_problem.get_nnz();
  if (!pre_presolve_primal_is_selectively_eligible(num_rows, nnz)) {
    CUOPT_LOG_INFO(
      "PRE_BNB_SUMMARY eligible=0 reason=rows_nnz_gate rows=%d nnz=%d "
      "small_nnz_limit=%d low_row_limit=%d max_nnz_limit=%d",
      num_rows,
      nnz,
      pre_presolve_primal_small_nnz_limit,
      pre_presolve_primal_low_row_limit,
      pre_presolve_primal_max_nnz_limit);
    return;
  }

  // This host snapshot is complete before the task is launched. PaPILO may subsequently mutate
  // op_problem without racing the CPU B&B arm.
  auto user_problem =
    std::make_shared<simplex::user_problem_t<i_t, f_t>>(snapshot_original_problem(op_problem));
  auto candidate_store           = std::make_shared<pre_presolve_best_candidate_store_t<f_t>>();
  auto bnb_settings              = make_pre_bnb_settings<i_t, f_t>(settings, num_rows, nnz);
  bnb_settings.solution_callback = [candidate_store](std::vector<f_t>& assignment,
                                                     f_t solver_objective) {
    candidate_store->consider(solver_objective, assignment);
  };

  const f_t objective_scale    = user_problem->obj_scale;
  const f_t objective_constant = user_problem->obj_constant;

  auto run_work = [user_problem,
                   candidate_store,
                   bnb_settings,
                   incumbent_callback = std::move(incumbent_callback),
                   objective_scale,
                   objective_constant](std::atomic<bool>& stop_requested) mutable {
    run_pre_bnb_best(*user_problem,
                     bnb_settings,
                     *candidate_store,
                     incumbent_callback,
                     objective_scale,
                     objective_constant,
                     stop_requested);
  };
  work_ = std::move(run_work);
}

template <typename i_t, typename f_t>
bool pre_presolve_primal_t<i_t, f_t>::start()
{
  if (mode_ == 0 || started_) { return started_; }
  if (!work_) {
    if (mode_ == 2) { return false; }
    CUOPT_LOG_WARN("Pre-presolve primal mode %d is not implemented on this branch", mode_);
    return false;
  }

  if (!thread_budget_.try_reserve(pre_presolve_primal_task_slots)) {
    CUOPT_LOG_INFO("PRE_BNB_SUMMARY eligible=1 status=NOT_STARTED reason=omp_budget task_slots=%d",
                   pre_presolve_primal_task_slots);
    return false;
  }

  started_            = true;
  auto* task_state    = task_state_.get();
  auto* thread_budget = &thread_budget_;
  auto* work          = &work_;

#pragma omp task default(none) firstprivate(task_state, thread_budget, work) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY) depend(out : *task_state)
  {
    task_slot_release_t release(thread_budget, pre_presolve_primal_task_slots);
    try {
      (*work)(task_state->stop_requested);
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
