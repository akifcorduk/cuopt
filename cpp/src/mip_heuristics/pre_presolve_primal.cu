/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "pre_presolve_primal.cuh"

#include <branch_and_bound/branch_and_bound.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/solve.hpp>
#include <math_optimization/tic_toc.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/utils.cuh>
#include <mip_heuristics/utils.hpp>
#include <utilities/logger.hpp>

#include <omp.h>

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
bool verify_pre_presolve_primal_solution(
  const simplex::user_problem_t<i_t, f_t>& problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  const std::vector<f_t>& assignment)
{
  if (assignment.size() != static_cast<size_t>(problem.num_cols)) return false;
  for (i_t col = 0; col < problem.num_cols; ++col) {
    const f_t value = assignment[col];
    if (!std::isfinite(value) || value < problem.lower[col] || value > problem.upper[col] ||
        (problem.var_types[col] == simplex::variable_type_t::BINARY &&
         (value < f_t{0} || value > f_t{1})) ||
        (problem.var_types[col] != simplex::variable_type_t::CONTINUOUS &&
         !is_integer(value, tolerances.integrality_tolerance)))
      return false;
  }
  const f_t objective =
    compensated_dot2(problem.objective.data(), assignment.data(), assignment.size());
  if (!std::isfinite(objective) ||
      !std::isfinite(problem.obj_scale * (objective + problem.obj_constant)))
    return false;

  // The snapshot is CSC. Accumulate each original row independently with the
  // same compensated summation used by private LNS candidate validation.
  std::vector<f_t> activity(problem.num_rows, f_t{0});
  std::vector<f_t> correction(problem.num_rows, f_t{0});
  for (i_t col = 0; col < problem.num_cols; ++col) {
    for (i_t pos = problem.A.col_start[col]; pos < problem.A.col_start[col + 1]; ++pos) {
      const i_t row   = problem.A.i[pos];
      const f_t term  = problem.A.x[pos] * assignment[col] - correction[row];
      const f_t next  = activity[row] + term;
      correction[row] = (next - activity[row]) - term;
      activity[row]   = next;
    }
  }
  std::vector<i_t> range_index(problem.num_rows, i_t{-1});
  for (size_t index = 0; index < problem.range_rows.size(); ++index)
    range_index[problem.range_rows[index]] = static_cast<i_t>(index);
  const f_t infinity = std::numeric_limits<f_t>::infinity();
  for (i_t row = 0; row < problem.num_rows; ++row) {
    const char sense = problem.row_sense[row];
    f_t lower = -infinity, upper = infinity;
    if (sense == 'E') {
      lower = upper = problem.rhs[row];
    } else if (sense == 'G') {
      lower = problem.rhs[row];
    } else if (sense == 'L') {
      upper = problem.rhs[row];
    } else {
      return false;
    }
    if (range_index[row] >= 0) {
      const auto bounds = simplex::get_range_bounds_from_sense(
        sense, problem.rhs[row], problem.range_value[range_index[row]]);
      lower = bounds.lower;
      upper = bounds.upper;
    }
    const f_t tolerance = get_cstr_tolerance<i_t, f_t>(
      lower, upper, tolerances.absolute_tolerance, tolerances.relative_tolerance);
    if (!std::isfinite(activity[row]) || activity[row] < lower - tolerance ||
        activity[row] > upper + tolerance)
      return false;
  }
  return true;
}

namespace {

constexpr const char* arm_name = "PRE-BNB-ROOT-DIVES";

template <typename i_t, typename f_t>
simplex::user_problem_t<i_t, f_t> snapshot_original_problem(const problem_t<i_t, f_t>& problem)
{
  simplex::user_problem_t<i_t, f_t> user_problem(problem.handle_ptr);
  problem.get_host_user_problem(user_problem);
  return user_problem;
}

template <typename i_t, typename f_t>
simplex::simplex_solver_settings_t<i_t, f_t> make_settings(
  const mip_solver_settings_t<i_t, f_t>& settings,
  std::function<bool(std::vector<f_t>&)> peer_incumbent_callback)
{
  simplex::simplex_solver_settings_t<i_t, f_t> bnb_settings;
  bnb_settings.iteration_limit                          = std::numeric_limits<i_t>::max();
  bnb_settings.node_limit                               = std::numeric_limits<i_t>::max();
  bnb_settings.time_limit                               = std::numeric_limits<f_t>::infinity();
  bnb_settings.work_limit                               = std::numeric_limits<f_t>::infinity();
  bnb_settings.branch_and_bound_simplex_iteration_limit = std::numeric_limits<int64_t>::max();
  bnb_settings.num_threads                              = pre_presolve_task_slots;
  bnb_settings.root_diving_only                         = true;
  bnb_settings.root_diving_peer_incumbent_callback      = std::move(peer_incumbent_callback);
  bnb_settings.print_presolve_stats                     = false;
  bnb_settings.preserve_advanced_basis_dimensions       = true;
  bnb_settings.primal_tol                               = settings.tolerances.absolute_tolerance;
  bnb_settings.dual_tol                                 = settings.tolerances.absolute_tolerance;
  bnb_settings.integer_tol                              = settings.tolerances.integrality_tolerance;
  bnb_settings.absolute_mip_gap_tol                     = settings.tolerances.absolute_mip_gap;
  bnb_settings.relative_mip_gap_tol                     = settings.tolerances.relative_mip_gap;
  bnb_settings.random_seed                              = settings.seed;

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

  bnb_settings.submip_settings.rins         = 0;
  bnb_settings.submip_settings.rens         = 0;
  bnb_settings.submip_settings.max_level    = 0;
  bnb_settings.submip_settings.enable_cpufj = false;

  bnb_settings.diving_settings                        = settings.diving_params;
  bnb_settings.diving_settings.line_search_diving     = 1;
  bnb_settings.diving_settings.pseudocost_diving      = 0;
  bnb_settings.diving_settings.guided_diving          = 1;
  bnb_settings.diving_settings.coefficient_diving     = 1;
  bnb_settings.diving_settings.farkas_diving          = 0;
  bnb_settings.diving_settings.vector_length_diving   = 1;
  bnb_settings.diving_settings.node_limit             = std::numeric_limits<i_t>::max();
  bnb_settings.diving_settings.iteration_limit_factor = std::numeric_limits<f_t>::infinity();
  bnb_settings.diving_settings.iteration_limit_offset = std::numeric_limits<int64_t>::max();

  bnb_settings.set_log(false);
  return bnb_settings;
}

}  // namespace

template <typename i_t, typename f_t>
struct pre_presolve_primal_t<i_t, f_t>::work_t {
  work_t(const problem_t<i_t, f_t>& problem,
         const mip_solver_settings_t<i_t, f_t>& settings,
         early_incumbent_callback_t<f_t> incumbent_callback,
         std::function<bool(std::vector<f_t>&)> peer_incumbent_callback)
    : user_problem(snapshot_original_problem(problem)),
      bnb_settings(make_settings(settings, std::move(peer_incumbent_callback)))
  {
    bnb_settings.root_diving_solution_validator =
      [this, tolerances = settings.tolerances](const std::vector<f_t>& assignment) {
        return verify_pre_presolve_primal_solution(user_problem, tolerances, assignment);
      };
    bnb_settings.solution_callback = [this, incumbent_callback = std::move(incumbent_callback)](
                                       std::vector<f_t>& assignment, f_t) {
      if (!incumbent_callback || !bnb_settings.root_diving_solution_validator(assignment)) {
        return;
      }
      const f_t solver_obj =
        compensated_dot2(user_problem.objective.data(), assignment.data(), assignment.size());
      const f_t user_obj = user_problem.obj_scale * (solver_obj + user_problem.obj_constant);
      try {
        incumbent_callback(solver_obj, user_obj, assignment, arm_name);
      } catch (const std::exception& e) {
        CUOPT_LOG_ERROR("Pre-presolve root-diving incumbent callback failed: %s", e.what());
      } catch (...) {
        CUOPT_LOG_ERROR("Pre-presolve root-diving incumbent callback failed");
      }
    };
  }

  void run(task_state_t& task_state)
  {
    if (task_state.cancel_requested.load(std::memory_order_acquire) != 0) { return; }

    probing_implied_bound_t<i_t, f_t> probing_implied_bound(user_problem.num_cols);
    branch_and_bound_t<i_t, f_t> branch_and_bound(
      user_problem, bnb_settings, tic(), probing_implied_bound);
    branch_and_bound.set_concurrent_lp_root_solve(false);
    branch_and_bound.set_external_halt(
      &task_state.cancel_requested, &task_state.root_halt, &task_state.node_halt);

    // Share an incumbent that arrived before the auxiliary B&B existed. The callback and the
    // copy held by B&B share their generation cursor, so each mailbox generation is consumed once.
    if (bnb_settings.root_diving_peer_incumbent_callback) {
      std::vector<f_t> peer_incumbent;
      if (bnb_settings.root_diving_peer_incumbent_callback(peer_incumbent) &&
          !peer_incumbent.empty()) {
        branch_and_bound.set_solution_from_heuristics(peer_incumbent,
                                                      heuristics_origin_t::HEURISTICS);
      }
    }

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
  const problem_t<i_t, f_t>& problem,
  const mip_solver_settings_t<i_t, f_t>& settings,
  early_incumbent_callback_t<f_t> aux_callback)
  : task_state_(std::make_unique<task_state_t>()),
    objective_sense_(problem.maximize ? f_t{-1} : f_t{1})
{
  const bool locally_eligible =
    is_pre_presolve_primal_eligible(problem.n_constraints,
                                    problem.n_variables,
                                    problem.nnz,
                                    !problem.Q_values.empty(),
                                    false,
                                    problem.n_integer_vars > 0,
                                    settings.presolver != presolver_t::None,
                                    settings.determinism_mode == CUOPT_MODE_OPPORTUNISTIC,
                                    omp_get_num_threads());
  if (!locally_eligible) { return; }

  auto last_generation = std::make_shared<uint64_t>(0);
  auto peer_incumbent_callback =
    [this, last_generation = std::move(last_generation)](std::vector<f_t>& assignment) {
      return poll_external_incumbent(*last_generation, assignment);
    };

  work_ = std::make_unique<work_t>(
    problem, settings, std::move(aux_callback), std::move(peer_incumbent_callback));
}

template <typename i_t, typename f_t>
pre_presolve_primal_t<i_t, f_t>::~pre_presolve_primal_t()
{
  stop();
}

template <typename i_t, typename f_t>
bool pre_presolve_primal_t<i_t, f_t>::start()
{
  if (started_) { return true; }
  if (!work_ || task_state_->cancel_requested.load(std::memory_order_acquire) != 0) {
    return false;
  }

  CUOPT_LOG_INFO("PRE_PRESOLVE_START name=refinement-w3-restricted workers=%d cooperative=1",
                 pre_presolve_diving_workers);

  started_         = true;
  auto* task_state = task_state_.get();
  auto* work       = work_.get();

#pragma omp task default(none) firstprivate(task_state, work) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY) depend(out : *task_state)
  {
    try {
      work->run(*task_state);
    } catch (...) {
      task_state->exception = std::current_exception();
    }
  }
  return true;
}

template <typename i_t, typename f_t>
void pre_presolve_primal_t<i_t, f_t>::stop() noexcept
{
  if (!started_) { return; }

  task_state_->cancel_requested.store(1, std::memory_order_release);
  task_state_->root_halt.store(1, std::memory_order_release);
  task_state_->node_halt.store(1, std::memory_order_release);
  auto* task_state = task_state_.get();
#pragma omp taskwait depend(in : *task_state)
  started_ = false;

  if (task_state_->exception) {
    try {
      std::rethrow_exception(task_state_->exception);
    } catch (const std::exception& e) {
      CUOPT_LOG_ERROR("Pre-presolve root-diving task failed: %s", e.what());
    } catch (...) {
      CUOPT_LOG_ERROR("Pre-presolve root-diving task failed with an unknown exception");
    }
    task_state_->exception = nullptr;
  }
}

template <typename i_t, typename f_t>
void pre_presolve_primal_t<i_t, f_t>::offer_external_incumbent(
  f_t user_obj, const std::vector<f_t>& assignment) noexcept
{
  try {
    if (!work_ || task_state_->cancel_requested.load(std::memory_order_acquire) != 0 ||
        !std::isfinite(user_obj) ||
        assignment.size() != static_cast<std::size_t>(work_->user_problem.num_cols)) {
      return;
    }

    if (!work_->bnb_settings.root_diving_solution_validator(assignment)) { return; }
    const f_t solver_obj =
      compensated_dot2(work_->user_problem.objective.data(), assignment.data(), assignment.size());
    const f_t verified_user_obj =
      work_->user_problem.obj_scale * (solver_obj + work_->user_problem.obj_constant);
    const f_t score = objective_sense_ * verified_user_obj;
    if (!std::isfinite(score)) { return; }

    std::lock_guard<std::mutex> lock(mailbox_mutex_);
    if (score >= mailbox_best_score_) { return; }
    mailbox_assignment_ = assignment;
    mailbox_best_score_ = score;
    ++mailbox_generation_;
  } catch (const std::exception& error) {
    task_state_->cancel_requested.store(1, std::memory_order_release);
    task_state_->root_halt.store(1, std::memory_order_release);
    task_state_->node_halt.store(1, std::memory_order_release);
    CUOPT_LOG_ERROR("Pre-presolve root-diving mailbox failed: %s", error.what());
  } catch (...) {
    task_state_->cancel_requested.store(1, std::memory_order_release);
    task_state_->root_halt.store(1, std::memory_order_release);
    task_state_->node_halt.store(1, std::memory_order_release);
    CUOPT_LOG_ERROR("Pre-presolve root-diving mailbox failed with an unknown exception");
  }
}

template <typename i_t, typename f_t>
bool pre_presolve_primal_t<i_t, f_t>::poll_external_incumbent(uint64_t& last_generation,
                                                              std::vector<f_t>& assignment)
{
  std::lock_guard<std::mutex> lock(mailbox_mutex_);
  if (mailbox_generation_ == 0 || mailbox_generation_ <= last_generation) { return false; }
  assignment      = mailbox_assignment_;
  last_generation = mailbox_generation_;
  return true;
}

#if MIP_INSTANTIATE_FLOAT
template bool verify_pre_presolve_primal_solution<int, float>(
  const simplex::user_problem_t<int, float>&,
  const mip_solver_settings_t<int, float>::tolerances_t&,
  const std::vector<float>&);
template class pre_presolve_primal_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template bool verify_pre_presolve_primal_solution<int, double>(
  const simplex::user_problem_t<int, double>&,
  const mip_solver_settings_t<int, double>::tolerances_t&,
  const std::vector<double>&);
template class pre_presolve_primal_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
