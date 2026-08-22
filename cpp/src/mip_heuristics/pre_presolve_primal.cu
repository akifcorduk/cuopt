/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "pre_presolve_primal.cuh"

#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/local_search/rounding/constraint_prop.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/solver_context.cuh>
#include <mip_heuristics/utils.cuh>
#include <pdlp/pdlp.cuh>
#include <pdlp/solve.cuh>
#include <utilities/logger.hpp>
#include <utilities/scope_guard.hpp>
#include <utilities/timer.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cublas_macros.hpp>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/error.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>

#include <thrust/fill.h>

#include <algorithm>
#include <limits>
#include <mutex>
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

constexpr int pdlp_cp_gpufj_mode               = 4;
constexpr double pdlp_time_limit               = 0.05;
constexpr int pdlp_iteration_limit             = 1000;
constexpr double pdlp_optimality_tolerance     = 1e-2;
constexpr double constraint_prop_time_limit    = 0.02;
constexpr double constraint_prop_lp_time_limit = 0.0;
constexpr double fj_time_limit                 = 0.08;

void set_cuda_device(int device_id) { RAFT_CUDA_TRY(cudaSetDevice(device_id)); }

void initialize_private_handle(const raft::handle_t& handle)
{
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle.get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle.get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle.get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle.get_stream()));
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

template <typename i_t, typename f_t>
class pdlp_cp_gpufj_state_t {
 public:
  pdlp_cp_gpufj_state_t(const optimization_problem_t<i_t, f_t>& op_problem,
                        const mip_solver_settings_t<i_t, f_t>& settings,
                        early_incumbent_callback_t<f_t> incumbent_callback)
    : op_problem_snapshot_(op_problem),
      settings_(settings),
      incumbent_callback_(std::move(incumbent_callback))
  {
    timer_t setup_timer(std::numeric_limits<double>::max());
    RAFT_CUDA_TRY(cudaGetDevice(&device_id_));

    // Snapshot the immutable original input before PaPILO starts. The worker deep-copies these
    // snapshots onto a RAFT handle created and destroyed on the worker host thread.
    op_problem_snapshot_.get_handle_ptr()->sync_stream();
    original_problem_snapshot_ = std::make_unique<problem_t<i_t, f_t>>(
      op_problem_snapshot_, settings_.get_tolerances(), false);
    original_problem_snapshot_->handle_ptr->sync_stream();
    problem_snapshot_ = std::make_unique<problem_t<i_t, f_t>>(*original_problem_snapshot_);
    problem_snapshot_->preprocess_problem();
    problem_snapshot_->handle_ptr->sync_stream();
    stats_.snapshot_setup_seconds = setup_timer.elapsed_time();
  }

  void run(std::atomic<bool>& stop_requested)
  {
    auto summary_guard = cuopt::scope_guard([this]() { log_summary(); });
    stats_.ran         = true;
    set_cuda_device(device_id_);
    if (should_stop(stop_requested)) { return; }

    std::unique_ptr<worker_resources_t> worker;
    {
      timer_t worker_setup_timer(std::numeric_limits<double>::max());
      auto worker_setup_guard = cuopt::scope_guard([this, &worker_setup_timer]() {
        stats_.worker_setup_seconds = worker_setup_timer.elapsed_time();
      });
      worker                  = std::make_unique<worker_resources_t>(
        *original_problem_snapshot_, *problem_snapshot_, settings_);
    }

    publish_context(worker->context_ptr_.get());
    auto context_guard = cuopt::scope_guard(
      [this, context = worker->context_ptr_.get()]() { clear_context(context); });
    if (should_stop(stop_requested)) { return; }

    if (!run_pdlp(*worker, stop_requested)) { return; }
    if (should_stop(stop_requested)) { return; }

    run_constraint_prop(*worker);
    if (should_stop(stop_requested)) { return; }

    run_fj(*worker, stop_requested);
  }

  void request_stop()
  {
    stop_signalled_.store(true, std::memory_order_release);
    pdlp_concurrent_halt_.store(1, std::memory_order_release);
    std::lock_guard<std::mutex> lock(active_context_mutex_);
    if (active_context_ != nullptr) {
      active_context_->preempt_heuristic_solver_.store(true, std::memory_order_release);
    }
  }

 private:
  struct stats_t {
    double snapshot_setup_seconds{0.0};
    double worker_setup_seconds{0.0};
    double pdlp_seconds{0.0};
    double constraint_prop_seconds{0.0};
    double fj_seconds{0.0};
    int pdlp_status{-1};
    int incumbents_reported{0};
    bool ran{false};
    bool pdlp_ran{false};
    bool pdlp_usable{false};
    bool constraint_prop_ran{false};
    bool constraint_prop_feasible{false};
    bool fj_ran{false};
    bool fj_feasible{false};
  };

  struct worker_resources_t {
    worker_resources_t(const problem_t<i_t, f_t>& original_problem_snapshot,
                       const problem_t<i_t, f_t>& problem_snapshot,
                       const mip_solver_settings_t<i_t, f_t>& settings)
    {
      // Fresh RAFT handles default to host pointer mode, while PDLP passes device alpha/beta
      // scalars into cuBLAS/cuSPARSE.
      initialize_private_handle(handle_);
      original_problem_ptr_ =
        std::make_unique<problem_t<i_t, f_t>>(original_problem_snapshot, &handle_);
      original_audited_solution_ptr_ =
        std::make_unique<solution_t<i_t, f_t>>(*original_problem_ptr_);

      problem_ptr_  = std::make_unique<problem_t<i_t, f_t>>(problem_snapshot, &handle_);
      solution_ptr_ = std::make_unique<solution_t<i_t, f_t>>(*problem_ptr_);
      thrust::fill(handle_.get_thrust_policy(),
                   solution_ptr_->assignment.begin(),
                   solution_ptr_->assignment.end(),
                   f_t{0});
      solution_ptr_->clamp_within_bounds();
      context_ptr_ =
        std::make_unique<mip_solver_context_t<i_t, f_t>>(&handle_, problem_ptr_.get(), settings);
      constraint_prop_ptr_ = std::make_unique<constraint_prop_t<i_t, f_t>>(*context_ptr_);
      constraint_prop_ptr_->max_time_for_bounds_prop = constraint_prop_time_limit;
      constraint_prop_ptr_->bounds_update.resize(*problem_ptr_);
      audited_solution_ptr_ = std::make_unique<solution_t<i_t, f_t>>(*problem_ptr_);
      handle_.sync_stream();
    }

    // The handle is constructed first and destroyed last, all on the OpenMP worker host thread.
    raft::handle_t handle_;
    std::unique_ptr<problem_t<i_t, f_t>> original_problem_ptr_;
    std::unique_ptr<solution_t<i_t, f_t>> original_audited_solution_ptr_;
    std::unique_ptr<problem_t<i_t, f_t>> problem_ptr_;
    std::unique_ptr<solution_t<i_t, f_t>> solution_ptr_;
    std::unique_ptr<mip_solver_context_t<i_t, f_t>> context_ptr_;
    std::unique_ptr<constraint_prop_t<i_t, f_t>> constraint_prop_ptr_;
    std::unique_ptr<solution_t<i_t, f_t>> audited_solution_ptr_;
  };

  void log_summary() const
  {
    CUOPT_LOG_INFO(
      "Pre-presolve PDLP+CP+GPUFJ summary: ran=%d stopped=%d snapshot_s=%.6f worker_setup_s=%.6f "
      "pdlp_ran=%d pdlp_s=%.6f pdlp_status=%d pdlp_usable=%d cp_ran=%d cp_s=%.6f "
      "cp_feasible=%d fj_ran=%d fj_s=%.6f fj_feasible=%d incumbents=%d",
      static_cast<int>(stats_.ran),
      static_cast<int>(stop_signalled_.load(std::memory_order_acquire)),
      stats_.snapshot_setup_seconds,
      stats_.worker_setup_seconds,
      static_cast<int>(stats_.pdlp_ran),
      stats_.pdlp_seconds,
      stats_.pdlp_status,
      static_cast<int>(stats_.pdlp_usable),
      static_cast<int>(stats_.constraint_prop_ran),
      stats_.constraint_prop_seconds,
      static_cast<int>(stats_.constraint_prop_feasible),
      static_cast<int>(stats_.fj_ran),
      stats_.fj_seconds,
      static_cast<int>(stats_.fj_feasible),
      stats_.incumbents_reported);
  }

  void publish_context(mip_solver_context_t<i_t, f_t>* context)
  {
    std::lock_guard<std::mutex> lock(active_context_mutex_);
    active_context_ = context;
    if (stop_signalled_.load(std::memory_order_acquire)) {
      active_context_->preempt_heuristic_solver_.store(true, std::memory_order_release);
    }
  }

  void clear_context(mip_solver_context_t<i_t, f_t>* context)
  {
    std::lock_guard<std::mutex> lock(active_context_mutex_);
    if (active_context_ == context) { active_context_ = nullptr; }
  }

  bool should_stop(const std::atomic<bool>& stop_requested)
  {
    if (!stop_requested.load(std::memory_order_acquire)) { return false; }
    request_stop();
    return true;
  }

  pdlp_solver_settings_t<i_t, f_t> make_pdlp_settings()
  {
    pdlp_solver_settings_t<i_t, f_t> settings{};
    settings.set_optimality_tolerance(static_cast<f_t>(pdlp_optimality_tolerance));
    settings.detect_infeasibility    = false;
    settings.iteration_limit         = pdlp_iteration_limit;
    settings.time_limit              = static_cast<f_t>(pdlp_time_limit);
    settings.log_to_console          = false;
    settings.per_constraint_residual = true;
    settings.first_primal_feasible   = false;
    settings.pdlp_solver_mode        = pdlp_solver_mode_t::Stable2;
    settings.presolver               = presolver_t::None;
    settings.method                  = method_t::PDLP;
    settings.inside_mip              = true;
    settings.concurrent_halt         = &pdlp_concurrent_halt_;
    set_pdlp_solver_mode(settings);
    return settings;
  }

  bool run_pdlp(worker_resources_t& worker, std::atomic<bool>& stop_requested)
  {
    timer_t phase_timer(std::numeric_limits<double>::max());
    auto phase_guard = cuopt::scope_guard(
      [this, &phase_timer]() { stats_.pdlp_seconds = phase_timer.elapsed_time(); });
    stats_.pdlp_ran = true;
    if (should_stop(stop_requested)) { return false; }

    timer_t timer(pdlp_time_limit);
    auto settings = make_pdlp_settings();
    pdlp::pdlp_solver_t<i_t, f_t> solver(*worker.problem_ptr_, settings);
    if (should_stop(stop_requested)) { return false; }

    solver.set_inside_mip(true);
    auto response      = solver.run_solver(timer);
    stats_.pdlp_status = static_cast<int>(response.get_termination_status());
    if (should_stop(stop_requested)) { return false; }

    const auto& primal = response.get_primal_solution();
    const bool usable =
      response.get_termination_status() != pdlp_termination_status_t::NumericalError &&
      primal.size() == worker.solution_ptr_->assignment.size();
    if (usable) {
      worker.solution_ptr_->copy_new_assignment(primal);
      worker.solution_ptr_->clamp_within_bounds();
      stats_.pdlp_usable = true;
    } else {
      CUOPT_LOG_DEBUG("Pre-presolve PDLP returned no usable CP point (status %d, size %zu)",
                      static_cast<int>(response.get_termination_status()),
                      primal.size());
    }
    return usable;
  }

  void run_constraint_prop(worker_resources_t& worker)
  {
    timer_t phase_timer(std::numeric_limits<double>::max());
    auto phase_guard = cuopt::scope_guard(
      [this, &phase_timer]() { stats_.constraint_prop_seconds = phase_timer.elapsed_time(); });
    stats_.constraint_prop_ran = true;

    timer_t constraint_prop_timer(constraint_prop_time_limit);
    stats_.constraint_prop_feasible =
      worker.constraint_prop_ptr_->apply_round(*worker.solution_ptr_,
                                               static_cast<f_t>(constraint_prop_lp_time_limit),
                                               constraint_prop_timer);
  }

  fj_settings_t make_fj_settings() const
  {
    fj_settings_t settings;
    settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
    settings.time_limit             = fj_time_limit;
    settings.iteration_limit        = std::numeric_limits<int>::max();
    settings.n_of_minimums_for_exit = std::numeric_limits<int>::max();
    settings.update_weights         = true;
    settings.feasibility_run        = false;
    return settings;
  }

  void run_fj(worker_resources_t& worker, std::atomic<bool>& stop_requested)
  {
    timer_t phase_timer(std::numeric_limits<double>::max());
    auto phase_guard = cuopt::scope_guard(
      [this, &phase_timer]() { stats_.fj_seconds = phase_timer.elapsed_time(); });
    stats_.fj_ran = true;
    if (should_stop(stop_requested)) { return; }

    fj_t<i_t, f_t> fj(*worker.context_ptr_, make_fj_settings());
    fj.improvement_callback = [this, &worker](f_t, const std::vector<f_t>& assignment) {
      report_feasible_improvement(worker, assignment);
    };

    if (should_stop(stop_requested)) { return; }
    stats_.fj_feasible = fj.solve(*worker.solution_ptr_);
    if (!should_stop(stop_requested) && stats_.fj_feasible) { report_current_solution(worker); }
  }

  void report_current_solution(worker_resources_t& worker)
  {
    report_feasible_improvement(worker, worker.solution_ptr_->get_host_assignment());
  }

  void report_feasible_improvement(worker_resources_t& worker, const std::vector<f_t>& assignment)
  {
    if (stop_signalled_.load(std::memory_order_acquire) ||
        assignment.size() != worker.audited_solution_ptr_->assignment.size()) {
      return;
    }

    // The FJ callback is synchronous after its stream sync. Audit the exact callback assignment on
    // that same private stream while FJ is paused, rather than trusting mutable FJ state.
    worker.audited_solution_ptr_->copy_new_assignment(assignment);
    if (has_variable_bounds_violation(
          &worker.handle_, worker.audited_solution_ptr_->assignment, worker.problem_ptr_.get()) ||
        !worker.audited_solution_ptr_->compute_feasibility(true)) {
      return;
    }

    if (worker.audited_solution_ptr_->get_objective() >= best_solver_objective_) { return; }

    auto stream = worker.handle_.get_stream();
    rmm::device_uvector<f_t> postprocessed_assignment(assignment.size(), stream);
    raft::copy(postprocessed_assignment.data(),
               worker.audited_solution_ptr_->assignment.data(),
               assignment.size(),
               stream);
    worker.problem_ptr_->post_process_assignment(postprocessed_assignment, true, stream);
    auto original_assignment = cuopt::host_copy(postprocessed_assignment, stream);

    // Re-audit after postprocessing against a private, unpreprocessed copy of the user's model.
    // This catches numerical drift while removing standardized free-variable auxiliaries.
    if (original_assignment.size() != worker.original_audited_solution_ptr_->assignment.size()) {
      return;
    }
    worker.original_audited_solution_ptr_->copy_new_assignment(original_assignment);
    if (has_variable_bounds_violation(&worker.handle_,
                                      worker.original_audited_solution_ptr_->assignment,
                                      worker.original_problem_ptr_.get()) ||
        !worker.original_audited_solution_ptr_->compute_feasibility(true)) {
      return;
    }

    const f_t solver_obj = worker.original_audited_solution_ptr_->get_objective();
    if (solver_obj >= best_solver_objective_) { return; }
    const f_t user_obj     = worker.original_audited_solution_ptr_->get_user_objective();
    best_solver_objective_ = solver_obj;
    if (incumbent_callback_) {
      incumbent_callback_(solver_obj, user_obj, original_assignment, "PDLP+CP+GPUFJ");
      ++stats_.incumbents_reported;
    }
  }

  int device_id_{0};
  optimization_problem_t<i_t, f_t> op_problem_snapshot_;
  mip_solver_settings_t<i_t, f_t> settings_;
  std::unique_ptr<problem_t<i_t, f_t>> original_problem_snapshot_;
  std::unique_ptr<problem_t<i_t, f_t>> problem_snapshot_;
  early_incumbent_callback_t<f_t> incumbent_callback_;
  f_t best_solver_objective_{std::numeric_limits<f_t>::infinity()};
  stats_t stats_;
  std::mutex active_context_mutex_;
  mip_solver_context_t<i_t, f_t>* active_context_{nullptr};
  std::atomic<bool> stop_signalled_{false};
  std::atomic<int> pdlp_concurrent_halt_{0};
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
  if (mode_ != pdlp_cp_gpufj_mode) { return; }

  auto state = std::make_shared<pdlp_cp_gpufj_state_t<i_t, f_t>>(
    op_problem, settings, std::move(incumbent_callback));
  work_         = [state](std::atomic<bool>& stop_requested) { state->run(stop_requested); };
  request_stop_ = [state]() { state->request_stop(); };
}

template <typename i_t, typename f_t>
bool pre_presolve_primal_t<i_t, f_t>::start()
{
  if (mode_ == 0 || started_) { return started_; }
  if (!work_) {
    CUOPT_LOG_WARN("Pre-presolve primal mode %d is not implemented on this branch", mode_);
    return false;
  }

  constexpr int task_slots = 1;
  if (!thread_budget_.try_reserve(task_slots)) {
    if (mode_ == pdlp_cp_gpufj_mode) {
      CUOPT_LOG_INFO("Pre-presolve PDLP+CP+GPUFJ summary: ran=0 stopped=0 reason=no_omp_task_slot");
    }
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
  if (request_stop_) { request_stop_(); }
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
