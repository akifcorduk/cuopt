/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "pre_presolve_primal.cuh"

#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/solver_context.cuh>
#include <pdlp/pdlp.cuh>
#include <pdlp/solve.cuh>
#include <utilities/logger.hpp>
#include <utilities/scope_guard.hpp>
#include <utilities/timer.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/error.hpp>

#include <thrust/fill.h>

#include <algorithm>
#include <chrono>
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

constexpr int pdlp_gpufj_mode                   = 3;
constexpr double pdlp_time_limit                = 0.05;
constexpr int pdlp_iteration_limit              = 1000;
constexpr double pdlp_optimality_tolerance      = 1e-2;
constexpr double fj_total_time_limit            = 0.10;
constexpr double fj_rounding_time_limit         = 0.02;
constexpr double fj_rounding_second_stage_split = 0.10;

void set_cuda_device(int device_id) { RAFT_CUDA_TRY(cudaSetDevice(device_id)); }

void initialize_private_gpu_handle(const raft::handle_t& handle)
{
  // A fresh RAFT handle defaults to host pointer mode. PDLP passes device-resident alpha/beta
  // scalars to cuBLAS/cuSPARSE, so leaving the defaults makes cuSPARSE host-dereference a device
  // address during SpMM setup.
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
class pdlp_gpufj_state_t {
 public:
  pdlp_gpufj_state_t(const optimization_problem_t<i_t, f_t>& op_problem,
                     const mip_solver_settings_t<i_t, f_t>& settings,
                     early_incumbent_callback_t<f_t> incumbent_callback)
    : settings_(settings), incumbent_callback_(std::move(incumbent_callback))
  {
    RAFT_CUDA_TRY(cudaGetDevice(&device_id_));

    // Finish all work on the caller's handle before PaPILO starts. The worker deep-copies this
    // immutable snapshot onto a RAFT handle constructed on the worker host thread.
    snapshot_problem_ =
      std::make_unique<problem_t<i_t, f_t>>(op_problem, settings.get_tolerances(), false);
    snapshot_problem_->preprocess_problem();
    snapshot_problem_->handle_ptr->sync_stream();
  }

  void run(std::atomic<bool>& stop_requested)
  {
    const auto arm_start = std::chrono::steady_clock::now();
    set_cuda_device(device_id_);

    // cuSPARSE resources inside raft::handle_t are host-thread sensitive. Construct the complete
    // private GPU object graph here instead of constructing it on the masked thread and handing it
    // to this OpenMP worker.
    task_resources_t resources(*snapshot_problem_, settings_);
    {
      std::lock_guard<std::mutex> lock(active_context_mutex_);
      active_context_ = resources.context.get();
      if (stop_requested.load(std::memory_order_acquire)) {
        active_context_->preempt_heuristic_solver_.store(true, std::memory_order_release);
      }
    }
    cuopt::scope_guard clear_active_context([this]() {
      std::lock_guard<std::mutex> lock(active_context_mutex_);
      active_context_ = nullptr;
    });

    if (!should_stop(stop_requested)) {
      run_pdlp(stop_requested, resources);
      if (!should_stop(stop_requested)) { run_fj(stop_requested, resources); }
    }
    const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - arm_start).count();
    CUOPT_LOG_INFO(
      "Pre-presolve PDLP+GPUFJ: elapsed=%.3fs stopped=%d pdlp_status=%d "
      "pdlp_point=%s feasible_candidates=%d",
      elapsed,
      stop_requested.load(std::memory_order_acquire),
      pdlp_status_,
      pdlp_point_usable_ ? "usable" : "unavailable",
      feasible_candidates_);
  }

  void request_stop()
  {
    pdlp_concurrent_halt_.store(1, std::memory_order_release);
    std::lock_guard<std::mutex> lock(active_context_mutex_);
    if (active_context_ != nullptr) {
      active_context_->preempt_heuristic_solver_.store(true, std::memory_order_release);
    }
  }

 private:
  struct task_resources_t {
    task_resources_t(const problem_t<i_t, f_t>& snapshot,
                     const mip_solver_settings_t<i_t, f_t>& settings)
      : problem(snapshot, &handle), solution(problem)
    {
      initialize_private_gpu_handle(handle);
      thrust::fill(
        handle.get_thrust_policy(), solution.assignment.begin(), solution.assignment.end(), f_t{0});
      solution.clamp_within_bounds();
      handle.sync_stream();
      context = std::make_unique<mip_solver_context_t<i_t, f_t>>(&handle, &problem, settings);
    }

    // Destruction is reverse declaration order: context and solution are gone before their handle.
    raft::handle_t handle;
    problem_t<i_t, f_t> problem;
    solution_t<i_t, f_t> solution;
    std::unique_ptr<mip_solver_context_t<i_t, f_t>> context;
  };

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

  void run_pdlp(std::atomic<bool>& stop_requested, task_resources_t& resources)
  {
    timer_t timer(pdlp_time_limit);
    auto settings = make_pdlp_settings();
    pdlp::pdlp_solver_t<i_t, f_t> solver(resources.problem, settings);
    if (should_stop(stop_requested)) { return; }

    solver.set_inside_mip(true);
    auto response = solver.run_solver(timer);
    if (should_stop(stop_requested)) { return; }

    const auto& primal = response.get_primal_solution();
    pdlp_status_       = static_cast<int>(response.get_termination_status());
    const bool usable =
      response.get_termination_status() != pdlp_termination_status_t::NumericalError &&
      primal.size() == resources.solution.assignment.size();
    pdlp_point_usable_ = usable;
    if (usable) {
      resources.solution.copy_new_assignment(primal);
      resources.solution.clamp_within_bounds();
    } else {
      CUOPT_LOG_DEBUG("Pre-presolve PDLP returned no usable GPUFJ seed (status %d, size %zu)",
                      static_cast<int>(response.get_termination_status()),
                      primal.size());
    }
  }

  fj_settings_t make_fj_settings(fj_mode_t mode, double time_limit) const
  {
    fj_settings_t settings;
    settings.mode                                   = mode;
    settings.time_limit                             = time_limit;
    settings.iteration_limit                        = std::numeric_limits<int>::max();
    settings.n_of_minimums_for_exit                 = std::numeric_limits<int>::max();
    settings.update_weights                         = mode != fj_mode_t::ROUNDING;
    settings.feasibility_run                        = false;
    settings.parameters.rounding_second_stage_split = fj_rounding_second_stage_split;
    return settings;
  }

  void run_fj(std::atomic<bool>& stop_requested, task_resources_t& resources)
  {
    timer_t timer(fj_total_time_limit);
    auto rounding_settings =
      make_fj_settings(fj_mode_t::ROUNDING, timer.clamp_remaining_time(fj_rounding_time_limit));
    fj_t<i_t, f_t> fj(*resources.context, rounding_settings);
    fj.improvement_callback = [this, &resources](f_t user_obj, const std::vector<f_t>& assignment) {
      report_feasible_improvement(resources, user_obj, assignment);
    };

    if (should_stop(stop_requested) || timer.check_time_limit()) { return; }
    const bool rounding_feasible = fj.solve(resources.solution);
    if (rounding_feasible) { report_current_solution(resources); }

    if (should_stop(stop_requested) || timer.check_time_limit()) { return; }
    fj.set_fj_settings(make_fj_settings(fj_mode_t::EXIT_NON_IMPROVING, timer.remaining_time()));
    const bool improving_feasible = fj.solve(resources.solution);
    if (improving_feasible) { report_current_solution(resources); }
  }

  void report_current_solution(task_resources_t& resources)
  {
    report_feasible_improvement(
      resources, resources.solution.get_user_objective(), resources.solution.get_host_assignment());
  }

  void report_feasible_improvement(task_resources_t& resources,
                                   f_t user_obj,
                                   const std::vector<f_t>& assignment)
  {
    // fj_t currently invokes its callback only after this same full check. Recheck at the shared
    // callback boundary so neither a fractional LP point nor an integer-but-constraint-infeasible
    // rounding point can escape if the internal callback contract changes.
    if (!resources.solution.compute_feasibility()) { return; }

    const f_t solver_obj = resources.problem.get_solver_obj_from_user_obj(user_obj);
    if (solver_obj >= best_solver_objective_) { return; }

    auto stream = resources.handle.get_stream();
    rmm::device_uvector<f_t> postprocessed_assignment(assignment.size(), stream);
    raft::copy(postprocessed_assignment.data(), assignment.data(), assignment.size(), stream);
    resources.problem.post_process_assignment(postprocessed_assignment, true, stream);
    auto original_assignment = cuopt::host_copy(postprocessed_assignment, stream);

    best_solver_objective_ = solver_obj;
    ++feasible_candidates_;
    if (incumbent_callback_) {
      incumbent_callback_(solver_obj, user_obj, original_assignment, "PDLP+GPUFJ");
    }
  }

  int device_id_{0};
  const mip_solver_settings_t<i_t, f_t>& settings_;
  std::unique_ptr<problem_t<i_t, f_t>> snapshot_problem_;
  early_incumbent_callback_t<f_t> incumbent_callback_;
  f_t best_solver_objective_{std::numeric_limits<f_t>::infinity()};
  int pdlp_status_{-1};
  int feasible_candidates_{0};
  bool pdlp_point_usable_{false};
  std::atomic<int> pdlp_concurrent_halt_{0};
  std::mutex active_context_mutex_;
  mip_solver_context_t<i_t, f_t>* active_context_{nullptr};
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
  if (mode_ != pdlp_gpufj_mode) { return; }

  auto state = std::make_shared<pdlp_gpufj_state_t<i_t, f_t>>(
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
    CUOPT_LOG_INFO("Skipping pre-presolve primal mode %d: no OpenMP task slot", mode_);
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
