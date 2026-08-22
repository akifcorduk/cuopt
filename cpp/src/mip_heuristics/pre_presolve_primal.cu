/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "pre_presolve_primal.cuh"

#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/local_search/feasibility_pump/feasibility_pump.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/solver_context.cuh>
#include <pdlp/pdlp.cuh>
#include <pdlp/solve.cuh>
#include <utilities/logger.hpp>
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

constexpr int pdlp_fp_mode                 = 5;
constexpr double pdlp_fp_total_time_limit  = 0.10;
constexpr double pdlp_time_limit           = 0.05;
constexpr int pdlp_iteration_limit         = 1000;
constexpr double pdlp_optimality_tolerance = 1e-2;

void set_cuda_device(int device_id) { RAFT_CUDA_TRY(cudaSetDevice(device_id)); }

void initialize_private_handle(const raft::handle_t& handle)
{
  // PDLP supplies device-resident alpha/beta scalars. A fresh RAFT handle defaults to host pointer
  // mode, which makes cuSPARSE host-dereference a device address during SpMM setup.
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
class pdlp_fp_state_t {
 public:
  pdlp_fp_state_t(const optimization_problem_t<i_t, f_t>& op_problem,
                  const mip_solver_settings_t<i_t, f_t>& settings,
                  early_incumbent_callback_t<f_t> incumbent_callback)
    : settings_(settings), incumbent_callback_(std::move(incumbent_callback))
  {
    RAFT_CUDA_TRY(cudaGetDevice(&device_id_));

    // Keep only a synchronized snapshot before dispatch. Stream-owning solver state is created on
    // the OpenMP worker that uses it, while the main thread is free to run PaPILO.
    snapshot_problem_ptr_ =
      std::make_unique<problem_t<i_t, f_t>>(op_problem, settings.get_tolerances(), false);
    snapshot_problem_ptr_->preprocess_problem();
    snapshot_problem_ptr_->handle_ptr->sync_stream();
  }

  void run(std::atomic<bool>& stop_requested)
  {
    timer_t worker_timer(std::numeric_limits<double>::max());
    set_cuda_device(device_id_);
    raft::handle_t handle;
    initialize_private_handle(handle);
    problem_t<i_t, f_t> problem(*snapshot_problem_ptr_, &handle);
    solution_t<i_t, f_t> solution(problem);
    thrust::fill(
      handle.get_thrust_policy(), solution.assignment.begin(), solution.assignment.end(), f_t{0});
    solution.clamp_within_bounds();
    handle.sync_stream();
    rmm::device_uvector<f_t> lp_solution(problem.n_variables, handle.get_stream());
    mip_solver_context_t<i_t, f_t> context(&handle, &problem, settings_);
    gpu_setup_elapsed_ = worker_timer.elapsed_time();

    publish_context(context);
    try {
      run_private_state(
        problem, solution, lp_solution, context, handle, stop_requested, worker_timer);
    } catch (...) {
      clear_context(context);
      throw;
    }
    clear_context(context);
  }

  void request_stop()
  {
    concurrent_halt_.store(1, std::memory_order_release);
    std::lock_guard<std::mutex> lock(active_context_mutex_);
    if (active_context_ptr_ != nullptr) {
      active_context_ptr_->preempt_heuristic_solver_.store(true, std::memory_order_release);
    }
  }

 private:
  void publish_context(mip_solver_context_t<i_t, f_t>& context)
  {
    std::lock_guard<std::mutex> lock(active_context_mutex_);
    active_context_ptr_ = &context;
    if (concurrent_halt_.load(std::memory_order_acquire) != 0) {
      context.preempt_heuristic_solver_.store(true, std::memory_order_release);
    }
  }

  void clear_context(mip_solver_context_t<i_t, f_t>& context)
  {
    std::lock_guard<std::mutex> lock(active_context_mutex_);
    if (active_context_ptr_ == &context) { active_context_ptr_ = nullptr; }
  }

  void run_private_state(problem_t<i_t, f_t>& problem,
                         solution_t<i_t, f_t>& solution,
                         rmm::device_uvector<f_t>& lp_solution,
                         mip_solver_context_t<i_t, f_t>& context,
                         raft::handle_t& handle,
                         std::atomic<bool>& stop_requested,
                         const timer_t& worker_timer)
  {
    timer_t arm_timer(pdlp_fp_total_time_limit);

    bool pdlp_usable = false;
    if (!should_stop(stop_requested)) {
      pdlp_usable = run_pdlp(problem,
                             solution,
                             lp_solution,
                             stop_requested,
                             arm_timer.clamp_remaining_time(pdlp_time_limit));
    }
    if (pdlp_usable && !should_stop(stop_requested) && !arm_timer.check_time_limit()) {
      run_fp(problem, solution, lp_solution, context, handle, stop_requested, arm_timer);
    }

    const bool stopped = should_stop(stop_requested);
    CUOPT_LOG_INFO(
      "Pre-presolve PDLP+FP summary: ran=1 stopped=%d pdlp_status=%d "
      "pdlp_elapsed=%.6f fp_attempts=%d fp_feasible=%d fp_setup_elapsed=%.6f "
      "gpu_setup_elapsed=%.6f total_elapsed=%.6f",
      static_cast<int>(stopped),
      pdlp_status_,
      pdlp_elapsed_,
      fp_attempts_,
      static_cast<int>(fp_feasible_),
      fp_setup_elapsed_,
      gpu_setup_elapsed_,
      worker_timer.elapsed_time());
  }
  bool should_stop(const std::atomic<bool>& stop_requested)
  {
    if (!stop_requested.load(std::memory_order_acquire)) { return false; }
    request_stop();
    return true;
  }

  pdlp_solver_settings_t<i_t, f_t> make_pdlp_settings(double time_limit)
  {
    pdlp_solver_settings_t<i_t, f_t> settings{};
    settings.set_optimality_tolerance(static_cast<f_t>(pdlp_optimality_tolerance));
    settings.detect_infeasibility    = false;
    settings.iteration_limit         = pdlp_iteration_limit;
    settings.time_limit              = static_cast<f_t>(time_limit);
    settings.log_to_console          = false;
    settings.per_constraint_residual = true;
    settings.first_primal_feasible   = false;
    settings.pdlp_solver_mode        = pdlp_solver_mode_t::Stable2;
    settings.presolver               = presolver_t::None;
    settings.method                  = method_t::PDLP;
    settings.inside_mip              = true;
    settings.concurrent_halt         = &concurrent_halt_;
    set_pdlp_solver_mode(settings);
    return settings;
  }

  bool run_pdlp(problem_t<i_t, f_t>& problem,
                solution_t<i_t, f_t>& solution,
                rmm::device_uvector<f_t>& lp_solution,
                std::atomic<bool>& stop_requested,
                double time_limit)
  {
    timer_t timer(time_limit);
    auto settings = make_pdlp_settings(time_limit);
    pdlp::pdlp_solver_t<i_t, f_t> solver(problem, settings);
    if (should_stop(stop_requested)) {
      pdlp_elapsed_ = timer.elapsed_time();
      return false;
    }

    solver.set_inside_mip(true);
    auto response = solver.run_solver(timer);
    pdlp_elapsed_ = timer.elapsed_time();
    pdlp_status_  = static_cast<int>(response.get_termination_status());
    if (should_stop(stop_requested)) { return false; }

    const auto& primal = response.get_primal_solution();
    const bool usable =
      response.get_termination_status() != pdlp_termination_status_t::NumericalError &&
      primal.size() == solution.assignment.size();
    if (usable) {
      solution.copy_new_assignment(primal);
      solution.clamp_within_bounds();
      raft::copy(lp_solution.data(),
                 solution.assignment.data(),
                 solution.assignment.size(),
                 problem.handle_ptr->get_stream());
    } else {
      CUOPT_LOG_DEBUG("Pre-presolve PDLP returned no usable FP point (status %d, size %zu)",
                      pdlp_status_,
                      primal.size());
    }
    return usable;
  }

  void run_fp(problem_t<i_t, f_t>& problem,
              solution_t<i_t, f_t>& solution,
              rmm::device_uvector<f_t>& lp_solution,
              mip_solver_context_t<i_t, f_t>& context,
              raft::handle_t& handle,
              std::atomic<bool>& stop_requested,
              timer_t& arm_timer)
  {
    timer_t setup_timer(std::numeric_limits<double>::max());
    fj_t<i_t, f_t> fj(context);
    constraint_prop_t<i_t, f_t> constraint_prop(context);
    line_segment_search_t<i_t, f_t> line_segment_search(fj, constraint_prop);
    feasibility_pump_t<i_t, f_t> fp(context, fj, constraint_prop, line_segment_search, lp_solution);
    fp_setup_elapsed_ = setup_timer.elapsed_time();

    if (should_stop(stop_requested) || arm_timer.check_time_limit()) { return; }
    fp.reset_for_standalone_run(solution, arm_timer.remaining_time(), &concurrent_halt_);
    fp_attempts_         = 1;
    const bool fp_result = fp.run_single_fp_descent(solution);
    if (should_stop(stop_requested) || !fp_result) { return; }

    // The standalone FP path has no population callback. Audit all MIP feasibility conditions at
    // this boundary before publishing its direct fractional-LP-derived result.
    fp_feasible_ = report_current_solution(problem, solution, handle);
  }

  bool report_current_solution(problem_t<i_t, f_t>& problem,
                               solution_t<i_t, f_t>& solution,
                               raft::handle_t& handle)
  {
    if (!solution.compute_feasibility()) { return false; }

    const f_t user_obj   = solution.get_user_objective();
    const f_t solver_obj = problem.get_solver_obj_from_user_obj(user_obj);
    if (solver_obj >= best_solver_objective_) { return true; }

    const auto assignment = solution.get_host_assignment();
    auto stream           = handle.get_stream();
    rmm::device_uvector<f_t> postprocessed_assignment(assignment.size(), stream);
    raft::copy(postprocessed_assignment.data(), assignment.data(), assignment.size(), stream);
    problem.post_process_assignment(postprocessed_assignment, true, stream);
    auto original_assignment = cuopt::host_copy(postprocessed_assignment, stream);

    best_solver_objective_ = solver_obj;
    if (incumbent_callback_) {
      incumbent_callback_(solver_obj, user_obj, original_assignment, "PDLP+FP");
    }
    return true;
  }

  int device_id_{0};

  std::unique_ptr<problem_t<i_t, f_t>> snapshot_problem_ptr_;
  mip_solver_settings_t<i_t, f_t> settings_;
  early_incumbent_callback_t<f_t> incumbent_callback_;
  f_t best_solver_objective_{std::numeric_limits<f_t>::infinity()};
  std::atomic<int> concurrent_halt_{0};
  std::mutex active_context_mutex_;
  mip_solver_context_t<i_t, f_t>* active_context_ptr_{nullptr};
  int pdlp_status_{-1};
  double pdlp_elapsed_{0.};
  int fp_attempts_{0};
  bool fp_feasible_{false};
  double fp_setup_elapsed_{0.};
  double gpu_setup_elapsed_{0.};
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
  if (mode_ != pdlp_fp_mode) { return; }

  auto state = std::make_shared<pdlp_fp_state_t<i_t, f_t>>(
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
