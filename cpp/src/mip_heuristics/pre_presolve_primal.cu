/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "pre_presolve_primal.cuh"

#include <mip_heuristics/feasibility_jump/fj_cpu.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/logger.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>

#include <thrust/fill.h>

#include <algorithm>
#include <limits>
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

template <typename f_t>
struct cpufj_worker_result_t {
  void reset()
  {
    has_solution    = false;
    best_objective  = std::numeric_limits<f_t>::infinity();
    best_assignment = {};
  }

  bool has_solution{false};
  f_t best_objective{std::numeric_limits<f_t>::infinity()};
  std::vector<f_t> best_assignment;
};

template <typename i_t, typename f_t>
class pdlp_cpufj_runtime_t {
 public:
  pdlp_cpufj_runtime_t(problem_t<i_t, f_t>& problem,
                       fj_cpu_climber_t<i_t, f_t>& worker,
                       std::atomic<bool>& stop_requested,
                       std::atomic<bool>& cpufj_preemption,
                       early_incumbent_callback_t<f_t> incumbent_callback,
                       pre_presolve_thread_budget_t& thread_budget)
    : problem_(problem),
      worker_(worker),
      stop_requested_(stop_requested),
      cpufj_preemption_(cpufj_preemption),
      incumbent_callback_(std::move(incumbent_callback)),
      thread_budget_(thread_budget)
  {
    auto* result                 = &result_;
    worker_.improvement_callback = [result](
                                     f_t objective, const std::vector<f_t>& assignment, double) {
      if (!result->has_solution || objective < result->best_objective) {
        result->has_solution    = true;
        result->best_objective  = objective;
        result->best_assignment = assignment;
      }
    };
    worker_.log_prefix = "PDLP-CPUFJ: ";
  }

  void on_major_iteration(i_t,
                          raft::device_span<const f_t> unscaled_primal,
                          rmm::cuda_stream_view stream)
  {
    ++firings_;

    if (stop_requested_.load(std::memory_order_acquire) ||
        cpufj_preemption_.load(std::memory_order_acquire)) {
      ++skipped_stopping_;
      return;
    }
    if (!thread_budget_.try_reserve(1)) {
      ++skipped_no_slot_;
      return;
    }

    task_slot_release_t release(&thread_budget_, 1);
    auto host_primal = cuopt::host_copy(unscaled_primal, stream);
    reseed_fj_cpu_from_host(worker_, host_primal);
    result_.reset();
    ++launched_;

    // Run on the already executing dispatcher task. The reservation is an admission-control token,
    // not a deferred OpenMP task, so a callback is skipped instead of entering the task queue.
    constexpr f_t worker_time_limit{0.05};
    cpufj_solve(&worker_, worker_time_limit);
    report_worker_result(stream);
  }

  void log_counts() const
  {
    CUOPT_LOG_INFO(
      "Pre-presolve mode 6 (PDLP-CPUFJ): firings=%d launched=%d skipped_no_slot=%d "
      "skipped_stopping=%d reports=%d",
      static_cast<int>(firings_),
      static_cast<int>(launched_),
      static_cast<int>(skipped_no_slot_),
      static_cast<int>(skipped_stopping_),
      static_cast<int>(reports_));
  }

 private:
  void report_worker_result(rmm::cuda_stream_view stream)
  {
    if (!result_.has_solution) { return; }

    rmm::device_uvector<f_t> assignment(result_.best_assignment.size(), stream);
    raft::copy(assignment.data(), result_.best_assignment.data(), assignment.size(), stream);
    // CPUFJ's callback only reports assignments that passed its feasibility checks. GPU work is
    // deferred until PDLP is paused in this callback.
    problem_.post_process_assignment(assignment, true, stream);
    auto user_assignment     = cuopt::host_copy(assignment, stream);
    const f_t user_objective = problem_.get_user_obj_from_solver_obj(result_.best_objective);
    if (incumbent_callback_) {
      incumbent_callback_(result_.best_objective, user_objective, user_assignment, "PDLP-CPUFJ");
      ++reports_;
    }
  }

  problem_t<i_t, f_t>& problem_;
  fj_cpu_climber_t<i_t, f_t>& worker_;
  std::atomic<bool>& stop_requested_;
  std::atomic<bool>& cpufj_preemption_;
  early_incumbent_callback_t<f_t> incumbent_callback_;
  pre_presolve_thread_budget_t& thread_budget_;
  cpufj_worker_result_t<f_t> result_;
  i_t firings_{0};
  i_t launched_{0};
  i_t skipped_no_slot_{0};
  i_t skipped_stopping_{0};
  i_t reports_{0};
};

template <typename i_t, typename f_t>
class pdlp_cpufj_arm_t {
 public:
  pdlp_cpufj_arm_t(const optimization_problem_t<i_t, f_t>& op_problem,
                   const mip_solver_settings_t<i_t, f_t>& settings,
                   early_incumbent_callback_t<f_t> incumbent_callback,
                   pre_presolve_thread_budget_t& thread_budget)
    : incumbent_callback_(std::move(incumbent_callback)), thread_budget_(thread_budget)
  {
    RAFT_CUDA_TRY(cudaGetDevice(&device_id_));

    // Complete source-side copies before PaPILO starts. The worker deep-copies this immutable
    // preprocessed snapshot onto a RAFT handle constructed on the worker host thread.
    snapshot_problem_ =
      std::make_unique<problem_t<i_t, f_t>>(op_problem, settings.get_tolerances(), false);
    snapshot_problem_->preprocess_problem();
    snapshot_problem_->handle_ptr->sync_stream();
  }

  void request_stop()
  {
    pdlp_halt_.store(1, std::memory_order_release);
    cpufj_preemption_.store(true, std::memory_order_release);
  }

  void run(std::atomic<bool>& stop_requested)
  {
    if (stop_requested.load(std::memory_order_acquire)) {
      request_stop();
      return;
    }
    RAFT_CUDA_TRY(cudaSetDevice(device_id_));

    raft::handle_t handle;
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
      handle.get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle.get_stream()));
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
      handle.get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle.get_stream()));
    problem_t<i_t, f_t> problem(*snapshot_problem_, &handle);
    handle.sync_stream();

    solution_t<i_t, f_t> solution(problem);
    thrust::fill(
      handle.get_thrust_policy(), solution.assignment.begin(), solution.assignment.end(), f_t{0});
    solution.clamp_within_bounds();

    fj_settings_t fj_settings;
    fj_settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
    fj_settings.n_of_minimums_for_exit = std::numeric_limits<int>::max();
    fj_settings.time_limit             = std::numeric_limits<f_t>::infinity();
    fj_settings.iteration_limit        = std::numeric_limits<int>::max();
    fj_settings.update_weights         = true;
    fj_settings.feasibility_run        = false;
    auto worker =
      init_fj_cpu_standalone(problem, solution, cpufj_preemption_, std::move(fj_settings));

    pdlp_cpufj_runtime_t<i_t, f_t> runtime(
      problem, *worker, stop_requested, cpufj_preemption_, incumbent_callback_, thread_budget_);
    relaxed_lp_settings_t lp_settings;
    lp_settings.time_limit      = std::numeric_limits<double>::infinity();
    lp_settings.save_state      = false;
    lp_settings.concurrent_halt = &pdlp_halt_;

    auto* runtime_ptr = &runtime;
    relaxed_lp_major_iteration_callback_t<i_t, f_t> major_iteration_callback =
      [runtime_ptr](
        i_t iteration, raft::device_span<const f_t> primal, rmm::cuda_stream_view stream) {
        runtime_ptr->on_major_iteration(iteration, primal, stream);
      };

    try {
      constexpr i_t callback_interval{1};
      get_relaxed_lp_solution(problem,
                              solution.assignment,
                              solution.lp_state,
                              lp_settings,
                              std::move(major_iteration_callback),
                              callback_interval);
    } catch (...) {
      cpufj_preemption_.store(true, std::memory_order_release);
      runtime.log_counts();
      throw;
    }
    cpufj_preemption_.store(true, std::memory_order_release);
    runtime.log_counts();
  }

 private:
  int device_id_{0};
  std::unique_ptr<problem_t<i_t, f_t>> snapshot_problem_;
  early_incumbent_callback_t<f_t> incumbent_callback_;
  pre_presolve_thread_budget_t& thread_budget_;
  std::atomic<int> pdlp_halt_{0};
  std::atomic<bool> cpufj_preemption_{false};
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
  if (mode_ != 6) { return; }

  auto arm = std::make_shared<pdlp_cpufj_arm_t<i_t, f_t>>(
    op_problem, settings, std::move(incumbent_callback), thread_budget_);
  work_         = [arm](std::atomic<bool>& stop_requested) { arm->run(stop_requested); };
  request_stop_ = [arm]() { arm->request_stop(); };
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
