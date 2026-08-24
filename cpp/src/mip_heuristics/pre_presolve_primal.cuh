/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/early_heuristic.cuh>

#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Each experiment branch changes this internal constant so checking out the branch activates its
// treatment without adding a product setting or command-line selector. The scaffold remains idle.
inline constexpr int pre_presolve_primal_branch_mode = 2;
inline constexpr int pre_presolve_primal_task_slots  = 2;

inline bool pre_presolve_primal_mode_replaces_early_gpufj(int mode)
{
  return mode >= 3 && mode <= 6;
}

/**
 * @brief Accounts for long-running tasks inside the existing MIP OpenMP team.
 *
 * This deliberately does not account for PaPILO's TBB workers. It only prevents the new
 * pre-presolve experiments from reserving more OpenMP task slots than the MIP team has left.
 */
class pre_presolve_thread_budget_t {
 public:
  explicit pre_presolve_thread_budget_t(int available_slots);

  bool try_reserve(int slots);
  void release(int slots);

  int capacity() const { return capacity_; }
  int reserved() const { return capacity_ - available_.load(std::memory_order_relaxed); }
  int available() const { return available_.load(std::memory_order_relaxed); }

 private:
  const int capacity_;
  std::atomic<int> available_;
};

template <typename f_t>
class pre_presolve_best_candidate_store_t {
 public:
  struct snapshot_t {
    bool has_candidate{false};
    std::size_t generated{0};
    f_t solver_objective{std::numeric_limits<f_t>::infinity()};
    std::vector<f_t> assignment;
  };

  void consider(f_t solver_objective, const std::vector<f_t>& assignment)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++snapshot_.generated;
    if (!snapshot_.has_candidate || solver_objective < snapshot_.solver_objective) {
      snapshot_.has_candidate    = true;
      snapshot_.solver_objective = solver_objective;
      snapshot_.assignment       = assignment;
    }
  }

  snapshot_t snapshot() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_;
  }

 private:
  mutable std::mutex mutex_;
  snapshot_t snapshot_;
};

/**
 * @brief Shared lifecycle for independent pre-PaPILO primal experiments.
 *
 * Experiment branches install one work function in the constructor. The scaffold owns the stop
 * flag, exception propagation, dependency join, and task-slot release so each arm only implements
 * its solver sequence.
 */
template <typename i_t, typename f_t>
class pre_presolve_primal_t {
 public:
  pre_presolve_primal_t(const optimization_problem_t<i_t, f_t>& op_problem,
                        const mip_solver_settings_t<i_t, f_t>& settings,
                        early_incumbent_callback_t<f_t> incumbent_callback,
                        pre_presolve_thread_budget_t& thread_budget);
  ~pre_presolve_primal_t();

  pre_presolve_primal_t(const pre_presolve_primal_t&)            = delete;
  pre_presolve_primal_t& operator=(const pre_presolve_primal_t&) = delete;

  bool start();
  void stop();

  // Only the GPU experiment branches override the normal early GPUFJ task.
  bool replaces_early_gpufj() const { return replaces_early_gpufj_; }

 private:
  struct task_state_t {
    std::atomic<bool> stop_requested{false};
    std::exception_ptr exception;
  };

  using work_t = std::function<void(std::atomic<bool>&)>;

  void configure_work(const optimization_problem_t<i_t, f_t>& op_problem,
                      const mip_solver_settings_t<i_t, f_t>& settings,
                      early_incumbent_callback_t<f_t> incumbent_callback);
  void stop_no_throw() noexcept;

  const i_t mode_;
  pre_presolve_thread_budget_t& thread_budget_;
  std::unique_ptr<task_state_t> task_state_;
  work_t work_;
  bool replaces_early_gpufj_{false};
  bool started_{false};
};

}  // namespace cuopt::mathematical_optimization::mip
