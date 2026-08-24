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
#include <memory>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Each experiment branch changes this internal constant so checking out the branch activates its
// treatment without adding a product setting or command-line selector. The scaffold remains idle.
inline constexpr int pre_presolve_primal_branch_mode = 2;
inline constexpr int pre_presolve_primal_task_slots  = 2;
inline constexpr int pre_bnb_compact_nnz_limit       = 5'000;
inline constexpr int pre_bnb_small_row_limit         = 64;
inline constexpr int pre_bnb_absolute_nnz_limit      = 50'000;

inline bool pre_presolve_primal_mode_replaces_early_gpufj(int mode)
{
  return mode >= 3 && mode <= 6;
}

inline bool pre_bnb_selective_problem_is_eligible(int num_constraints, int nnz)
{
  return nnz <= pre_bnb_compact_nnz_limit ||
         (num_constraints <= pre_bnb_small_row_limit && nnz <= pre_bnb_absolute_nnz_limit);
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
class pre_presolve_first_candidate_publisher_t {
 public:
  struct snapshot_t {
    std::size_t generated{0};
    std::size_t submitted{0};
    bool candidate_stop_requested{false};
  };

  pre_presolve_first_candidate_publisher_t(early_incumbent_callback_t<f_t> incumbent_callback,
                                           f_t objective_scale,
                                           f_t objective_constant,
                                           const char* heuristic_name)
    : incumbent_callback_(std::move(incumbent_callback)),
      objective_scale_(objective_scale),
      objective_constant_(objective_constant),
      heuristic_name_(heuristic_name)
  {
  }

  void consider(f_t solver_objective, const std::vector<f_t>& assignment)
  {
    generated_.fetch_add(1, std::memory_order_relaxed);
    bool expected = false;
    if (!candidate_claimed_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      return;
    }

    try {
      if (incumbent_callback_) {
        const f_t user_objective = objective_scale_ * (solver_objective + objective_constant_);
        incumbent_callback_(solver_objective, user_objective, assignment, heuristic_name_);
        submitted_.store(1, std::memory_order_release);
      }
    } catch (...) {
      candidate_stop_requested_.store(true, std::memory_order_release);
      throw;
    }
    candidate_stop_requested_.store(true, std::memory_order_release);
  }

  bool candidate_stop_requested() const
  {
    return candidate_stop_requested_.load(std::memory_order_acquire);
  }

  snapshot_t snapshot() const
  {
    return {generated_.load(std::memory_order_acquire),
            submitted_.load(std::memory_order_acquire),
            candidate_stop_requested()};
  }

 private:
  early_incumbent_callback_t<f_t> incumbent_callback_;
  const f_t objective_scale_;
  const f_t objective_constant_;
  const char* heuristic_name_;
  std::atomic<std::size_t> generated_{0};
  std::atomic<std::size_t> submitted_{0};
  std::atomic<bool> candidate_claimed_{false};
  std::atomic<bool> candidate_stop_requested_{false};
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
