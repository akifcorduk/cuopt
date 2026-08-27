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
#include <exception>
#include <memory>

namespace cuopt::mathematical_optimization::mip {

// Benchmark leaves change only this compile-time width; no product setting selects the treatment.
inline constexpr int pre_bnb_max_diving_workers = 6;

// Accounts only for long-running tasks in the existing MIP OpenMP team. PaPILO's TBB workers are
// intentionally outside this budget.
class pre_presolve_thread_budget_t {
 public:
  explicit pre_presolve_thread_budget_t(int available_slots);

  bool try_reserve(int slots);
  void release(int slots);

 private:
  const int capacity_;
  std::atomic<int> available_;
};

// Runs the fixed, default-on selective pre-B&B treatment while PaPILO is active.
template <typename i_t, typename f_t>
class pre_presolve_primal_t {
 public:
  pre_presolve_primal_t(const optimization_problem_t<i_t, f_t>& op_problem,
                        const mip_solver_settings_t<i_t, f_t>& settings,
                        early_incumbent_callback_t<f_t> incumbent_callback,
                        pre_presolve_thread_budget_t& thread_budget,
                        int task_slots);
  ~pre_presolve_primal_t();

  pre_presolve_primal_t(const pre_presolve_primal_t&)            = delete;
  pre_presolve_primal_t& operator=(const pre_presolve_primal_t&) = delete;

  bool start();
  void stop();

 private:
  struct task_state_t {
    std::atomic<bool> stop_requested{false};
    std::exception_ptr exception;
  };
  struct work_t;

  void stop_no_throw() noexcept;

  pre_presolve_thread_budget_t& thread_budget_;
  std::unique_ptr<task_state_t> task_state_;
  std::unique_ptr<work_t> work_;
  int task_slots_;
  bool started_{false};
};

}  // namespace cuopt::mathematical_optimization::mip
