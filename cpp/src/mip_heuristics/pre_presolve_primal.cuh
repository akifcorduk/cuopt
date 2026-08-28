/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/early_heuristic.cuh>
#include <mip_heuristics/problem/problem.cuh>

#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>

#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>

namespace cuopt::mathematical_optimization::mip {

// Internal fixed policy for the root-diving experiment; no public setting selects the treatment.
inline constexpr int pre_bnb_max_diving_workers  = 6;
inline constexpr int top_level_papilo_thread_cap = 4;
inline constexpr int pre_bnb_max_rows            = 20'000;
inline constexpr int pre_bnb_max_cols            = 25'000;
inline constexpr int pre_bnb_max_nnz             = 50'000;

template <typename i_t>
inline bool is_pre_bnb_eligible(i_t num_rows, i_t num_cols, i_t nnz, bool has_quadratic_objective)
{
  return !has_quadratic_objective && num_rows <= pre_bnb_max_rows && num_cols <= pre_bnb_max_cols &&
         nnz <= pre_bnb_max_nnz;
}

inline int resolve_top_level_papilo_threads(int configured_threads, int omp_team_threads)
{
  const int resolved = configured_threads < 0 ? omp_team_threads : configured_threads;
  return std::min(std::max(resolved, 1), top_level_papilo_thread_cap);
}

inline int pre_bnb_task_slots(int omp_team_threads)
{
  constexpr int max_task_slots = 1 + pre_bnb_max_diving_workers;
  return std::min(max_task_slots, std::max(2, omp_team_threads - 3));
}

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

// Runs fixed, default-on root-seeded dives through PaPILO and cuOpt presolve.
template <typename i_t, typename f_t>
class pre_presolve_primal_t {
 public:
  pre_presolve_primal_t(const problem_t<i_t, f_t>& problem,
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
    std::atomic<int> cancel_requested{0};
    std::atomic<int> root_halt{0};
    std::atomic<int> node_halt{0};
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
