/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/user_problem.hpp>
#include <mip_heuristics/early_heuristic.cuh>
#include <mip_heuristics/problem/problem.cuh>

#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>

#include <atomic>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

inline constexpr int pre_presolve_max_rows         = 20'000;
inline constexpr int pre_presolve_max_cols         = 25'000;
inline constexpr int pre_presolve_max_nnz          = 50'000;
inline constexpr int pre_presolve_min_team_threads = 12;

// The fixed refinement portfolio uses three dive tasks plus one coordinator.
inline constexpr int pre_presolve_diving_workers = 3;
inline constexpr int pre_presolve_task_slots     = pre_presolve_diving_workers + 1;

/**
 * @brief Gate the auxiliary root-diving portfolio before reserving its task slots.
 */
template <typename i_t>
inline bool is_pre_presolve_primal_eligible(i_t num_rows,
                                            i_t num_cols,
                                            i_t nnz,
                                            bool has_quadratic_objective,
                                            bool has_quadratic_constraints,
                                            bool is_mip,
                                            bool presolve_enabled,
                                            bool opportunistic,
                                            int team_threads)
{
  return is_mip && presolve_enabled && opportunistic &&
         team_threads >= pre_presolve_min_team_threads && !has_quadratic_objective &&
         !has_quadratic_constraints && num_rows > 0 && num_cols > 0 && nnz >= 0 &&
         num_rows <= pre_presolve_max_rows && num_cols <= pre_presolve_max_cols &&
         nnz <= pre_presolve_max_nnz;
}

// Validate in the auxiliary solver's original column space before accepting or
// publishing an incumbent. Bounds remain strict; rows and integrality use MIP tolerances.
template <typename i_t, typename f_t>
bool verify_pre_presolve_primal_solution(
  const simplex::user_problem_t<i_t, f_t>& problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  const std::vector<f_t>& assignment);

/**
 * @brief Runs the fixed cooperative refinement portfolio concurrently with presolve.
 */
template <typename i_t, typename f_t>
class pre_presolve_primal_t {
 public:
  pre_presolve_primal_t(const problem_t<i_t, f_t>& problem,
                        const mip_solver_settings_t<i_t, f_t>& settings,
                        early_incumbent_callback_t<f_t> aux_callback);
  ~pre_presolve_primal_t();

  pre_presolve_primal_t(const pre_presolve_primal_t&)            = delete;
  pre_presolve_primal_t& operator=(const pre_presolve_primal_t&) = delete;

  bool eligible() const { return work_ != nullptr; }
  bool start();
  // Start and stop from the same OpenMP task so task dependencies join the worker.
  void stop() noexcept;

  // Rechecks incumbents against the original host model before offering them.
  // Safe before start, while the coordinator runs, and after it has naturally completed.
  void offer_external_incumbent(f_t user_obj, const std::vector<f_t>& assignment) noexcept;

 private:
  struct task_state_t {
    std::atomic<int> cancel_requested{0};
    std::atomic<int> root_halt{0};
    std::atomic<int> node_halt{0};
    std::exception_ptr exception;
  };
  struct work_t;

  bool poll_external_incumbent(uint64_t& last_generation, std::vector<f_t>& assignment);

  std::unique_ptr<task_state_t> task_state_;
  std::unique_ptr<work_t> work_;

  std::mutex mailbox_mutex_;
  f_t objective_sense_{f_t{1}};
  f_t mailbox_best_score_{std::numeric_limits<f_t>::infinity()};
  std::vector<f_t> mailbox_assignment_;
  uint64_t mailbox_generation_{0};
  bool started_{false};
};

}  // namespace cuopt::mathematical_optimization::mip
