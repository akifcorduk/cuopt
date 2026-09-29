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

#include <atomic>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

inline constexpr int pre_presolve_max_rows         = 20'000;
inline constexpr int pre_presolve_max_cols         = 25'000;
inline constexpr int pre_presolve_max_nnz          = 50'000;
inline constexpr int pre_presolve_min_team_threads = 12;

/**
 * @brief Internal benchmark configuration selected by CUOPT_CONFIG_ID.
 *
 * An unset environment leaves `enabled == false` and preserves the stock PR behavior. The four
 * enabled profiles are deliberately internal to this experiment and do not add a public setting.
 */
struct pre_presolve_config_t {
  bool enabled{false};
  int raw_id{-1};
  int max_config{-1};
  int repeat{-1};
  int config_id{-1};

  int diving_workers() const;
  int task_slots() const { return enabled ? diving_workers() + 1 : 0; }
  int cpufj_reserved_threads() const;
  bool cooperative() const;
  const char* name() const;
};

/**
 * @brief Strict, pure parser used by the environment-facing resolver and unit tests.
 */
pre_presolve_config_t resolve_pre_presolve_config(std::optional<std::string_view> config_id,
                                                  std::optional<std::string_view> max_config);

/**
 * @brief Resolve the benchmark profile from CUOPT_CONFIG_ID and CUOPT_MAX_CONFIG.
 */
pre_presolve_config_t resolve_pre_presolve_config_from_env();

/**
 * @brief Gate the experimental arm without changing any stock solver policy.
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

/**
 * @brief Runs root-seeded dives concurrently with presolve for one enabled experiment profile.
 */
template <typename i_t, typename f_t>
class pre_presolve_primal_t {
 public:
  pre_presolve_primal_t(const problem_t<i_t, f_t>& problem,
                        const mip_solver_settings_t<i_t, f_t>& settings,
                        pre_presolve_config_t config,
                        early_incumbent_callback_t<f_t> aux_callback);
  ~pre_presolve_primal_t();

  pre_presolve_primal_t(const pre_presolve_primal_t&)            = delete;
  pre_presolve_primal_t& operator=(const pre_presolve_primal_t&) = delete;

  bool eligible() const { return work_ != nullptr; }
  bool start();
  void stop() noexcept;

  // Safe before start, while the coordinator runs, and after it has naturally completed.
  void offer_external_incumbent(f_t user_obj, const std::vector<f_t>& assignment);

 private:
  struct task_state_t {
    std::atomic<int> cancel_requested{0};
    std::atomic<int> root_halt{0};
    std::atomic<int> node_halt{0};
    std::exception_ptr exception;
  };
  struct work_t;

  bool poll_external_incumbent(uint64_t& last_generation, std::vector<f_t>& assignment);

  pre_presolve_config_t config_;
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
