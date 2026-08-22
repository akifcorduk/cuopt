/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <atomic>
#include <chrono>

namespace cuopt::mathematical_optimization {

/**
 * @brief Run-local timing data for the MIP path through the first root LP.
 *
 * Critical-path milestones use one steady clock. Diagnostic durations may be nested in those
 * intervals or describe background work and therefore must not be included in the residual.
 */
struct startup_profile_t {
  explicit startup_profile_t(bool enabled_) : enabled(enabled_), solve_start(enabled_ ? now() : 0.0)
  {
  }

  static double now() noexcept
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
  }

  static double elapsed(double start) noexcept { return now() - start; }

  bool active() const noexcept
  {
    return enabled && !root_profile_emitted.load(std::memory_order_acquire);
  }

  static double interval(double begin, double end) noexcept
  {
    return begin > 0.0 && end >= begin ? end - begin : 0.0;
  }

  bool enabled{false};
  std::atomic<bool> root_profile_emitted{false};

  // Adjacent critical-path milestones. These are the only values used to compute the residual.
  double solve_start{0.0};
  double papilo_start{0.0};
  double papilo_end{0.0};
  double run_mip_start{0.0};
  double cuopt_presolve_start{0.0};
  double cuopt_presolve_end{0.0};
  double bnb_export_start{0.0};
  double bnb_export_end{0.0};
  double bnb_dispatch{0.0};
  double bnb_solve_start{0.0};
  double root_lp_start{0.0};
  double root_lp_end{0.0};

  // Nested critical-path diagnostics.
  double mip_scaling{0.0};
  double initial_problem_ctor{0.0};
  double sort_csr{0.0};
  double papilo_pipeline{0.0};
  double reduced_problem_ctor{0.0};
  double scaled_problem_copy{0.0};
  double mip_preprocess{0.0};
  double cuopt_to_user_problem{0.0};
  double bnb_problem_export{0.0};
  double bnb_ctor{0.0};
  double bnb_convert_user_problem{0.0};
  double bnb_full_variable_types{0.0};
  double bnb_arow{0.0};
  double bnb_variable_bounds{0.0};
  double dual_presolve{0.0};
  double dual_scaling{0.0};
  double dual_create_phase1{0.0};
  double dual_phase1{0.0};
  double dual_phase2{0.0};
  double dual_first_refactor{0.0};

  // Background lifetimes and critical-path waits are reported separately.
  double original_cpufj_launch{0.0};
  double original_gpufj_launch{0.0};
  double original_cpufj_start{0.0};
  double original_gpufj_start{0.0};
  double original_cpufj_background{0.0};
  double original_gpufj_background{0.0};
  double original_cpufj_wait{0.0};
  double original_gpufj_wait{0.0};
  double presolved_cpufj_launch{0.0};
  double presolved_cpufj_start{0.0};
  double presolved_cpufj_background{0.0};
  double presolved_cpufj_wait{0.0};
  double symmetry_wait{0.0};
};

}  // namespace cuopt::mathematical_optimization
