/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt::linear_programming::detail {

struct bp_recombiner_config_t {
  static constexpr double bounds_prop_time_limit          = 2.;
  static constexpr double lp_after_bounds_prop_time_limit = 2.;
  // number of repair iterations even if it fails during the repair
  static constexpr size_t n_repair_iterations          = 10;
  static constexpr size_t initial_n_of_vars_from_other = 200;
  static constexpr size_t max_different_var_limit      = 10000;
  static constexpr size_t min_different_var_limit      = 20;
  static size_t max_n_of_vars_from_other;
  static constexpr double n_var_ratio_increase_factor = 1.1;
  static constexpr double n_var_ratio_decrease_factor = 0.99;
  static void increase_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::min(
      static_cast<size_t>(std::ceil(max_n_of_vars_from_other * n_var_ratio_increase_factor)),
      max_different_var_limit);
    CUOPT_LOG_DEBUG("Increased max_n_of_vars_from_other in BP recombiner to %lu",
                    max_n_of_vars_from_other);
  }
  static void decrease_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::max(
      static_cast<size_t>(std::floor(max_n_of_vars_from_other * n_var_ratio_decrease_factor)),
      min_different_var_limit);
    CUOPT_LOG_DEBUG("Decreased max_n_of_vars_from_other in BP recombiner to %lu",
                    max_n_of_vars_from_other);
  }
};

struct feasibility_lns_config_t {
  static constexpr size_t ruin_count                      = 128;
  static constexpr size_t max_attempts                    = 1000000;
  static constexpr double seed_repair_time_limit          = 1.;
  static constexpr double bounds_prop_time_limit          = 0.25;
  static constexpr double lp_after_bounds_prop_time_limit = 0.25;
  static constexpr double alpha                           = 1.;
  static constexpr double beta                            = 1.;
  // Related-neighbor similarity metric experiment (see
  // design_summaries/lns_feasibility/SIMILARITY_METRIC.md). Enabled only when CUOPT_LNS_SIM_ALPHA
  // is set; default scoring is value divergence. similarity_alpha in [0,1] trades structural vs
  // state similarity; similarity_jaccard_weight scales the shared-constraint Jaccard term.
  static constexpr double similarity_alpha                     = 0.5;
  static constexpr double similarity_jaccard_weight            = 1.0;
  static constexpr size_t n_repair_iterations                  = 100;
  static constexpr size_t violated_constraint_ruin_unsat_limit = 128;
  // Per-repair constraint-propagation wall-clock budget. The repair loop is meant to
  // iterate very fast, so each ruin/repair gets a small slice rather than all remaining time.
  static constexpr double repair_time_limit = 0.2;
  // Max failed repair iterations inside a single bounded constraint-propagation call.
  static constexpr size_t repair_max_failed_iterations = 8;
  // Probability of seeding the related-ruin from a variable that sits in a violated
  // constraint (vs. a uniformly random integer variable). Seeding inside violated
  // constraints lets the repair actually reduce the number of unsatisfied constraints.
  static constexpr double violated_seed_probability = 0.9;
  // Feasibility Jump is cuOpt's strongest feasibility local search; the LNS uses it both to
  // build a strong initial solution and to "finish off" near-feasible incumbents that the
  // bounds-propagation repair grinds down too slowly. These bound the FJ work.
  static constexpr double fj_polish_initial_time_limit = 1.0;
  static constexpr double fj_polish_time_limit         = 0.5;
  // Only invoke the (relatively expensive) FJ polish when the incumbent is close to feasible.
  static constexpr size_t fj_polish_unsat_threshold = 64;
  // Throttle: minimum number of attempts between consecutive FJ polish calls.
  static constexpr size_t fj_polish_min_attempts_between = 20;
  // Mini-MIP repair: when the incumbent is stuck at a small number of violated constraints that
  // neither bounds-propagation nor FJ can close, free the integer variables of the violated
  // constraints, fix the rest, and solve the resulting small sub-MIP exactly (dual-simplex B&B)
  // with a strict time budget.
  // Disabled by default: with the incumbent's values fixed, the residual sub-MIP is almost
  // always infeasible on the tested instances (the fixed part is itself part of the
  // infeasibility), so the dual-simplex B&B finds no solution and only wastes time. The 1-hop
  // related-variable neighborhood expansion did not change this. Kept behind a flag for future
  // iteration (e.g. relaxing fixed constraints / smarter free-set selection).
  static constexpr bool sub_mip_repair_enabled                = false;
  static constexpr size_t sub_mip_repair_unsat_threshold      = 16;
  static constexpr size_t sub_mip_repair_max_free_vars        = 400;
  static constexpr double sub_mip_repair_time_limit           = 1.5;
  static constexpr size_t sub_mip_repair_min_attempts_between = 30;
  // The sub-MIP repair rebuilds the full (bounds-fixed) problem on the host for dual-simplex
  // branch and bound, so its per-call cost scales with the *whole* problem size, not the freed
  // subset. On large models a single call can burn many seconds, so only enable it for problems
  // small enough that the host build + B&B is cheap.
  static constexpr size_t sub_mip_repair_max_constraints  = 8000;
  static constexpr size_t sub_mip_repair_max_problem_vars = 15000;
};

struct ls_recombiner_config_t {
  // line segment related configs
  // FIXME: not implemented yet
  static constexpr bool use_fj_for_rounding            = false;
  static constexpr int n_points_to_search              = 20;
  static constexpr double time_limit                   = 2.;
  static constexpr size_t initial_n_of_vars_from_other = 200;
  static constexpr size_t max_different_var_limit      = 10000;
  static constexpr size_t min_different_var_limit      = 20;
  static size_t max_n_of_vars_from_other;
  static constexpr double n_var_ratio_increase_factor = 1.1;
  static constexpr double n_var_ratio_decrease_factor = 0.99;
  static void increase_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::min(
      static_cast<size_t>(std::ceil(max_n_of_vars_from_other * n_var_ratio_increase_factor)),
      max_different_var_limit);
    CUOPT_LOG_DEBUG("Increased max_n_of_vars_from_other in LS recombiner to %lu",
                    max_n_of_vars_from_other);
  }
  static void decrease_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::max(
      static_cast<size_t>(std::floor(max_n_of_vars_from_other * n_var_ratio_decrease_factor)),
      min_different_var_limit);
    CUOPT_LOG_DEBUG("Decreased max_n_of_vars_from_other in LS recombiner to %lu",
                    max_n_of_vars_from_other);
  }
};

struct fp_recombiner_config_t {
  static constexpr double infeasibility_detection_time_limit = 0.05;
  static constexpr double fp_time_limit                      = 2.;
  static constexpr double alpha                              = 0.99;
  static constexpr double alpha_decrease_factor              = 0.9;
  static constexpr size_t initial_n_of_vars_from_other       = 200;
  static constexpr size_t max_different_var_limit            = 10000;
  static constexpr size_t min_different_var_limit            = 20;
  static size_t max_n_of_vars_from_other;
  static constexpr double n_var_ratio_increase_factor = 1.1;
  static constexpr double n_var_ratio_decrease_factor = 0.99;
  static void increase_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::min(
      static_cast<size_t>(std::ceil(max_n_of_vars_from_other * n_var_ratio_increase_factor)),
      max_different_var_limit);
    CUOPT_LOG_DEBUG("Increased max_n_of_vars_from_other in FP recombiner to %lu",
                    max_n_of_vars_from_other);
  }
  static void decrease_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::max(
      static_cast<size_t>(std::floor(max_n_of_vars_from_other * n_var_ratio_decrease_factor)),
      min_different_var_limit);
    CUOPT_LOG_DEBUG("Decreased max_n_of_vars_from_other in FP recombiner to %lu",
                    max_n_of_vars_from_other);
  }
};

struct sub_mip_recombiner_config_t {
  static constexpr size_t max_continuous_vars                = 5000;
  static constexpr double sub_mip_time_limit                 = 2.;
  static constexpr double infeasibility_detection_time_limit = 0.05;
  static constexpr size_t initial_n_of_vars_from_other       = 40;
  static constexpr size_t max_different_var_limit            = 500;
  static constexpr size_t min_different_var_limit            = 10;
  static size_t max_n_of_vars_from_other;
  static constexpr double n_var_ratio_increase_factor = 1.1;
  static constexpr double n_var_ratio_decrease_factor = 0.99;
  static void increase_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::min(
      static_cast<size_t>(std::ceil(max_n_of_vars_from_other * n_var_ratio_increase_factor)),
      max_different_var_limit);
    CUOPT_LOG_DEBUG("Increased max_n_of_vars_from_other in SUB_MIP recombiner to %lu",
                    max_n_of_vars_from_other);
  }
  static void decrease_max_n_of_vars_from_other()
  {
    max_n_of_vars_from_other = std::max(
      static_cast<size_t>(std::floor(max_n_of_vars_from_other * n_var_ratio_decrease_factor)),
      min_different_var_limit);
    CUOPT_LOG_DEBUG("Decreased max_n_of_vars_from_other in SUB_MIP recombiner to %lu",
                    max_n_of_vars_from_other);
  }
};

}  // namespace cuopt::linear_programming::detail
