/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/local_search/line_segment_search/line_segment_search.cuh>
#include <mip_heuristics/local_search/rounding/constraint_prop.cuh>
#include <mip_heuristics/relaxed_lp/batched_relaxed_lp.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <utilities/timer.hpp>

#include <rmm/device_uvector.hpp>

#include <deque>
#include <memory>
#include <random>
#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class population_t;

struct batched_fp_config_t {
  int batch_size                        = 8;
  double alpha                          = 0.99;
  double alpha_decrease_factor          = 0.9;
  int max_iterations                    = 1000000;
  int cycle_detection_window            = 30;
  double cycle_distance_reduction_ratio = 0.1;
  double perturbation_ratio             = 0.1;
};

/**
 * Multi-path feasibility pump that runs N projection-round cycles in parallel,
 * leveraging a batched PDLP solver (SPMM-based) where the constraint matrix A
 * is shared across all subproblems.
 *
 * Each path independently:
 *  - builds a distance-minimization objective toward its own rounding
 *  - solves via batched LP (shared A, per-path obj/bounds)
 *  - rounds the projection to integers
 *  - detects cycles via distance history
 *
 * Feasible solutions from any path are submitted to the population immediately.
 * The best candidate gets a final FJ repair pass.
 */
template <typename i_t, typename f_t>
class batched_feasibility_pump_t {
 public:
  batched_feasibility_pump_t() = delete;
  batched_feasibility_pump_t(mip_solver_context_t<i_t, f_t>& context,
                             fj_t<i_t, f_t>& fj,
                             constraint_prop_t<i_t, f_t>& constraint_prop,
                             line_segment_search_t<i_t, f_t>& line_segment_search,
                             rmm::device_uvector<f_t>& lp_optimal_solution);

  /**
   * Main entry point. Runs the multi-path FP loop until a feasible solution
   * is found or the timer expires.
   * @return true if a feasible solution was found (written into solution)
   */
  bool run(solution_t<i_t, f_t>& solution, timer_t timer, population_t<i_t, f_t>* population_ptr);

  void reset();
  void resize_vectors(problem_t<i_t, f_t>& problem, const raft::handle_t* handle_ptr);

  batched_fp_config_t config;
  cuopt::timer_t timer;

  void initialize_paths(solution_t<i_t, f_t>& solution);

  /**
   * Build the augmented problem (original + distance variables/constraints) once.
   * The constraint matrix A is fixed for the entire FP run; only per-path
   * objectives, bounds, and RHS change between iterations.
   */
  void build_augmented_problem(solution_t<i_t, f_t>& solution);

  /**
   * Fill per-path objectives, bounds, RHS, and initial primals into a
   * batched_lp_input_t. Does NOT rebuild the constraint matrix.
   */
  batched_lp_input_t<i_t, f_t> update_batched_input(solution_t<i_t, f_t>& solution);

  void extract_projections(const batched_lp_output_t<i_t, f_t>& output, i_t n_orig_vars);

  void round_all_paths(solution_t<i_t, f_t>& solution);

  std::vector<bool> check_path_cycles(solution_t<i_t, f_t>& solution);

  void replace_cycling_paths(solution_t<i_t, f_t>& solution, const std::vector<bool>& cycling_mask);

  i_t check_and_submit_feasible(solution_t<i_t, f_t>& solution,
                                population_t<i_t, f_t>* population_ptr);

  bool try_fj_repair(solution_t<i_t, f_t>& solution, i_t path_idx);

  mip_solver_context_t<i_t, f_t>& context;
  fj_t<i_t, f_t>& fj;
  constraint_prop_t<i_t, f_t>& constraint_prop;
  line_segment_search_t<i_t, f_t>& line_segment_search;
  rmm::device_uvector<f_t>& lp_optimal_solution;

  // Per-path state: column-major [batch_size x n_vars]
  // Element (path k, var j) is at index [k + j * batch_size]
  rmm::device_uvector<f_t> path_roundings;
  rmm::device_uvector<f_t> path_projections;

  // Per-path distance history for cycle detection
  std::vector<std::deque<f_t>> path_distances;

  // Per-path alpha for objective blending
  std::vector<double> path_alphas;

  // Augmented problem built once per run() call, reused across iterations.
  // Contains the original constraints + distance linearization constraints.
  std::unique_ptr<problem_t<i_t, f_t>> augmented_problem_;
  // Map from integer index (in h_integer_indices) to distance variable id, or -1
  std::vector<i_t> dist_var_id_;
  // Number of original variables (before augmentation)
  i_t n_orig_vars_{0};
  // Index in augmented problem where distance constraints start
  i_t dist_cstr_base_{0};

  std::mt19937 rng;
};

}  // namespace cuopt::linear_programming::detail
