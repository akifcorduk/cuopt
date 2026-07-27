/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/local_search/line_segment_search/line_segment_search.cuh>
#include <mip_heuristics/local_search/rounding/constraint_prop.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <utilities/timer.hpp>

#include <cuopt/mathematical_optimization/optimization_problem.hpp>

#include <thrust/count.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class population_t;

constexpr double default_alpha                  = 0.99;
constexpr double distance_to_check_for_feasible = 0.01;

template <typename i_t, typename f_t>
struct cycle_queue_t {
  cycle_queue_t(problem_t<i_t, f_t>& problem, i_t cycle_len = 30)
    : cycle_detection_length(cycle_len), curr_recent_sol(cycle_detection_length - 1)
  {
    for (i_t i = 0; i < cycle_detection_length; ++i) {
      recent_solutions.emplace_back(
        rmm::device_uvector<f_t>(problem.n_variables, problem.handle_ptr->get_stream()));
    }
  }

  bool check_same_solution(solution_t<i_t, f_t>& solution,
                           rmm::device_uvector<f_t>& recent_solution)
  {
    const f_t* other_ptr        = recent_solution.data();
    const f_t* curr_assignement = solution.assignment.data();
    i_t n_equal_integers        = thrust::count_if(
      solution.handle_ptr->get_thrust_policy(),
      solution.problem_ptr->integer_indices.begin(),
      solution.problem_ptr->integer_indices.end(),
      [other_ptr, curr_assignement, pb = solution.problem_ptr->view()] __device__(i_t idx) {
        return pb.integer_equal(other_ptr[idx], curr_assignement[idx]);
      });
    return n_equal_integers == solution.problem_ptr->n_integer_vars;
  }

  bool check_cycle(solution_t<i_t, f_t>& solution)
  {
    for (i_t i = 0; i < cycle_detection_length; ++i) {
      i_t sol_idx  = (curr_recent_sol + cycle_detection_length - i) % cycle_detection_length;
      bool is_same = check_same_solution(solution, recent_solutions[sol_idx]);
      if (is_same) {
        CUOPT_LOG_DEBUG(
          "Detected same solution recent order %d",
          (curr_recent_sol + cycle_detection_length - sol_idx) % cycle_detection_length);
        return true;
      }
    }
    return false;
  }

  void update_recent_solutions(solution_t<i_t, f_t>& solution)
  {
    curr_recent_sol++;
    if (curr_recent_sol == cycle_detection_length) { curr_recent_sol = 0; }
    // update current integer solution
    raft::copy(recent_solutions[curr_recent_sol].data(),
               solution.assignment.data(),
               solution.problem_ptr->n_variables,
               solution.handle_ptr->get_stream());
  }

  void reset(solution_t<i_t, f_t>& solution)
  {
    for (i_t i = 0; i < cycle_detection_length; ++i) {
      recent_solutions[i].resize(solution.problem_ptr->n_variables,
                                 solution.handle_ptr->get_stream());
      thrust::fill(solution.handle_ptr->get_thrust_policy(),
                   recent_solutions[i].begin(),
                   recent_solutions[i].end(),
                   NAN);
    }
  }

  std::vector<rmm::device_uvector<f_t>> recent_solutions;
  const i_t cycle_detection_length;
  i_t curr_recent_sol;
  i_t n_iterations_without_cycle = 0;
};

struct fp_config_t {
  double alpha                           = default_alpha;
  double alpha_decrease_factor           = 0.9;
  bool check_distance_cycle              = true;
  int first_stage_kk                     = 70;
  double cycle_distance_reduction_ration = 0.1;
};

enum class fp_batched_exit_t : int { feasible, climber_cycle, batch_exhausted, time_limit };

struct fp_batched_result_t {
  bool feasible;
  fp_batched_exit_t exit;
  int trajectories;
};

// Target range for the batched-PDLP feasibility-pump cloud size. compute_optimal_batch_size is
// clamped into [min, max]; if it falls below fallback_threshold we fall back to single-point FP.
struct fp_batch_config_t {
  int target_min_batch_size           = 8;
  int target_max_batch_size           = 8;
  int latency_max_batch_size          = 8;
  int fallback_threshold              = 8;
  int max_trajectories_before_restart = 8;
  int phase_restart_trajectory_limit  = 8;
  int max_work_without_feasible       = 8;
  int work_limit_min_variables        = 50000;
  bool restart_batch_exhausted        = true;
  bool restart_batch_before_feasible  = false;
  bool phase_restart_large_pure_only  = false;
  bool skip_restart_for_large_pure    = true;
  bool use_trajectory_stagnation      = false;
  bool reset_stage_state              = false;
  int objective_cut_mode              = 1;
  int quality_config_id               = 0;
  int diversity_frequency             = 1;
  int structural_selector             = 0;
  bool adaptive_cloud                 = false;
  bool feedback_cloud                 = false;
  bool legacy_diversity               = false;
  double fj_ratio                     = 0.2;  // 20% FJ run

  bool operator==(const fp_batch_config_t&) const = default;
};

constexpr int n_fp_quality_configs      = 9;
constexpr int default_fp_quality_config = 8;

int resolve_fp_quality_config_id(const char* config);
fp_batch_config_t make_fp_batch_config(int quality_config_id);

struct fp_structure_metrics_t {
  double unified_growth;
  double nonbinary_integer_share;
};

inline fp_structure_metrics_t compute_fp_structure_metrics(int n_variables,
                                                           int n_constraints,
                                                           int n_integer_variables,
                                                           int n_nonbinary_integer_auxiliaries)
{
  cuopt_assert(n_variables >= 0 && n_constraints >= 0 && n_integer_variables >= 0,
               "FP structure counts must be nonnegative");
  cuopt_assert(
    n_nonbinary_integer_auxiliaries >= 0 && n_nonbinary_integer_auxiliaries <= n_integer_variables,
    "FP auxiliary count must describe non-binary integers");
  const int base_size = n_variables + n_constraints;
  return {
    (double)(base_size + 3 * n_nonbinary_integer_auxiliaries) / (double)std::max(1, base_size),
    (double)n_nonbinary_integer_auxiliaries / (double)std::max(1, n_integer_variables)};
}

inline bool fp_prefers_classic_single(const fp_structure_metrics_t& metrics)
{
  return metrics.unified_growth > 1.75 || metrics.nonbinary_integer_share > 0.40;
}

inline bool fp_feedback_regressed(double adjusted_violation,
                                  double previous_adjusted_violation,
                                  int pdhg_work,
                                  int previous_pdhg_work)
{
  return (std::isfinite(previous_adjusted_violation) &&
          adjusted_violation > previous_adjusted_violation * 1.25) ||
         (previous_pdhg_work > 0 && pdhg_work > previous_pdhg_work + previous_pdhg_work / 2);
}

inline int reduce_fp_cloud_after_regression(int cloud_cap)
{
  cuopt_assert(cloud_cap >= 1, "FP cloud cap must be positive");
  return std::max(1, cloud_cap / 2);
}

inline bool fp_should_restart_batch(bool restart_batch_exhausted,
                                    bool restart_batch_before_feasible,
                                    bool phase_restart_large_pure_only,
                                    bool any_feasible,
                                    bool large_pure_integer,
                                    bool skip_restart_for_large_pure)
{
  return (restart_batch_exhausted && !(large_pure_integer && skip_restart_for_large_pure)) ||
         (restart_batch_before_feasible && !any_feasible &&
          (!phase_restart_large_pure_only || large_pure_integer));
}

inline int fp_trajectory_limit(int regular_limit,
                               int phase_limit,
                               bool phase_restart_large_pure_only,
                               bool large_pure_integer)
{
  cuopt_assert(regular_limit > 0 && phase_limit > 0, "FP trajectory limits must be positive");
  return phase_restart_large_pure_only && large_pure_integer ? phase_limit : regular_limit;
}

template <typename i_t, typename f_t>
struct fp_iteration_metrics_t {
  i_t iteration                       = 0;
  i_t outer_iteration                 = -1;
  i_t batch_size                      = 1;
  i_t projected_integers              = 0;
  i_t projection_violated_constraints = 0;
  f_t projection_l1_distance          = 0.;
  f_t projection_total_violation      = 0.;
  f_t projection_adjusted_violation   = 0.;
  f_t projection_objective            = 0.;
  double projection_time              = 0.;
  i_t pdlp_iterations                 = -1;
  i_t pdhg_iterations                 = -1;
  i_t projection_termination_status   = -1;
  f_t pdlp_primal_residual            = std::numeric_limits<f_t>::quiet_NaN();
  f_t pdlp_dual_residual              = std::numeric_limits<f_t>::quiet_NaN();
  f_t pdlp_gap                        = std::numeric_limits<f_t>::quiet_NaN();
  double batch_mean_pdlp_iterations   = -1.;
  i_t batch_max_pdlp_iterations       = -1;
  i_t rounded_integers                = 0;
  i_t rounding_violated_constraints   = 0;
  f_t rounding_l1_movement            = 0.;
  f_t rounding_total_violation        = 0.;
  f_t rounding_adjusted_violation     = 0.;
  double rounding_time                = 0.;
  bool projection_feasible            = false;
  bool rounding_feasible              = false;
  bool cycle                          = false;
  i_t cycle_kind                      = 0;
  bool perturbed                      = false;
  bool timed_out                      = false;
  bool feasible                       = false;
  i_t diversity_feasible_candidates   = 0;
  i_t diversity_best_updates          = 0;
  i_t diversity_published             = 0;
  i_t effective_cloud_size            = 1;
  i_t diversity_frequency             = 1;
  double diversity_postprocess_time   = 0.;
  i_t climber1_feasible_candidates    = 0;
  i_t climber1_best_updates           = 0;
};

template <typename i_t, typename f_t>
struct fp_outer_iteration_metrics_t {
  i_t iteration           = 0;
  i_t trajectories        = 0;
  i_t exit_reason         = -1;
  bool feasible_return    = false;
  bool objective_improved = false;
  bool restarted          = false;
  bool recombiner_run     = false;
  f_t best_objective      = std::numeric_limits<f_t>::infinity();
  f_t objective_cut_rhs   = std::numeric_limits<f_t>::infinity();
  double elapsed          = 0.;
};

template <typename i_t, typename f_t>
struct fp_run_metrics_t {
  std::vector<fp_iteration_metrics_t<i_t, f_t>> iterations;
  std::vector<fp_outer_iteration_metrics_t<i_t, f_t>> outer_iterations;
  i_t feasible_events = 0;
  double total_time   = 0.;
};

template <typename i_t, typename f_t>
class feasibility_pump_t {
 public:
  feasibility_pump_t() = delete;
  feasibility_pump_t(mip_solver_context_t<i_t, f_t>& context,
                     fj_t<i_t, f_t>& fj,
                     //                     fj_tree_t<i_t, f_t>& fj_tree_,
                     constraint_prop_t<i_t, f_t>& constraint_prop_,
                     line_segment_search_t<i_t, f_t>& line_segment_search_,
                     rmm::device_uvector<f_t>& lp_optimal_solution_);

  void adjust_objective_with_original(solution_t<i_t, f_t>& solution,
                                      std::vector<f_t>& dist_objective,
                                      bool longer_lp_run = false);
  bool linear_project_onto_polytope(solution_t<i_t, f_t>& solution,
                                    f_t proximity_to_polytope,
                                    bool longer_lp_run = false);

  void perturbate(solution_t<i_t, f_t>& solution);
  bool run_fj_cycle_escape(solution_t<i_t, f_t>& solution);
  bool run_single_fp_descent(solution_t<i_t, f_t>& solution);

  fp_batched_result_t run_batched_fp_cloud(solution_t<i_t, f_t>& solution);
  // Builds the fixed unified projection problem once. Binary distance is represented directly in
  // each climber's objective; non-binary integers use one aux distance var and 2 abs-value rows.
  void build_unified_projection_problem(solution_t<i_t, f_t>& solution);
  // Pre-expands per-climber PDLP problem fields and warm-start buffers to batch_capacity (call once
  // after compute_cloud_batch_size).
  void expand_unified_projection_batch_buffers(solution_t<i_t, f_t>& solution, i_t batch_capacity);
  // Shrinks pre-expanded batch views when OOM retry runs fewer climbers than cloud_batch_capacity.
  void ensure_batch_problem_views(solution_t<i_t, f_t>& solution, i_t try_n);
  // Memory-aware cloud size via compute_optimal_batch_size, clamped into [target_min, target_max];
  // 0 means "fall back to single".
  i_t compute_cloud_batch_size(solution_t<i_t, f_t>& solution);
  fp_batched_result_t run_batched_fp_cloud_descent(solution_t<i_t, f_t>& solution,
                                                   i_t batch_size,
                                                   rmm::device_uvector<f_t>& d_batch_assignments,
                                                   std::vector<char>& flagged,
                                                   std::vector<size_t>& climber_hashes);
  // Runs the batch projection in place: d_batch_assignments is integer-side input on entry and
  // projected primal output on success. On OOM, halves n_points and retries; if even batch size 1
  // returns no usable solution it sets n_points = 0 so the caller falls back to a single rounding
  // step.
  void project_cloud(solution_t<i_t, f_t>& solution,
                     i_t& n_points,
                     rmm::device_uvector<f_t>& d_batch_assignments,
                     i_t trajectory);
  // Runs the original single-point FP logic for one round on climber 0's projection (already in
  // solution.assignment): distance-cycle check, full-integer + near-feasible LP-verify, CP round,
  // then the 20% FJ fallback. Uses the shared FP trajectory state (last_rounding, last_projection,
  // last_distances, alpha, cycle_queue). Returns whether climber 0 reached feasibility; sets
  // climber0_cycle when climber 0 cycled and should be restarted.
  bool run_climber0_step(solution_t<i_t, f_t>& solution,
                         f_t proj_begin,
                         bool& climber0_cycle,
                         i_t batch_size);
  void advance_diversity_climbers_gpu(solution_t<i_t, f_t>& solution,
                                      i_t n_points,
                                      rmm::device_uvector<f_t>& d_batch_assignments,
                                      std::vector<char>& flagged,
                                      std::vector<size_t>& climber_hashes);
  void seed_cloud_from_assignment_gpu(solution_t<i_t, f_t>& solution,
                                      i_t n_points,
                                      rmm::device_uvector<f_t>& d_cloud);
  void replace_flagged_climbers_diverse(solution_t<i_t, f_t>& solution,
                                        i_t n_points,
                                        rmm::device_uvector<f_t>& d_cloud,
                                        const std::vector<char>& flagged,
                                        i_t& n_diverse,
                                        i_t& n_fallback);
  // Host-side constraint-activity feasibility filter for an already-integer assignment (uses the
  // cached CSR). A cheap pre-filter; device compute_feasibility confirms the hits.
  bool host_assignment_feasible(const f_t* h_assignment);

  bool round(solution_t<i_t, f_t>& solution, bool update_last_rounding = true);
  bool handle_cycle(solution_t<i_t, f_t>& solution);
  bool restart_fp(solution_t<i_t, f_t>& solution);
  bool test_number_all_integer(solution_t<i_t, f_t>& solution);
  bool check_distance_cycle(solution_t<i_t, f_t>& solution);
  void reset();
  void resize_vectors(problem_t<i_t, f_t>& problem, const raft::handle_t* handle_ptr);
  bool random_round_with_fj(solution_t<i_t, f_t>& solution, timer_t& round_timer);
  bool round_multiple_points(solution_t<i_t, f_t>& solution);
  void relax_general_integers(solution_t<i_t, f_t>& solution);
  void revert_relaxation(solution_t<i_t, f_t>& solution);
  bool test_fj_feasible(solution_t<i_t, f_t>& solution,
                        f_t time_limit,
                        i_t trajectory_capacity = 0);
  void record_projection_metrics(solution_t<i_t, f_t>& solution,
                                 i_t n_integers,
                                 i_t batch_size,
                                 double elapsed);
  void set_projection_solver_metrics(i_t pdlp_iterations,
                                     i_t pdhg_iterations,
                                     i_t termination_status,
                                     f_t primal_residual,
                                     f_t dual_residual,
                                     f_t gap,
                                     double batch_mean_pdlp_iterations,
                                     i_t batch_max_pdlp_iterations);
  void record_rounding_metrics(solution_t<i_t, f_t>& solution, bool is_feasible, double elapsed);
  void finish_iteration_metrics(bool cycle, bool timed_out, bool feasible);

  mip_solver_context_t<i_t, f_t>& context;
  // keep a reference from upstream local search
  fj_t<i_t, f_t>& fj;
  // fj_tree_t<i_t, f_t>& fj_tree;
  line_segment_search_t<i_t, f_t>& line_segment_search;
  cycle_queue_t<i_t, f_t> cycle_queue;
  constraint_prop_t<i_t, f_t>& constraint_prop;
  fp_config_t config;
  rmm::device_uvector<f_t> last_rounding;
  rmm::device_uvector<f_t> last_projection;
  rmm::device_uvector<var_t> orig_variable_types;

  // ---- Batched-PDLP feasibility pump state ----
  fp_batch_config_t batch_config;
  // Cached unified projection problem (fixed structure across climbers and outer iterations).
  std::unique_ptr<cuopt::mathematical_optimization::optimization_problem_t<i_t, f_t>>
    unified_problem;
  i_t unified_n_int           = 0;  // number of integer variables
  i_t unified_n_aux           = 0;  // non-binary integers represented by auxiliary distance vars
  i_t unified_n_vars          = 0;  // original n_variables (without aux distance vars)
  i_t unified_n_vars_total    = 0;  // original + aux distance vars
  i_t unified_n_constr        = 0;  // original n_constraints (without abs-value constraints)
  i_t unified_n_constr_total  = 0;  // original + 2 * n_aux abs-value constraints
  i_t cloud_batch_capacity    = 0;  // per-climber PDLP buffers expanded to this many climbers
  i_t cached_cloud_batch_size = -1;
  i_t structural_cloud_cap    = 1;
  i_t feedback_cloud_cap      = 1;
  i_t cloud_invocation        = 0;
  f_t previous_climber0_adjusted_violation = std::numeric_limits<f_t>::infinity();
  i_t previous_climber0_pdhg_work          = -1;
  // Pre-allocated warm-start / projection buffers (sized once in
  // expand_unified_projection_batch_buffers).
  rmm::device_uvector<f_t> batch_primal_init;
  rmm::device_uvector<i_t> d_aux_integer_indices_cache;
  // Host copies needed to rebuild per-iteration objective / per-climber constraint bounds.
  std::vector<i_t> h_integer_indices_cache;      // integer variable column indices
  std::vector<i_t> h_aux_integer_indices_cache;  // non-binary integer column indices
  std::vector<f_t> h_base_constraint_lower;      // original constraint lower bounds
  std::vector<f_t> h_base_constraint_upper;      // original constraint upper bounds
  std::vector<f_t> h_var_lower;                  // original variable lower bounds
  std::vector<f_t> h_var_upper;                  // original variable upper bounds
  f_t best_excess;
  rmm::device_uvector<f_t>& lp_optimal_solution;
  std::mt19937 diversity_rng;
  std::deque<f_t> last_distances;
  f_t last_lp_time;
  f_t total_fp_time_until_cycle;
  f_t fp_fj_cycle_time_begin;
  f_t proj_and_round_time;
  f_t proj_begin;
  i_t n_fj_single_descents;
  i_t max_n_of_integers = 0;
  cuopt::timer_t timer;
  fp_run_metrics_t<i_t, f_t>* metrics    = nullptr;
  i_t current_outer_iteration            = -1;
  i_t last_pdlp_iterations               = -1;
  i_t last_pdhg_iterations               = -1;
  i_t last_projection_status             = -1;
  f_t last_pdlp_primal_residual          = std::numeric_limits<f_t>::quiet_NaN();
  f_t last_pdlp_dual_residual            = std::numeric_limits<f_t>::quiet_NaN();
  f_t last_pdlp_gap                      = std::numeric_limits<f_t>::quiet_NaN();
  double last_batch_mean_pdlp_iterations = -1.;
  i_t last_batch_max_pdlp_iterations     = -1;
  // Per-climber alpha for the distance/original-objective blend in batch projection. Slot 0 mirrors
  // config.alpha; reset to default_alpha when a diversity climber is freshly seeded (FJ trajectory
  // or perturbed LP-optimal padding).
  std::vector<f_t> climber_alphas;
  // Climber 0's integer ratio from its previous projection (fraction of integer vars that are
  // integral). Still recorded but no longer read: the batch projection runs every climber at the
  // full absolute tolerance instead of loosening it early via get_tolerance_from_ratio.
  f_t climber0_int_ratio                 = 0.;
  population_t<i_t, f_t>* population_ptr = nullptr;

  // ---- Per-climber persistent diversity trajectories (cloud slots 1..N) ----
  // Ring buffer of recent integer-rounding hashes per climber for integer-assignment cycle
  // detection; a diversity climber is replaced when its new rounding repeats one of these.
  std::vector<std::deque<size_t>> climber_hash_history;
  // Host copy of the (presolved) problem CSR, cached for the host-side feasibility filter of the
  // diversity climbers' integer roundings.
  std::vector<i_t> h_csr_offsets;
  std::vector<i_t> h_csr_indices;
  std::vector<f_t> h_csr_values;
};

}  // namespace cuopt::mathematical_optimization::mip
