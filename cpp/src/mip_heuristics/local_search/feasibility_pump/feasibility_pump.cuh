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

#include <deque>
#include <memory>

namespace cuopt::mathematical_optimization::mip {

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

// Target range for the batched-PDLP feasibility-pump cloud size. compute_optimal_batch_size is
// clamped into [min, max]; if it falls below fallback_threshold we fall back to single-point FP.
struct fp_batch_config_t {
  int target_min_batch_size = 8;
  int target_max_batch_size = 2048;
  // Extra cap on cloud size for per-iteration latency (projection + rounding wall clock).
  int latency_max_batch_size   = 64;
  int fallback_threshold       = 8;
  double fj_seed_time_ratio    = 0.2;  // 20% FJ run to seed the cloud trajectory
  double projection_time_limit = 1.0;
  // Cap on the fraction of the cloud filled with reseed (previous-iteration projected) points, so
  // fresh FJ-trajectory points and perturbed-LP-optimal padding always contribute diversity.
  double reseed_fraction = 0.8;
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

  // ---- Batched-PDLP feasibility pump (cloud projection) ----
  // Outer-loop entry point: projects a cloud of integer points simultaneously onto the LP polytope
  // with batched PDLP, collapses to one point (min L1, tie-break max integer count) for
  // round/cycle-break, and reseeds the next cloud. Falls back to run_single_fp_descent when the
  // memory-aware batch size is too small. Returns true if a feasible solution was found.
  bool run_batched_fp_cloud(solution_t<i_t, f_t>& solution);
  // Builds the fixed unified projection problem once (one aux distance var + 2 abs-value
  // constraints per integer variable). Per-climber variation is in the added constraints'
  // lower bounds (+/- val_j) and per-climber alpha-blended objectives.
  void build_unified_projection_problem(solution_t<i_t, f_t>& solution);
  // Pre-expands per-climber PDLP problem fields and warm-start buffers to batch_capacity (call once
  // after compute_cloud_batch_size).
  void expand_unified_projection_batch_buffers(solution_t<i_t, f_t>& solution, i_t batch_capacity);
  // Shrinks pre-expanded batch views when OOM retry runs fewer climbers than cloud_batch_capacity.
  void ensure_batch_problem_views(solution_t<i_t, f_t>& solution, i_t try_n);
  // Memory-aware cloud size via compute_optimal_batch_size, clamped into [target_min, target_max];
  // 0 means "fall back to single".
  i_t compute_cloud_batch_size(solution_t<i_t, f_t>& solution);
  // Assembles up to batch_size integer cloud points into d_cloud (concatenated [batch_size *
  // n_variables]); returns the number of distinct points actually written.
  i_t assemble_cloud(solution_t<i_t, f_t>& solution,
                     i_t batch_size,
                     bool first_iteration,
                     rmm::device_uvector<f_t>& d_cloud,
                     bool& seed_found_feasible);
  // Runs the batch projection for the assembled cloud and writes the per-climber projected primals
  // back into d_projected. On OOM, halves n_points and retries; throws if batch size 1 still fails.
  void project_cloud(solution_t<i_t, f_t>& solution,
                     i_t& n_points,
                     const rmm::device_uvector<f_t>& d_cloud,
                     rmm::device_uvector<f_t>& d_projected);
  // Selects the best projected point (min L1 distance to its seed, tie-break max integer count)
  // among climbers [start_climber, n_points) and copies it into solution.assignment. Returns the
  // selected climber index. start_climber = 1 skips climber 0 (the dedicated classic-FP
  // trajectory).
  i_t select_cloud_point(solution_t<i_t, f_t>& solution,
                         i_t n_points,
                         const rmm::device_uvector<f_t>& d_cloud,
                         const rmm::device_uvector<f_t>& d_projected,
                         i_t start_climber = 0);
  // Runs the original single-point FP logic for one round on climber 0's projection (already in
  // solution.assignment): distance-cycle check, full-integer + near-feasible LP-verify, CP round,
  // then the 20% FJ fallback. Uses the shared FP trajectory state (last_rounding, last_projection,
  // last_distances, alpha, cycle_queue). Returns whether climber 0 reached feasibility; sets
  // climber0_cycle when climber 0 cycled and should be restarted.
  bool run_climber0_step(solution_t<i_t, f_t>& solution,
                         f_t proj_begin,
                         bool& climber0_cycle,
                         i_t fj_traj_capacity = 0);
  void advance_diversity_climbers_gpu(solution_t<i_t, f_t>& solution,
                                      i_t n_points,
                                      const rmm::device_uvector<f_t>& d_projected,
                                      rmm::device_uvector<f_t>& d_seeds,
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
  // Fill flagged diversity slots from climber 0's most recent FJ trajectory (same run as the 20%
  // fallback). Returns how many flagged climbers received a trajectory point.
  i_t reseed_flagged_from_fj_trajectory(solution_t<i_t, f_t>& solution,
                                        i_t n_points,
                                        rmm::device_uvector<f_t>& d_cloud,
                                        const std::vector<char>& flagged);
  // Sequentially rounds one projected climber to integers using the probing cache to resolve
  // conflicts (inspired by constraint_prop's bulk rounding) with NO bound propagation: only the
  // precomputed probing cache is consulted, and its implied bounds accumulate across variables in
  // the h_lb/h_ub scratch (giving the sequential behavior). Integer columns of out_assignment
  // (host, size n_variables) are overwritten; continuous columns are left untouched.
  void probing_cache_sequential_round(solution_t<i_t, f_t>& solution,
                                      const f_t* h_projection,
                                      std::vector<f_t>& h_lb,
                                      std::vector<f_t>& h_ub,
                                      f_t* out_assignment);
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
  i_t unified_n_int          = 0;  // number of integer variables (== number of aux distance vars)
  i_t unified_n_vars         = 0;  // original n_variables (without aux distance vars)
  i_t unified_n_vars_total   = 0;  // original + aux distance vars
  i_t unified_n_constr       = 0;  // original n_constraints (without abs-value constraints)
  i_t unified_n_constr_total = 0;  // original + 2 * n_int abs-value constraints
  i_t cloud_batch_capacity   = 0;  // per-climber PDLP buffers expanded to this many climbers
  // Host scratch for per-climber objectives [cloud_batch_capacity * unified_n_vars_total].
  std::vector<f_t> h_batch_obj;
  // Pre-allocated warm-start / projection buffers (sized once in
  // expand_unified_projection_batch_buffers).
  rmm::device_uvector<f_t> batch_primal_init;
  rmm::device_uvector<f_t> batch_dual_init;
  // Host copies needed to rebuild per-iteration objective / per-climber constraint bounds.
  std::vector<i_t> h_integer_indices_cache;  // integer variable column indices
  std::vector<f_t> h_base_constraint_lower;  // original constraint lower bounds
  std::vector<f_t> h_base_constraint_upper;  // original constraint upper bounds
  std::vector<f_t> h_var_lower;              // original variable lower bounds
  std::vector<f_t> h_var_upper;              // original variable upper bounds
  i_t reseed_count = 0;
  // L1 distance of the last selected projected cloud point to its seed (logged per trajectory).
  f_t last_selected_l1 = 0.;
  f_t best_excess;
  rmm::device_uvector<f_t>& lp_optimal_solution;
  std::mt19937 rng;
  std::deque<f_t> last_distances;
  f_t last_lp_time;
  f_t total_fp_time_until_cycle;
  f_t fp_fj_cycle_time_begin;
  f_t proj_and_round_time;
  f_t proj_begin;
  i_t n_fj_single_descents;
  i_t max_n_of_integers = 0;
  cuopt::timer_t timer;
  // Promising points carried from the previous iteration (nearest-rounded projected cloud),
  // concatenated [reseed_count * n_variables]. Declared last so its stream-aware initialization
  // ordering in the constructor is unambiguous.
  rmm::device_uvector<f_t> reseed_points;
  // Per-climber alpha for the distance/original-objective blend in batch projection. Slot 0 mirrors
  // config.alpha; reset to default_alpha when a diversity climber is freshly seeded (FJ trajectory
  // or perturbed LP-optimal padding).
  std::vector<f_t> climber_alphas;
  // Per-climber PDLP dual warm start carried across projections within a single batched FP descent:
  // the previous projection's dual [warm_start_n_points * unified_n_constr_total], the previous
  // best (selected) climber index, and how many leading climbers of the current cloud are
  // carried-over reseed points (so they can reuse their own previous dual; the rest reuse the best
  // climber's dual). The primal is always seeded from the current point, so it is not stored.
  // Reset (warm_start_n_points = 0) whenever the unified problem is rebuilt or a new descent
  // starts.
  rmm::device_uvector<f_t> warm_start_dual;
  i_t warm_start_n_points   = 0;
  i_t warm_start_best_c     = 0;
  i_t n_carried_over_points = 0;
  // Climber 0's integer ratio from its previous projection (fraction of integer vars that are
  // integral). Drives the batch projection's LP tolerance via get_tolerance_from_ratio, mirroring
  // the original single-point FP: loose early (ratio low), tightening as climber 0 nears integral.
  f_t climber0_int_ratio = 0.;

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
