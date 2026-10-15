/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <branch_and_bound/constants.hpp>

#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <dual_simplex/presolve.hpp>
#include <mip_heuristics/gpu_submip_fp.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace cuopt {
class pcgenerator_t;
}

namespace cuopt::mathematical_optimization::mip {

enum class gpu_submip_mode_t { NODE_QUEUE, CPUFJ_FULL_GPU };

struct gpu_submip_options_t {
  gpu_submip_mode_t mode{gpu_submip_mode_t::NODE_QUEUE};
  double fix_rate{0.25};
  double time_limit{60.0};
  bool independent{true};
  int max_projections{20};  // Total across perturbed descents; zero disables the limit.
  int max_descents{0};      // Zero disables the limit; one runs a single descent.
  int node_stride{8};
  int min_depth{0};
  int max_fractional{std::numeric_limits<int>::max()};
  int initial_calls{4};
  bool cached_primal{false};
  bool select_warm_start{false};
  double unproductive_seconds{0.};  // Zero keeps uninterrupted sub-MIP ownership.
  double regular_gpu_seconds{5.};
  double pending_max_age{0.};  // Zero preserves the existing pending-job admission policy.
  double min_improve{0.};  // Fraction of the valid objective gap; zero retains strict cutoff only.
  bool enable_fp{true};
  bool compact_fixed{true};
  bool fused_pdlp{true};
  double incumbent_blend{0.0};
  gpu_submip_fp_config_t fp_config;
};

gpu_submip_options_t gpu_submip_options_from_environment();

// Observe completed jobs only. Tiny accepted updates do not reset the stall
// window; useful improvement restores the shortest regular-worker turn.
class gpu_submip_schedule_t {
 public:
  gpu_submip_schedule_t(double stall_seconds, double regular_seconds)
    : stall_seconds_(stall_seconds), regular_seconds_(regular_seconds), next_turn_(regular_seconds)
  {
  }
  void observe(double seconds, double relative_improvement)
  {
    if (relative_improvement >= 1e-6) {
      stalled_   = 0.;
      next_turn_ = regular_seconds_;
    } else {
      stalled_ += std::max(0., seconds);
    }
  }
  bool should_yield() const { return stall_seconds_ > 0 && stalled_ >= stall_seconds_; }
  double regular_turn()
  {
    const double seconds = next_turn_;
    next_turn_           = std::min(4 * regular_seconds_, 2 * next_turn_);
    stalled_             = 0.;
    return seconds;
  }
  double stalled_seconds() const { return stalled_; }

 private:
  double stall_seconds_;
  double regular_seconds_;
  double next_turn_;
  double stalled_{0.};
};

enum class gpu_submip_source_t { MAIN_TREE, CPU_SUBMIP, DIVE };
inline constexpr size_t gpu_submip_source_count = 3;

// Quick RINS/RENS/mutation DFS belongs to the outer B&B object, so inside_submip alone
// cannot identify the source. Ordinary diving keeps its separate main-model fixing policy.
inline gpu_submip_source_t gpu_submip_source(bool inside_submip, search_strategy_t strategy)
{
  if (inside_submip || strategy == search_strategy_t::RINS || strategy == search_strategy_t::RENS ||
      strategy == search_strategy_t::MUTATION) {
    return gpu_submip_source_t::CPU_SUBMIP;
  }
  return strategy == search_strategy_t::BEST_FIRST ? gpu_submip_source_t::MAIN_TREE
                                                   : gpu_submip_source_t::DIVE;
}

inline bool gpu_submip_source_applies_rins(gpu_submip_source_t source)
{
  return source != gpu_submip_source_t::CPU_SUBMIP;
}

const char* gpu_submip_source_name(gpu_submip_source_t source);
std::string gpu_submip_job_snapshot_path(gpu_submip_source_t source,
                                         const char* path,
                                         const char* source_filter,
                                         const char* directory);
// Outcome-independent collection: one job per solve-time/size stratum and source.
std::string gpu_submip_job_collection_path(gpu_submip_source_t source,
                                           const char* directory,
                                           double elapsed,
                                           double solve_seconds,
                                           int free_integers);

template <typename i_t, typename f_t>
struct gpu_submip_job_t {
  // All node data and the mapping closure are owned. A producer may return its worker to
  // the pool, change sibling bounds, or destroy its entire sub-MIP immediately after enqueue.
  simplex::lp_problem_t<i_t, f_t> lp;
  std::vector<simplex::variable_type_t> var_types;
  std::vector<f_t> primal;
  std::vector<f_t> incumbent;
  f_t cutoff;
  f_t proven_lower_bound{-std::numeric_limits<f_t>::infinity()};
  uint64_t job_id{0};
  uint64_t source_id{0};
  i_t tree_node{0};
  i_t depth{0};
  i_t fractional{0};
  i_t fixed{0};
  gpu_submip_source_t source{gpu_submip_source_t::MAIN_TREE};
  search_strategy_t strategy{search_strategy_t::BEST_FIRST};
  double deadline{std::numeric_limits<double>::infinity()};
  double enqueued_at{0.};
  double solve_start{0.};  // Shared outer-solve clock, used only by opt-in collection.
  std::function<bool(const std::vector<f_t>&, std::vector<f_t>&)> to_user;
};

template <typename f_t>
struct gpu_submip_cutoff_t {
  f_t target;
  const char* basis;
};

template <typename i_t, typename f_t>
gpu_submip_cutoff_t<f_t> gpu_submip_improving_cutoff(
  const gpu_submip_job_t<i_t, f_t>& job,
  f_t cutoff,
  const mip_solver_settings_t<i_t, f_t>& settings,
  const gpu_submip_options_t& options);

// Host-only, bounded history. Exact model/source matching keeps column identities
// intact; bounds may change because every selected point is clamped and rescored.
template <typename i_t, typename f_t>
class gpu_submip_seed_cache_t {
 public:
  struct entry_t {
    simplex::lp_problem_t<i_t, f_t> lp;
    std::vector<simplex::variable_type_t> types;
    uint64_t source_id;
    std::vector<f_t> projection;
    std::vector<f_t> candidate;
  };
  const entry_t* find(const gpu_submip_job_t<i_t, f_t>& job) const;
  void remember(const gpu_submip_job_t<i_t, f_t>& job,
                std::vector<f_t> projection,
                std::vector<f_t> candidate);

 private:
  std::vector<entry_t> entries_;
};

template <typename i_t, typename f_t>
struct gpu_submip_seed_t {
  std::vector<f_t> values;
  const char* name{"node_primal"};
};

template <typename i_t, typename f_t>
gpu_submip_seed_t<i_t, f_t> select_gpu_submip_seed(
  const gpu_submip_job_t<i_t, f_t>& job,
  f_t target,
  const mip_solver_settings_t<i_t, f_t>& settings,
  const gpu_submip_seed_cache_t<i_t, f_t>* history);

template <typename i_t, typename f_t>
gpu_submip_job_t<i_t, f_t> make_gpu_submip_job(
  const simplex::lp_problem_t<i_t, f_t>& lp,
  const std::vector<simplex::variable_type_t>& var_types,
  const std::vector<f_t>& primal,
  const std::vector<f_t>& incumbent,
  f_t cutoff,
  const gpu_submip_options_t& options,
  const mip_solver_settings_t<i_t, f_t>& settings,
  uint64_t job_id,
  bool apply_rins);

template <typename i_t, typename f_t>
bool gpu_submip_candidate_is_valid(const simplex::lp_problem_t<i_t, f_t>& lp,
                                   const std::vector<simplex::variable_type_t>& types,
                                   const std::vector<f_t>& candidate,
                                   const mip_solver_settings_t<i_t, f_t>& settings);

template <typename i_t, typename f_t>
class gpu_submip_fp_t;

// Clamp a finite full-coordinate seed, then perturb at most eight free integer
// coordinates by one integer step. The caller owns the independent restart RNG.
template <typename i_t, typename f_t>
i_t perturb_gpu_submip_restart_seed(const simplex::lp_problem_t<i_t, f_t>& lp,
                                    const std::vector<simplex::variable_type_t>& types,
                                    std::vector<f_t>& seed,
                                    pcgenerator_t& rng);

// Pure GPU FP execution: no CPU branch-and-bound, root LP, or node LP solve is launched here.
// A missing mapper publishes in job coordinates (direct replay). current_user_cutoff, when
// supplied, is in the original user objective space; it is refreshed before each queued job.
template <typename i_t, typename f_t>
bool run_gpu_submip_job(const gpu_submip_job_t<i_t, f_t>& job,
                        const mip_solver_settings_t<i_t, f_t>& settings,
                        const raft::handle_t* handle,
                        std::atomic<int>* halt,
                        const gpu_submip_options_t& options,
                        std::function<f_t()> current_user_cutoff,
                        std::function<void(const std::vector<f_t>&, f_t)> publish,
                        std::unique_ptr<gpu_submip_fp_t<i_t, f_t>>* session_cache = nullptr,
                        const std::vector<f_t>* primal_seed_override              = nullptr,
                        const char* primal_seed_name_override         = "perturbed_projection",
                        gpu_submip_fp_run_info_t* run_info            = nullptr,
                        int projection_limit                          = 0,
                        gpu_submip_seed_cache_t<i_t, f_t>* seed_cache = nullptr);

}  // namespace cuopt::mathematical_optimization::mip
