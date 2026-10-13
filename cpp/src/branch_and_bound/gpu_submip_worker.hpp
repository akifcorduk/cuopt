/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <branch_and_bound/constants.hpp>

#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <dual_simplex/presolve.hpp>
#include <mip_heuristics/gpu_submip_fp.hpp>

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
  int node_stride{8};
  int min_depth{0};
  int max_fractional{std::numeric_limits<int>::max()};
  int initial_calls{4};
  bool cached_primal{false};
  bool enable_fp{true};
  bool compact_fixed{true};
  bool fused_pdlp{true};
  double incumbent_blend{0.0};
  gpu_submip_fp_config_t fp_config;
};

gpu_submip_options_t gpu_submip_options_from_environment();

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

template <typename i_t, typename f_t>
struct gpu_submip_job_t {
  // All node data and the mapping closure are owned. A producer may return its worker to
  // the pool, change sibling bounds, or destroy its entire sub-MIP immediately after enqueue.
  simplex::lp_problem_t<i_t, f_t> lp;
  std::vector<simplex::variable_type_t> var_types;
  std::vector<f_t> primal;
  std::vector<f_t> incumbent;
  f_t cutoff;
  f_t lower_bound;
  uint64_t job_id{0};
  uint64_t source_id{0};
  i_t tree_node{0};
  i_t depth{0};
  i_t fractional{0};
  i_t fixed{0};
  gpu_submip_source_t source{gpu_submip_source_t::MAIN_TREE};
  search_strategy_t strategy{search_strategy_t::BEST_FIRST};
  double deadline{std::numeric_limits<double>::infinity()};
  std::function<bool(const std::vector<f_t>&, std::vector<f_t>&)> to_user;
};

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
                        const char* primal_seed_name_override = "perturbed_projection",
                        gpu_submip_fp_run_info_t* run_info    = nullptr);

}  // namespace cuopt::mathematical_optimization::mip
