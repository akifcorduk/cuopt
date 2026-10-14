/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace raft {
class handle_t;
}

namespace cuopt::mathematical_optimization {
template <typename i_t, typename f_t>
class mip_solver_settings_t;

namespace simplex {
template <typename i_t, typename f_t>
struct lp_problem_t;
enum class variable_type_t : int8_t;
}  // namespace simplex

namespace mip {

// Named experiment settings shared by integrated workers and direct-job replay.
// Rounding uses a repair/backtracking budget followed by a total propagation cap.
struct gpu_submip_fp_config_t {
  std::string name{"baseline"};
  double initial_alpha{0.99};
  double pdlp_seconds{1.};
  double rounding_seconds{0.5};
  double propagation_seconds{0.7};

  bool operator==(const gpu_submip_fp_config_t&) const = default;
};

gpu_submip_fp_config_t gpu_submip_fp_config_from_name(std::string_view name);
// CUOPT_CONFIG_ID selects a preset modulo four, validated against CUOPT_MAX_CONFIG (default four).
// Numeric IDs override the legacy named environment selector; neither changes the solver seed.
gpu_submip_fp_config_t gpu_submip_fp_config_from_environment();

struct gpu_submip_fp_run_info_t {
  bool started{false};
  int projections{0};
  double descent_seconds{0.};
  double projection_seconds{0.};
  const char* exit_reason{"not_started"};
};

// One private GPU FP session for a sub-MIP's fixed equality/slack representation.
// Construct and destroy on the worker thread with its GPU context current. The
// handle must outlive the session. Recreate the session if matrix, RHS, objective,
// variable types, or column/row identities change. Compaction removes columns
// fixed at construction; their bounds must remain fixed at those values. Bounds
// of retained columns may change freely. Public vectors retain full coordinates.
template <typename i_t, typename f_t>
class gpu_submip_fp_t {
 public:
  gpu_submip_fp_t(const raft::handle_t* handle,
                  const simplex::lp_problem_t<i_t, f_t>& lp,
                  const std::vector<simplex::variable_type_t>& variable_types,
                  const mip_solver_settings_t<i_t, f_t>& settings,
                  bool compact_fixed                   = true,
                  bool fused_adaptive_step             = false,
                  const gpu_submip_fp_config_t& config = {});
  ~gpu_submip_fp_t();

  gpu_submip_fp_t(const gpu_submip_fp_t&)            = delete;
  gpu_submip_fp_t& operator=(const gpu_submip_fp_t&) = delete;

  // Host-only check for reusing storage and previous projection/PDLP seeds.
  // Equality includes matrix, RHS, objective and types, not just dimensions.
  bool can_reuse(const simplex::lp_problem_t<i_t, f_t>& node_lp,
                 const std::vector<simplex::variable_type_t>& variable_types) const;

  // Export the last finite projection (or ending assignment) in full coordinates.
  // The caller can perturb it for a new descent without rebuilding the GPU model.
  bool get_restart_seed(std::vector<f_t>& seed) const;
  gpu_submip_fp_run_info_t last_run_info() const;
  const gpu_submip_fp_config_t& config() const;

  // cutoff and candidate objective use c^T x in the local minimization space,
  // excluding the simplex objective constant. Runs one descent without restarts.
  bool run(const simplex::lp_problem_t<i_t, f_t>& node_lp,
           const std::vector<f_t>& node_primal,
           f_t improving_cutoff,
           double remaining_time,
           std::atomic<int>* halt,
           std::vector<f_t>& candidate,
           bool use_cached_primal       = false,
           const char* primal_seed_name = "node_primal",
           uint64_t job_id              = 0,
           int projection_limit         = 0);

 private:
  struct impl_t;
  std::unique_ptr<impl_t> impl_;
};

}  // namespace mip
}  // namespace cuopt::mathematical_optimization
