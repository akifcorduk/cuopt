/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/deterministic_calibrator/bp_harness.hpp>
#include <mip_heuristics/deterministic_calibrator/work_features.hpp>

#include <cuopt/linear_programming/io/parser.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mip_heuristics/presolve/bounds_presolve.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solver.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <raft/core/handle.hpp>

#include <thrust/reduce.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace cuopt::linear_programming::detail::calib {

std::vector<std::string> bp_feature_names()
{
  auto names = base_features_t::names();
  names.push_back("n_changed_constraints");    // dynamic: changed constraints this iteration
  names.push_back("changed_constraints_nnz");  // dynamic: row-size-weighted changed set
  return names;
}

namespace {

constexpr int kRepeats  = 5;
constexpr int kMaxIters = 256;  // bound presolve converges well before this on these instances

}  // namespace

std::vector<calibration_sample_t> run_bp_calibration_samples(const std::string& mps_path,
                                                             const std::string& instance_name)
{
  const raft::handle_t handle_{};
  auto mps_problem = cuopt::linear_programming::io::read_mps<int, double>(mps_path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);

  problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();

  mip_solver_settings_t<int, double> settings;
  settings.heuristics_only = true;
  cuopt::timer_t solver_timer(std::numeric_limits<double>::infinity());
  mip_solver_t<int, double> solver(problem, settings, solver_timer);
  auto* pb    = solver.context.problem_ptr;
  auto stream = pb->handle_ptr->get_stream();

  // ---- structural features (post-preprocess) ----
  auto h_offsets     = cuopt::host_copy(pb->offsets, stream);
  auto h_rev_offsets = cuopt::host_copy(pb->reverse_offsets, stream);
  pb->handle_ptr->sync_stream();

  base_features_t bf;
  bf.n_vars        = static_cast<double>(pb->n_variables);
  bf.n_constraints = static_cast<double>(pb->n_constraints);
  bf.nnz           = static_cast<double>(pb->nnz);
  bf.row_nnz_var   = row_nnz_mean_var(h_offsets).second;
  bf.col_nnz_var   = row_nnz_mean_var(h_rev_offsets).second;

  std::vector<double> h_row_size(h_offsets.size() > 0 ? h_offsets.size() - 1 : 0);
  for (std::size_t r = 0; r + 1 < h_offsets.size(); ++r) {
    h_row_size[r] = static_cast<double>(h_offsets[r + 1] - h_offsets[r]);
  }

  bound_presolve_t<int, double> bp(solver.context);
  bp.resize(*pb);
  bp.settings.iteration_limit = kMaxIters;
  bp.settings.time_limit      = std::numeric_limits<double>::infinity();

  // Group iterations by their distinct dynamic-feature point (n_changed_constraints,
  // changed_constraints_nnz). A converging loop revisits the same state many times (e.g. a long
  // tail of identical iterations); each DISTINCT (features -> time) observation should weigh once,
  // not once per visit, otherwise one instance's tail dominates the fit.
  std::map<std::pair<long, long>, std::vector<double>> by_point;
  for (int rep = 0; rep < kRepeats; ++rep) {
    bp.copy_input_bounds(*pb);
    bp.upd.init_changed_constraints(pb->handle_ptr);

    for (int iter = 0; iter < kMaxIters; ++iter) {
      // Dynamic features = state of the changed-constraints mask at the start of this iteration.
      auto h_changed = cuopt::host_copy(bp.upd.changed_constraints, stream);
      pb->handle_ptr->sync_stream();
      long n_changed   = 0;
      long changed_nnz = 0;
      for (std::size_t c = 0; c < h_changed.size(); ++c) {
        if (h_changed[c] != 0) {
          n_changed += 1;
          changed_nnz += (long)h_row_size[c];
        }
      }

      pb->handle_ptr->sync_stream();
      auto t0 = std::chrono::steady_clock::now();
      bp.calculate_activity(*pb);
      bool updated = bp.calculate_bounds_update(*pb);
      pb->handle_ptr->sync_stream();
      auto t1           = std::chrono::steady_clock::now();
      const double secs = std::chrono::duration<double>(t1 - t0).count();

      by_point[{n_changed, changed_nnz}].push_back(secs);

      if (!updated) { break; }
      bp.upd.prepare_for_next_iteration(pb->handle_ptr);
    }
  }

  std::vector<calibration_sample_t> samples;
  int point_idx = 0;
  for (auto& [key, times] : by_point) {
    std::sort(times.begin(), times.end());
    const double median = times.empty() ? 0.0 : times[times.size() / 2];
    if (median <= 0.0) { continue; }

    std::vector<double> features;
    bf.append_to(features);
    features.push_back((double)key.first);   // n_changed_constraints
    features.push_back((double)key.second);  // changed_constraints_nnz

    calibration_sample_t s;
    s.instance               = instance_name + "#p" + std::to_string(point_idx++);
    s.features               = std::move(features);
    s.measured_time_per_iter = median;
    samples.push_back(std::move(s));

    std::printf("    [%s] changed_cons=%ld changed_nnz=%ld -> %.3e s (x%zu)\n",
                instance_name.c_str(),
                key.first,
                key.second,
                median,
                times.size());
    std::fflush(stdout);
  }
  return samples;
}

}  // namespace cuopt::linear_programming::detail::calib
