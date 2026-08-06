/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/deterministic_calibrator/bp_harness.hpp>
#include <mip_heuristics/deterministic_calibrator/work_features.hpp>

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
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
#include <tuple>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip::calib {

std::vector<std::string> bp_feature_names()
{
  auto names = base_features_t::names();
  names.push_back("max_row_nnz");              // static: deepest constraint row (reduction depth)
  names.push_back("n_changed_constraints");    // dynamic: changed constraints (block launches)
  names.push_back("changed_constraints_nnz");  // dynamic: total nnz of the changed set
  names.push_back("changed_warp_loads");       // dynamic: sum ceil(rowsize/32) over changed set
  return names;
}

namespace {

constexpr int kRepeats  = 5;
constexpr int kMaxIters = 256;  // bound presolve converges well before this on these instances

}  // namespace

bp_samples_t run_bp_calibration_samples(const std::string& mps_path,
                                        const std::string& instance_name)
{
  const raft::handle_t handle_{};
  auto mps_problem = cuopt::mathematical_optimization::io::read_mps<int, double>(mps_path, false);
  handle_.sync_stream();
  // Intentionally heap-allocated and leaked: destroying this optimization_problem_t after a
  // mip_solver_t has been constructed segfaults in rmm::device_buffer::deallocate_async (an RMM
  // stream/MR teardown-ordering issue exposed on newer CUDA). This is an offline calibration tool
  // that processes a fixed instance list and exits, so leaking one problem per instance is fine and
  // lets the multi-instance run complete.
  auto* op_problem_ptr = new auto(mps_data_model_to_optimization_problem(&handle_, mps_problem));
  auto& op_problem     = *op_problem_ptr;

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
  double max_row_nnz = 0.0;
  for (std::size_t r = 0; r + 1 < h_offsets.size(); ++r) {
    h_row_size[r] = static_cast<double>(h_offsets[r + 1] - h_offsets[r]);
    max_row_nnz   = std::max(max_row_nnz, h_row_size[r]);
  }
  bound_presolve_t<int, double> bp(solver.context);
  bp.resize(*pb);
  bp.settings.iteration_limit = kMaxIters;
  bp.settings.time_limit      = std::numeric_limits<double>::infinity();

  // Group iterations by their distinct dynamic-feature point (n_changed_constraints,
  // changed_constraints_nnz). A converging loop revisits the same state many times (e.g. a long
  // tail of identical iterations); each DISTINCT (features -> time) observation should weigh once,
  // not once per visit, otherwise one instance's tail dominates the fit.
  // Per distinct dynamic-feature point: collect activity-kernel and bounds-update-kernel times
  // separately (each timed with its own sync barriers) so the two sub-models can be fit
  // independently.
  struct point_times_t {
    std::vector<double> activity;
    std::vector<double> update;
    long max_changed_row{0};  // longest row in the changed set (warp serialization depth)
  };
  std::map<std::tuple<long, long, long>, point_times_t> by_point;
  for (int rep = 0; rep < kRepeats; ++rep) {
    bp.copy_input_bounds(*pb);
    bp.upd.init_changed_constraints(pb->handle_ptr);

    for (int iter = 0; iter < kMaxIters; ++iter) {
      // Dynamic features = state of the changed-constraints mask at the start of this iteration.
      auto h_changed = cuopt::host_copy(bp.upd.changed_constraints, stream);
      pb->handle_ptr->sync_stream();
      long n_changed   = 0;
      long changed_nnz = 0;
      long warp_loads  = 0;  // sum ceil(rowsize/32): per-row reduction depth at warp granularity
      long max_changed_row = 0;  // longest changed row: serialized by ~one warp -> latency-bound
      for (std::size_t c = 0; c < h_changed.size(); ++c) {
        if (h_changed[c] != 0) {
          const long rs = (long)h_row_size[c];
          n_changed += 1;
          changed_nnz += rs;
          warp_loads += (rs + 31) / 32;
          max_changed_row = std::max(max_changed_row, rs);
        }
      }

      pb->handle_ptr->sync_stream();
      auto t0 = std::chrono::steady_clock::now();
      bp.calculate_activity(*pb);
      pb->handle_ptr->sync_stream();
      auto t1      = std::chrono::steady_clock::now();
      bool updated = bp.calculate_bounds_update(*pb);
      pb->handle_ptr->sync_stream();
      auto t2 = std::chrono::steady_clock::now();

      auto& pt = by_point[{n_changed, changed_nnz, warp_loads}];
      pt.activity.push_back(std::chrono::duration<double>(t1 - t0).count());
      pt.update.push_back(std::chrono::duration<double>(t2 - t1).count());
      pt.max_changed_row = max_changed_row;

      if (!updated) { break; }
      bp.upd.prepare_for_next_iteration(pb->handle_ptr);
    }
  }

  bp_samples_t out;
  int point_idx = 0;
  for (auto& [key, pt] : by_point) {
    std::sort(pt.activity.begin(), pt.activity.end());
    std::sort(pt.update.begin(), pt.update.end());
    const double med_act = pt.activity.empty() ? 0.0 : pt.activity[pt.activity.size() / 2];
    const double med_upd = pt.update.empty() ? 0.0 : pt.update[pt.update.size() / 2];
    if (med_act <= 0.0 || med_upd <= 0.0) { continue; }
    const long n_changed   = std::get<0>(key);
    const long changed_nnz = std::get<1>(key);
    const long warp_loads  = std::get<2>(key);

    std::vector<double> features;
    bf.append_to(features);
    features.push_back(max_row_nnz);
    features.push_back((double)n_changed);
    features.push_back((double)changed_nnz);
    features.push_back((double)warp_loads);

    const std::string tag = instance_name + "#p" + std::to_string(point_idx++);
    // Serialization term = the problem's static deepest row (max_row_nnz). The runtime work model
    // can supply this cheaply (work_features.max_row_nnz); the dynamic longest-changed-row is not
    // available per iteration without an extra reduction, so fit and runtime both use the static
    // value to keep the device coefficients transferable.
    calibration_sample_t a;
    a.instance               = tag;
    a.features               = features;
    a.measured_time_per_iter = med_act;
    a.max_row_nnz            = max_row_nnz;
    out.activity.push_back(std::move(a));

    calibration_sample_t u;
    u.instance               = tag;
    u.features               = std::move(features);
    u.measured_time_per_iter = med_upd;
    u.max_row_nnz            = max_row_nnz;
    out.update.push_back(std::move(u));

    std::printf(
      "    [%s] changed_cons=%ld changed_nnz=%ld warp_loads=%ld -> act %.3e s upd %.3e s\n",
      instance_name.c_str(),
      n_changed,
      changed_nnz,
      warp_loads,
      med_act,
      med_upd);
    std::fflush(stdout);
  }
  return out;
}

}  // namespace cuopt::mathematical_optimization::mip::calib
