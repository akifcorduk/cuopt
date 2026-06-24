/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// NOTE (calibration finding): multi_probe does NOT reach the <10% per-iteration CV target; the best
// achieved is ~0.28 combined. This is not a feature-set problem -- it was tried with:
//   (1) per-kernel split + constraint-warp saturation hinge            -> CV 0.280
//   (2) per-kernel split + changed-variable features (2D hinge)        -> CV 0.296
//   (3) single model on total per-iteration time + 2D hinge           -> CV 0.284
// Tracking changed variables (this file) fixed the large reduction-bound instances (bab2#p3 0.95,
// nw04 0.96, uccase9 ok) but the remaining residual is dual-probe per-iteration heterogeneity:
// multi_probe runs two probes (upd_0, upd_1) through a SINGLE joint kernel launch, but the probes
// converge at different rates, so within one instance the per-iteration cost varies ~3x (e.g.
// neos-3004026-krka: 6.5e-5 vs 2.04e-4 s/iter). Aggregate (summed-over-both-probes) features
// predict a middle value -> cheap iterations over-predicted (~0.7x), expensive ones under-predicted
// (~2x). Because the kernels time both probes in one launch, the per-probe contributions can't be
// separated without kernel-level instrumentation. Practical runtime fallback: record multi_probe
// work using the bound-presolve model on the combined changed set (captures the large instances
// that dominate the budget). Infra kept for a future per-probe-instrumented attempt.

#include <mip_heuristics/deterministic_calibrator/mp_harness.hpp>
#include <mip_heuristics/deterministic_calibrator/work_features.hpp>

#include <cuopt/linear_programming/io/parser.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mip_heuristics/presolve/multi_probe.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solver.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <raft/core/handle.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

namespace cuopt::linear_programming::detail::calib {

std::vector<std::string> mp_feature_names()
{
  auto names = base_features_t::names();
  names.push_back("max_row_nnz");
  names.push_back("n_changed_cons");
  names.push_back("changed_cons_nnz");
  names.push_back("changed_cons_warp_loads");  // activity hinge feature
  names.push_back("n_changed_vars");
  names.push_back("changed_vars_warp_loads");  // update hinge feature
  return names;
}

std::size_t mp_cons_warp_index() { return base_features_t::size() + 3; }  // changed_cons_warp_loads
std::size_t mp_vars_warp_index() { return base_features_t::size() + 5; }  // changed_vars_warp_loads

namespace {

constexpr int kRepeats   = 5;
constexpr int kMaxIters  = 256;
constexpr int kMaxProbes = 64;  // number of integer variables probed simultaneously

}  // namespace

bp_samples_t run_mp_calibration_samples(const std::string& mps_path,
                                        const std::string& instance_name)
{
  const raft::handle_t handle_{};
  auto mps_problem = cuopt::linear_programming::io::read_mps<int, double>(mps_path, false);
  handle_.sync_stream();
  // Leaked on purpose: destroying this after a mip_solver_t exists segfaults in RMM async free
  // (teardown-order issue on newer CUDA). Offline tool with a fixed instance list -> leak is fine.
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

  auto h_offsets     = cuopt::host_copy(pb->offsets, stream);
  auto h_rev_offsets = cuopt::host_copy(pb->reverse_offsets, stream);
  auto h_int_idx     = cuopt::host_copy(pb->integer_indices, stream);
  auto h_bounds      = cuopt::host_copy(pb->variable_bounds, stream);
  pb->handle_ptr->sync_stream();

  if (h_int_idx.empty()) { return {}; }

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
  // Column (variable) warp-load weights for the bounds-update kernel: ceil(col_degree/32).
  std::vector<long> h_col_warps(h_rev_offsets.size() > 0 ? h_rev_offsets.size() - 1 : 0);
  for (std::size_t v = 0; v + 1 < h_rev_offsets.size(); ++v) {
    h_col_warps[v] = ((h_rev_offsets[v + 1] - h_rev_offsets[v]) + 31) / 32;
  }

  // Build a batch of two-valued probes: probe0 fixes var to its lower bound, probe1 to its upper
  // (or lb+1 if upper is unbounded), over the first kMaxProbes integer variables.
  std::vector<int> p_vars;
  std::vector<double> p_v0, p_v1;
  const int n_probe = std::min<int>(kMaxProbes, (int)h_int_idx.size());
  for (int k = 0; k < n_probe; ++k) {
    const int v     = h_int_idx[k];
    const double lb = get_lower(h_bounds[v]);
    const double ub = get_upper(h_bounds[v]);
    double v0       = std::isfinite(lb) ? lb : 0.0;
    double v1       = (std::isfinite(ub) && ub > v0) ? ub : v0 + 1.0;
    p_vars.push_back(v);
    p_v0.push_back(v0);
    p_v1.push_back(v1);
  }
  auto var_probe_vals = std::make_tuple(p_vars, p_v0, p_v1);

  multi_probe_t<int, double> mp(solver.context);
  mp.resize(*pb);
  mp.settings.iteration_limit = kMaxIters;
  mp.settings.time_limit      = std::numeric_limits<double>::infinity();
  mp.compute_stats            = false;

  struct point_times_t {
    std::vector<double> activity;
    std::vector<double> update;
  };
  // Key: (n_changed_cons, changed_cons_nnz, changed_cons_warp, n_changed_vars, changed_vars_warp).
  std::map<std::tuple<long, long, long, long, long>, point_times_t> by_point;

  auto count_changed_cons = [&](const std::vector<int>& mask, long& n, long& nnz, long& warps) {
    for (std::size_t c = 0; c < mask.size() && c < h_row_size.size(); ++c) {
      if (mask[c] != 0) {
        const long rs = (long)h_row_size[c];
        n += 1;
        nnz += rs;
        warps += (rs + 31) / 32;
      }
    }
  };
  // Count variables whose bounds changed between pre/post snapshots (a probe), and their column
  // warp-loads. Drives the bounds-update kernel cost.
  auto count_changed_vars = [&](const std::vector<double>& lb0,
                                const std::vector<double>& ub0,
                                const std::vector<double>& lb1,
                                const std::vector<double>& ub1,
                                long& n,
                                long& warps) {
    for (std::size_t v = 0; v < lb0.size() && v < h_col_warps.size(); ++v) {
      if (lb0[v] != lb1[v] || ub0[v] != ub1[v]) {
        n += 1;
        warps += h_col_warps[v];
      }
    }
  };

  for (int rep = 0; rep < kRepeats; ++rep) {
    mp.copy_problem_into_probing_buffers(*pb, pb->handle_ptr);
    mp.set_bounds(var_probe_vals, pb->handle_ptr);
    mp.upd_0.init_changed_constraints(pb->handle_ptr);
    mp.upd_1.init_changed_constraints(pb->handle_ptr);
    mp.skip_0 = false;
    mp.skip_1 = false;

    for (int iter = 0; iter < kMaxIters; ++iter) {
      auto h0 = cuopt::host_copy(mp.upd_0.changed_constraints, stream);
      auto h1 = cuopt::host_copy(mp.upd_1.changed_constraints, stream);
      // pre-update bounds (activity does not modify bounds; only bounds-update does).
      auto lb0_pre = cuopt::host_copy(mp.upd_0.lb, stream);
      auto ub0_pre = cuopt::host_copy(mp.upd_0.ub, stream);
      auto lb1_pre = cuopt::host_copy(mp.upd_1.lb, stream);
      auto ub1_pre = cuopt::host_copy(mp.upd_1.ub, stream);
      pb->handle_ptr->sync_stream();
      long n_cons = 0, cons_nnz = 0, cons_warp = 0;
      count_changed_cons(h0, n_cons, cons_nnz, cons_warp);
      count_changed_cons(h1, n_cons, cons_nnz, cons_warp);

      pb->handle_ptr->sync_stream();
      auto t0 = std::chrono::steady_clock::now();
      mp.calculate_activity(*pb, pb->handle_ptr);
      pb->handle_ptr->sync_stream();
      auto t1      = std::chrono::steady_clock::now();
      bool updated = mp.calculate_bounds_update(*pb, pb->handle_ptr);
      pb->handle_ptr->sync_stream();
      auto t2 = std::chrono::steady_clock::now();

      auto lb0_post = cuopt::host_copy(mp.upd_0.lb, stream);
      auto ub0_post = cuopt::host_copy(mp.upd_0.ub, stream);
      auto lb1_post = cuopt::host_copy(mp.upd_1.lb, stream);
      auto ub1_post = cuopt::host_copy(mp.upd_1.ub, stream);
      pb->handle_ptr->sync_stream();
      long n_vars_changed = 0, vars_warp = 0;
      count_changed_vars(lb0_pre, ub0_pre, lb0_post, ub0_post, n_vars_changed, vars_warp);
      count_changed_vars(lb1_pre, ub1_pre, lb1_post, ub1_post, n_vars_changed, vars_warp);

      auto& pt = by_point[{n_cons, cons_nnz, cons_warp, n_vars_changed, vars_warp}];
      pt.activity.push_back(std::chrono::duration<double>(t1 - t0).count());
      pt.update.push_back(std::chrono::duration<double>(t2 - t1).count());

      if (!updated) { break; }
      if (!mp.skip_0) { mp.upd_0.prepare_for_next_iteration(pb->handle_ptr); }
      if (!mp.skip_1) { mp.upd_1.prepare_for_next_iteration(pb->handle_ptr); }
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

    std::vector<double> features;
    bf.append_to(features);
    features.push_back(max_row_nnz);
    features.push_back((double)std::get<0>(key));  // n_changed_cons
    features.push_back((double)std::get<1>(key));  // changed_cons_nnz
    features.push_back((double)std::get<2>(key));  // changed_cons_warp_loads
    features.push_back((double)std::get<3>(key));  // n_changed_vars
    features.push_back((double)std::get<4>(key));  // changed_vars_warp_loads

    const std::string tag = instance_name + "#p" + std::to_string(point_idx++);
    calibration_sample_t a;
    a.instance               = tag;
    a.features               = features;
    a.measured_time_per_iter = med_act;
    out.activity.push_back(std::move(a));
    calibration_sample_t u;
    u.instance               = tag;
    u.features               = std::move(features);
    u.measured_time_per_iter = med_upd;
    out.update.push_back(std::move(u));

    std::printf("    [%s] cons=%ld cons_warp=%ld vars=%ld vars_warp=%ld -> act %.3e upd %.3e\n",
                instance_name.c_str(),
                std::get<0>(key),
                std::get<2>(key),
                std::get<3>(key),
                std::get<4>(key),
                med_act,
                med_upd);
    std::fflush(stdout);
  }
  return out;
}

}  // namespace cuopt::linear_programming::detail::calib
