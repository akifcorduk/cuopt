/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/deterministic_calibrator/repair_harness.hpp>
#include <mip_heuristics/deterministic_calibrator/work_features.hpp>

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <mip_heuristics/local_search/rounding/bounds_repair.cuh>
#include <mip_heuristics/presolve/bounds_presolve.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solver.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <raft/core/handle.hpp>

#include <thrust/for_each.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip::calib {

std::vector<std::string> repair_feature_names()
{
  auto names = base_features_t::names();
  names.push_back("n_candidates");      // compute_damages launches + sort
  names.push_back("cur_cstr_rowsize");  // compute_best_shift over the chosen constraint row
  names.push_back("total_warp_loads");  // full-matrix ii-violation pass (hinge feature, last)
  return names;
}

namespace {

constexpr int kRepeats  = 5;
constexpr int kMaxMoves = 400;

double median(std::vector<double> v)
{
  if (v.empty()) { return 0.0; }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

std::vector<calibration_sample_t> run_repair_calibration_sample(const std::string& mps_path,
                                                                const std::string& instance_name)
{
  const raft::handle_t handle_{};
  auto mps_problem = cuopt::mathematical_optimization::io::read_mps<int, double>(mps_path, false);
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

  auto h_offsets = cuopt::host_copy(pb->offsets, stream);
  pb->handle_ptr->sync_stream();
  base_features_t bf;
  bf.n_vars        = static_cast<double>(pb->n_variables);
  bf.n_constraints = static_cast<double>(pb->n_constraints);
  bf.nnz           = static_cast<double>(pb->nnz);
  bf.row_nnz_var   = row_nnz_mean_var(h_offsets).second;
  // total warp loads of the full ii-violation activity pass (sum ceil(rowsize/32) over
  // constraints).
  double total_warp_loads = 0.0;
  double max_row_nnz      = 0.0;
  for (std::size_t r = 0; r + 1 < h_offsets.size(); ++r) {
    const double rs = (double)(h_offsets[r + 1] - h_offsets[r]);
    total_warp_loads += (double)(((h_offsets[r + 1] - h_offsets[r]) + 31) / 32);
    max_row_nnz = std::max(max_row_nnz, rs);
  }
  {
    auto h_rev = cuopt::host_copy(pb->reverse_offsets, stream);
    pb->handle_ptr->sync_stream();
    bf.col_nnz_var = row_nnz_mean_var(h_rev).second;
  }

  // original problem keeps the true variable bounds (used by compute_best_shift / apply_move).
  problem_t<int, double> original(*pb);

  bound_presolve_t<int, double> bp(solver.context);
  bp.resize(*pb);
  bounds_repair_t<int, double> repair(*pb, bp);
  repair.handle_ptr = pb->handle_ptr;  // normally set inside repair_problem(); we drive the loop
  repair.timer      = cuopt::timer_t(std::numeric_limits<double>::infinity());
  repair.resize(*pb);

  // Per-move times bucketed by the dynamic feature point (n_candidates, chosen-constraint rowsize).
  // Emitting one sample per distinct point (instead of a single per-instance median) lets the
  // regression actually observe the per-move cost slope in n_candidates and rowsize, which are the
  // dynamic inputs the runtime repair work model varies. Mirrors the bp/mp per-point bucketing.
  std::map<std::pair<long, long>, std::vector<double>> by_point;
  long total_moves = 0;
  for (int rep = 0; rep < kRepeats; ++rep) {
    // Recreate the infeasible state: fix every integer variable to its lower bound.
    thrust::for_each(pb->handle_ptr->get_thrust_policy(),
                     pb->integer_indices.begin(),
                     pb->integer_indices.end(),
                     [vb = pb->variable_bounds.data()] __device__(int v) {
                       auto b = vb[v];
                       vb[v]  = {get_lower(b), get_lower(b)};
                     });
    pb->handle_ptr->sync_stream();

    repair.reset();
    double vio = repair.get_ii_violation(*pb);
    if (repair.h_n_violated_cstr == 0) { break; }  // not infeasible -> no repair work
    (void)vio;

    for (int mv = 0; mv < kMaxMoves && repair.h_n_violated_cstr > 0; ++mv) {
      auto t0       = std::chrono::steady_clock::now();
      int curr_cstr = repair.get_random_cstr();
      int n_cand    = repair.compute_best_shift(*pb, original, curr_cstr);
      if (n_cand == 0) {
        pb->handle_ptr->sync_stream();
        continue;
      }
      repair.compute_damages(*pb, n_cand);
      repair.apply_move(*pb, original, 0);
      repair.reset();
      repair.get_ii_violation(*pb);
      pb->handle_ptr->sync_stream();
      auto t1 = std::chrono::steady_clock::now();

      const long rs = (curr_cstr + 1 < (int)h_offsets.size())
                        ? (long)(h_offsets[curr_cstr + 1] - h_offsets[curr_cstr])
                        : 0L;
      by_point[{(long)n_cand, rs}].push_back(std::chrono::duration<double>(t1 - t0).count());
      ++total_moves;
    }
    // restore original bounds for the next repeat
    raft::copy(pb->variable_bounds.data(),
               original.variable_bounds.data(),
               pb->variable_bounds.size(),
               stream);
    pb->handle_ptr->sync_stream();
  }

  if (total_moves < 3) {
    std::printf(
      "    [%s] no/low repair activity (%ld moves) -> skip\n", instance_name.c_str(), total_moves);
    return {};
  }

  std::vector<calibration_sample_t> out;
  int point_idx = 0;
  for (auto& [key, ts] : by_point) {
    const double med = median(ts);
    if (med <= 0.0) { continue; }
    const double n_cand = (double)key.first;
    const double rs     = (double)key.second;
    std::vector<double> features;
    bf.append_to(features);
    features.push_back(n_cand);
    features.push_back(rs);
    features.push_back(total_warp_loads);

    calibration_sample_t s;
    s.instance               = instance_name + "#p" + std::to_string(point_idx++);
    s.features               = std::move(features);
    s.measured_time_per_iter = med;
    s.max_row_nnz            = max_row_nnz;
    out.push_back(std::move(s));
  }
  std::printf("    [%s] %zu distinct (n_cand,rowsize) points over %ld moves, warp=%.0f\n",
              instance_name.c_str(),
              out.size(),
              total_moves,
              total_warp_loads);
  std::fflush(stdout);
  return out;
}

}  // namespace cuopt::mathematical_optimization::mip::calib
