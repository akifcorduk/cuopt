/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/deterministic_calibrator/pdlp_harness.hpp>
#include <mip_heuristics/deterministic_calibrator/work_features.hpp>

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/problem/problem_helpers.cuh>
#include <mip_heuristics/utils.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <pdlp/pdlp.cuh>
#include <pdlp/solve.cuh>

#include <raft/core/handle.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <vector>

namespace cuopt::mathematical_optimization::mip::calib {

std::vector<std::string> pdlp_feature_names()
{
  auto names = base_features_t::names();
  names.push_back("row_warp_loads");    // sum ceil(rowsize/32): A*x SpMV depth
  names.push_back("col_warp_loads");    // sum ceil(colsize/32): A^T*y SpMV depth
  names.push_back("total_warp_loads");  // row+col warp loads (hinge feature)
  return names;
}

namespace {

constexpr int kIterLow  = 200;
constexpr int kIterHigh = 1200;
constexpr int kRepeats  = 5;

std::pair<double, int> time_pdlp_run(problem_t<int, double>& problem, int iter_limit)
{
  pdlp_solver_settings_t<int, double> s{};
  s.time_limit           = std::numeric_limits<double>::infinity();
  s.iteration_limit      = iter_limit;
  s.pdlp_solver_mode     = pdlp_solver_mode_t::Stable2;
  s.presolver            = presolver_t::None;
  s.detect_infeasibility = false;
  set_pdlp_solver_mode(s);
  // Unreachable tolerance so PDLP never converges early: the two-point measurement needs it to run
  // the full iteration_limit (otherwise s_hi == s_lo and the per-iter estimate is contaminated by
  // one-time setup).
  s.set_optimality_tolerance(1e-14);

  pdlp::pdlp_solver_t<int, double> lp_solver(problem, s);
  lp_solver.set_inside_mip(true);
  auto stream = problem.handle_ptr->get_stream();
  problem.handle_ptr->sync_stream();

  auto t0    = std::chrono::steady_clock::now();
  auto timer = timer_t(std::numeric_limits<double>::infinity());
  auto resp  = lp_solver.run_solver(timer);
  problem.handle_ptr->sync_stream();
  auto t1 = std::chrono::steady_clock::now();

  const double secs = std::chrono::duration<double>(t1 - t0).count();
  const int steps   = resp.get_additional_termination_information(0).number_of_steps_taken;
  (void)stream;
  return {secs, steps};
}

}  // namespace

calibration_sample_t run_pdlp_calibration_sample(const std::string& mps_path,
                                                 const std::string& instance_name)
{
  const raft::handle_t handle_{};
  auto mps_problem = cuopt::mathematical_optimization::io::read_mps<int, double>(mps_path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);

  problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  // PDLP expects <= rows; mirror the heuristic LP path.
  convert_greater_to_less(problem);

  auto stream        = problem.handle_ptr->get_stream();
  auto h_offsets     = cuopt::host_copy(problem.offsets, stream);
  auto h_rev_offsets = cuopt::host_copy(problem.reverse_offsets, stream);
  problem.handle_ptr->sync_stream();

  base_features_t bf;
  bf.n_vars        = static_cast<double>(problem.n_variables);
  bf.n_constraints = static_cast<double>(problem.n_constraints);
  bf.nnz           = static_cast<double>(problem.nnz);
  bf.row_nnz_var   = row_nnz_mean_var(h_offsets).second;
  bf.col_nnz_var   = row_nnz_mean_var(h_rev_offsets).second;

  double row_warp_loads = 0.0;
  double max_row_nnz    = 0.0;
  for (std::size_t r = 0; r + 1 < h_offsets.size(); ++r) {
    row_warp_loads += (double)(((h_offsets[r + 1] - h_offsets[r]) + 31) / 32);
    max_row_nnz = std::max(max_row_nnz, (double)(h_offsets[r + 1] - h_offsets[r]));
  }
  double col_warp_loads = 0.0;
  for (std::size_t v = 0; v + 1 < h_rev_offsets.size(); ++v) {
    col_warp_loads += (double)(((h_rev_offsets[v + 1] - h_rev_offsets[v]) + 31) / 32);
  }

  std::vector<double> features;
  bf.append_to(features);
  features.push_back(row_warp_loads);
  features.push_back(col_warp_loads);
  features.push_back(row_warp_loads + col_warp_loads);

  std::vector<double> per_iter;
  for (int rep = 0; rep < kRepeats; ++rep) {
    auto [t_lo, s_lo] = time_pdlp_run(problem, kIterLow);
    auto [t_hi, s_hi] = time_pdlp_run(problem, kIterHigh);
    double v;
    if (s_hi - s_lo > 0) {
      v = (t_hi - t_lo) / (double)(s_hi - s_lo);
    } else {
      v = s_hi > 0 ? t_hi / (double)s_hi : 0.0;
    }
    if (v > 0.0) { per_iter.push_back(v); }
    std::printf("    [%s] rep %d: lo(%d,%.4fs) hi(%d,%.4fs) -> %.3e s/iter\n",
                instance_name.c_str(),
                rep,
                s_lo,
                t_lo,
                s_hi,
                t_hi,
                v);
    std::fflush(stdout);
  }
  std::sort(per_iter.begin(), per_iter.end());
  const double median = per_iter.empty() ? 0.0 : per_iter[per_iter.size() / 2];

  calibration_sample_t s;
  s.instance               = instance_name;
  s.features               = std::move(features);
  s.measured_time_per_iter = median;
  s.max_row_nnz            = max_row_nnz;
  return s;
}

}  // namespace cuopt::mathematical_optimization::mip::calib
