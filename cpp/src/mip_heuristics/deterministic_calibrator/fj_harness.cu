/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/deterministic_calibrator/fj_harness.hpp>
#include <mip_heuristics/deterministic_calibrator/work_features.hpp>

#include <cuopt/linear_programming/io/parser.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/solver.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <raft/core/handle.hpp>

#include <thrust/fill.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <vector>

namespace cuopt::linear_programming::detail::calib {

std::vector<std::string> fj_feature_names()
{
  auto names = base_features_t::names();
  names.push_back("frontier_work_mean");  // FJ dynamic feature: avg per-step frontier work
  return names;
}

namespace {

// Configuration of the two-point timing measurement.
constexpr int kIterLow  = 2000;
constexpr int kIterHigh = 22000;
constexpr int kRepeats  = 5;

// Build solver + solution + FJ from a preprocessed problem and run FJ to `iter_limit` steps,
// timing only the solve. Returns {wall_seconds, steps_performed}.
std::pair<double, int> time_fj_run(problem_t<int, double>& problem,
                                   const mip_solver_settings_t<int, double>& settings,
                                   int iter_limit)
{
  cuopt::timer_t timer(std::numeric_limits<double>::infinity());
  mip_solver_t<int, double> solver(problem, settings, timer);

  solution_t<int, double> solution(*solver.context.problem_ptr);
  auto stream = solution.handle_ptr->get_stream();
  thrust::fill(solution.handle_ptr->get_thrust_policy(),
               solution.assignment.begin(),
               solution.assignment.end(),
               0.0);
  solution.clamp_within_bounds();

  fj_settings_t fj_settings;
  fj_settings.time_limit             = std::numeric_limits<double>::infinity();
  fj_settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = 20000 * 1000;
  fj_settings.update_weights         = true;
  fj_settings.feasibility_run        = false;
  fj_settings.iteration_limit        = iter_limit;

  fj_t<int, double> fj(solver.context, fj_settings);
  fj.reset_weights(stream, 1.0);
  double zero_obj_weight = 0.0;
  fj.objective_weight.set_value_async(zero_obj_weight, stream);
  solution.handle_ptr->sync_stream();

  auto t0 = std::chrono::steady_clock::now();
  fj.solve(solution);
  solution.handle_ptr->sync_stream();
  auto t1 = std::chrono::steady_clock::now();

  const double secs = std::chrono::duration<double>(t1 - t0).count();
  const int steps   = fj.climbers[0]->iterations.value(stream);
  return {secs, steps};
}

}  // namespace

calibration_sample_t run_fj_calibration_sample(const std::string& mps_path,
                                               const std::string& instance_name)
{
  const raft::handle_t handle_{};
  auto mps_problem = cuopt::linear_programming::io::read_mps<int, double>(mps_path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);

  problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();

  mip_solver_settings_t<int, double> settings;
  settings.heuristics_only                    = true;
  settings.heuristic_params.num_cpufj_threads = 0;

  // ---- structural features (post-preprocess) ----
  auto h_offsets     = cuopt::host_copy(problem.offsets, handle_.get_stream());
  auto h_rev_offsets = cuopt::host_copy(problem.reverse_offsets, handle_.get_stream());
  handle_.sync_stream();

  base_features_t bf;
  bf.n_vars        = static_cast<double>(problem.n_variables);
  bf.n_constraints = static_cast<double>(problem.n_constraints);
  bf.nnz           = static_cast<double>(problem.nnz);
  bf.row_nnz_var   = row_nnz_mean_var(h_offsets).second;      // variance of nnz per constraint (A)
  bf.col_nnz_var   = row_nnz_mean_var(h_rev_offsets).second;  // variance of nnz per variable (A^T)

  // FJ dynamic feature: expected per-step frontier work. Flipping a variable v scans, for each
  // constraint c containing v, the whole row of c. Averaged uniformly over variables this is
  // (sum_c rowsize_c^2) / n_vars. (Empirically a better per-step predictor than FJ's A^T*A*degree
  // estimator, which over-weights high-degree variables that FJ rarely flips.)
  double sum_sq_rowsize = 0.0;
  double max_row_nnz    = 0.0;
  for (std::size_t r = 0; r + 1 < h_offsets.size(); ++r) {
    const double rs = static_cast<double>(h_offsets[r + 1] - h_offsets[r]);
    sum_sq_rowsize += rs * rs;
    max_row_nnz = std::max(max_row_nnz, rs);
  }
  const double frontier_work_mean =
    problem.n_variables > 0 ? sum_sq_rowsize / static_cast<double>(problem.n_variables) : 0.0;

  std::vector<double> features;
  bf.append_to(features);
  features.push_back(frontier_work_mean);

  // ---- two-point timing ----
  std::vector<double> per_iter_samples;
  per_iter_samples.reserve(kRepeats);
  for (int rep = 0; rep < kRepeats; ++rep) {
    auto [t_lo, s_lo] = time_fj_run(problem, settings, kIterLow);
    auto [t_hi, s_hi] = time_fj_run(problem, settings, kIterHigh);
    double per_iter;
    if (s_hi - s_lo > 0) {
      per_iter = (t_hi - t_lo) / static_cast<double>(s_hi - s_lo);
    } else {
      per_iter = s_hi > 0 ? t_hi / static_cast<double>(s_hi) : 0.0;
    }
    if (per_iter > 0.0) { per_iter_samples.push_back(per_iter); }
    std::printf("    [%s] rep %d: lo(%d steps, %.4fs) hi(%d steps, %.4fs) -> %.3e s/step\n",
                instance_name.c_str(),
                rep,
                s_lo,
                t_lo,
                s_hi,
                t_hi,
                per_iter);
    std::fflush(stdout);
  }

  std::sort(per_iter_samples.begin(), per_iter_samples.end());
  const double median_per_iter =
    per_iter_samples.empty() ? 0.0 : per_iter_samples[per_iter_samples.size() / 2];

  calibration_sample_t sample;
  sample.instance               = instance_name;
  sample.features               = std::move(features);
  sample.measured_time_per_iter = median_per_iter;
  sample.max_row_nnz            = max_row_nnz;
  return sample;
}

}  // namespace cuopt::linear_programming::detail::calib
