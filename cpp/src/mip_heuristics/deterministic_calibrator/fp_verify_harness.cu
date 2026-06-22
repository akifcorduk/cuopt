/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/deterministic_calibrator/fp_verify_harness.hpp>

#include <cuopt/linear_programming/io/parser.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <mip_heuristics/problem/problem_helpers.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/solver.cuh>
#include <mip_heuristics/utils.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

#include <pdlp/pdlp.cuh>
#include <pdlp/solve.cuh>

#include <raft/core/handle.hpp>

#include <thrust/fill.h>

#include <chrono>
#include <cstdio>
#include <limits>
#include <vector>

namespace cuopt::linear_programming::detail::calib {

namespace {

struct pdlp_run_t {
  int steps;
  double obj;
  double secs;
};

pdlp_run_t run_pdlp_to_iters(problem_t<int, double>& problem, int iter_limit)
{
  pdlp_solver_settings_t<int, double> s{};
  s.time_limit           = std::numeric_limits<double>::infinity();
  s.iteration_limit      = iter_limit;
  s.pdlp_solver_mode     = pdlp_solver_mode_t::Stable2;
  s.presolver            = presolver_t::None;
  s.detect_infeasibility = false;
  set_pdlp_solver_mode(s);
  s.set_optimality_tolerance(1e-14);  // never converge early; run the full work budget

  pdlp_solver_t<int, double> lp(problem, s);
  lp.set_inside_mip(true);
  problem.handle_ptr->sync_stream();
  cuopt::timer_t lp_timer(std::numeric_limits<double>::infinity());
  auto t0   = std::chrono::steady_clock::now();
  auto resp = lp.run_solver(lp_timer);
  problem.handle_ptr->sync_stream();
  auto t1 = std::chrono::steady_clock::now();
  return {resp.get_additional_termination_information(0).number_of_steps_taken,
          resp.get_objective_value(),
          std::chrono::duration<double>(t1 - t0).count()};
}

}  // namespace

bool run_fp_verify(const std::string& mps_path, const std::string& instance_name)
{
  const raft::handle_t handle_{};
  auto mps_problem = cuopt::linear_programming::io::read_mps<int, double>(mps_path, false);
  handle_.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle_, mps_problem);

  problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  convert_greater_to_less(problem);

  mip_solver_settings_t<int, double> settings;
  settings.heuristics_only  = true;
  settings.determinism_mode = CUOPT_MODE_DETERMINISTIC;
  cuopt::timer_t solver_timer(std::numeric_limits<double>::infinity());
  mip_solver_t<int, double> solver(problem, settings, solver_timer);
  auto& ctx = solver.context;

  const double wpi = calib::pdlp_work_per_iter((ctx.ensure_work_features(), ctx.work_features));
  std::printf("\n[%s] pdlp_work_per_iter=%.3e s\n", instance_name.c_str(), wpi);

  bool ok = true;
  for (double budget : {0.05, 0.2, 0.5}) {
    const int lim = ctx.pdlp_iters_for_budget(budget);
    auto r1       = run_pdlp_to_iters(problem, lim);
    auto r2       = run_pdlp_to_iters(problem, lim);
    const bool reproducible =
      (r1.steps == r2.steps) && (std::abs(r1.obj - r2.obj) <= 1e-6 * (1 + std::abs(r1.obj)));
    const double time_ratio = budget > 0 ? r1.secs / budget : 0.0;
    if (!reproducible) { ok = false; }
    std::printf(
      "  budget=%.3f s -> iter_limit=%d | run1(steps=%d obj=%.6g %.4fs) run2(steps=%d %.4fs) | "
      "reproducible=%s  wall/budget=%.2f\n",
      budget,
      lim,
      r1.steps,
      r1.obj,
      r1.secs,
      r2.steps,
      r2.secs,
      reproducible ? "YES" : "NO",
      time_ratio);
    std::fflush(stdout);
  }
  return ok;
}

}  // namespace cuopt::linear_programming::detail::calib
