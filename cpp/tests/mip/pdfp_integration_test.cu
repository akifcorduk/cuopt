/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/local_search/feasibility_pump/feasibility_pump.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>
#include <raft/core/handle.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::test {

TEST(PdfpIntegration, DispatchPreservesObjectiveMixingAndAlphaSchedule)
{
  raft::handle_t handle;
  optimization_problem_t<int, double> input(&handle);
  std::vector<int> offsets{0, 2}, indices{0, 1};
  std::vector<double> coefficients{1., 1.}, objective{1., 0.};
  std::vector<double> lower{0., 0.}, upper{5., 5.}, rhs{4.};
  std::vector<var_t> types{var_t::INTEGER, var_t::CONTINUOUS};
  input.set_csr_constraint_matrix(coefficients.data(), 2, indices.data(), 2, offsets.data(), 2);
  input.set_objective_coefficients(objective.data(), 2);
  input.set_variable_lower_bounds(lower.data(), 2);
  input.set_variable_upper_bounds(upper.data(), 2);
  input.set_constraint_lower_bounds(rhs.data(), 1);
  input.set_constraint_upper_bounds(rhs.data(), 1);
  input.set_variable_types(types.data(), 2);

  mip::problem_t<int, double> problem(input);
  mip_solver_settings_t<int, double> settings;
  settings.tolerances.absolute_tolerance          = 1e-7;
  settings.heuristic_params.relaxed_lp_time_limit = 5.;
  mip::mip_solver_context_t<int, double> context(&handle, &problem, settings);
  mip::fj_t<int, double> fj(context);
  mip::constraint_prop_t<int, double> propagation(context);
  mip::line_segment_search_t<int, double> line_search(fj, propagation);
  rmm::device_uvector<double> root_point(2, handle.get_stream());
  mip::feasibility_pump_t<int, double> pump(context, fj, propagation, line_search, root_point);
  mip::solution_t<int, double> solution(problem);
  pump.config.use_pdfp_projection = true;
  pump.timer                      = cuopt::timer_t(60.);

  // The normalized objective is (1-alpha)*|x-3| + alpha*x on x+y=4.
  // High alpha uniquely minimizes it at (0,4); low alpha at (3,1).
  const std::vector<double> target{3., 1.};
  solution.copy_new_assignment(target);
  pump.config.alpha = .99;
  pump.linear_project_onto_polytope(solution, 1.);
  EXPECT_NEAR(pump.config.alpha, .891, 1e-14);
  EXPECT_GT(pump.last_pdfp_result.iterations, 0);
  EXPECT_TRUE(pump.last_pdfp_result.converged);
  auto projected = solution.get_host_assignment();
  EXPECT_NEAR(projected[0], 0., 1e-5);
  EXPECT_NEAR(projected[1], 4., 1e-5);
  EXPECT_NEAR(pump.last_pdfp_result.objective, .109 * 3., 1e-5);

  solution.copy_new_assignment(target);
  pump.config.alpha = .1;
  pump.linear_project_onto_polytope(solution, 1.);
  EXPECT_NEAR(pump.config.alpha, .09, 1e-14);
  EXPECT_TRUE(pump.last_pdfp_result.converged);
  projected = solution.get_host_assignment();
  EXPECT_NEAR(projected[0], 3., 1e-5);
  EXPECT_NEAR(projected[1], 1., 1e-5);
  EXPECT_NEAR(pump.last_pdfp_result.objective, .09 * 3., 1e-5);

  solution.copy_new_assignment(target);
  pump.linear_project_onto_polytope(solution, 1., true);
  EXPECT_NEAR(pump.config.alpha, .09, 1e-14);
  EXPECT_TRUE(pump.last_pdfp_result.converged);
  EXPECT_EQ(problem.n_variables, 2);
  EXPECT_EQ(problem.n_constraints, 1);
  EXPECT_EQ(problem.nnz, 2);
  EXPECT_EQ(solution.assignment.size(), 2);
}

}  // namespace cuopt::mathematical_optimization::test
