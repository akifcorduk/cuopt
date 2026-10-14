/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <dual_simplex/presolve.hpp>
#include <mip_heuristics/gpu_submip_fp.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace cuopt::mathematical_optimization::mip {
namespace {

class GpuSubmipFp : public ::testing::Test {
 protected:
  GpuSubmipFp() : lp(&handle, 1, 2, 2)
  {
    // x + s = 0; x is integer, s is a retained continuous simplex slack.
    lp.A.col_start = {0, 1, 2};
    lp.A.i         = {0, 0};
    lp.A.x         = {1., 1.};
    lp.rhs         = {0.};
    lp.objective   = {1., 0.};
    lp.lower       = {0., -1.};
    lp.upper       = {1., 0.};
    lp.obj_scale   = 1.;
    // The session operates in internal objective space, irrespective of this constant.
    lp.obj_constant = 1000.;
  }

  raft::handle_t handle;
  simplex::lp_problem_t<int, double> lp;
  std::vector<simplex::variable_type_t> types{simplex::variable_type_t::INTEGER,
                                              simplex::variable_type_t::CONTINUOUS};
  mip_solver_settings_t<int, double> settings;
};

TEST_F(GpuSubmipFp, DistanceConfigSurvivesResetOnEveryDescent)
{
  // This LP has no integer solution. Distance-only FP should detect the repeated
  // integer target before the baseline objective-weight schedule allows cycling.
  simplex::lp_problem_t<int, double> fractional_lp(&handle, 1, 1, 1);
  fractional_lp.A.col_start = {0, 1};
  fractional_lp.A.i         = {0};
  fractional_lp.A.x         = {1.};
  fractional_lp.rhs         = {0.5};
  fractional_lp.objective   = {1.};
  fractional_lp.lower       = {0.};
  fractional_lp.upper       = {1.};
  std::vector<simplex::variable_type_t> fractional_types{simplex::variable_type_t::INTEGER};
  gpu_submip_fp_t<int, double> baseline(&handle, fractional_lp, fractional_types, settings);
  gpu_submip_fp_t<int, double> distance(&handle,
                                        fractional_lp,
                                        fractional_types,
                                        settings,
                                        true,
                                        false,
                                        gpu_submip_fp_config_from_name("distance"));
  std::vector<double> candidate;
  EXPECT_FALSE(baseline.run(fractional_lp, {0.5}, 1., 10., nullptr, candidate));
  const auto baseline_projections = baseline.last_run_info().projections;
  ASSERT_GT(baseline_projections, 0);
  for (int restart = 0; restart < 2; ++restart) {
    EXPECT_FALSE(distance.run(fractional_lp, {0.5}, 1., 10., nullptr, candidate));
    EXPECT_STREQ(distance.last_run_info().exit_reason, "integer_cycle");
    EXPECT_GT(distance.last_run_info().projections, 0);
    EXPECT_LT(distance.last_run_info().projections, baseline_projections);
  }
}

TEST_F(GpuSubmipFp, RestoresSiblingBoundsAndRetainsSlackCoordinates)
{
  gpu_submip_fp_t<int, double> session(&handle, lp, types, settings);
  std::vector<double> candidate;
  lp.upper[0] = 0.;
  ASSERT_TRUE(session.run(lp, {0., 0.}, 0.25, 10., nullptr, candidate));
  ASSERT_EQ(candidate.size(), 2);
  EXPECT_NEAR(candidate[0], 0., 1e-6);
  EXPECT_NEAR(candidate[1], 0., 1e-6);

  // Reuse the previous projection, but move to the opposite child. Updating
  // both bounds must remove the previous child's fixing before clamping.
  lp.lower[0] = 1.;
  lp.upper[0] = 1.;
  ASSERT_TRUE(session.run(lp, {1., -1.}, 1.25, 10., nullptr, candidate, true));
  ASSERT_EQ(candidate.size(), 2);
  EXPECT_NEAR(candidate[0], 1., 1e-6);
  EXPECT_NEAR(candidate[1], -1., 1e-6);

  lp.lower[0] = 0.;
  lp.upper[0] = 0.;
  ASSERT_TRUE(session.run(lp, {0., 0.}, 0.25, 10., nullptr, candidate, true));
  EXPECT_NEAR(candidate[0], 0., 1e-6);
  EXPECT_NEAR(candidate[1], 0., 1e-6);
}

TEST_F(GpuSubmipFp, DoesNotReturnNonImprovingNodeSolution)
{
  gpu_submip_fp_t<int, double> session(&handle, lp, types, settings);
  lp.lower[0] = 1.;
  lp.upper[0] = 1.;
  std::vector<double> candidate;
  EXPECT_FALSE(session.run(lp, {1., -1.}, 0.5, 0.1, nullptr, candidate));
  EXPECT_TRUE(candidate.empty());
}

TEST_F(GpuSubmipFp, InitialRoundingFindsFeasibilityBeforeProjectionAndHonorsCutoff)
{
  auto config             = gpu_submip_fp_config_from_name("distance");
  config.initial_rounding = true;
  gpu_submip_fp_t<int, double> session(&handle, lp, types, settings, true, true, config);
  std::vector<double> candidate;
  ASSERT_TRUE(session.run(lp, {0., 0.}, 0.25, 10., nullptr, candidate));
  EXPECT_EQ(session.last_run_info().projections, 0);
  EXPECT_STREQ(session.last_run_info().exit_reason, "feasible_initial_rounding");
  ASSERT_EQ(candidate.size(), 2);
  EXPECT_NEAR(candidate[0], 0., 1e-6);
  EXPECT_NEAR(candidate[1], 0., 1e-6);
  EXPECT_FALSE(session.run(lp, {0., 0.}, -0.25, 0.1, nullptr, candidate));
  EXPECT_TRUE(candidate.empty());
}

TEST_F(GpuSubmipFp, UnlimitedDeadlineRetainsFiniteTimingAndExportsRestartProjection)
{
  lp.lower[1] = -0.49;
  gpu_submip_fp_t<int, double> session(&handle, lp, types, settings);
  std::vector<double> candidate, restart_seed;
  EXPECT_FALSE(session.get_restart_seed(restart_seed));
  ASSERT_TRUE(session.run(
    lp, {0.45, -0.45}, 0.25, std::numeric_limits<double>::infinity(), nullptr, candidate));
  const auto info = session.last_run_info();
  EXPECT_TRUE(info.started);
  EXPECT_GT(info.projections, 0);
  EXPECT_TRUE(std::isfinite(info.descent_seconds));
  EXPECT_GT(info.descent_seconds, 0.);
  EXPECT_TRUE(std::isfinite(info.projection_seconds));
  EXPECT_GT(info.projection_seconds, 0.);
  ASSERT_TRUE(session.get_restart_seed(restart_seed));
  ASSERT_EQ(restart_seed.size(), 2);
  for (double value : restart_seed) {
    EXPECT_TRUE(std::isfinite(value));
  }
  EXPECT_NEAR(candidate[0], 0., 1e-6);
  EXPECT_NEAR(candidate[1], 0., 1e-6);
}

TEST_F(GpuSubmipFp, ReusesProjectionAcrossGeneralIntegerDistanceTargets)
{
  // x + s = 2 with x in [0, 4]. Integer targets 1 and 2 are interior,
  // so FP creates two distance rows with the same sizes but different RHSs.
  lp.rhs[0] = 2.;
  lp.lower  = {0., 0.};
  lp.upper  = {4., 1.};
  gpu_submip_fp_t<int, double> session(&handle, lp, types, settings);
  std::vector<double> candidate;
  lp.upper[1] = 0.;
  ASSERT_TRUE(session.run(lp, {2., 0.}, 2.25, 10., nullptr, candidate));
  EXPECT_NEAR(candidate[0], 2., 1e-6);
  EXPECT_NEAR(candidate[1], 0., 1e-6);

  lp.lower[1] = 1.;
  lp.upper[1] = 1.;
  ASSERT_TRUE(session.run(lp, {1., 1.}, 2.25, 10., nullptr, candidate, true));
  EXPECT_NEAR(candidate[0], 1., 1e-6);
  EXPECT_NEAR(candidate[1], 1., 1e-6);

  lp.lower[1] = 0.;
  lp.upper[1] = 0.;
  ASSERT_TRUE(session.run(lp, {2., 0.}, 2.25, 10., nullptr, candidate, true));
  EXPECT_NEAR(candidate[0], 2., 1e-6);
  EXPECT_NEAR(candidate[1], 0., 1e-6);
}

TEST_F(GpuSubmipFp, CancellationAndExpiredDeadlineClearCandidate)
{
  gpu_submip_fp_t<int, double> session(&handle, lp, types, settings);
  std::atomic<int> halt{1};
  std::vector<double> candidate{5., 5.};
  EXPECT_FALSE(session.run(lp, {0., 0.}, 0.25, 10., &halt, candidate));
  EXPECT_TRUE(candidate.empty());
  halt.store(0);
  candidate = {5., 5.};
  EXPECT_FALSE(session.run(lp, {0., 0.}, 0.25, 0., &halt, candidate));
  EXPECT_TRUE(candidate.empty());
}

TEST_F(GpuSubmipFp, RejectsChangedModelDimensions)
{
  gpu_submip_fp_t<int, double> session(&handle, lp, types, settings);
  std::vector<double> candidate;
  ++lp.num_rows;
  EXPECT_ANY_THROW(session.run(lp, {0., 0.}, 0.25, 10., nullptr, candidate));
}

TEST_F(GpuSubmipFp, ReuseRequiresTheSameModelAndVariableTypes)
{
  gpu_submip_fp_t<int, double> session(&handle, lp, types, settings);
  EXPECT_TRUE(session.can_reuse(lp, types));
  auto sibling     = lp;
  sibling.lower[0] = sibling.upper[0] = 1.;
  EXPECT_TRUE(session.can_reuse(sibling, types));

  auto changed_types = types;
  changed_types[0]   = simplex::variable_type_t::CONTINUOUS;
  EXPECT_FALSE(session.can_reuse(lp, changed_types));

  auto changed_model   = lp;
  changed_model.A.x[0] = 2.;
  EXPECT_FALSE(session.can_reuse(changed_model, types));
  std::vector<double> candidate{42.};
  EXPECT_ANY_THROW(session.run(changed_model, {0., 0.}, 0.25, 10., nullptr, candidate));
  EXPECT_TRUE(candidate.empty());

  changed_model        = lp;
  changed_model.rhs[0] = 1.;
  EXPECT_FALSE(session.can_reuse(changed_model, types));
  EXPECT_ANY_THROW(session.run(changed_model, {0., 0.}, 0.25, 10., nullptr, candidate));

  changed_model              = lp;
  changed_model.objective[0] = -1.;
  EXPECT_FALSE(session.can_reuse(changed_model, types));
  EXPECT_ANY_THROW(session.run(changed_model, {0., 0.}, 0.25, 10., nullptr, candidate));
}

TEST_F(GpuSubmipFp, FusedProjectionPreservesFractionalSeedResultAndSiblingBounds)
{
  gpu_submip_fp_t<int, double> separate(&handle, lp, types, settings, true, false);
  gpu_submip_fp_t<int, double> fused(&handle, lp, types, settings, true, true);
  std::vector<double> separate_candidate, fused_candidate;
  // Rounding the integer x leaves s fractional and violates the equality, so
  // the result requires a projection even though the continuous seed is feasible.
  ASSERT_TRUE(separate.run(lp, {0.45, -0.45}, 0.25, 10., nullptr, separate_candidate));
  ASSERT_TRUE(fused.run(lp, {0.45, -0.45}, 0.25, 10., nullptr, fused_candidate));
  ASSERT_EQ(separate_candidate.size(), 2);
  ASSERT_EQ(fused_candidate.size(), 2);
  for (size_t j = 0; j < fused_candidate.size(); ++j) {
    EXPECT_NEAR(fused_candidate[j], separate_candidate[j], 1e-6);
    EXPECT_NEAR(fused_candidate[j], 0., 1e-6);
  }

  // An incumbent-blended seed can lie outside the new child. Both paths must
  // clamp to the node bounds before projecting, independently of cached seeds.
  lp.lower[0] = lp.upper[0] = 1.;
  ASSERT_TRUE(separate.run(lp, {0., 0.}, 1.25, 10., nullptr, separate_candidate));
  ASSERT_TRUE(fused.run(lp, {0., 0.}, 1.25, 10., nullptr, fused_candidate));
  EXPECT_NEAR(fused_candidate[0], 1., 1e-6);
  EXPECT_NEAR(fused_candidate[1], -1., 1e-6);
  for (size_t j = 0; j < fused_candidate.size(); ++j) {
    EXPECT_NEAR(fused_candidate[j], separate_candidate[j], 1e-6);
  }
}

TEST_F(GpuSubmipFp, FixedColumnCompactionPreservesCutoffAndFullCoordinates)
{
  simplex::lp_problem_t<int, double> full_lp(&handle, 1, 4, 4);
  // Fixed integer a=2 and continuous b=1.5 contribute 12 to the objective.
  // After removing them, a+b+x+s=3.5 becomes x+s=0.
  full_lp.A.col_start  = {0, 1, 2, 3, 4};
  full_lp.A.i          = {0, 0, 0, 0};
  full_lp.A.x          = {1., 1., 1., 1.};
  full_lp.rhs          = {3.5};
  full_lp.objective    = {3., 4., 1., 0.};
  full_lp.lower        = {2., 1.5, 0., -1.};
  full_lp.upper        = {2., 1.5, 1., 0.};
  full_lp.obj_constant = 1000.;
  full_lp.obj_scale    = 1.;
  const std::vector<simplex::variable_type_t> full_types{simplex::variable_type_t::INTEGER,
                                                         simplex::variable_type_t::CONTINUOUS,
                                                         simplex::variable_type_t::INTEGER,
                                                         simplex::variable_type_t::CONTINUOUS};
  gpu_submip_fp_t<int, double> compact(&handle, full_lp, full_types, settings, true);
  gpu_submip_fp_t<int, double> identity(&handle, full_lp, full_types, settings, false);
  std::vector<double> compact_candidate, identity_candidate;

  full_lp.upper[2] = 0.;
  ASSERT_TRUE(compact.run(full_lp, {2., 1.5, 0., 0.}, 12.25, 10., nullptr, compact_candidate));
  ASSERT_TRUE(identity.run(full_lp, {2., 1.5, 0., 0.}, 12.25, 10., nullptr, identity_candidate));
  ASSERT_EQ(compact_candidate.size(), 4);
  ASSERT_EQ(identity_candidate.size(), 4);
  for (size_t j = 0; j < compact_candidate.size(); ++j) {
    EXPECT_NEAR(compact_candidate[j], identity_candidate[j], 1e-6);
  }
  EXPECT_NEAR(compact_candidate[0], 2., 1e-6);
  EXPECT_NEAR(compact_candidate[1], 1.5, 1e-6);
  EXPECT_NEAR(compact_candidate[2], 0., 1e-6);
  EXPECT_NEAR(compact_candidate[3], 0., 1e-6);

  // A retained column may move to the opposite child without recreating either
  // session. The full cutoff 13.25 corresponds to reduced cutoff 1.25.
  full_lp.lower[2] = full_lp.upper[2] = 1.;
  EXPECT_TRUE(compact.can_reuse(full_lp, full_types));
  EXPECT_TRUE(identity.can_reuse(full_lp, full_types));
  ASSERT_TRUE(
    compact.run(full_lp, {2., 1.5, 1., -1.}, 13.25, 10., nullptr, compact_candidate, true));
  ASSERT_TRUE(
    identity.run(full_lp, {2., 1.5, 1., -1.}, 13.25, 10., nullptr, identity_candidate, true));
  ASSERT_EQ(compact_candidate.size(), 4);
  ASSERT_EQ(identity_candidate.size(), 4);
  for (size_t j = 0; j < compact_candidate.size(); ++j) {
    EXPECT_NEAR(compact_candidate[j], identity_candidate[j], 1e-6);
  }
  EXPECT_NEAR(compact_candidate[2], 1., 1e-6);
  EXPECT_NEAR(compact_candidate[3], -1., 1e-6);

  EXPECT_FALSE(compact.run(full_lp, {2., 1.5, 1., -1.}, 12.5, 0.1, nullptr, compact_candidate));
  EXPECT_FALSE(identity.run(full_lp, {2., 1.5, 1., -1.}, 12.5, 0.1, nullptr, identity_candidate));
  EXPECT_TRUE(compact_candidate.empty());
  EXPECT_TRUE(identity_candidate.empty());

  // An excluded column cannot be unfixed behind the mapping's back.
  full_lp.upper[0] = 3.;
  EXPECT_FALSE(compact.can_reuse(full_lp, full_types));
  EXPECT_TRUE(identity.can_reuse(full_lp, full_types));
  EXPECT_ANY_THROW(
    compact.run(full_lp, {2., 1.5, 1., -1.}, 13.25, 10., nullptr, compact_candidate));
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::mip
