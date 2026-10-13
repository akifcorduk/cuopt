/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mip_heuristics/gpu_submip_fixed_columns.hpp>

#include <gtest/gtest.h>

#include <limits>

namespace cuopt::mathematical_optimization::mip {
namespace {

using type_t = simplex::variable_type_t;

class GpuSubmipFixedColumns : public ::testing::Test {
 protected:
  GpuSubmipFixedColumns() : lp(nullptr, 2, 5, 8)
  {
    // Keep x and s. Eliminate integer z=2, continuous t=-1.5, and a zero artificial.
    // x + 3z + 2t + a = 4; 2x - z + 4t + s = -3.
    lp.A.col_start  = {0, 2, 4, 6, 7, 8};
    lp.A.i          = {0, 1, 0, 1, 0, 1, 1, 0};
    lp.A.x          = {1, 2, 3, -1, 2, 4, 1, 1};
    lp.rhs          = {4, -3};
    lp.lower        = {0, 2, -1.5, -10, 0};
    lp.upper        = {5, 2, -1.5, 10, 0};
    lp.objective    = {5, 7, 2, -1, 0};
    lp.obj_constant = 1000;
    lp.obj_scale    = -1;
  }

  simplex::lp_problem_t<int, double> lp;
  // The appended columns have no explicit type and must remain continuous.
  std::vector<type_t> types{type_t::INTEGER, type_t::INTEGER, type_t::CONTINUOUS};
};

TEST_F(GpuSubmipFixedColumns, PreservesRowsObjectiveAndFullCandidateCoordinates)
{
  gpu_submip_fixed_columns_t<int, double> compact(lp, types);
  EXPECT_EQ(compact.free_to_full, (std::vector<int>{0, 3}));
  EXPECT_EQ(compact.full_to_free, (std::vector<int>{0, -1, -1, 1, -1}));
  EXPECT_EQ(compact.fixed_columns, (std::vector<int>{1, 2, 4}));
  EXPECT_EQ(compact.reduced_types, (std::vector<type_t>{type_t::INTEGER, type_t::CONTINUOUS}));
  EXPECT_EQ(compact.reduced_lp.num_rows, 2);
  EXPECT_EQ(compact.reduced_lp.num_cols, 2);
  EXPECT_EQ(compact.reduced_lp.A.col_start, (std::vector<int>{0, 2, 3}));
  EXPECT_EQ(compact.reduced_lp.A.i, (std::vector<int>{0, 1, 1}));
  EXPECT_EQ(compact.reduced_lp.A.x, (std::vector<double>{1, 2, 1}));
  EXPECT_EQ(compact.reduced_lp.rhs, (std::vector<double>{1, 5}));
  EXPECT_EQ(compact.reduced_lp.objective, (std::vector<double>{5, -1}));
  EXPECT_DOUBLE_EQ(compact.reduced_lp.obj_constant, 0);
  EXPECT_DOUBLE_EQ(compact.reduced_lp.obj_scale, 1);
  EXPECT_DOUBLE_EQ(compact.objective_shift, 11);
  EXPECT_DOUBLE_EQ(compact.reduced_cutoff(12), 1);

  const std::vector<double> full{1, 2, -1.5, 3, 0};
  const auto reduced = compact.gather(full);
  EXPECT_EQ(reduced, (std::vector<double>{1, 3}));
  EXPECT_EQ(compact.scatter(reduced), full);
  EXPECT_DOUBLE_EQ(
    compensated_dot2(lp.objective.data(), full.data(), full.size()),
    compensated_dot2(compact.reduced_lp.objective.data(), reduced.data(), reduced.size()) +
      compact.objective_shift);
  // The CPU matrix, objective, and original coordinates have not been modified.
  EXPECT_EQ(lp.num_cols, 5);
  EXPECT_EQ(lp.A.nnz(), 8);
  EXPECT_EQ(lp.rhs, (std::vector<double>{4, -3}));
  EXPECT_DOUBLE_EQ(lp.obj_constant, 1000);
}

TEST_F(GpuSubmipFixedColumns, RejectsRelaxedOrChangedExcludedFixing)
{
  gpu_submip_fixed_columns_t<int, double> compact(lp, types);
  auto node     = lp;
  node.upper[1] = 3;
  EXPECT_THROW(compact.validate_node_bounds(node), std::invalid_argument);
  node.lower[1] = node.upper[1] = 1;
  EXPECT_THROW(compact.validate_node_bounds(node), std::invalid_argument);
  node          = lp;
  node.lower[2] = -std::numeric_limits<double>::infinity();
  EXPECT_THROW(compact.validate_node_bounds(node), std::invalid_argument);
  node = lp;
  ++node.num_rows;
  EXPECT_THROW(compact.validate_node_bounds(node), std::invalid_argument);
}

TEST_F(GpuSubmipFixedColumns, RestoresRetainedBoundsAcrossOppositeSiblings)
{
  gpu_submip_fixed_columns_t<int, double> compact(lp, types);
  auto node     = lp;
  node.lower[0] = node.upper[0] = 0;
  EXPECT_NO_THROW(compact.validate_node_bounds(node));
  EXPECT_EQ(compact.gather(node.lower), (std::vector<double>{0, -10}));
  EXPECT_EQ(compact.gather(node.upper), (std::vector<double>{0, 10}));
  node.lower[0] = node.upper[0] = 1;
  EXPECT_NO_THROW(compact.validate_node_bounds(node));
  EXPECT_EQ(compact.gather(node.lower), (std::vector<double>{1, -10}));
  EXPECT_EQ(compact.gather(node.upper), (std::vector<double>{1, 10}));
  node = lp;
  EXPECT_NO_THROW(compact.validate_node_bounds(node));
  EXPECT_EQ(compact.gather(node.upper), (std::vector<double>{5, 10}));
}

TEST_F(GpuSubmipFixedColumns, KeepsNarrowBoundsAndFractionallyFixedIntegers)
{
  lp.upper[1] = lp.lower[1] + 1e-12;
  lp.lower[0] = lp.upper[0] = 0.5;
  gpu_submip_fixed_columns_t<int, double> compact(lp, types);
  EXPECT_EQ(compact.free_to_full, (std::vector<int>{0, 1, 3}));
  EXPECT_EQ(compact.fixed_columns, (std::vector<int>{2, 4}));
  EXPECT_EQ(compact.reduced_types,
            (std::vector<type_t>{type_t::INTEGER, type_t::INTEGER, type_t::CONTINUOUS}));
}

TEST_F(GpuSubmipFixedColumns, DisabledAndAllFixedModelsRetainIdentityCoordinates)
{
  gpu_submip_fixed_columns_t<int, double> disabled(lp, types, false);
  EXPECT_EQ(disabled.free_to_full, (std::vector<int>{0, 1, 2, 3, 4}));
  EXPECT_TRUE(disabled.fixed_columns.empty());
  EXPECT_EQ(disabled.reduced_lp.A.x, lp.A.x);
  EXPECT_EQ(disabled.reduced_lp.rhs, lp.rhs);
  EXPECT_DOUBLE_EQ(disabled.reduced_cutoff(12), 12);
  EXPECT_EQ(disabled.scatter(disabled.gather(lp.lower)), lp.lower);

  lp.upper = lp.lower;
  gpu_submip_fixed_columns_t<int, double> all_fixed(lp, types);
  EXPECT_EQ(all_fixed.reduced_lp.num_cols, lp.num_cols);
  EXPECT_TRUE(all_fixed.fixed_columns.empty());
  EXPECT_EQ(all_fixed.reduced_lp.rhs, lp.rhs);
  EXPECT_EQ(all_fixed.scatter(all_fixed.gather(lp.lower)), lp.lower);
}

TEST_F(GpuSubmipFixedColumns, RejectsMismatchedVectorDimensions)
{
  gpu_submip_fixed_columns_t<int, double> compact(lp, types);
  EXPECT_THROW(compact.gather({1}), std::invalid_argument);
  EXPECT_THROW(compact.scatter({1}), std::invalid_argument);
  lp.lower.pop_back();
  EXPECT_THROW(compact.validate_node_bounds(lp), std::invalid_argument);
  EXPECT_THROW((gpu_submip_fixed_columns_t<int, double>(lp, types)), std::invalid_argument);
}

TEST(GpuSubmipFixedColumnsNumerics, PreservesCancellationInRhsAndCutoff)
{
  simplex::lp_problem_t<int, double> lp(nullptr, 1, 4, 4);
  lp.A.col_start = {0, 1, 2, 3, 4};
  lp.A.i         = {0, 0, 0, 0};
  lp.A.x         = {1e16, 1, -1e16, 1};
  lp.rhs         = {0};
  lp.lower       = {1, 1, 1, -5};
  lp.upper       = {1, 1, 1, 5};
  lp.objective   = {1e16, 1, -1e16, 1};
  gpu_submip_fixed_columns_t<int, double> compact(lp, {});
  EXPECT_EQ(compact.reduced_lp.rhs, (std::vector<double>{-1}));
  EXPECT_DOUBLE_EQ(compact.objective_shift, 1);
  EXPECT_DOUBLE_EQ(compact.reduced_cutoff(2), 1);

  // The correctly rounded fixed objective alone loses the +1. Preserve its low part when
  // subtracting from a nearly equal cutoff, so an infeasible target is not admitted.
  lp.objective = {1e16, 1, 0, 1};
  gpu_submip_fixed_columns_t<int, double> large_shift(lp, {});
  EXPECT_DOUBLE_EQ(large_shift.objective_shift, 1e16);
  EXPECT_DOUBLE_EQ(large_shift.reduced_cutoff(1e16), -1);
  EXPECT_EQ(large_shift.reduced_cutoff(std::numeric_limits<double>::infinity()),
            std::numeric_limits<double>::infinity());
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::mip
