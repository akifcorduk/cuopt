/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/post_root_snapshot.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::mip {
namespace {

class PostRootSnapshot : public ::testing::Test {
 protected:
  void SetUp() override
  {
    path = std::filesystem::temp_directory_path() /
           ("cuopt_post_root_snapshot_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    snapshot.num_rows            = 2;
    snapshot.num_cols            = 3;
    snapshot.col_start           = {0, 1, 3, 4};
    snapshot.row_indices         = {0, 0, 1, 1};
    snapshot.coefficients        = {1.0, 2.0, -1.0, 1.0};
    snapshot.rhs                 = {3.0, 0.0};
    snapshot.objective           = {3.0, -0.25, 0.0};
    snapshot.lower               = {0.0, -std::numeric_limits<double>::infinity(), 0.0};
    snapshot.upper               = {1.0, std::numeric_limits<double>::infinity(), 4.0};
    snapshot.var_types           = {1, 2, 0};
    snapshot.basis_status        = {-1, 0, 0};
    snapshot.root_primal         = {-0.0, 1.5, 1.5};
    snapshot.root_dual           = {-0.125, 0.0};
    snapshot.root_reduced_cost   = {3.125, 0.0, 0.0};
    snapshot.edge_norms          = {1.0, 2.0};
    snapshot.incumbent           = {1.0, 1.0, 1.0};
    snapshot.new_slacks          = {2};
    snapshot.obj_constant        = 11.0;
    snapshot.obj_scale           = -2.0;
    snapshot.objective_step_size = 0.25;
    snapshot.objective_step_bias = 0.0;
    snapshot.max_abs_obj_coeff   = 3.0;
    snapshot.min_abs_obj_coeff   = 0.25;
    snapshot.root_objective      = -0.375;
    snapshot.upper_bound         = 2.75;
  }

  void TearDown() override
  {
    std::error_code error;
    std::filesystem::remove(path, error);
  }

  std::filesystem::path path;
  post_root_snapshot_t<int, double> snapshot;
};

TEST_F(PostRootSnapshot, AugmentedProblemRoundTripsExactly)
{
  write_post_root_snapshot(path.string(), snapshot);
  const auto restored = read_post_root_snapshot<int, double>(path.string());
  EXPECT_EQ(restored.fields(), snapshot.fields());
  EXPECT_TRUE(std::signbit(restored.root_primal.front()));
  EXPECT_EQ(restored.new_slacks, std::vector<int>({2}));
  EXPECT_EQ(restored.var_types, std::vector<int8_t>({1, 2, 0}));
}

TEST_F(PostRootSnapshot, PreservesMissingIncumbentAndInfiniteCutoff)
{
  snapshot.incumbent.clear();
  snapshot.upper_bound = std::numeric_limits<double>::infinity();
  write_post_root_snapshot(path.string(), snapshot);
  const auto restored = read_post_root_snapshot<int, double>(path.string());
  EXPECT_TRUE(restored.incumbent.empty());
  EXPECT_EQ(restored.upper_bound, std::numeric_limits<double>::infinity());
}

TEST_F(PostRootSnapshot, RestoresObjectiveScaleAndConstant)
{
  struct root_solution_t {
    std::vector<double> x, y, z;
    double objective, user_objective;
    void resize(int m, int n)
    {
      x.resize(n);
      y.resize(m);
      z.resize(n);
    }
  } solution;
  snapshot.restore_root_solution(solution);
  EXPECT_DOUBLE_EQ(solution.user_objective, -21.25);
  EXPECT_EQ(solution.x, snapshot.root_primal);
  EXPECT_EQ(solution.y, snapshot.root_dual);
  EXPECT_EQ(solution.z, snapshot.root_reduced_cost);
}

TEST_F(PostRootSnapshot, RejectsInvalidColumnAndSolutionDimensions)
{
  snapshot.col_start.back() = 5;
  EXPECT_THROW(snapshot.validate(), std::runtime_error);
  snapshot.col_start.back() = 4;
  snapshot.root_dual.pop_back();
  EXPECT_THROW(snapshot.validate(), std::runtime_error);
}

TEST_F(PostRootSnapshot, RejectsWrongScalarTypeAndTruncatedFile)
{
  write_post_root_snapshot(path.string(), snapshot);
  EXPECT_THROW((read_post_root_snapshot<int, float>(path.string())), std::runtime_error);
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
  EXPECT_THROW((read_post_root_snapshot<int, double>(path.string())), std::runtime_error);
}

TEST_F(PostRootSnapshot, RejectsTrailingDataAndInvalidMagic)
{
  write_post_root_snapshot(path.string(), snapshot);
  {
    std::ofstream stream(path, std::ios::binary | std::ios::app);
    stream.put('X');
  }
  EXPECT_THROW((read_post_root_snapshot<int, double>(path.string())), std::runtime_error);
  {
    std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
    stream.put('X');
  }
  EXPECT_THROW((read_post_root_snapshot<int, double>(path.string())), std::runtime_error);
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::mip
