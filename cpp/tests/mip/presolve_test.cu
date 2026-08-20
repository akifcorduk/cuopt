/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <mip_heuristics/presolve/presolve_budget_policy.hpp>
#include <mip_heuristics/presolve/third_party_presolve.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <pdlp/utils.cuh>
#include <utilities/common_utils.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::test {

TEST(problem, find_implied_integers)
{
  const raft::handle_t handle_{};

  auto path           = make_path_absolute("mip/fiball.mps");
  auto mps_data_model = cuopt::mathematical_optimization::io::read_mps<int, double>(path, false);
  auto op_problem     = mps_data_model_to_optimization_problem(&handle_, mps_data_model);
  auto presolver      = std::make_unique<mip::third_party_presolve_t<int, double>>();
  auto result         = presolver->apply_presolve_from_op_problem(
    op_problem,
    cuopt::mathematical_optimization::problem_category_t::MIP,
    cuopt::mathematical_optimization::presolver_t::Papilo,
    false,
    1e-6,
    1e-12,
    20,
    1);
  ASSERT_NE(result.status, mip::third_party_presolve_status_t::INFEASIBLE);
  ASSERT_NE(result.status, mip::third_party_presolve_status_t::UNBNDORINFEAS);

  auto problem = mip::problem_t<int, double>(result.reduced_problem);
  problem.set_implied_integers(result.implied_integer_indices);
  ASSERT_TRUE(result.implied_integer_indices.size() > 0);
  auto var_types = host_copy(problem.variable_types, handle_.get_stream());
  // Find the index of the one continuous variable
  auto it = std::find_if(var_types.begin(), var_types.end(), [](var_t var_type) {
    return var_type == var_t::CONTINUOUS;
  });
  ASSERT_NE(it, var_types.end());
  ASSERT_EQ(problem.presolve_data.var_flags.size(), var_types.size());
  // Ensure it is an implied integer
  EXPECT_EQ(problem.presolve_data.var_flags.element(it - var_types.begin(), handle_.get_stream()),
            ((int)mip::problem_t<int, double>::var_flags_t::VAR_IMPLIED_INTEGER));
}

TEST(presolve_budget_policy, prioritize_large_tall_zero_objective_mixed_model)
{
  mip::presolve_features_t features{};
  features.n_vars = 11928;
  features.n_cons = 131865;
  features.n_int  = 7482;

  EXPECT_TRUE(mip::should_prioritize_early_feasibility(features, true));
  EXPECT_FALSE(mip::should_prioritize_early_feasibility(features, false));

  features.n_vars = 1000;
  features.n_cons = 99999;
  EXPECT_FALSE(mip::should_prioritize_early_feasibility(features, true));

  features.n_vars = 20000;
  features.n_cons = 8 * features.n_vars - 1;
  EXPECT_FALSE(mip::should_prioritize_early_feasibility(features, true));

  features.n_vars = 11928;
  features.n_cons = 131865;
  features.n_int  = features.n_vars;
  EXPECT_FALSE(mip::should_prioritize_early_feasibility(features, true));
}

TEST(presolve_budget_policy, bound_large_probing_startup_latency)
{
  mip::mip_heuristics_hyper_params_t<int, double> hp{};
  mip::presolve_features_t features{};
  features.n_vars = mip::early_primal_probing_size_threshold - 1;
  features.n_cons = 1000;
  features.nnz    = 10000;
  features.n_int  = features.n_vars;

  auto budget = mip::evaluate_presolve_budget(hp, features);
  EXPECT_TRUE(std::isinf(budget.probing_wall_limit));

  features.n_vars = mip::early_primal_probing_size_threshold;
  budget          = mip::evaluate_presolve_budget(hp, features);
  EXPECT_EQ(budget.probing_wall_limit, mip::early_primal_probing_wall_limit);
}

TEST(presolve_budget_policy, run_pre_root_quick_repair_only_with_constraint_guidance)
{
  mip::presolve_features_t features{};
  features.n_vars = 1000;
  features.n_cons = 499;
  EXPECT_FALSE(mip::should_run_pre_root_quick_repair(features));

  features.n_cons = 500;
  EXPECT_TRUE(mip::should_run_pre_root_quick_repair(features));
}

}  // namespace cuopt::mathematical_optimization::test
