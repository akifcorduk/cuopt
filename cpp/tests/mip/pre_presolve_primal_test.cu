/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/pre_presolve_primal.cuh>

#include <gtest/gtest.h>

#include <vector>

namespace cuopt::mathematical_optimization::mip::test {

TEST(PrePresolveThreadBudget, ReservesBothPreBnbSlotsAtomically)
{
  pre_presolve_thread_budget_t budget(pre_presolve_primal_task_slots);

  EXPECT_FALSE(budget.try_reserve(pre_presolve_primal_task_slots + 1));
  EXPECT_TRUE(budget.try_reserve(pre_presolve_primal_task_slots));
  EXPECT_EQ(budget.available(), 0);
  EXPECT_FALSE(budget.try_reserve(1));

  budget.release(pre_presolve_primal_task_slots);
  EXPECT_EQ(budget.available(), pre_presolve_primal_task_slots);
}

TEST(PrePresolveSelection, UsesSmallOrLowRowGate)
{
  EXPECT_TRUE(pre_presolve_primal_selected(1'000, 5'000));
  EXPECT_TRUE(pre_presolve_primal_selected(64, 50'000));
  EXPECT_FALSE(pre_presolve_primal_selected(65, 50'000));
  EXPECT_FALSE(pre_presolve_primal_selected(64, 50'001));
}

TEST(PrePresolveFirstFinalCandidateStore, RetainsFirstAndBestCandidates)
{
  pre_presolve_first_final_candidate_store_t<double> candidates;
  EXPECT_TRUE(candidates.consider(4.0, std::vector<double>{4.0}));
  candidates.mark_submitted();
  EXPECT_FALSE(candidates.consider(6.0, std::vector<double>{6.0}));
  EXPECT_FALSE(candidates.consider(2.0, std::vector<double>{2.0}));

  const auto retained = candidates.snapshot();
  EXPECT_TRUE(retained.has_candidate);
  EXPECT_EQ(retained.generated, 3);
  EXPECT_EQ(retained.submitted, 1);
  EXPECT_DOUBLE_EQ(retained.first_solver_objective, 4.0);
  EXPECT_DOUBLE_EQ(retained.solver_objective, 2.0);
  EXPECT_EQ(retained.assignment, std::vector<double>({2.0}));
}

}  // namespace cuopt::mathematical_optimization::mip::test
