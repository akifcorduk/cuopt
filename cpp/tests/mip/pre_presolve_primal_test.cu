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

TEST(PrePresolveSelectiveGate, AdmitsSmallAndLowRowProblems)
{
  EXPECT_TRUE(pre_presolve_primal_is_selectively_eligible(10'000, 5'000));
  EXPECT_TRUE(pre_presolve_primal_is_selectively_eligible(64, 50'000));
}

TEST(PrePresolveSelectiveGate, RejectsProblemsOutsideBothRegions)
{
  EXPECT_FALSE(pre_presolve_primal_is_selectively_eligible(65, 5'001));
  EXPECT_FALSE(pre_presolve_primal_is_selectively_eligible(64, 50'001));
}

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

TEST(PrePresolveBestCandidateStore, RetainsOnlyTheBestCandidate)
{
  pre_presolve_best_candidate_store_t<double> candidates;
  candidates.consider(4.0, std::vector<double>{4.0});
  candidates.consider(6.0, std::vector<double>{6.0});
  candidates.consider(2.0, std::vector<double>{2.0});

  const auto retained = candidates.snapshot();
  EXPECT_TRUE(retained.has_candidate);
  EXPECT_EQ(retained.generated, 3);
  EXPECT_DOUBLE_EQ(retained.solver_objective, 2.0);
  EXPECT_EQ(retained.assignment, std::vector<double>({2.0}));
}

}  // namespace cuopt::mathematical_optimization::mip::test
