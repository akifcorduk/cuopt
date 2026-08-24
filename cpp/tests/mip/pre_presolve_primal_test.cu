/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/pre_presolve_primal.cuh>

#include <gtest/gtest.h>

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

}  // namespace cuopt::mathematical_optimization::mip::test
