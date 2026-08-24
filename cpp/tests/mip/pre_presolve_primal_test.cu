/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/pre_presolve_primal.cuh>

#include <gtest/gtest.h>

namespace cuopt::mathematical_optimization::mip::test {

TEST(PrePresolveSelectiveGate, IncludesBothBoundaryLanes)
{
  EXPECT_TRUE(pre_bnb_selective_problem_is_eligible(65, pre_bnb_small_nnz_limit));
  EXPECT_TRUE(
    pre_bnb_selective_problem_is_eligible(pre_bnb_low_row_limit, pre_bnb_low_row_nnz_limit));
}

TEST(PrePresolveSelectiveGate, ExcludesProblemsOutsideBothLanes)
{
  EXPECT_FALSE(
    pre_bnb_selective_problem_is_eligible(pre_bnb_low_row_limit + 1, pre_bnb_small_nnz_limit + 1));
  EXPECT_FALSE(
    pre_bnb_selective_problem_is_eligible(pre_bnb_low_row_limit, pre_bnb_low_row_nnz_limit + 1));
}

}  // namespace cuopt::mathematical_optimization::mip::test
