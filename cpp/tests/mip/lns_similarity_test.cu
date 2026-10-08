/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mip_heuristics/lns/cpufj_similarity.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace cuopt::lns::test {
namespace mip = cuopt::mathematical_optimization::mip;

TEST(LnsSimilarity, IndependentRowScalingPreservesScoresAndSelection)
{
  mip::fj_cpu_problem_t<int, double> problem;
  problem.n_variables   = 4;
  problem.n_constraints = 2;
  problem.offsets       = {0, 3, 6};
  problem.variables     = {0, 1, 2, 0, 1, 3};
  problem.coefficients  = {1, 2, -1, 2, 1, 3};
  const std::vector<int> eligible{0, 1, 2, 3};
  const std::vector<double> assignment{2, 1, 2, 3};
  mip::cpufj_lns_similarity_t<int, double> original(problem, eligible);
  for (int p = 0; p < 3; ++p)
    problem.coefficients[p] *= 1e9;
  for (int p = 3; p < 6; ++p)
    problem.coefficients[p] *= -1e-9;
  mip::cpufj_lns_similarity_t<int, double> scaled(problem, eligible);
  for (const double alpha : {1.0, 0.5}) {
    for (int v = 1; v < 4; ++v)
      EXPECT_NEAR(
        original.score(0, v, assignment, alpha), scaled.score(0, v, assignment, alpha), 1e-14);
    std::mt19937 first_rng(42), second_rng(42);
    std::vector<int> first, second;
    original.select(0, assignment, 3, alpha, first_rng, first);
    scaled.select(0, assignment, 3, alpha, second_rng, second);
    EXPECT_EQ(first, second);
  }
}

TEST(LnsSimilarity, StateCanChangeStructuralRankingAndPreservesCoefficientSigns)
{
  mip::fj_cpu_problem_t<int, double> problem;
  problem.n_variables   = 4;
  problem.n_constraints = 1;
  problem.offsets       = {0, 4};
  problem.variables     = {0, 1, 2, 3};
  problem.coefficients  = {1, 1, 0.5, -1};
  const std::vector<double> assignment{1, 0, 2, 1};
  mip::cpufj_lns_similarity_t<int, double> selector(problem, {0, 1, 2, 3});
  EXPECT_GT(selector.score(0, 1, assignment, 1), selector.score(0, 2, assignment, 1));
  EXPECT_LT(selector.score(0, 1, assignment, 0.5), selector.score(0, 2, assignment, 0.5));
  EXPECT_EQ(selector.score(0, 3, assignment, 1), 0);
  std::mt19937 rng(42);
  std::vector<int> selected;
  selector.select(0, assignment, 2, 1, rng, selected);
  EXPECT_EQ(selected, (std::vector<int>{0, 1}));
  selector.select(0, assignment, 2, 0.5, rng, selected);
  EXPECT_EQ(selected, (std::vector<int>{0, 2}));
}

TEST(LnsSimilarity, JaccardRewardsSharedIncidenceAndDuplicatesAreCoalesced)
{
  mip::fj_cpu_problem_t<int, double> problem;
  problem.n_variables   = 4;
  problem.n_constraints = 4;
  problem.offsets       = {0, 4, 6, 9, 9};
  problem.variables     = {0, 0, 1, 2, 0, 1, 2, 3, 3};
  problem.coefficients  = {0.25, 0.75, 1, 1, 1, 1, 1, 2, -2};
  const std::vector<double> assignment(4, 0);
  mip::cpufj_lns_similarity_t<int, double> selector(problem, {0, 1, 2, 3});
  EXPECT_DOUBLE_EQ(selector.score(0, 1, assignment, 1), 1);
  EXPECT_DOUBLE_EQ(selector.score(0, 2, assignment, 1), 1.0 / 3);
  EXPECT_DOUBLE_EQ(selector.score(0, 3, assignment, 1), 0);
  std::mt19937 rng(42);
  std::vector<int> selected;
  selector.select(3, assignment, 4, 0.5, rng, selected);
  EXPECT_EQ(selected, (std::vector<int>{3}));
  selector.select(0, assignment, 4, 1, rng, selected);
  EXPECT_EQ(selected, (std::vector<int>{0, 1, 2}));
}

TEST(LnsSimilarity, DenseRowSamplingIsBoundedDistinctAndNotAnIndexPrefix)
{
  mip::fj_cpu_problem_t<int, double> problem;
  problem.n_variables   = 1000;
  problem.n_constraints = 1;
  problem.offsets       = {0, 1000};
  problem.variables.resize(1000);
  std::iota(problem.variables.begin(), problem.variables.end(), 0);
  problem.coefficients.assign(1000, 1);
  auto eligible = problem.variables;
  eligible.erase(eligible.begin() + 1);  // An ineligible continuous or fixed variable.
  mip::cpufj_lns_similarity_t<int, double> selector(problem, eligible);
  std::mt19937 rng(42);
  std::vector<int> selected;
  selector.select(0, std::vector<double>(1000, 1), 1000, 0.5, rng, selected);
  ASSERT_EQ(selected.size(), 101);
  EXPECT_EQ(selected.front(), 0);
  EXPECT_EQ(std::count(selected.begin(), selected.end(), 1), 0);
  EXPECT_GT(*std::max_element(selected.begin(), selected.end()), 500);
  std::sort(selected.begin(), selected.end());
  EXPECT_EQ(std::unique(selected.begin(), selected.end()), selected.end());
}

}  // namespace cuopt::lns::test
