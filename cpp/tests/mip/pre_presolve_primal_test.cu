/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/pre_presolve_primal.cuh>

#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <thread>
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

  EXPECT_THROW(budget.release(1), std::logic_error);
  EXPECT_EQ(budget.available(), pre_presolve_primal_task_slots);
}

TEST(PreBnbSelectiveGate, CoversCompactAndSmallRowProblems)
{
  EXPECT_TRUE(pre_bnb_selective_problem_is_eligible(1'000, pre_bnb_compact_nnz_limit));
  EXPECT_FALSE(pre_bnb_selective_problem_is_eligible(1'000, pre_bnb_compact_nnz_limit + 1));
  EXPECT_TRUE(
    pre_bnb_selective_problem_is_eligible(pre_bnb_small_row_limit, pre_bnb_absolute_nnz_limit));
  EXPECT_FALSE(pre_bnb_selective_problem_is_eligible(pre_bnb_small_row_limit + 1,
                                                     pre_bnb_compact_nnz_limit + 1));
  EXPECT_FALSE(
    pre_bnb_selective_problem_is_eligible(pre_bnb_small_row_limit, pre_bnb_absolute_nnz_limit + 1));
}

TEST(PrePresolveFirstCandidatePublisher, SubmitsFirstCandidateThenRequestsStop)
{
  int callback_count = 0;
  double callback_solver_objective{};
  double callback_user_objective{};
  std::vector<double> callback_assignment;
  const char* callback_name = nullptr;
  pre_presolve_first_candidate_publisher_t<double>* publisher_ptr{};
  pre_presolve_first_candidate_publisher_t<double> publisher(
    [&](double solver_objective,
        double user_objective,
        const std::vector<double>& assignment,
        const char* heuristic_name) {
      EXPECT_FALSE(publisher_ptr->candidate_stop_requested());
      ++callback_count;
      callback_solver_objective = solver_objective;
      callback_user_objective   = user_objective;
      callback_assignment       = assignment;
      callback_name             = heuristic_name;
    },
    2.0,
    3.0,
    "FIRST-STOP-TEST");
  publisher_ptr = &publisher;

  publisher.consider(4.0, std::vector<double>{4.0});
  publisher.consider(2.0, std::vector<double>{2.0});

  const auto snapshot = publisher.snapshot();
  EXPECT_EQ(callback_count, 1);
  EXPECT_DOUBLE_EQ(callback_solver_objective, 4.0);
  EXPECT_DOUBLE_EQ(callback_user_objective, 14.0);
  EXPECT_EQ(callback_assignment, std::vector<double>({4.0}));
  EXPECT_STREQ(callback_name, "FIRST-STOP-TEST");
  EXPECT_EQ(snapshot.generated, 2);
  EXPECT_EQ(snapshot.submitted, 1);
  EXPECT_TRUE(snapshot.candidate_stop_requested);
}

TEST(PrePresolveFirstCandidatePublisher, RequestsStopWhenSubmissionThrows)
{
  pre_presolve_first_candidate_publisher_t<double> publisher(
    [](double, double, const std::vector<double>&, const char*) {
      throw std::runtime_error("callback failure");
    },
    1.0,
    0.0,
    "FIRST-STOP-TEST");

  EXPECT_THROW(publisher.consider(1.0, std::vector<double>{1.0}), std::runtime_error);

  const auto snapshot = publisher.snapshot();
  EXPECT_EQ(snapshot.generated, 1);
  EXPECT_EQ(snapshot.submitted, 0);
  EXPECT_TRUE(snapshot.candidate_stop_requested);
}

TEST(PrePresolveFirstCandidatePublisher, ConcurrentCandidatesSubmitExactlyOnce)
{
  constexpr int num_candidates = 16;
  std::atomic<int> callback_count{0};
  std::atomic<bool> start{false};
  pre_presolve_first_candidate_publisher_t<double> publisher(
    [&](double, double, const std::vector<double>&, const char*) {
      callback_count.fetch_add(1, std::memory_order_relaxed);
    },
    1.0,
    0.0,
    "FIRST-STOP-TEST");

  std::vector<std::thread> candidates;
  candidates.reserve(num_candidates);
  for (int i = 0; i < num_candidates; ++i) {
    candidates.emplace_back([&publisher, &start, i] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      publisher.consider(static_cast<double>(i), std::vector<double>{static_cast<double>(i)});
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& candidate : candidates) {
    candidate.join();
  }

  const auto snapshot = publisher.snapshot();
  EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(snapshot.generated, num_candidates);
  EXPECT_EQ(snapshot.submitted, 1);
  EXPECT_TRUE(snapshot.candidate_stop_requested);
}

}  // namespace cuopt::mathematical_optimization::mip::test
