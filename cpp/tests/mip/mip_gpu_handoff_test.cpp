/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <utilities/mip_gpu_handoff.hpp>
#include <utilities/scope_guard.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace cuopt {
namespace {
using namespace std::chrono_literals;
using event_t = std::tuple<std::string, uint64_t, uint64_t>;

void park_regular_worker(mip_gpu_handoff_t& handoff,
                         std::promise<void>& drain_started,
                         std::shared_future<void> drain_release)
{
  handoff.park_if_requested([&]() {
    drain_started.set_value();
    drain_release.wait();
  });
}

void park_without_work(mip_gpu_handoff_t& handoff)
{
  handoff.park_if_requested([]() {});
}

bool wait_until_ready(const mip_gpu_handoff_t& handoff)
{
  const auto deadline = std::chrono::steady_clock::now() + 1s;
  while (!handoff.ready() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return handoff.ready();
}

void run_regular_until_stopped(mip_gpu_handoff_t& handoff,
                               const std::atomic<bool>& stop,
                               std::atomic<int>& overlap)
{
  while (!stop.load()) {
    handoff.park_if_requested([]() {});
    if (handoff.ready()) { ++overlap; }
    std::this_thread::yield();
  }
}

TEST(MipGpuHandoff, TreeEntryOnlyEnablesEligibility)
{
  mip_gpu_handoff_t handoff;
  EXPECT_FALSE(handoff.acquire());
  handoff.enable();
  bool drained = false;
  handoff.park_if_requested([&]() { drained = true; });
  EXPECT_FALSE(drained);
  EXPECT_FALSE(handoff.ready());
  EXPECT_FALSE(handoff.requested());
  EXPECT_EQ(handoff.halt_flag()->load(), 0);
}

TEST(MipGpuHandoff, WaitsForDrainAndLastDemandBeforeResuming)
{
  mip_gpu_handoff_t handoff;
  std::promise<void> drain_started;
  std::promise<void> drain_release;
  auto release = drain_release.get_future().share();
  handoff.enable();
  EXPECT_TRUE(handoff.acquire());
  EXPECT_TRUE(handoff.acquire());
  auto worker = std::async(
    std::launch::async, park_regular_worker, std::ref(handoff), std::ref(drain_started), release);
  drain_started.get_future().wait();
  EXPECT_FALSE(handoff.ready());
  drain_release.set_value();
  EXPECT_TRUE(wait_until_ready(handoff));
  handoff.release();
  EXPECT_TRUE(handoff.ready());
  EXPECT_EQ(worker.wait_for(0ms), std::future_status::timeout);
  EXPECT_EQ(handoff.halt_flag()->load(), 1);
  handoff.release();
  EXPECT_EQ(worker.wait_for(1s), std::future_status::ready);
  worker.get();
  EXPECT_FALSE(handoff.ready());
  EXPECT_EQ(handoff.halt_flag()->load(), 0);
}

TEST(MipGpuHandoff, ANewBurstAfterResumeRequiresAnotherDrain)
{
  mip_gpu_handoff_t handoff;
  std::vector<event_t> events;
  handoff.set_observer([&](const char* phase, uint64_t generation, uint64_t demand) {
    events.emplace_back(phase, generation, demand);
  });
  handoff.enable();
  for (int burst = 0; burst < 2; ++burst) {
    EXPECT_TRUE(handoff.acquire());
    EXPECT_FALSE(handoff.ready());
    auto worker = std::async(std::launch::async, park_without_work, std::ref(handoff));
    EXPECT_TRUE(wait_until_ready(handoff));
    handoff.release();
    worker.get();
    EXPECT_FALSE(handoff.requested());
  }
  const std::vector<event_t> expected{{"tree", 0, 0},
                                      {"request", 1, 1},
                                      {"drain_begin", 1, 1},
                                      {"drain_end", 1, 1},
                                      {"ready", 1, 1},
                                      {"resume", 1, 0},
                                      {"request", 2, 1},
                                      {"drain_begin", 2, 1},
                                      {"drain_end", 2, 1},
                                      {"ready", 2, 1},
                                      {"resume", 2, 0}};
  EXPECT_EQ(events, expected);
}

TEST(MipGpuHandoff, ConsecutiveDemandAroundWakeNeverUsesStaleReadiness)
{
  mip_gpu_handoff_t handoff;
  std::atomic<bool> stop{false};
  std::atomic<int> overlap{0};
  handoff.enable();
  auto regular = std::async(std::launch::async,
                            run_regular_until_stopped,
                            std::ref(handoff),
                            std::cref(stop),
                            std::ref(overlap));
  for (int burst = 0; burst < 100; ++burst) {
    EXPECT_TRUE(handoff.acquire());
    EXPECT_TRUE(wait_until_ready(handoff));
    EXPECT_TRUE(handoff.acquire());
    handoff.release();
    EXPECT_TRUE(handoff.ready());
    handoff.release();
  }
  stop = true;
  regular.get();
  EXPECT_EQ(overlap.load(), 0);
  EXPECT_EQ(handoff.halt_flag()->load(), 0);
}

TEST(MipGpuHandoff, SafePointObservationDoesNotGrantOwnership)
{
  mip_gpu_handoff_t handoff;
  std::vector<event_t> events;
  handoff.set_observer([&](const char* phase, uint64_t generation, uint64_t demand) {
    events.emplace_back(phase, generation, demand);
  });
  handoff.enable();
  handoff.observe_safe_point();  // No pending request: no event.
  ASSERT_TRUE(handoff.acquire());
  handoff.observe_safe_point();
  EXPECT_FALSE(handoff.ready());
  handoff.regular_finished();
  EXPECT_TRUE(handoff.ready());
  handoff.release();
  handoff.finish();
  handoff.observe_safe_point();  // Cancellation must not look like a new handoff.
  const std::vector<event_t> expected{{"tree", 0, 0},
                                      {"request", 1, 1},
                                      {"safe_point", 1, 1},
                                      {"regular_finished", 1, 1},
                                      {"ready", 1, 1},
                                      {"finished", 1, 0}};
  EXPECT_EQ(events, expected);
}

TEST(MipGpuHandoff, FinishedRegularWorkerStillWaitsForActualDemand)
{
  mip_gpu_handoff_t handoff;
  handoff.regular_finished();
  EXPECT_FALSE(handoff.acquire());
  handoff.enable();
  EXPECT_FALSE(handoff.ready());
  EXPECT_TRUE(handoff.acquire());
  EXPECT_TRUE(handoff.ready());
  handoff.release();
  EXPECT_FALSE(handoff.ready());
  EXPECT_TRUE(handoff.acquire());
  EXPECT_TRUE(handoff.ready());
  handoff.release();
  handoff.finish();
  EXPECT_FALSE(handoff.acquire());
}

TEST(MipGpuHandoff, DemandReleasedDuringDrainDoesNotPark)
{
  mip_gpu_handoff_t handoff;
  handoff.enable();
  EXPECT_TRUE(handoff.acquire());
  handoff.park_if_requested([&]() { handoff.release(); });
  EXPECT_FALSE(handoff.ready());
  EXPECT_FALSE(handoff.requested());
}

TEST(MipGpuHandoff, FinishDuringDrainNeverActivatesConsumer)
{
  mip_gpu_handoff_t handoff;
  handoff.enable();
  EXPECT_TRUE(handoff.acquire());
  handoff.park_if_requested([&]() { handoff.finish(); });
  EXPECT_FALSE(handoff.ready());
  EXPECT_FALSE(handoff.requested());
  EXPECT_FALSE(handoff.acquire());
  handoff.release();
}

TEST(MipGpuHandoff, FinishingStillWaitsForActiveLeaseToDrain)
{
  mip_gpu_handoff_t handoff;
  handoff.enable();
  EXPECT_TRUE(handoff.acquire());
  auto worker = std::async(std::launch::async, park_without_work, std::ref(handoff));
  EXPECT_TRUE(wait_until_ready(handoff));
  handoff.finish();
  EXPECT_FALSE(handoff.ready());
  EXPECT_FALSE(handoff.acquire());
  EXPECT_EQ(worker.wait_for(0ms), std::future_status::timeout);
  handoff.release();
  worker.get();
  EXPECT_FALSE(handoff.ready());
}

TEST(MipGpuHandoff, DrainFailureNeverActivatesConsumer)
{
  mip_gpu_handoff_t handoff;
  handoff.enable();
  EXPECT_TRUE(handoff.acquire());
  EXPECT_THROW(handoff.park_if_requested([]() { throw std::runtime_error("drain failed"); }),
               std::runtime_error);
  EXPECT_FALSE(handoff.ready());
  handoff.release();
  EXPECT_FALSE(handoff.requested());
  handoff.finish();
}

TEST(MipGpuHandoff, LeaseUnwindReleasesCancelledPendingWork)
{
  mip_gpu_handoff_t handoff;
  handoff.enable();
  try {
    EXPECT_TRUE(handoff.acquire());
    auto release = scope_guard([&]() { handoff.release(); });
    throw std::runtime_error("job startup failed");
  } catch (const std::runtime_error&) {
  }
  EXPECT_FALSE(handoff.ready());
  EXPECT_FALSE(handoff.requested());
}

TEST(MipGpuHandoff, RegularTurnResumesBeforeNextAcquisition)
{
  mip_gpu_handoff_t handoff;
  handoff.enable();
  ASSERT_TRUE(handoff.acquire());
  auto worker = std::async(std::launch::async, park_without_work, std::ref(handoff));
  EXPECT_TRUE(wait_until_ready(handoff));
  handoff.release();
  std::atomic<int> halt{0};
  const auto started = std::chrono::steady_clock::now();
  EXPECT_TRUE(handoff.wait_for_regular(0.03, &halt));
  EXPECT_GE(std::chrono::steady_clock::now() - started, 25ms);
  worker.get();
  EXPECT_FALSE(handoff.ready());
  ASSERT_TRUE(handoff.acquire());
  EXPECT_FALSE(handoff.ready());
  auto next = std::async(std::launch::async, park_without_work, std::ref(handoff));
  EXPECT_TRUE(wait_until_ready(handoff));
  handoff.release();
  next.get();
}

TEST(MipGpuHandoff, RegularTurnSkipsFinishedWorkerAndObservesRootCancellation)
{
  mip_gpu_handoff_t handoff;
  handoff.enable();
  std::atomic<int> halt{1};
  EXPECT_FALSE(handoff.wait_for_regular(60., &halt));
  halt.store(0);
  auto turn = std::async(std::launch::async, [&] { return handoff.wait_for_regular(60., &halt); });
  halt.store(1);
  EXPECT_EQ(turn.wait_for(1s), std::future_status::ready);
  turn.get();
  halt.store(0);
  handoff.regular_finished();
  auto finished =
    std::async(std::launch::async, [&] { return handoff.wait_for_regular(60., &halt); });
  EXPECT_EQ(finished.wait_for(1s), std::future_status::ready);
  EXPECT_FALSE(finished.get());
}

TEST(MipGpuHandoff, FinishingWakesRegularTurn)
{
  mip_gpu_handoff_t handoff;
  handoff.enable();
  auto turn =
    std::async(std::launch::async, [&] { return handoff.wait_for_regular(60., nullptr); });
  handoff.finish();
  EXPECT_EQ(turn.wait_for(1s), std::future_status::ready);
  turn.get();
}

TEST(MipGpuHandoff, DiagnosticFailureDoesNotInterruptOwnershipTransitions)
{
  mip_gpu_handoff_t handoff;
  handoff.set_observer(
    [](const char*, uint64_t, uint64_t) { throw std::runtime_error("diagnostic failed"); });
  handoff.enable();
  EXPECT_TRUE(handoff.acquire());
  EXPECT_TRUE(handoff.requested());
  handoff.regular_finished();
  EXPECT_TRUE(handoff.ready());
  handoff.release();
  EXPECT_FALSE(handoff.ready());
  EXPECT_FALSE(handoff.requested());
  handoff.finish();
}
}  // namespace
}  // namespace cuopt
