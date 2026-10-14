/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <branch_and_bound/gpu_submip_pair.hpp>
#include <branch_and_bound/gpu_submip_queue.hpp>
#include <branch_and_bound/gpu_submip_worker.hpp>
#include <branch_and_bound/post_root_snapshot.hpp>
#include <dual_simplex/solve.hpp>
#include <mip_heuristics/feasibility_jump/fj_cpu_worker.cuh>
#include <utilities/pcgenerator.hpp>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <limits>
#include <optional>
#include <stdexcept>

namespace cuopt::mathematical_optimization::mip {
namespace {
using namespace std::chrono_literals;
using type_t = simplex::variable_type_t;
using job_t  = gpu_submip_job_t<int, double>;

class GpuSubmipFpConfig : public ::testing::Test {
 protected:
  void SetUp() override
  {
    for (size_t i = 0; i < keys.size(); ++i) {
      if (const char* value = std::getenv(keys[i])) { saved[i] = value; }
      unsetenv(keys[i]);
    }
    setenv("CUOPT_GPU_SUBMIP_CONFIG_FAMILY", "effort", 1);
  }

  void TearDown() override
  {
    for (size_t i = 0; i < keys.size(); ++i) {
      if (saved[i]) {
        setenv(keys[i], saved[i]->c_str(), 1);
      } else {
        unsetenv(keys[i]);
      }
    }
  }

  static constexpr std::array<const char*, 13> keys{"CUOPT_CONFIG_ID",
                                                    "CUOPT_MAX_CONFIG",
                                                    "CUOPT_GPU_SUBMIP_CONFIG_FAMILY",
                                                    "CUOPT_GPU_SUBMIP_FP_CONFIG",
                                                    "CUOPT_GPU_SUBMIP_MODE",
                                                    "CUOPT_GPU_SUBMIP_LIFETIME",
                                                    "CUOPT_GPU_SUBMIP_SECONDS",
                                                    "CUOPT_GPU_SUBMIP_MAX_PROJECTIONS",
                                                    "CUOPT_GPU_SUBMIP_MAX_DESCENTS",
                                                    "CUOPT_GPU_SUBMIP_INITIAL_ROUNDING",
                                                    "CUOPT_GPU_SUBMIP_SELECT_WARM_START",
                                                    "CUOPT_GPU_SUBMIP_UNPRODUCTIVE_SECONDS",
                                                    "CUOPT_GPU_SUBMIP_REGULAR_SECONDS"};
  std::array<std::optional<std::string>, keys.size()> saved;
};

TEST_F(GpuSubmipFpConfig, PresetsChangeOnlyTheirNamedFactors)
{
  const auto baseline = gpu_submip_fp_config_from_name("baseline");
  EXPECT_EQ(baseline, gpu_submip_fp_config_t{});
  EXPECT_EQ(gpu_submip_options_t{}.fp_config, baseline);
  auto expected                = baseline;
  expected.name                = "round_long";
  expected.rounding_seconds    = 2.;
  expected.propagation_seconds = 2.;
  EXPECT_EQ(gpu_submip_fp_config_from_name("round_long"), expected);
  expected              = baseline;
  expected.name         = "pdlp_long";
  expected.pdlp_seconds = 3.;
  EXPECT_EQ(gpu_submip_fp_config_from_name("pdlp_long"), expected);
  expected               = baseline;
  expected.name          = "distance";
  expected.initial_alpha = 0.;
  EXPECT_EQ(gpu_submip_fp_config_from_name("distance"), expected);
  EXPECT_THROW(gpu_submip_fp_config_from_name(""), std::invalid_argument);
  EXPECT_THROW(gpu_submip_fp_config_from_name("pdlp-long"), std::invalid_argument);
}

TEST_F(GpuSubmipFpConfig, CheckpointRepeatsSingleDescentAndAllowsRoundingControl)
{
  unsetenv("CUOPT_GPU_SUBMIP_CONFIG_FAMILY");
  setenv("CUOPT_MAX_CONFIG", "8", 1);
  for (int id = 0; id < 8; ++id) {
    setenv("CUOPT_CONFIG_ID", std::to_string(id).c_str(), 1);
    const auto options = gpu_submip_options_from_environment();
    EXPECT_EQ(options.mode, gpu_submip_mode_t::CPUFJ_FULL_GPU);
    EXPECT_TRUE(options.independent);
    EXPECT_EQ(options.max_descents, 1);
    EXPECT_EQ(options.max_projections, 0);
    EXPECT_EQ(options.time_limit, 60.);
    EXPECT_EQ(options.fp_config.initial_alpha, 0.);
    EXPECT_TRUE(options.fp_config.initial_rounding);
    EXPECT_TRUE(options.select_warm_start);
    EXPECT_EQ(options.unproductive_seconds, 15.);
    EXPECT_EQ(options.regular_gpu_seconds, 5.);
  }
  setenv("CUOPT_GPU_SUBMIP_INITIAL_ROUNDING", "0", 1);
  EXPECT_FALSE(gpu_submip_options_from_environment().fp_config.initial_rounding);
  EXPECT_FALSE(gpu_submip_fp_config_from_environment().initial_rounding);
  setenv("CUOPT_GPU_SUBMIP_SELECT_WARM_START", "0", 1);
  EXPECT_FALSE(gpu_submip_options_from_environment().select_warm_start);
  setenv("CUOPT_GPU_SUBMIP_UNPRODUCTIVE_SECONDS", "0", 1);
  EXPECT_EQ(gpu_submip_options_from_environment().unproductive_seconds, 0.);
  setenv("CUOPT_GPU_SUBMIP_INITIAL_ROUNDING", "bad", 1);
  EXPECT_THROW(gpu_submip_options_from_environment(), std::invalid_argument);
}

TEST_F(GpuSubmipFpConfig, EnvironmentUsesTheSharedValidatedParser)
{
  const char* key = "CUOPT_GPU_SUBMIP_FP_CONFIG";
  EXPECT_EQ(gpu_submip_options_from_environment().fp_config, gpu_submip_fp_config_t{});
  setenv(key, "round_long", 1);
  EXPECT_EQ(gpu_submip_options_from_environment().fp_config,
            gpu_submip_fp_config_from_name("round_long"));
  setenv(key, "unknown", 1);
  EXPECT_THROW(gpu_submip_options_from_environment(), std::invalid_argument);
}

TEST_F(GpuSubmipFpConfig, BenchmarkIdsSelectPresetsAndRepeatAcrossIterations)
{
  setenv("CUOPT_GPU_SUBMIP_CONFIG_FAMILY", "fp", 1);
  constexpr std::array<const char*, 4> names{"baseline", "round_long", "pdlp_long", "distance"};
  setenv("CUOPT_MAX_CONFIG", "8", 1);
  for (int id = 0; id < 8; ++id) {
    SCOPED_TRACE(id);
    setenv("CUOPT_CONFIG_ID", std::to_string(id).c_str(), 1);
    const auto expected = gpu_submip_fp_config_from_name(names[id % names.size()]);
    EXPECT_EQ(gpu_submip_options_from_environment().fp_config, expected);
    EXPECT_EQ(gpu_submip_fp_config_from_environment(), expected);
  }
}

TEST_F(GpuSubmipFpConfig, NumericIdOverridesLegacyNameAndDefaultsToFourConfigurations)
{
  setenv("CUOPT_GPU_SUBMIP_CONFIG_FAMILY", "fp", 1);
  setenv("CUOPT_CONFIG_ID", "2", 1);
  setenv("CUOPT_GPU_SUBMIP_FP_CONFIG", "unknown", 1);
  EXPECT_EQ(gpu_submip_options_from_environment().fp_config,
            gpu_submip_fp_config_from_name("pdlp_long"));
  setenv("CUOPT_CONFIG_ID", "4", 1);
  EXPECT_THROW(gpu_submip_options_from_environment(), std::invalid_argument);
}

TEST_F(GpuSubmipFpConfig, MaximumConfigurationCountValidatesPartialSweeps)
{
  setenv("CUOPT_MAX_CONFIG", "2", 1);
  EXPECT_EQ(gpu_submip_options_from_environment().fp_config, gpu_submip_fp_config_t{});
  setenv("CUOPT_CONFIG_ID", "1", 1);
  EXPECT_EQ(gpu_submip_options_from_environment().fp_config,
            gpu_submip_fp_config_from_name("distance"));
  setenv("CUOPT_CONFIG_ID", "2", 1);
  EXPECT_THROW(gpu_submip_options_from_environment(), std::invalid_argument);
}

TEST_F(GpuSubmipFpConfig, RejectsMalformedNegativeOverflowingAndOutOfRangeIds)
{
  setenv("CUOPT_MAX_CONFIG", "4", 1);
  for (const char* id : {"", "-1", "4", "1tail", "1.5", "2147483648"}) {
    SCOPED_TRACE(id);
    setenv("CUOPT_CONFIG_ID", id, 1);
    EXPECT_THROW(gpu_submip_options_from_environment(), std::invalid_argument);
  }
  setenv("CUOPT_CONFIG_ID", "0", 1);
  for (const char* count : {"", "0", "-1", "4tail", "4.5", "2147483648"}) {
    SCOPED_TRACE(count);
    setenv("CUOPT_MAX_CONFIG", count, 1);
    EXPECT_THROW(gpu_submip_options_from_environment(), std::invalid_argument);
  }
}

TEST_F(GpuSubmipFpConfig, IndependentEffortDoesNotChangeDistancePreset)
{
  setenv("CUOPT_CONFIG_ID", "1", 1);
  auto options = gpu_submip_options_from_environment();
  EXPECT_TRUE(options.independent);
  EXPECT_EQ(options.max_projections, 20);
  EXPECT_EQ(options.max_descents, 0);
  setenv("CUOPT_GPU_SUBMIP_MAX_PROJECTIONS", "100", 1);
  setenv("CUOPT_GPU_SUBMIP_MAX_DESCENTS", "1", 1);
  setenv("CUOPT_GPU_SUBMIP_LIFETIME", "paired", 1);
  options = gpu_submip_options_from_environment();
  EXPECT_FALSE(options.independent);
  EXPECT_EQ(options.max_projections, 100);
  EXPECT_EQ(options.max_descents, 1);
  EXPECT_EQ(options.fp_config, gpu_submip_fp_config_from_name("distance"));
  for (const char* key : {"CUOPT_GPU_SUBMIP_MAX_PROJECTIONS", "CUOPT_GPU_SUBMIP_MAX_DESCENTS"}) {
    for (const char* value : {"-1", "1.5", "bad", "2147483648"}) {
      setenv(key, value, 1);
      EXPECT_THROW(gpu_submip_options_from_environment(), std::invalid_argument);
    }
    unsetenv(key);
  }
  setenv("CUOPT_GPU_SUBMIP_LIFETIME", "invalid", 1);
  EXPECT_THROW(gpu_submip_options_from_environment(), std::invalid_argument);
}

TEST_F(GpuSubmipFpConfig, EffortSweepUsesIndependentDistanceFpAndRepeatsEveryFourIds)
{
  constexpr std::array<int, 4> projections{0, 20, 100, 100};
  constexpr std::array<int, 4> descents{1, 0, 0, 0};
  constexpr std::array<double, 4> seconds{60., 60., 10., 60.};
  setenv("CUOPT_MAX_CONFIG", "8", 1);
  // Numeric selection wins over stale named FP settings, as in the original sweep.
  setenv("CUOPT_GPU_SUBMIP_FP_CONFIG", "unknown", 1);
  for (int id = 0; id < 8; ++id) {
    SCOPED_TRACE(id);
    setenv("CUOPT_CONFIG_ID", std::to_string(id).c_str(), 1);
    const auto options = gpu_submip_options_from_environment();
    EXPECT_EQ(options.mode, gpu_submip_mode_t::CPUFJ_FULL_GPU);
    EXPECT_TRUE(options.independent);
    EXPECT_EQ(options.max_projections, projections[id % 4]);
    EXPECT_EQ(options.max_descents, descents[id % 4]);
    EXPECT_EQ(options.time_limit, seconds[id % 4]);
    EXPECT_EQ(options.fp_config, gpu_submip_fp_config_from_name("distance"));
    EXPECT_EQ(gpu_submip_fp_config_from_environment(), options.fp_config);
  }
}

TEST_F(GpuSubmipFpConfig, ExplicitEffortAndModeOverridesTakePrecedenceOverPresetDefaults)
{
  setenv("CUOPT_CONFIG_ID", "2", 1);
  setenv("CUOPT_GPU_SUBMIP_CONFIG_FAMILY", "effort", 1);
  auto options = gpu_submip_options_from_environment();
  EXPECT_EQ(options.time_limit, 10.);
  setenv("CUOPT_GPU_SUBMIP_SECONDS", "7.5", 1);
  setenv("CUOPT_GPU_SUBMIP_MAX_PROJECTIONS", "37", 1);
  setenv("CUOPT_GPU_SUBMIP_MAX_DESCENTS", "3", 1);
  setenv("CUOPT_GPU_SUBMIP_LIFETIME", "paired", 1);
  setenv("CUOPT_GPU_SUBMIP_MODE", "node", 1);
  options = gpu_submip_options_from_environment();
  EXPECT_EQ(options.mode, gpu_submip_mode_t::NODE_QUEUE);
  EXPECT_FALSE(options.independent);
  EXPECT_EQ(options.max_projections, 37);
  EXPECT_EQ(options.max_descents, 3);
  EXPECT_EQ(options.time_limit, 7.5);
  EXPECT_EQ(options.fp_config, gpu_submip_fp_config_from_name("distance"));
}

TEST_F(GpuSubmipFpConfig, RejectsUnknownFamiliesAndInvalidEffortTimes)
{
  setenv("CUOPT_CONFIG_ID", "2", 1);
  for (const char* family : {"", "unknown"}) {
    setenv("CUOPT_GPU_SUBMIP_CONFIG_FAMILY", family, 1);
    EXPECT_THROW(gpu_submip_options_from_environment(), std::invalid_argument);
    EXPECT_THROW(gpu_submip_fp_config_from_environment(), std::invalid_argument);
  }
  setenv("CUOPT_GPU_SUBMIP_CONFIG_FAMILY", "effort", 1);
  for (const char* value : {"", "-1", "0", "nan", "inf", "bad", "10tail"}) {
    SCOPED_TRACE(value);
    setenv("CUOPT_GPU_SUBMIP_SECONDS", value, 1);
    EXPECT_THROW(gpu_submip_options_from_environment(), std::invalid_argument);
  }
}

TEST(GpuSubmipQueue, IndependentAdmissionBoundsCopiesAndDoesNotCancelPoppedJob)
{
  gpu_submip_queue_t<int> queue(1);
  ASSERT_TRUE(queue.try_reserve(false).publish(std::make_unique<int>(1)));
  auto active = queue.pop_for(0ms);
  ASSERT_TRUE(active);
  ASSERT_TRUE(queue.try_reserve(false).publish(std::make_unique<int>(2)));
  EXPECT_FALSE(queue.try_reserve(false));
  EXPECT_EQ(*active, 1);
  EXPECT_EQ(*queue.pop_for(0ms), 2);
  queue.close();
  EXPECT_EQ(*active, 1);
}

TEST(GpuSubmipQueue, ReservationsBoundConstructionAndReleaseUnusedSlots)
{
  gpu_submip_queue_t<int> queue(2);
  auto first  = queue.try_reserve();
  auto second = queue.try_reserve();
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_FALSE(queue.try_reserve());
  first            = {};
  auto replacement = queue.try_reserve();
  ASSERT_TRUE(replacement);
  ASSERT_TRUE(replacement.publish(std::make_unique<int>(7)));
  ASSERT_TRUE(second.publish(std::make_unique<int>(8)));
  auto a = queue.pop_for(0ms);
  auto b = queue.pop_for(0ms);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  EXPECT_EQ(*a, 7);
  EXPECT_EQ(*b, 8);
  EXPECT_FALSE(queue.pop_for(0ms));
}

TEST(GpuSubmipQueue, ReplacesPendingWorkWithoutChangingActiveConsumerJob)
{
  gpu_submip_queue_t<int> queue(2);
  ASSERT_TRUE(queue.try_reserve().publish(std::make_unique<int>(1)));
  auto active = queue.pop_for(0ms);
  ASSERT_TRUE(active);
  ASSERT_TRUE(queue.try_reserve().publish(std::make_unique<int>(2)));
  ASSERT_TRUE(queue.try_reserve().publish(std::make_unique<int>(3)));
  ASSERT_TRUE(queue.try_reserve().publish(std::make_unique<int>(4)));
  auto next = queue.pop_for(0ms);
  auto last = queue.pop_for(0ms);
  ASSERT_TRUE(next);
  ASSERT_TRUE(last);
  EXPECT_EQ(*active, 1);
  EXPECT_EQ(*next, 3);
  EXPECT_EQ(*last, 4);
}

TEST(GpuSubmipQueue, CloseWakesConsumerAndRejectsInflightProducer)
{
  gpu_submip_queue_t<int> queue(2);
  auto reservation = queue.try_reserve();
  std::promise<void> ready;
  auto result = std::async(std::launch::async, [&] {
    ready.set_value();
    return queue.pop_for(5s);
  });
  ready.get_future().wait();
  queue.close();
  ASSERT_EQ(result.wait_for(1s), std::future_status::ready);
  EXPECT_FALSE(result.get());
  EXPECT_FALSE(reservation.publish(std::make_unique<int>(4)));
  EXPECT_FALSE(queue.try_reserve());
  EXPECT_TRUE(queue.closed());
}

TEST(GpuSubmipQueue, ReservationAndActiveJobOutliveQueueOwner)
{
  gpu_submip_queue_t<int>::reservation_t reservation;
  std::unique_ptr<int> active;
  {
    gpu_submip_queue_t<int> queue(2);
    ASSERT_TRUE(queue.try_reserve().publish(std::make_unique<int>(11)));
    active      = queue.pop_for(0ms);
    reservation = queue.try_reserve();
    queue.close();
  }
  ASSERT_TRUE(active);
  EXPECT_EQ(*active, 11);
  EXPECT_FALSE(reservation.publish(std::make_unique<int>(12)));
}

TEST(GpuSubmipPair, ScopeExitCancelsAndJoinsOwnedCleanup)
{
  std::promise<void> constructed;
  std::promise<void> allow_gpu_completion;
  auto ready = allow_gpu_completion.get_future().share();
  std::atomic<bool> finished{false};
  std::weak_ptr<const int> model_lifetime;
  auto owner     = std::make_shared<const int>(17);
  model_lifetime = owner;
  auto scope     = std::async(std::launch::async, [&, owner = std::move(owner)]() mutable {
    gpu_submip_pair_t pair(
      [model = std::move(owner), ready, &finished](std::atomic<int>* halt) {
        while (!halt->load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        ready.wait();  // Simulate outstanding device cleanup after cancellation.
        EXPECT_EQ(*model, 17);
        finished = true;
      },
      [](std::exception_ptr) { ADD_FAILURE() << "unexpected paired task exception"; });
    constructed.set_value();
    // Ending the full CPU sub-MIP scope cancels GPU work but still joins device cleanup.
  });
  constructed.get_future().wait();
  EXPECT_FALSE(model_lifetime.expired());
  EXPECT_EQ(scope.wait_for(10ms), std::future_status::timeout);
  EXPECT_FALSE(finished);
  allow_gpu_completion.set_value();
  scope.get();
  EXPECT_TRUE(finished);
  EXPECT_TRUE(model_lifetime.expired());
}

TEST(GpuSubmipPair, AsyncFailureIsReportedAndJoinedBeforeOwnerContinues)
{
  std::exception_ptr failure;
  std::atomic<int> halt{0};
  {
    gpu_submip_pair_t pair([](std::atomic<int>*) { throw std::runtime_error("paired FP failure"); },
                           [&](std::exception_ptr error) {
                             failure = error;
                             halt    = 1;
                           });
  }
  EXPECT_EQ(halt.load(), 1);
  ASSERT_TRUE(failure);
  EXPECT_THROW(std::rethrow_exception(failure), std::runtime_error);
}

TEST(GpuSubmipPair, CpuScopeCompletionIsReportedAfterGlobalCancellation)
{
  std::promise<void> global_cancelled;
  int stops = 0;
  gpu_submip_pair_t pair(
    [&](std::atomic<int>* halt) {
      // The full-device wrapper mirrors the root halt into this pair's token.
      halt->store(1, std::memory_order_release);
      global_cancelled.set_value();
    },
    [](std::exception_ptr) { ADD_FAILURE() << "unexpected paired task exception"; },
    {},
    [&]() { ++stops; });
  global_cancelled.get_future().wait();
  EXPECT_EQ(stops, 0);
  pair.request_stop();
  pair.request_stop();
  pair.join();
  EXPECT_EQ(stops, 1);
}

TEST(GpuSubmipPair, RequestStopIsLocalAndIdempotent)
{
  std::atomic<int> root_halt{0};
  std::promise<void> entered;
  std::atomic<bool> saw_stop{false};
  int stops = 0;
  gpu_submip_pair_t pair(
    [&](std::atomic<int>* halt) {
      EXPECT_EQ(halt->load(), 0);
      entered.set_value();
      while (!halt->load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      saw_stop = true;
    },
    [](std::exception_ptr) { ADD_FAILURE() << "unexpected paired task exception"; },
    {},
    [&]() { ++stops; });
  entered.get_future().wait();
  // A short CPUFJ invocation ending does not end its owning CPU sub-MIP scope.
  fj_cpu_worker_t<int, double>::notify_lifecycle({}, false);
  EXPECT_FALSE(saw_stop.load());
  pair.request_stop();
  pair.request_stop();
  pair.join();
  EXPECT_TRUE(saw_stop.load());
  EXPECT_EQ(stops, 1);
  EXPECT_EQ(root_halt.load(), 0);
}

TEST(GpuSubmipPair, CancellationRetainsDemandUntilDeviceCleanup)
{
  std::atomic<int> active{0};
  std::promise<void> entered;
  std::promise<void> stopped;
  std::promise<void> allow_cleanup;
  auto ready  = allow_cleanup.get_future().share();
  auto demand = std::make_shared<gpu_submip_demand_t>(
    [&]() {
      ++active;
      return true;
    },
    [&]() { --active; });
  gpu_submip_pair_t pair(
    [&, demand, ready](std::atomic<int>* halt) {
      entered.set_value();
      while (!halt->load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      stopped.set_value();
      ready.wait();  // Cancellation has arrived, but device resources have not drained.
      demand->release();
    },
    [](std::exception_ptr) { ADD_FAILURE() << "unexpected paired task exception"; });
  entered.get_future().wait();
  pair.request_stop();
  stopped.get_future().wait();
  EXPECT_EQ(active.load(), 1);
  auto joining = std::async(std::launch::async, [&]() { pair.join(); });
  EXPECT_EQ(joining.wait_for(10ms), std::future_status::timeout);
  allow_cleanup.set_value();
  joining.get();
  EXPECT_EQ(active.load(), 0);
}

TEST(GpuSubmipPair, ThrowingStopDiagnosticsCannotBypassCancellation)
{
  std::atomic<bool> stopped{false};
  {
    gpu_submip_pair_t pair(
      [&](std::atomic<int>* halt) {
        while (!halt->load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        stopped = true;
      },
      [](std::exception_ptr) { ADD_FAILURE() << "unexpected paired task exception"; },
      [](bool, bool) { throw std::runtime_error("join diagnostic"); },
      []() { throw std::runtime_error("stop diagnostic"); });
  }
  EXPECT_TRUE(stopped.load());
}

TEST(GpuSubmipPair, DemandCountsOverlappingJobsAndReleasesExactlyOnce)
{
  std::atomic<int> active{0};
  std::atomic<int> releases{0};
  const auto acquire = [&]() {
    ++active;
    return true;
  };
  const auto release = [&]() {
    --active;
    ++releases;
  };
  {
    gpu_submip_demand_t first(acquire, release);
    gpu_submip_demand_t second(acquire, release);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(active.load(), 2);
    first.release();
    first.release();
    EXPECT_EQ(active.load(), 1);
    EXPECT_EQ(releases.load(), 1);
  }
  EXPECT_EQ(active.load(), 0);
  EXPECT_EQ(releases.load(), 2);
  {
    gpu_submip_demand_t rejected([]() { return false; }, release);
    EXPECT_FALSE(rejected);
  }
  EXPECT_EQ(releases.load(), 2);
}

TEST(GpuSubmipPair, DemandSurvivesProducerReturnUntilTaskCleanup)
{
  std::atomic<int> active{0};
  std::promise<void> release_gpu;
  auto ready = release_gpu.get_future().share();
  std::unique_ptr<gpu_submip_pair_t> pair;
  {
    auto demand = std::make_shared<gpu_submip_demand_t>(
      [&]() {
        ++active;
        return true;
      },
      [&]() { --active; });
    pair = std::make_unique<gpu_submip_pair_t>(
      [demand, ready](std::atomic<int>*) {
        ready.wait();
        demand->release();
      },
      [](std::exception_ptr) { ADD_FAILURE() << "unexpected failure"; });
  }
  EXPECT_EQ(active.load(), 1);
  release_gpu.set_value();
  pair->join();
  EXPECT_EQ(active.load(), 0);
}

TEST(GpuSubmipPair, AbandonedStartupAndExceptionalTaskReleaseDemand)
{
  std::atomic<int> active{0};
  const auto acquire = [&]() {
    ++active;
    return true;
  };
  const auto release = [&]() { --active; };
  try {
    auto demand                     = std::make_shared<gpu_submip_demand_t>(acquire, release);
    std::function<void()> unstarted = [demand]() {};
    throw std::runtime_error("simulated thread startup failure");
  } catch (const std::runtime_error&) {
  }
  EXPECT_EQ(active.load(), 0);
  {
    auto demand = std::make_shared<gpu_submip_demand_t>(acquire, release);
    gpu_submip_pair_t pair(
      [demand](std::atomic<int>*) {
        struct cleanup_t {
          std::shared_ptr<gpu_submip_demand_t> demand;
          ~cleanup_t() { demand->release(); }
        } cleanup{demand};
        throw std::runtime_error("simulated GPU failure after cleanup");
      },
      [&](std::exception_ptr) { EXPECT_EQ(active.load(), 0); });
  }
  EXPECT_EQ(active.load(), 0);
}

TEST(GpuSubmipPair, FatalCancellationPrecedesDemandRelease)
{
  std::atomic<bool> fatal_cancelled{false};
  std::atomic<int> releases{0};
  auto demand = std::make_shared<gpu_submip_demand_t>([]() { return true; },
                                                      [&]() {
                                                        EXPECT_TRUE(fatal_cancelled.load());
                                                        ++releases;
                                                      });
  {
    gpu_submip_pair_t pair(
      [demand, &fatal_cancelled](std::atomic<int>*) {
        try {
          throw std::runtime_error("simulated executor failure after device cleanup");
        } catch (...) {
          fatal_cancelled = true;
        }
        demand->release();
      },
      [](std::exception_ptr) {
        ADD_FAILURE() << "failure must be handled before releasing demand";
      });
  }
  EXPECT_TRUE(fatal_cancelled.load());
  EXPECT_EQ(releases.load(), 1);
  demand.reset();
  EXPECT_EQ(releases.load(), 1);
}

TEST(GpuSubmipPair, JoinObserverTracksOnlyActualScopeJoin)
{
  std::promise<void> release_gpu;
  auto ready = release_gpu.get_future().share();
  int starts = 0;
  int ends   = 0;
  gpu_submip_pair_t pair([ready](std::atomic<int>*) { ready.wait(); },
                         [](std::exception_ptr) { ADD_FAILURE() << "unexpected failure"; },
                         [&](bool starting, bool already_finished) {
                           EXPECT_FALSE(already_finished);
                           if (starting) {
                             ++starts;
                             release_gpu.set_value();
                           } else {
                             ++ends;
                           }
                         });
  pair.join();
  pair.join();
  EXPECT_EQ(starts, 1);
  EXPECT_EQ(ends, 1);
}

TEST(GpuSubmipPair, CpuLifecycleDiagnosticsCannotInterruptSearch)
{
  int calls                          = 0;
  std::function<void(bool)> callback = [&](bool) {
    ++calls;
    throw std::runtime_error("diagnostic failure");
  };
  fj_cpu_worker_t<int, double>::notify_lifecycle(callback, true);
  fj_cpu_worker_t<int, double>::notify_lifecycle(callback, false);
  EXPECT_EQ(calls, 2);
  fj_cpu_worker_t<int, double>::notify_lifecycle({}, true);
}

TEST(GpuSubmipSchedule, MaterialProgressResetsStallAndBackoff)
{
  gpu_submip_schedule_t schedule(15., 5.);
  schedule.observe(10., 0.);
  EXPECT_FALSE(schedule.should_yield());
  schedule.observe(5., 1e-9);
  EXPECT_TRUE(schedule.should_yield());
  EXPECT_DOUBLE_EQ(schedule.regular_turn(), 5.);
  EXPECT_FALSE(schedule.should_yield());
  schedule.observe(15., 0.);
  EXPECT_DOUBLE_EQ(schedule.regular_turn(), 10.);
  schedule.observe(15., 0.);
  EXPECT_DOUBLE_EQ(schedule.regular_turn(), 20.);
  schedule.observe(15., 0.);
  EXPECT_DOUBLE_EQ(schedule.regular_turn(), 20.);
  schedule.observe(1., 1e-5);
  EXPECT_FALSE(schedule.should_yield());
  schedule.observe(15., 0.);
  EXPECT_DOUBLE_EQ(schedule.regular_turn(), 5.);
}

TEST(GpuSubmipSchedule, DisabledPolicyNeverYields)
{
  gpu_submip_schedule_t schedule(0., 5.);
  schedule.observe(600., 0.);
  EXPECT_FALSE(schedule.should_yield());
}

class GpuSubmipWorker : public ::testing::Test {
 protected:
  GpuSubmipWorker() : lp(nullptr, 1, 3, 3)
  {
    lp.A.col_start   = {0, 1, 2, 3};
    lp.A.i           = {0, 0, 0};
    lp.A.x           = {1, 1, 1};
    lp.rhs           = {1};
    lp.objective     = {1, 2, 0};
    lp.lower         = {0, 0, 0};
    lp.upper         = {1, 1, 1};
    lp.obj_constant  = 7;
    lp.obj_scale     = -2;
    settings.seed    = 42;
    options.fix_rate = 0.5;
  }
  simplex::lp_problem_t<int, double> lp;
  mip_solver_settings_t<int, double> settings;
  gpu_submip_options_t options;
  std::vector<type_t> types{type_t::BINARY, type_t::BINARY};
};

TEST_F(GpuSubmipWorker, WarmStartPrefersCompatibleProjectionAndRescoresSiblingBounds)
{
  auto job = make_gpu_submip_job(lp, types, {0.5, 0.5, 0.}, {}, 2., options, settings, 1, false);
  gpu_submip_seed_cache_t<int, double> cache;
  cache.remember(job, {0., 0., 1.}, {});
  auto selected = select_gpu_submip_seed(job, 1.9, settings, &cache);
  EXPECT_STREQ(selected.name, "previous_projection");
  EXPECT_EQ(selected.values, (std::vector<double>{0., 0., 1.}));
  // Clamping the old point violates the equality. The feasible CPU seed must win.
  job.lp.upper[2] = 0.;
  selected        = select_gpu_submip_seed(job, 1.9, settings, &cache);
  EXPECT_STREQ(selected.name, "node_primal");
  EXPECT_EQ(selected.values, job.primal);
  EXPECT_DOUBLE_EQ(cache.find(job)->projection[2], 1.);
}

TEST_F(GpuSubmipWorker, WarmStartUsesIncumbentBlendOrPreviousFpSolution)
{
  auto job =
    make_gpu_submip_job(lp, types, {0.5, 0.5, 0.}, {0., 0., 1.}, 2., options, settings, 1, false);
  gpu_submip_seed_cache_t<int, double> cache;
  auto selected = select_gpu_submip_seed(job, 1.9, settings, &cache);
  EXPECT_STREQ(selected.name, "incumbent_guided");
  EXPECT_EQ(selected.values, (std::vector<double>{0.375, 0.375, 0.25}));
  cache.remember(job, {}, {0., 0., 1.});
  selected = select_gpu_submip_seed(job, 1.9, settings, &cache);
  EXPECT_STREQ(selected.name, "previous_fp_solution");
  // A tighter objective cutoff must outweigh the old point's integrality.
  job.lp.objective = {0., 0., 10.};
  cache.remember(job, {}, {0., 0., 1.});
  selected = select_gpu_submip_seed(job, 1., settings, &cache);
  EXPECT_STREQ(selected.name, "node_primal");
}

TEST_F(GpuSubmipWorker, WarmStartCacheRequiresModelAndSourceIdentityAndIsBounded)
{
  auto job = make_gpu_submip_job(lp, types, {0.5, 0.5, 0.}, {}, 2., options, settings, 1, false);
  gpu_submip_seed_cache_t<int, double> cache;
  cache.remember(job, {0., 0., 1.}, {});
  ASSERT_NE(cache.find(job), nullptr);
  auto changed = job;
  ++changed.source_id;
  EXPECT_EQ(cache.find(changed), nullptr);
  changed           = job;
  changed.lp.A.x[0] = 2.;
  EXPECT_EQ(cache.find(changed), nullptr);
  changed              = job;
  changed.var_types[0] = type_t::CONTINUOUS;
  EXPECT_EQ(cache.find(changed), nullptr);
  cache.remember(job, {std::numeric_limits<double>::quiet_NaN(), 0., 1.}, {});
  EXPECT_DOUBLE_EQ(cache.find(job)->projection[0], 0.);
  for (int id = 1; id <= 4; ++id) {
    changed           = job;
    changed.source_id = id;
    cache.remember(changed, {0., 0., 1.}, {});
  }
  EXPECT_EQ(cache.find(job), nullptr);
  EXPECT_NE(cache.find(changed), nullptr);
}

TEST_F(GpuSubmipWorker, JobOwnsNodeAndMappingAfterProducerExit)
{
  gpu_submip_queue_t<job_t> queue;
  std::weak_ptr<const std::vector<double>> mapping_lifetime;
  {
    auto mapping     = std::make_shared<const std::vector<double>>(std::vector<double>{2, 3});
    mapping_lifetime = mapping;
    auto reservation = queue.try_reserve();
    auto job         = std::make_unique<job_t>(make_gpu_submip_job(lp,
                                                           types,
                                                           std::vector<double>{0.5, 0.5, 0},
                                                           std::vector<double>{},
                                                           2.0,
                                                           options,
                                                           settings,
                                                           4,
                                                           false));
    job->to_user     = [mapping](const auto& x, auto& out) {
      out = {x[0] + mapping->at(0), x[1] + mapping->at(1)};
      return true;
    };
    ASSERT_TRUE(reservation.publish(std::move(job)));
    // Simulate the producer moving to an opposite sibling and reusing its node storage.
    lp.lower[0] = lp.upper[0] = 1;
    lp.A.x[0]                 = 99;
    types[0]                  = type_t::CONTINUOUS;
  }
  EXPECT_FALSE(mapping_lifetime.expired());
  auto job = queue.pop_for(0ms);
  ASSERT_TRUE(job);
  EXPECT_EQ(job->lp.lower[0], 0);
  EXPECT_EQ(job->lp.upper[0], 1);
  EXPECT_EQ(job->lp.A.x[0], 1);
  EXPECT_EQ(job->var_types,
            (std::vector<type_t>{type_t::BINARY, type_t::BINARY, type_t::CONTINUOUS}));
  std::vector<double> user;
  ASSERT_TRUE(job->to_user({0, 1, 0}, user));
  EXPECT_EQ(user, (std::vector<double>{2, 4}));
  job.reset();
  EXPECT_TRUE(mapping_lifetime.expired());
}

TEST_F(GpuSubmipWorker, RinsFixingDoesNotMutateCpuNodeOrSolveItsRelaxation)
{
  const std::vector<double> primal{0, 0.5, 0.5};
  const std::vector<double> incumbent{0, 1, 0};
  const auto job =
    make_gpu_submip_job(lp, types, primal, incumbent, 2.0, options, settings, 1, true);
  EXPECT_EQ(job.fixed, 1);
  EXPECT_EQ(job.lp.lower[0], 0);
  EXPECT_EQ(job.lp.upper[0], 0);
  EXPECT_EQ(lp.upper[0], 1);
  EXPECT_EQ(job.primal, primal);
  EXPECT_EQ(job.incumbent, incumbent);
  EXPECT_DOUBLE_EQ(job.lp.obj_scale, -2);
  EXPECT_DOUBLE_EQ(job.lp.obj_constant, 7);
  EXPECT_DOUBLE_EQ(simplex::compute_user_objective(job.lp, 2.0), -18);
  EXPECT_DOUBLE_EQ(simplex::compute_presolved_objective(job.lp, -18.0), 2);
}

TEST_F(GpuSubmipWorker, QuickSubmipDfsKeepsItsBoundsDespiteOuterBnbOwnership)
{
  const std::vector<double> primal{0, 0.5, 0.5};
  const std::vector<double> incumbent{0, 1, 0};
  for (auto strategy :
       {search_strategy_t::RINS, search_strategy_t::RENS, search_strategy_t::MUTATION}) {
    const auto source = gpu_submip_source(false, strategy);
    ASSERT_EQ(source, gpu_submip_source_t::CPU_SUBMIP);
    const auto job = make_gpu_submip_job(lp,
                                         types,
                                         primal,
                                         incumbent,
                                         2.0,
                                         options,
                                         settings,
                                         1,
                                         gpu_submip_source_applies_rins(source));
    EXPECT_EQ(job.fixed, 0);
    EXPECT_EQ(job.lp.lower, lp.lower);
    EXPECT_EQ(job.lp.upper, lp.upper);
    EXPECT_EQ(job.primal, primal);
  }
}

TEST_F(GpuSubmipWorker, OrdinaryDivesKeepSeparateSourceAndMainFixingPolicy)
{
  EXPECT_EQ(gpu_submip_source(false, search_strategy_t::BEST_FIRST),
            gpu_submip_source_t::MAIN_TREE);
  for (auto strategy : {search_strategy_t::PSEUDOCOST_DIVING,
                        search_strategy_t::LINE_SEARCH_DIVING,
                        search_strategy_t::GUIDED_DIVING,
                        search_strategy_t::COEFFICIENT_DIVING,
                        search_strategy_t::FARKAS_DIVING,
                        search_strategy_t::VECTOR_LENGTH_DIVING}) {
    const auto source = gpu_submip_source(false, strategy);
    EXPECT_EQ(source, gpu_submip_source_t::DIVE);
    EXPECT_TRUE(gpu_submip_source_applies_rins(source));
    EXPECT_EQ(gpu_submip_source(true, strategy), gpu_submip_source_t::CPU_SUBMIP);
  }
  EXPECT_EQ(gpu_submip_source(true, search_strategy_t::BEST_FIRST),
            gpu_submip_source_t::CPU_SUBMIP);
}

TEST_F(GpuSubmipWorker, SnapshotDirectoryKeepsSourcesSeparate)
{
  for (auto source : {gpu_submip_source_t::MAIN_TREE,
                      gpu_submip_source_t::CPU_SUBMIP,
                      gpu_submip_source_t::DIVE}) {
    EXPECT_EQ(gpu_submip_job_snapshot_path(source, "/single.bin", "submip", "/jobs"),
              "/jobs/" + std::string(gpu_submip_source_name(source)) + ".bin");
  }
  EXPECT_EQ(
    gpu_submip_job_snapshot_path(gpu_submip_source_t::CPU_SUBMIP, "/single.bin", "submip", nullptr),
    "/single.bin");
  EXPECT_TRUE(
    gpu_submip_job_snapshot_path(gpu_submip_source_t::DIVE, "/single.bin", "submip", nullptr)
      .empty());
  EXPECT_EQ(gpu_submip_job_snapshot_path(gpu_submip_source_t::DIVE, "/single.bin", "dive", nullptr),
            "/single.bin");
  EXPECT_THROW(
    gpu_submip_job_snapshot_path(gpu_submip_source_t::MAIN_TREE, "/single.bin", "invalid", nullptr),
    std::invalid_argument);
}

TEST_F(GpuSubmipWorker, FiniteCutoffWithoutIncumbentSnapshotsExactSelectedJob)
{
  const auto job = make_gpu_submip_job(lp,
                                       types,
                                       std::vector<double>{0.5, 0.5, 0},
                                       std::vector<double>{},
                                       1.9,
                                       options,
                                       settings,
                                       0,
                                       false);
  const auto snapshot =
    capture_gpu_submip_job_snapshot(job.lp, job.var_types, job.primal, job.incumbent, job.cutoff);
  const auto path = std::filesystem::temp_directory_path() /
                    ("cuopt_gpu_job_" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  write_post_root_snapshot(path.string(), snapshot);
  const auto restored = read_post_root_snapshot<int, double>(path.string());
  std::filesystem::remove(path);
  EXPECT_EQ(restored.fields(), snapshot.fields());
  EXPECT_EQ(restored.lower, job.lp.lower);
  EXPECT_EQ(restored.upper, job.lp.upper);
  EXPECT_EQ(restored.root_primal, job.primal);
  EXPECT_EQ(restored.var_types, (std::vector<int8_t>{1, 1, 0}));
  EXPECT_EQ(restored.basis_status, (std::vector<int8_t>{-1, -1, -1}));
  EXPECT_TRUE(restored.incumbent.empty());
  EXPECT_TRUE(restored.edge_norms.empty());
  EXPECT_DOUBLE_EQ(restored.upper_bound, 1.9);
}

TEST_F(GpuSubmipWorker, CandidateAuditRejectsChangedSiblingBoundsAndInvalidValues)
{
  EXPECT_TRUE(gpu_submip_candidate_is_valid(lp, types, {0, 1, 0}, settings));
  EXPECT_FALSE(gpu_submip_candidate_is_valid(lp, types, {0.5, 0.5, 0}, settings));
  EXPECT_FALSE(gpu_submip_candidate_is_valid(lp, types, {0, 0, 0}, settings));
  EXPECT_FALSE(gpu_submip_candidate_is_valid(
    lp, types, {0, 1, std::numeric_limits<double>::quiet_NaN()}, settings));
  lp.lower[0] = 1;
  EXPECT_FALSE(gpu_submip_candidate_is_valid(lp, types, {0, 1, 0}, settings));
  EXPECT_TRUE(gpu_submip_candidate_is_valid(lp, types, {1, 0, 0}, settings));
}

TEST_F(GpuSubmipWorker, HaltedOrExpiredJobNeverInitializesGpuOrPublishes)
{
  auto job = make_gpu_submip_job(lp,
                                 types,
                                 std::vector<double>{0.5, 0.5, 0},
                                 std::vector<double>{},
                                 2.0,
                                 options,
                                 settings,
                                 0,
                                 false);
  std::atomic<int> halt{1};
  bool published = false;
  auto publish   = [&](const std::vector<double>&, double) { published = true; };
  EXPECT_FALSE(
    (run_gpu_submip_job<int, double>(job, settings, nullptr, &halt, options, {}, publish)));
  halt         = 0;
  job.deadline = 0;
  EXPECT_FALSE(
    (run_gpu_submip_job<int, double>(job, settings, nullptr, &halt, options, {}, publish)));
  EXPECT_FALSE(published);
}

TEST(GpuSubmipRestartSeed, PerturbationIsSparseDeterministicAndRespectsBounds)
{
  simplex::lp_problem_t<int, double> lp(nullptr, 0, 1003, 0);
  lp.lower.assign(lp.num_cols, 0.);
  lp.upper.assign(lp.num_cols, 1.);
  lp.lower[0] = lp.upper[0] = 2.;
  lp.lower[1]               = -2.;
  lp.upper[1]               = 3.;
  lp.lower[2]               = 0.1;
  lp.upper[2]               = 0.9;  // No admissible integer value: never perturb this coordinate.
  std::vector<type_t> types(lp.num_cols, type_t::INTEGER);
  types[1] = type_t::CONTINUOUS;
  std::vector<double> seed(lp.num_cols, 0.25);
  seed[0]             = 2.;
  seed[1]             = -0.75;
  const auto original = seed;
  auto same_seed      = seed;
  pcgenerator_t rng(41, 7), same_rng(41, 7);
  EXPECT_EQ(perturb_gpu_submip_restart_seed(lp, types, seed, rng), 8);
  EXPECT_EQ(perturb_gpu_submip_restart_seed(lp, types, same_seed, same_rng), 8);
  EXPECT_EQ(seed, same_seed);
  EXPECT_EQ(seed[0], 2.);
  EXPECT_EQ(seed[1], -0.75);
  EXPECT_EQ(seed[2], original[2]);
  int changed = 0;
  for (int j = 0; j < lp.num_cols; ++j) {
    EXPECT_GE(seed[j], lp.lower[j]);
    EXPECT_LE(seed[j], lp.upper[j]);
    changed += seed[j] != original[j];
    EXPECT_LE(std::abs(seed[j] - original[j]), 1.5);
  }
  EXPECT_EQ(changed, 8);
}

TEST_F(GpuSubmipWorker, RestartSeedClampsFixedAndContinuousCoordinatesWithoutPerturbingThem)
{
  lp.lower[0] = lp.upper[0] = 1.;
  lp.lower[1] = lp.upper[1] = 0.;
  std::vector<double> seed{-10., 12., 3.};
  pcgenerator_t rng(1, 2);
  EXPECT_EQ(perturb_gpu_submip_restart_seed(lp, types, seed, rng), 0);
  EXPECT_EQ(seed, (std::vector<double>{1., 0., 1.}));
  seed[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(perturb_gpu_submip_restart_seed(lp, types, seed, rng), std::invalid_argument);
}
}  // namespace
}  // namespace cuopt::mathematical_optimization::mip
