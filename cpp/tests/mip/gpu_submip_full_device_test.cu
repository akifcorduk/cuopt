/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math_optimization/tic_toc.hpp>
#include <mip_heuristics/gpu_submip_full_device.hpp>

#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>

namespace cuopt::mathematical_optimization::mip {
namespace {

struct concurrent_job_state_t {
  void arrive()
  {
    std::unique_lock lock(mutex);
    ++started;
    condition.notify_all();
    if (!condition.wait_for(lock, std::chrono::seconds(10), [this] { return started == 2; })) {
      throw std::runtime_error("Concurrent FP jobs did not reach the start barrier");
    }
  }

  void publish(int slot, const std::vector<double>& candidate, double objective)
  {
    int device;
    RAFT_CUDA_TRY(cudaGetDevice(&device));
    std::unique_lock lock(mutex);
    candidates[slot]        = candidate;
    objectives[slot]        = objective;
    devices[slot]           = device;
    entered_callbacks[slot] = true;
    condition.notify_all();
    condition.wait_for(lock, std::chrono::seconds(10), [this, slot] {
      return entered_callbacks[1 - slot] || finished[1 - slot];
    });
    overlapped[slot] = entered_callbacks[1 - slot];
  }

  void finish(int slot)
  {
    std::lock_guard lock(mutex);
    finished[slot] = true;
    condition.notify_all();
  }

  std::mutex mutex;
  std::condition_variable condition;
  int started{0};
  std::array<bool, 2> entered_callbacks{false, false};
  std::array<bool, 2> finished{false, false};
  std::array<bool, 2> overlapped{false, false};
  std::array<std::vector<double>, 2> candidates;
  std::array<double, 2> objectives{0., 0.};
  std::array<int, 2> devices{-1, -1};
};

struct concurrent_job_result_t {
  bool found{false};
  int restored_device{-1};
};

concurrent_job_result_t run_concurrent_full_device_job(concurrent_job_state_t& state,
                                                       int slot,
                                                       gpu_submip_job_t<int, double> job,
                                                       mip_solver_settings_t<int, double> settings,
                                                       gpu_submip_options_t options,
                                                       int target_device,
                                                       int original_device)
{
  try {
    RAFT_CUDA_TRY(cudaSetDevice(original_device));
    state.arrive();
    concurrent_job_result_t result;
    result.found = run_gpu_submip_job_full_device<int, double>(
      job,
      settings,
      nullptr,
      options,
      {},
      [&state, slot](const std::vector<double>& candidate, double objective) {
        state.publish(slot, candidate, objective);
      },
      target_device);
    RAFT_CUDA_TRY(cudaGetDevice(&result.restored_device));
    state.finish(slot);
    return result;
  } catch (...) {
    state.finish(slot);
    throw;
  }
}

class GpuSubmipFullDevice : public ::testing::Test {
 protected:
  GpuSubmipFullDevice() : lp(&handle, 1, 2, 2)
  {
    lp.A.col_start     = {0, 1, 2};
    lp.A.i             = {0, 0};
    lp.A.x             = {1., 1.};
    lp.rhs             = {0.};
    lp.objective       = {1., 0.};
    lp.lower           = {0., -1.};
    lp.upper           = {1., 0.};
    lp.obj_scale       = 1.;
    options.fix_rate   = 0.;
    options.time_limit = 10.;
    RAFT_CUDA_TRY(cudaGetDevice(&original_device));
    int device_count;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&device_count));
    target_device = (original_device + 1) % device_count;
  }

  gpu_submip_job_t<int, double> make_job()
  {
    return make_gpu_submip_job(
      lp,
      {simplex::variable_type_t::INTEGER, simplex::variable_type_t::CONTINUOUS},
      {0., 0.},
      {},
      0.5,
      options,
      settings,
      0,
      false);
  }

  void publish(const std::vector<double>& candidate, double objective)
  {
    RAFT_CUDA_TRY(cudaGetDevice(&publication_device));
    published           = candidate;
    published_objective = objective;
  }

  void expect_original_device()
  {
    int device;
    RAFT_CUDA_TRY(cudaGetDevice(&device));
    EXPECT_EQ(device, original_device);
    handle.sync_stream();
  }

  raft::handle_t handle;
  simplex::lp_problem_t<int, double> lp;
  mip_solver_settings_t<int, double> settings;
  gpu_submip_options_t options;
  int original_device{-1};
  int target_device{-1};
  int publication_device{-1};
  std::vector<double> published;
  double published_objective{1.};
};

TEST_F(GpuSubmipFullDevice, PublishesOnRequestedDeviceAndRestoresCaller)
{
  EXPECT_TRUE((run_gpu_submip_job_full_device<int, double>(
    make_job(),
    settings,
    nullptr,
    options,
    {},
    [this](const std::vector<double>& candidate, double objective) {
      publish(candidate, objective);
    },
    target_device)));
  EXPECT_EQ(publication_device, target_device);
  EXPECT_EQ(published, (std::vector<double>{0., 0.}));
  EXPECT_NEAR(published_objective, 0., 1e-6);
  expect_original_device();
}

TEST_F(GpuSubmipFullDevice, CallbackFailureRestoresCallerAndAllowsAnotherJob)
{
  const auto fail = [](const std::vector<double>&, double) {
    throw std::runtime_error("publication failed");
  };
  EXPECT_THROW((run_gpu_submip_job_full_device<int, double>(
                 make_job(), settings, nullptr, options, {}, fail, target_device)),
               std::runtime_error);
  expect_original_device();

  EXPECT_TRUE((run_gpu_submip_job_full_device<int, double>(
    make_job(),
    settings,
    nullptr,
    options,
    {},
    [this](const std::vector<double>& candidate, double objective) {
      publish(candidate, objective);
    },
    target_device)));
  EXPECT_EQ(published.size(), 2);
  expect_original_device();
}

TEST_F(GpuSubmipFullDevice, CancelledJobDoesNotPublish)
{
  std::atomic<int> halt{1};
  EXPECT_FALSE((run_gpu_submip_job_full_device<int, double>(
    make_job(),
    settings,
    &halt,
    options,
    {},
    [this](const std::vector<double>& candidate, double objective) {
      publish(candidate, objective);
    },
    target_device)));
  EXPECT_TRUE(published.empty());
  expect_original_device();
}

TEST_F(GpuSubmipFullDevice, ConcurrentCallsKeepBoundsAndSettingsIndependent)
{
  // Fractional LP seeds require projection after rounding. Opposite slack bounds
  // give the two simultaneous sessions different unique integer solutions.
  auto first_job                                = make_job();
  first_job.lp.lower[1]                         = -0.49;
  first_job.primal                              = {0.45, -0.45};
  auto second_job                               = make_job();
  second_job.job_id                             = 1;
  second_job.lp.upper[1]                        = -0.51;
  second_job.primal                             = {0.55, -0.55};
  second_job.cutoff                             = 1.5;
  auto first_settings                           = settings;
  auto second_settings                          = settings;
  first_settings.tolerances.absolute_tolerance  = 1e-8;
  second_settings.tolerances.absolute_tolerance = 1e-5;
  auto first_options                            = options;
  auto second_options                           = options;
  first_options.compact_fixed                   = true;
  second_options.compact_fixed                  = false;
  first_options.fused_pdlp                      = true;
  second_options.fused_pdlp                     = false;

  concurrent_job_state_t state;
  auto first  = std::async(std::launch::async,
                          run_concurrent_full_device_job,
                          std::ref(state),
                          0,
                          first_job,
                          first_settings,
                          first_options,
                          target_device,
                          original_device);
  auto second = std::async(std::launch::async,
                           run_concurrent_full_device_job,
                           std::ref(state),
                           1,
                           second_job,
                           second_settings,
                           second_options,
                           target_device,
                           original_device);
  concurrent_job_result_t first_result, second_result;
  EXPECT_NO_THROW(first_result = first.get());
  EXPECT_NO_THROW(second_result = second.get());
  EXPECT_TRUE(first_result.found);
  EXPECT_TRUE(second_result.found);
  EXPECT_TRUE(state.overlapped[0]);
  EXPECT_TRUE(state.overlapped[1]);
  ASSERT_EQ(state.candidates[0].size(), 2);
  ASSERT_EQ(state.candidates[1].size(), 2);
  EXPECT_NEAR(state.candidates[0][0], 0., 1e-7);
  EXPECT_NEAR(state.candidates[0][1], 0., 1e-7);
  EXPECT_NEAR(state.candidates[1][0], 1., 1e-5);
  EXPECT_NEAR(state.candidates[1][1], -1., 1e-5);
  EXPECT_NEAR(state.objectives[0], 0., 1e-6);
  EXPECT_NEAR(state.objectives[1], 1., 1e-6);
  EXPECT_EQ(state.devices[0], target_device);
  EXPECT_EQ(state.devices[1], target_device);
  EXPECT_EQ(first_result.restored_device, original_device);
  EXPECT_EQ(second_result.restored_device, original_device);
  EXPECT_EQ(first_job.lp.lower[1], -0.49);
  EXPECT_EQ(second_job.lp.upper[1], -0.51);
  EXPECT_EQ(first_settings.tolerances.absolute_tolerance, 1e-8);
  EXPECT_EQ(second_settings.tolerances.absolute_tolerance, 1e-5);
  expect_original_device();
}

TEST_F(GpuSubmipFullDevice, ContinuousSessionRestartsAfterSuccessAndStopsOnPairCancellation)
{
  auto job     = make_job();
  job.deadline = tic() + 10.;
  std::atomic<int> root_halt{0}, pair_halt{0};
  int publications = 0;
  int cutoff_reads = 0;
  // Keep the same cutoff to make two feasible descents deterministic. The second
  // starts from a perturbed prior projection and must repair x + slack = 0 again.
  EXPECT_TRUE((run_gpu_submip_job_full_device_continuous<int, double>(
    job,
    settings,
    &root_halt,
    &pair_halt,
    options,
    [&] {
      ++cutoff_reads;
      return 0.5;
    },
    [&](const std::vector<double>& candidate, double objective) {
      publish(candidate, objective);
      EXPECT_NEAR(candidate[0], 0., 1e-6);
      EXPECT_NEAR(candidate[1], 0., 1e-6);
      if (++publications == 2) { pair_halt.store(1, std::memory_order_release); }
    },
    target_device)));
  EXPECT_EQ(publications, 2);
  EXPECT_GE(cutoff_reads, 4);
  EXPECT_EQ(pair_halt.load(), 1);
  EXPECT_EQ(root_halt.load(), 0);
  EXPECT_EQ(job.primal, (std::vector<double>{0., 0.}));
  expect_original_device();
}

TEST_F(GpuSubmipFullDevice, ContinuousSessionRefreshesCutoffBeforeRestart)
{
  auto job     = make_job();
  job.deadline = tic() + 10.;
  std::atomic<int> pair_halt{0};
  double live_cutoff = 0.5;
  int cutoff_reads   = 0;
  int publications   = 0;
  bool refreshed     = false;
  const auto cutoff  = [&] {
    ++cutoff_reads;
    if (publications > 0) {
      refreshed = live_cutoff == 0.;
      pair_halt.store(1, std::memory_order_release);
    }
    return live_cutoff;
  };
  EXPECT_TRUE((run_gpu_submip_job_full_device_continuous<int, double>(
    job,
    settings,
    nullptr,
    &pair_halt,
    options,
    cutoff,
    [&](const std::vector<double>&, double objective) {
      ++publications;
      live_cutoff = objective;
    },
    target_device)));
  EXPECT_EQ(publications, 1);
  EXPECT_TRUE(refreshed);
  EXPECT_EQ(cutoff_reads, 3);
  expect_original_device();
}

TEST_F(GpuSubmipFullDevice, ContinuousSessionRestartsAfterCycleAndPairCancelLeavesRootRunning)
{
  // x = 0.5 has an LP solution but no integer solution. Its first natural FP
  // descent must fail; reaching the second cutoff refresh proves it restarted.
  simplex::lp_problem_t<int, double> fractional_lp(&handle, 1, 1, 1);
  fractional_lp.A.col_start = {0, 1};
  fractional_lp.A.i         = {0};
  fractional_lp.A.x         = {1.};
  fractional_lp.rhs         = {0.5};
  fractional_lp.objective   = {1.};
  fractional_lp.lower       = {0.};
  fractional_lp.upper       = {1.};
  fractional_lp.obj_scale   = 1.;
  auto job                  = make_gpu_submip_job(fractional_lp,
                                                  {simplex::variable_type_t::INTEGER},
                                 std::vector<double>{0.5},
                                                  {},
                                 2.,
                                 options,
                                 settings,
                                 7,
                                 false);
  job.deadline              = tic() + 10.;
  std::atomic<int> root_halt{0}, pair_halt{0};
  int cutoff_reads = 0;
  int publications = 0;
  EXPECT_FALSE((run_gpu_submip_job_full_device_continuous<int, double>(
    job,
    settings,
    &root_halt,
    &pair_halt,
    options,
    [&] {
      if (++cutoff_reads == 2) { pair_halt.store(1, std::memory_order_release); }
      return 2.;
    },
    [&](const std::vector<double>&, double) { ++publications; },
    target_device)));
  EXPECT_EQ(cutoff_reads, 2);
  EXPECT_EQ(publications, 0);
  EXPECT_EQ(pair_halt.load(), 1);
  EXPECT_EQ(root_halt.load(), 0);
  expect_original_device();
}

TEST_F(GpuSubmipFullDevice, ContinuousSessionWaitsForAnIncumbentThenUsesInitialSeed)
{
  auto job     = make_job();
  job.cutoff   = std::numeric_limits<double>::infinity();
  job.deadline = tic() + 10.;
  std::atomic<int> pair_halt{0};
  int cutoff_reads = 0;
  int publications = 0;
  EXPECT_TRUE((run_gpu_submip_job_full_device_continuous<int, double>(
    job,
    settings,
    nullptr,
    &pair_halt,
    options,
    [&] { return ++cutoff_reads < 3 ? std::numeric_limits<double>::infinity() : 0.5; },
    [&](const std::vector<double>& candidate, double objective) {
      ++publications;
      publish(candidate, objective);
      pair_halt.store(1, std::memory_order_release);
    },
    target_device)));
  EXPECT_EQ(publications, 1);
  EXPECT_GE(cutoff_reads, 5);
  EXPECT_EQ(published, (std::vector<double>{0., 0.}));
  expect_original_device();
}

TEST_F(GpuSubmipFullDevice, UnperturbableSessionWaitsForCpuAndMirrorsRootCancellation)
{
  auto job        = make_job();
  job.lp.upper[0] = 0.;
  job.lp.lower[1] = 0.;
  job.deadline    = tic() + 10.;
  std::atomic<int> root_halt{0}, pair_halt{0};
  std::promise<void> published_first;
  auto result      = std::async(std::launch::async, [&] {
    return run_gpu_submip_job_full_device_continuous<int, double>(
      job,
      settings,
      &root_halt,
      &pair_halt,
      options,
      {},
      [&](const std::vector<double>&, double) { published_first.set_value(); },
      target_device);
  });
  const auto ready = published_first.get_future().wait_for(std::chrono::seconds(3));
  EXPECT_EQ(ready, std::future_status::ready);
  // The GPU session has no free integer to perturb but must keep its ownership
  // until the paired CPU scope stops, rather than returning after its first FP.
  EXPECT_EQ(result.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  root_halt.store(1, std::memory_order_release);
  EXPECT_EQ(result.wait_for(std::chrono::seconds(3)), std::future_status::ready);
  EXPECT_TRUE(result.get());
  EXPECT_EQ(pair_halt.load(), 1);
  EXPECT_EQ(root_halt.load(), 1);
  expect_original_device();
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::mip
