/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <branch_and_bound/worker_pool_diagnostics.hpp>

#include <gtest/gtest.h>

namespace cuopt::mathematical_optimization::mip {
namespace {

TEST(WorkerPoolDiagnostics, IntegratesEachRealSlotAcrossOverlappingLeases)
{
  worker_pool_occupancy_t pool(2, 100);
  pool.acquire(0, 102);
  pool.acquire(1, 103);
  pool.release(0, 107);
  EXPECT_TRUE(pool.close(110));
  const auto summary = pool.summary();
  EXPECT_EQ(summary.elapsed_ns, 10);
  EXPECT_EQ(summary.capacity_ns, 20);
  EXPECT_EQ(summary.idle_ns, 8);
  EXPECT_EQ(summary.busy_ns, 12);
  EXPECT_EQ(summary.acquires, 2);
  EXPECT_EQ(summary.closes, 1);
  EXPECT_EQ(summary.busy_at_close, 1);
  EXPECT_EQ(pool.slots()[0].idle_ns, 5);
  EXPECT_EQ(pool.slots()[0].busy_ns, 5);
  EXPECT_EQ(pool.slots()[1].idle_ns, 3);
  EXPECT_EQ(pool.slots()[1].busy_ns, 7);
}

TEST(WorkerPoolDiagnostics, UnusedSlotsContributeTheirEntireWindowAsIdle)
{
  worker_pool_occupancy_t pool(3, 25);
  pool.close(75);
  const auto summary = pool.summary();
  EXPECT_EQ(summary.idle_ns, 150);
  EXPECT_EQ(summary.capacity_ns, 150);
  EXPECT_EQ(summary.busy_ns, 0);
  EXPECT_EQ(summary.acquires, 0);
  EXPECT_EQ(summary.closes, 0);
}

TEST(WorkerPoolDiagnostics, ReusedSlotCountsEveryLeaseWithoutDoubleCountingTime)
{
  worker_pool_occupancy_t pool(1, 0);
  pool.acquire(0, 1);
  pool.release(0, 4);
  pool.acquire(0, 4);
  pool.release(0, 7);
  pool.close(10);
  const auto summary = pool.summary();
  EXPECT_EQ(summary.idle_ns, 4);
  EXPECT_EQ(summary.busy_ns, 6);
  EXPECT_EQ(summary.capacity_ns, summary.idle_ns + summary.busy_ns);
  EXPECT_EQ(summary.acquires, 2);
  EXPECT_EQ(summary.closes, 2);
  EXPECT_EQ(summary.busy_at_close, 0);
}

TEST(WorkerPoolDiagnostics, CloseClipsOutstandingLeaseAndIsIdempotent)
{
  worker_pool_occupancy_t pool(1, 10);
  pool.acquire(0, 12);
  EXPECT_TRUE(pool.close(20));
  EXPECT_FALSE(pool.close(100));
  pool.release(0, 120);
  const auto summary = pool.summary();
  EXPECT_EQ(summary.idle_ns, 2);
  EXPECT_EQ(summary.busy_ns, 8);
  EXPECT_EQ(summary.closes, 0);
  EXPECT_EQ(summary.busy_at_close, 1);
}

TEST(WorkerPoolDiagnostics, SeparatePoolsKeepTheirOwnCapacityDenominators)
{
  worker_pool_occupancy_t root(2, 0);
  worker_pool_occupancy_t child(1, 3);
  root.acquire(0, 1);
  child.acquire(0, 4);
  child.release(0, 7);
  child.close(8);
  root.release(0, 9);
  root.close(10);
  EXPECT_EQ(root.summary().capacity_ns, 20);
  EXPECT_EQ(root.summary().idle_ns, 12);
  EXPECT_EQ(child.summary().capacity_ns, 5);
  EXPECT_EQ(child.summary().idle_ns, 2);
}

TEST(WorkerPoolDiagnostics, ZeroLengthWindowPreservesLeaseCounts)
{
  worker_pool_occupancy_t pool(1, 7);
  pool.acquire(0, 7);
  pool.release(0, 7);
  pool.close(7);
  const auto summary = pool.summary();
  EXPECT_EQ(summary.capacity_ns, 0);
  EXPECT_EQ(summary.idle_ns, 0);
  EXPECT_EQ(summary.busy_ns, 0);
  EXPECT_EQ(summary.acquires, 1);
  EXPECT_EQ(summary.closes, 1);
}

TEST(WorkerPoolDiagnostics, TaskQueueDelayIsSeparateFromInclusiveTaskLifetime)
{
  worker_pool_task_t task;
  task.acquire(100);
  task.queue(105);
  task.start(120);
  const auto duration = task.durations(170);
  EXPECT_EQ(duration.preparation_ns, 5);
  EXPECT_EQ(duration.queue_ns, 15);
  EXPECT_EQ(duration.task_ns, 50);
  EXPECT_EQ(duration.preparation_ns + duration.queue_ns + duration.task_ns, 70);
  EXPECT_EQ(task.lease, 1);
}

TEST(WorkerPoolDiagnostics, UndispatchedAndQueuedLeasesHaveNoExecutionTime)
{
  worker_pool_task_t task;
  task.acquire(10);
  auto duration = task.durations(20);
  EXPECT_EQ(duration.preparation_ns, 10);
  EXPECT_EQ(duration.queue_ns, 0);
  EXPECT_EQ(duration.task_ns, 0);
  task.queue(20);
  duration = task.durations(40);
  EXPECT_EQ(duration.preparation_ns, 10);
  EXPECT_EQ(duration.queue_ns, 20);
  EXPECT_EQ(duration.task_ns, 0);
}

TEST(WorkerPoolDiagnostics, ReusingSlotResetsTimingStateAndAdvancesLease)
{
  worker_pool_task_t task;
  task.acquire(0);
  task.queue(2);
  task.start(5);
  task.acquire(100);
  EXPECT_FALSE(task.queued);
  EXPECT_FALSE(task.started);
  EXPECT_EQ(task.lease, 2);
  auto duration = task.durations(110);
  EXPECT_EQ(duration.preparation_ns, 10);
  EXPECT_EQ(duration.queue_ns, 0);
  EXPECT_EQ(duration.task_ns, 0);
  task.queue(110);
  task.start(110);
  duration = task.durations(120);
  EXPECT_EQ(duration.preparation_ns, 10);
  EXPECT_EQ(duration.queue_ns, 0);
  EXPECT_EQ(duration.task_ns, 10);
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::mip
