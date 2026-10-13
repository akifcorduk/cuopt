/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <utilities/logger.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// All timestamps use one monotonic clock. The caller serializes transitions with the pool
// lock, so slot time measures availability rather than CPU execution or task scheduling time.
class worker_pool_occupancy_t {
 public:
  struct slot_t {
    int64_t idle_ns{0};
    int64_t busy_ns{0};
    int64_t last_transition_ns{0};
    uint64_t acquires{0};
    uint64_t closes{0};
    bool acquired{false};
  };

  struct summary_t {
    int64_t elapsed_ns{0};
    int64_t capacity_ns{0};
    int64_t idle_ns{0};
    int64_t busy_ns{0};
    uint64_t acquires{0};
    uint64_t closes{0};
    uint64_t busy_at_close{0};
  };

  worker_pool_occupancy_t(size_t slots, int64_t start_ns) : slots_(slots), start_ns_(start_ns)
  {
    for (auto& slot : slots_) {
      slot.last_transition_ns = start_ns;
    }
  }

  void acquire(size_t index, int64_t now_ns)
  {
    if (closed_) { return; }
    auto& slot = slots_.at(index);
    assert(!slot.acquired);
    accumulate(slot, now_ns);
    slot.acquired = true;
    ++slot.acquires;
  }

  void release(size_t index, int64_t now_ns)
  {
    if (closed_) { return; }
    auto& slot = slots_.at(index);
    assert(slot.acquired);
    accumulate(slot, now_ns);
    slot.acquired = false;
    ++slot.closes;
  }

  bool close(int64_t now_ns)
  {
    if (closed_) { return false; }
    assert(now_ns >= start_ns_);
    for (auto& slot : slots_) {
      accumulate(slot, now_ns);
    }
    end_ns_ = now_ns;
    closed_ = true;
    return true;
  }

  summary_t summary() const
  {
    assert(closed_);
    summary_t result;
    result.elapsed_ns  = end_ns_ - start_ns_;
    result.capacity_ns = result.elapsed_ns * static_cast<int64_t>(slots_.size());
    for (const auto& slot : slots_) {
      result.idle_ns += slot.idle_ns;
      result.busy_ns += slot.busy_ns;
      result.acquires += slot.acquires;
      result.closes += slot.closes;
      result.busy_at_close += slot.acquired;
    }
    return result;
  }

  const std::vector<slot_t>& slots() const { return slots_; }

 private:
  static void accumulate(slot_t& slot, int64_t now_ns)
  {
    assert(now_ns >= slot.last_transition_ns);
    auto& duration = slot.acquired ? slot.busy_ns : slot.idle_ns;
    duration += now_ns - slot.last_transition_ns;
    slot.last_transition_ns = now_ns;
  }

  std::vector<slot_t> slots_;
  const int64_t start_ns_;
  int64_t end_ns_{0};
  bool closed_{false};
};

// These observations describe scheduler decisions, not wall-time attribution. In particular,
// a suppressed launch records the existing once-per-plunge policy without changing it.
enum class submip_schedule_reason_t : size_t {
  NODE_WITHOUT_CHILDREN,
  NODE_BOUND_PRUNED,
  PLUNGE_ALREADY_LAUNCHED,
  REQUEST,
  SOLVER_STOPPED,
  HALTED,
  DISABLED,
  NO_INCUMBENT,
  NO_SLOT,
  ACQUIRE_RACE,
  NO_STRATEGY,
  DISPATCHED,
  COUNT
};

inline const char* submip_schedule_reason_name(submip_schedule_reason_t reason)
{
  constexpr std::array names{"node_without_children",
                             "node_bound_pruned",
                             "plunge_already_launched",
                             "request",
                             "solver_stopped",
                             "halted",
                             "disabled",
                             "no_incumbent",
                             "no_slot",
                             "acquire_race",
                             "no_strategy",
                             "dispatched"};
  static_assert(names.size() == static_cast<size_t>(submip_schedule_reason_t::COUNT));
  return names.at(static_cast<size_t>(reason));
}

// Per-lease timestamps separate dispatch latency from the inclusive task scope. A task can
// execute nested CPU tasks and wait for GPU cleanup, so task time is not CPU execution time.
struct worker_pool_task_t {
  int64_t acquired_ns{0};
  int64_t queued_ns{0};
  int64_t started_ns{0};
  uint64_t lease{0};
  bool queued{false};
  bool started{false};

  struct durations_t {
    int64_t preparation_ns;
    int64_t queue_ns;
    int64_t task_ns;
  };

  void acquire(int64_t now_ns)
  {
    acquired_ns = now_ns;
    queued      = false;
    started     = false;
    ++lease;
  }

  void queue(int64_t now_ns)
  {
    assert(!queued && !started && now_ns >= acquired_ns);
    queued_ns = now_ns;
    queued    = true;
  }

  void start(int64_t now_ns)
  {
    assert(queued && !started && now_ns >= queued_ns);
    started_ns = now_ns;
    started    = true;
  }

  durations_t durations(int64_t now_ns) const
  {
    assert(now_ns >= (started ? started_ns : queued ? queued_ns : acquired_ns));
    return {(queued ? queued_ns : now_ns) - acquired_ns,
            queued ? (started ? started_ns : now_ns) - queued_ns : 0,
            started ? now_ns - started_ns : 0};
  }
};

class worker_pool_diagnostics_t {
 public:
  worker_pool_diagnostics_t(size_t slots, uint64_t pool_id, bool inside_submip, double root_start)
    : start_ns_(now_ns()),
      occupancy_(slots, start_ns_),
      tasks_(slots),
      pool_id_(pool_id),
      owner_(inside_submip ? "submip" : "root"),
      root_start_(root_start)
  {
    CUOPT_LOG_INFO(
      "SubmipPoolBegin pool_id=%llu owner=%s slots=%zu elapsed=%.6f "
      "host_monotonic_seconds=%.9f "
      "idle_definition=available_in_pool busy_definition=acquired_to_return "
      "busy_includes=preparation,queued_task,cpu_search,gpu_join "
      "queue_execution_split=instrumented task_time_definition=inclusive_not_cpu_busy",
      (unsigned long long)pool_id_,
      owner_,
      slots,
      elapsed(),
      seconds(start_ns_));
  }

  void acquire(size_t slot)
  {
    const auto now = now_ns();
    occupancy_.acquire(slot, now);
    tasks_.at(slot).acquire(now);
    task_event(slot, "acquire", now);
  }

  void release(size_t slot)
  {
    const auto now = now_ns();
    occupancy_.release(slot, now);
    const auto duration = tasks_.at(slot).durations(now);
    preparation_ns_ += duration.preparation_ns;
    queue_ns_ += duration.queue_ns;
    task_ns_ += duration.task_ns;
    if (!tasks_.at(slot).started) { ++returned_without_start_; }
    task_event(slot, "return", now);
  }

  void queued(size_t slot, const char* strategy, bool deferred)
  {
    const auto now = now_ns();
    tasks_.at(slot).queue(now);
    ++queued_;
    task_event(slot, "queued", now, strategy, deferred);
  }

  void started(size_t slot)
  {
    const auto now = now_ns();
    tasks_.at(slot).start(now);
    ++started_;
    task_event(slot, "start", now);
  }

  void task_stage(size_t slot, const char* stage) { task_event(slot, stage, now_ns()); }

  void observe(submip_schedule_reason_t reason, size_t idle_slots)
  {
    auto& observation = observations_.at(static_cast<size_t>(reason));
    ++observation.count;
    observation.with_idle_slot += idle_slots > 0;
    if (observation_events_ >= observation_event_limit_) { return; }
    ++observation_events_;
    CUOPT_LOG_INFO(
      "SubmipScheduleObservation pool_id=%llu owner=%s reason=%s idle_slots=%zu "
      "elapsed=%.6f host_monotonic_seconds=%.9f",
      (unsigned long long)pool_id_,
      owner_,
      submip_schedule_reason_name(reason),
      idle_slots,
      elapsed(),
      seconds(now_ns()));
  }

  void close()
  {
    const auto end_ns = now_ns();
    if (!occupancy_.close(end_ns)) { return; }
    const double end_elapsed = elapsed();
    const auto summary       = occupancy_.summary();
    const auto& slots        = occupancy_.slots();
    for (size_t index = 0; index < slots.size(); ++index) {
      const auto& slot = slots[index];
      CUOPT_LOG_INFO(
        "SubmipPoolSlot pool_id=%llu owner=%s slot=%zu idle_seconds=%.9f "
        "busy_seconds=%.9f capacity_seconds=%.9f idle_ratio=%.9f "
        "acquires=%llu closes=%llu busy_at_close=%d",
        (unsigned long long)pool_id_,
        owner_,
        index,
        seconds(slot.idle_ns),
        seconds(slot.busy_ns),
        seconds(summary.elapsed_ns),
        ratio(slot.idle_ns, summary.elapsed_ns),
        (unsigned long long)slot.acquires,
        (unsigned long long)slot.closes,
        int(slot.acquired));
    }
    CUOPT_LOG_INFO(
      "SubmipPoolSummary pool_id=%llu owner=%s slots=%zu elapsed=%.6f "
      "host_monotonic_seconds=%.9f "
      "window_seconds=%.9f idle_seconds=%.9f busy_seconds=%.9f "
      "capacity_seconds=%.9f idle_ratio=%.9f acquires=%llu closes=%llu "
      "busy_at_close=%llu",
      (unsigned long long)pool_id_,
      owner_,
      slots.size(),
      end_elapsed,
      seconds(end_ns),
      seconds(summary.elapsed_ns),
      seconds(summary.idle_ns),
      seconds(summary.busy_ns),
      seconds(summary.capacity_ns),
      ratio(summary.idle_ns, summary.capacity_ns),
      (unsigned long long)summary.acquires,
      (unsigned long long)summary.closes,
      (unsigned long long)summary.busy_at_close);
    CUOPT_LOG_INFO(
      "SubmipTaskSummary pool_id=%llu owner=%s queued=%llu started=%llu "
      "returned_without_start=%llu preparation_seconds=%.9f queue_seconds=%.9f "
      "task_seconds=%.9f completed_leases_only=1 task_events=%llu task_events_dropped=%llu "
      "observation_events=%llu observation_event_limit=%llu",
      (unsigned long long)pool_id_,
      owner_,
      (unsigned long long)queued_,
      (unsigned long long)started_,
      (unsigned long long)returned_without_start_,
      seconds(preparation_ns_),
      seconds(queue_ns_),
      seconds(task_ns_),
      (unsigned long long)std::min(task_events_, task_event_limit_),
      (unsigned long long)(task_events_ - std::min(task_events_, task_event_limit_)),
      (unsigned long long)observation_events_,
      (unsigned long long)observation_event_limit_);
    for (size_t i = 0; i < observations_.size(); ++i) {
      const auto& observation = observations_[i];
      CUOPT_LOG_INFO(
        "SubmipScheduleSummary pool_id=%llu owner=%s reason=%s count=%llu "
        "with_idle_slot=%llu",
        (unsigned long long)pool_id_,
        owner_,
        submip_schedule_reason_name(static_cast<submip_schedule_reason_t>(i)),
        (unsigned long long)observation.count,
        (unsigned long long)observation.with_idle_slot);
    }
  }

 private:
  void task_event(size_t slot,
                  const char* phase,
                  int64_t now,
                  const char* strategy = "unspecified",
                  bool deferred        = false)
  {
    if (++task_events_ > task_event_limit_) { return; }
    const auto& task    = tasks_.at(slot);
    const auto duration = task.durations(now);
    CUOPT_LOG_INFO(
      "SubmipTask pool_id=%llu owner=%s slot=%zu lease=%llu phase=%s strategy=%s "
      "deferred=%d elapsed=%.6f host_monotonic_seconds=%.9f "
      "preparation_seconds=%.9f queue_seconds=%.9f task_seconds=%.9f",
      (unsigned long long)pool_id_,
      owner_,
      slot,
      (unsigned long long)task.lease,
      phase,
      strategy,
      int(deferred),
      elapsed(),
      seconds(now),
      seconds(duration.preparation_ns),
      seconds(duration.queue_ns),
      seconds(duration.task_ns));
  }

  static int64_t now_ns()
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
  }

  double elapsed() const
  {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
             .count() -
           root_start_;
  }

  static double seconds(int64_t ns) { return double(ns) * 1e-9; }
  static double ratio(int64_t part, int64_t capacity)
  {
    return capacity > 0 ? double(part) / double(capacity) : 0.0;
  }

  int64_t start_ns_;
  worker_pool_occupancy_t occupancy_;
  std::vector<worker_pool_task_t> tasks_;
  struct observation_t {
    uint64_t count{0};
    uint64_t with_idle_slot{0};
  };
  std::array<observation_t, static_cast<size_t>(submip_schedule_reason_t::COUNT)> observations_{};
  static constexpr uint64_t observation_event_limit_ = 256;
  static constexpr uint64_t task_event_limit_        = 2048;
  uint64_t observation_events_{0};
  uint64_t task_events_{0};
  uint64_t queued_{0};
  uint64_t started_{0};
  uint64_t returned_without_start_{0};
  int64_t preparation_ns_{0};
  int64_t queue_ns_{0};
  int64_t task_ns_{0};
  uint64_t pool_id_;
  const char* owner_;
  double root_start_;
};

}  // namespace cuopt::mathematical_optimization::mip
