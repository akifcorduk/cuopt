/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <utility>

namespace cuopt {

// Each pending or active GPU job owns one demand lease. The regular GPU worker drains and
// parks while demand exists, preserving its stack and CPU heuristic lifetimes between bursts.
class mip_gpu_handoff_t {
 public:
  using observer_t = std::function<void(const char*, uint64_t, uint64_t)>;

  void set_observer(observer_t observer)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    observer_ = std::move(observer);
  }

  void enable()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finished_ || enabled_) { return; }
    enabled_ = true;
    notify("tree");
  }

  bool acquire()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || finished_) { return false; }
    if (++demand_ == 1) {
      ++generation_;
      notify("request");
      requested_.store(1, std::memory_order_release);
      // A new job can arrive after the last release but before the regular worker wakes.
      // Its GPU is still drained in that case, so preserve the parked ownership interval.
      if (parked_ || regular_finished_) { publish_ready(); }
    }
    return true;
  }

  // Call only after every stream of the leased job has drained, including exception cleanup.
  void release()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (demand_ == 0) { return; }
      if (--demand_ != 0) { return; }
      ready_.store(false, std::memory_order_release);
      requested_.store(0, std::memory_order_release);
    }
    state_cv_.notify_all();
  }

  template <typename drain_t>
  void park_if_requested(drain_t&& drain)
  {
    if (!requested()) { return; }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (finished_ || demand_ == 0) { return; }
    }
    drain();
    std::unique_lock<std::mutex> lock(mutex_);
    if (finished_ || demand_ == 0) { return; }
    parked_ = true;
    publish_ready();
    // Cancellation invalidates readiness immediately, but existing leases still drain before
    // the regular worker can leave this safe point and perform its own cleanup submissions.
    state_cv_.wait(lock, [this]() { return demand_ == 0; });
    ready_.store(false, std::memory_order_release);
    parked_ = false;
    // This event belongs to the regular worker and precedes any resumed GPU submissions.
    if (!finished_) { notify("resume"); }
  }

  // The regular solver can return before demand exists. Its owner drains after assigning
  // the returned solution, before announcing that it will submit no further GPU work.
  void regular_finished()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    regular_finished_ = true;
    if (demand_ != 0 && !finished_) { publish_ready(); }
  }

  void finish()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (finished_) { return; }
      finished_ = true;
      ready_.store(false, std::memory_order_release);
      requested_.store(0, std::memory_order_release);
      notify("finished");
    }
    state_cv_.notify_all();
  }

  bool ready() const { return ready_.load(std::memory_order_acquire); }
  bool requested() const { return requested_.load(std::memory_order_acquire) != 0; }
  std::atomic<int>* halt_flag() { return &requested_; }

 private:
  void notify(const char* phase) noexcept
  {
    // Diagnostics cannot interrupt a lease transition and leave demand unrepresented.
    try {
      if (observer_) { observer_(phase, generation_, demand_); }
    } catch (...) {
    }
  }

  void publish_ready()
  {
    notify("ready");
    ready_.store(true, std::memory_order_release);
  }

  std::atomic<int> requested_{0};
  std::atomic<bool> ready_{false};
  std::mutex mutex_;
  std::condition_variable state_cv_;
  observer_t observer_;
  uint64_t demand_{0};
  uint64_t generation_{0};
  bool enabled_{false};
  bool parked_{false};
  bool regular_finished_{false};
  bool finished_{false};
};

}  // namespace cuopt
