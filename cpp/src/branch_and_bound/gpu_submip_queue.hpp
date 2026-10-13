/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace cuopt::mathematical_optimization::mip {

// A producer reserves space before constructing an expensive immutable node copy. Producers
// reserve without waiting. Publishing holds only the short metadata lock. Only pending jobs can
// be superseded; the
// consumer owns popped jobs independently of this queue and of their original CPU workers.
template <typename Job>
class gpu_submip_queue_t {
  struct state_t {
    explicit state_t(size_t capacity_) : capacity(capacity_) {}
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::unique_ptr<Job>> pending;
    size_t reserved{0};
    size_t capacity;
    bool closed{false};
  };

 public:
  class reservation_t {
   public:
    reservation_t() = default;
    explicit reservation_t(std::shared_ptr<state_t> state) : state_(std::move(state)) {}
    reservation_t(const reservation_t&)            = delete;
    reservation_t& operator=(const reservation_t&) = delete;
    reservation_t(reservation_t&&)                 = default;
    reservation_t& operator=(reservation_t&& other) noexcept
    {
      release();
      state_ = std::move(other.state_);
      return *this;
    }
    ~reservation_t() { release(); }
    explicit operator bool() const { return bool(state_); }

    bool publish(std::unique_ptr<Job> job)
    {
      if (!state_ || !job) { return false; }
      auto state = std::move(state_);
      std::lock_guard lock(state->mutex);
      --state->reserved;
      if (state->closed) { return false; }
      state->pending.push_back(std::move(job));
      state->changed.notify_one();
      return true;
    }

   private:
    void release()
    {
      if (!state_) { return; }
      auto state = std::move(state_);
      std::lock_guard lock(state->mutex);
      --state->reserved;
    }
    std::shared_ptr<state_t> state_;
  };

  explicit gpu_submip_queue_t(size_t capacity = 2) : state_(std::make_shared<state_t>(capacity))
  {
    if (capacity == 0) { throw std::invalid_argument("GPU job queue capacity must be positive"); }
  }

  reservation_t try_reserve()
  {
    std::unique_lock lock(state_->mutex, std::try_to_lock);
    if (!lock.owns_lock() || state_->closed) { return {}; }
    std::unique_ptr<Job> replaced;
    if (state_->reserved + state_->pending.size() == state_->capacity) {
      if (state_->pending.empty()) { return {}; }
      replaced = std::move(state_->pending.front());
      state_->pending.pop_front();
    }
    ++state_->reserved;
    lock.unlock();
    return reservation_t(state_);
  }

  std::unique_ptr<Job> pop_for(std::chrono::milliseconds timeout)
  {
    std::unique_lock lock(state_->mutex);
    state_->changed.wait_for(
      lock, timeout, [&] { return state_->closed || !state_->pending.empty(); });
    if (state_->closed || state_->pending.empty()) { return {}; }
    auto job = std::move(state_->pending.front());
    state_->pending.pop_front();
    return job;
  }

  void close()
  {
    std::deque<std::unique_ptr<Job>> discarded;
    {
      std::lock_guard lock(state_->mutex);
      state_->closed = true;
      discarded.swap(state_->pending);
    }
    state_->changed.notify_all();
  }

  bool closed() const
  {
    std::lock_guard lock(state_->mutex);
    return state_->closed;
  }

 private:
  std::shared_ptr<state_t> state_;
};

}  // namespace cuopt::mathematical_optimization::mip
