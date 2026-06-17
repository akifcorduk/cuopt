/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <chrono>
#include <limits>
#include <string>

namespace cuopt {

// TODO extend this for the whole solver.
// we are currently using this for diversity and adapters
class timer_t {
  using steady_clock = std::chrono::steady_clock;

 public:
  timer_t()               = delete;
  timer_t(const timer_t&) = default;
  timer_t(double time_limit_)
  {
    time_limit = time_limit_;
    begin      = steady_clock::now();
  }

  void print_debug(std::string msg) const
  {
    printf("%s time_limit: %f remaining_time: %f elapsed_time: %f \n",
           msg.c_str(),
           time_limit,
           remaining_time(),
           elapsed_time());
  }

  // Stop on the absolute work-unit budget (deterministic, work-clock only) OR the local time/work
  // budget. The absolute cap (@p work_limit_abs_) is an absolute value of the shared work counter,
  // so every timer sharing the counter stops at the same global work_limit regardless of when it
  // was created -- this is what makes `--work-limit` a reproducible global heuristic stop.
  bool check_time_limit() const noexcept
  {
    if (work_units_ != nullptr && *work_units_ >= work_limit_abs_) { return true; }
    return elapsed_time() >= time_limit;
  }

  bool check_half_time() const noexcept { return elapsed_time() >= time_limit / 2; }

  // Switch this timer to a virtual "work clock": elapsed/remaining are derived from accumulated
  // work units (deterministic) instead of wall-clock time. @p work_units points at a monotonically
  // increasing work counter (e.g. work_limit_context_t::global_work_units_elapsed); @p
  // work_per_second converts work units back into seconds so existing time-based budgets keep
  // working. @p work_limit_abs is an absolute global work-unit cap (infinity => disabled).
  // Passed as a raw double* to avoid a header dependency cycle.
  void use_work_clock(const double* work_units,
                      double work_per_second,
                      double work_limit_abs = std::numeric_limits<double>::infinity()) noexcept
  {
    work_units_      = work_units;
    work_per_second_ = work_per_second > 0.0 ? work_per_second : 1.0;
    work_begin_      = work_units != nullptr ? *work_units : 0.0;
    work_limit_abs_  = work_limit_abs;
  }

  double elapsed_time() const noexcept
  {
    if (work_units_ != nullptr) { return (*work_units_ - work_begin_) / work_per_second_; }
    return std::chrono::duration<double>(steady_clock::now() - begin).count();
  }

  double remaining_time() const noexcept
  {
    return std::max<double>(0.0, time_limit - elapsed_time());
  }

  double clamp_remaining_time(double desired_time) const noexcept
  {
    return std::min<double>(desired_time, remaining_time());
  }

  double get_time_limit() const noexcept { return time_limit; }

  double get_tic_start() const noexcept
  {
    /**
     * Converts a std::chrono::steady_clock::time_point to a struct timeval.
     * This is an approximate conversion because steady_clock is relative to an
     * unspecified epoch (e.g., system boot time), not the system clock epoch (UTC).
     */
    // Get the current time from both clocks at approximately the same instant
    std::chrono::system_clock::time_point sys_now    = std::chrono::system_clock::now();
    std::chrono::steady_clock::time_point steady_now = std::chrono::steady_clock::now();

    // Calculate the difference between the given steady_clock time point and the current steady
    // time
    auto diff_from_now = begin - steady_now;

    // Apply that same difference to the current system clock time point
    std::chrono::system_clock::time_point sys_t = sys_now + diff_from_now;

    // Convert the resulting system_clock time point to microseconds since the system epoch
    auto us_since_epoch =
      std::chrono::duration_cast<std::chrono::microseconds>(sys_t.time_since_epoch());

    // Populate the timeval struct
    double tv_sec  = us_since_epoch.count() / 1000000;
    double tv_usec = us_since_epoch.count() % 1000000;

    return tv_sec + 1e-6 * tv_usec;
  }

 private:
  double time_limit;
  steady_clock::time_point begin;
  // Optional virtual work-clock (deterministic mode). nullptr => wall-clock.
  const double* work_units_{nullptr};
  double work_per_second_{1.0};
  double work_begin_{0.0};
  // Absolute global work-unit cap (work-clock only). infinity => no work-unit limit.
  double work_limit_abs_{std::numeric_limits<double>::infinity()};
};

}  // namespace cuopt
