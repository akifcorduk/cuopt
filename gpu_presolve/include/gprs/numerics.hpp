/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

// Every translation unit that includes this file is compiled with -ffp-contract=off: the only
// fused multiply-add is the explicit one in two_prod.

namespace gprs {

constexpr double inf           = std::numeric_limits<double>::infinity();
constexpr double rounding_unit = 0x1p-53;

struct tolerances_t {
  double feasibility = 1e-6;
  double dual        = 1e-7;
  double zero        = 1e-9;
  double integrality = 1e-5;
  // A term a * bound with a larger absolute value is counted as infinite in an activity.
  double infinite_term = 1e20;
  // Share of the feasibility tolerance that the error bound of an activity may reach: above it
  // the row is recomputed, and an integer bound is accepted only if error / |a| stays below it.
  double activity_error_share = 0.1;
  // An implied bound of a continuous column is recorded when it improves the record by more than
  // this many feasibility tolerances.
  double implied_bound_step = 1000.;
  // No bound is derived from a side b with |b| * side_error > feasibility.
  double side_error = 1e-14;
  // t + s * p is a structural zero when |t + s * p| <= cancellation * (|t| + |s * p|).
  double cancellation = 1e-12;
};

// s = fl(a + b) and error = (a + b) - s, exactly.
inline double two_sum(double a, double b, double& error)
{
  double s = a + b;
  double z = s - a;
  error    = (a - (s - z)) + (b - z);
  return s;
}

// p = fl(a * b) and error = a * b - p, exactly.
inline double two_prod(double a, double b, double& error)
{
  double p = a * b;
  error    = std::fma(a, b, -p);
  return p;
}

// Compensated accumulator: hi is the running sum in double arithmetic, lo collects the rounding
// errors of the additions and products. The value is hi + lo.
struct dd_t {
  double hi = 0.;
  double lo = 0.;
};

inline double dd_value(dd_t s) { return s.hi + s.lo; }

// s += a * b. The product must be finite.
inline void dd_add_product(dd_t& s, double a, double b)
{
  double product_error, sum_error;
  double p = two_prod(a, b, product_error);
  double t = two_sum(s.hi, p, sum_error);
  s.lo += sum_error + product_error;
  s.hi = t;
}

// s -= a * b. The product must be finite.
inline void dd_sub_product(dd_t& s, double a, double b)
{
  double product_error, sum_error;
  double p = two_prod(a, b, product_error);
  double t = two_sum(s.hi, -p, sum_error);
  s.lo += sum_error - product_error;
  s.hi = t;
}

// a + b; the merge step of the activity tree.
inline dd_t dd_add(dd_t a, dd_t b)
{
  double error;
  double t = two_sum(a.hi, b.hi, error);
  return dd_t{t, (a.lo + b.lo) + error};
}

// side - s as one double, for a finite side (an infinite one gives NaN). An implied bound is
// dd_residual(side, activity without the column's own term) / a, never
// (side - activity) / a + own bound.
inline double dd_residual(double side, dd_t s)
{
  double error;
  double d = two_sum(side, -s.hi, error);
  return d + (error - s.lo);
}

inline bool is_infinite_term(double a, double bound, const tolerances_t& tol)
{
  return !(std::fabs(a * bound) <= tol.infinite_term);
}

// The bound of a column with coefficient a that enters the minimum / maximum activity of a row.
inline double min_activity_bound(double a, double lower, double upper)
{
  return a > 0 ? lower : upper;
}

inline double max_activity_bound(double a, double lower, double upper)
{
  return a > 0 ? upper : lower;
}

// Bound of the error of a compensated activity after `updates` added terms of absolute value at
// most max_term.
inline double activity_error(uint32_t updates, double max_term)
{
  return updates * rounding_unit * rounding_unit * max_term;
}

// Bounds of an integer column implied by a real lower / upper bound.
inline double round_lower_integer(double lower, const tolerances_t& tol)
{
  return std::ceil(lower - tol.feasibility);
}

inline double round_upper_integer(double upper, const tolerances_t& tol)
{
  return std::floor(upper + tol.feasibility);
}

// Summation order of an activity computed from scratch, the same in every engine: the n live
// entries of the row, in storage order, are dealt to activity_lanes(n) lanes (entry k to lane
// k mod lanes), each lane adds its terms in order with dd_add_product, and the lanes are merged
// by  for (off = lanes / 2; off > 0; off /= 2) lane[l] = dd_add(lane[l], lane[l + off]), l < off.
// Up to max_light_row entries a lane holds at most two entries.
constexpr int32_t max_light_row      = 64;
constexpr int32_t max_activity_lanes = 256;

inline int32_t activity_lanes(int32_t n)
{
  if (n > max_light_row) { return max_activity_lanes; }
  int32_t lanes = 1;
  while (2 * lanes < n) {
    lanes *= 2;
  }
  return lanes;
}

}  // namespace gprs
