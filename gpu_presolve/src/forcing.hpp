/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// The forcing test (rules.hpp (a)) on a set of free entries: an activity of the entries meets
// its side, so every one of their columns sits at the bound that gives that activity. Shared by
// the forcing row, which evaluates it on the free entries of a row, and by simple probing
// (rules.hpp (g)), which evaluates it on the free entries of an equation other than its binary,
// once for each value of the binary.

#include <gprs/numerics.hpp>

#include <cmath>

namespace gprs {

// A bound of a continuous column lies up to 4 rounding units of its size outside the value that
// its row gives (bound propagation), so the activity of a row that is forcing in exact terms
// misses its side by up to that many units of the sum of its terms; twice that is allowed.
constexpr double forcing_rounding_units = 8.;

// What the test reads of the entries: the smallest and the largest magnitude of their
// coefficients, the sum of |a * value| over them, value being the bound of the column that
// gives the activity, and whether one of the columns is continuous.
struct forcing_entries_t {
  double smallest = inf;
  double largest  = 0.;
  double size     = 0.;
  bool continuous = false;

  void add(double a, double value, bool is_continuous)
  {
    double magnitude = std::fabs(a);
    if (magnitude < smallest) { smallest = magnitude; }
    if (magnitude > largest) { largest = magnitude; }
    size += std::fabs(a * value);
    continuous = continuous || is_continuous;
  }
};

// The part of the test that needs no entry: no set of entries is forcing with a gap beyond the
// feasibility tolerance.
inline bool forcing_possible(const tolerances_t& tol, double gap, double error)
{
  return std::fabs(gap) + error <= tol.feasibility;
}

// gap is side - activity, error the error bound of the activity, and the activity is the minimum
// (at_minimum) or the maximum one of the entries. They are forcing when the gap is zero within
// feasibility / dynamism. That alone keeps a feasible point within feasibility / max|a| of the
// fixed one, which is above the tolerance when every coefficient is below 1, so the tolerance
// is feasibility * min|a| / max(1, max|a|).
// An integer column has no point between its bound and that distance. A continuous column has:
// the points lie up to slack / |a| off the bound, where the slack is what the activity can
// still move towards the side, and putting the column on the bound moves them all onto it.
// Another row that holds the column with a larger coefficient is then violated by more than the
// slack; with bounds that propagation left a step short of their limit that makes a feasible
// model infeasible. So entries with a continuous column are forcing only when the slack is
// rounding: the error bound of the activity and forcing_rounding_units of its terms. An activity
// beyond the side, within the tolerance, has no such point and is forcing as before.
inline bool forcing(
  const tolerances_t& tol, const forcing_entries_t& f, double gap, double error, bool at_minimum)
{
  double tolerance = tol.feasibility * f.smallest / (f.largest > 1. ? f.largest : 1.);
  if (!(std::fabs(gap) + error <= tolerance)) { return false; }
  double slack = at_minimum ? gap : -gap;
  return !(f.continuous && slack > error + forcing_rounding_units * rounding_unit * f.size);
}

}  // namespace gprs
