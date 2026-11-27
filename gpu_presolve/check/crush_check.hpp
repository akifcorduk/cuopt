/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// The crush check: known feasible points of the original model against one presolve result.
// It needs no solver and costs O(nonzeros) per point.
//
// For a feasible point x of the original model:
//   1. the status carries a reduced model;
//   2. crush(x) meets rows, bounds and integrality of the reduced model;
//   3. z = postsolve(crush(x)) meets rows, bounds and integrality of the original model;
//   4. objective(z) equals the reduced objective of crush(x), and is not above objective(x);
//   5. primal-only mode: objective(z) equals objective(x) and z equals x.
//
// Why 2 to 4 hold in default mode although postsolve_t::crush promises them in primal-only mode
// only. crush(x) is x on the columns that are left. Let M_t be the model after t reductions and
// p_t the point x restricted to the columns of M_t. Every Stage 1 rule but one keeps every
// feasible point of M_t: bound facts (start step, singleton and forcing rows, propagation,
// recorded implied bounds, bounds from a doubleton equation) hold at every feasible point, so a
// column they fix has its fixed value at p_t; row edits (redundancy, integral scaling, side
// rounding, coefficient strengthening, parallel rows) keep the integer-feasible set of the row
// under the explicit bounds; a doubleton substitution removes a column whose value the equation
// gives. The exception is dual fixing by locks, which fixes column j at a bound although
// p_t[j] may lie elsewhere. Dropping j from p_t is the same as moving it to that bound, and no
// live row of M_t objects to the move (no lock in that direction) and the objective of M_t
// does not rise with it. So p_(t+1) is feasible in M_(t+1) and its objective there is not above
// that of p_t; a bound fact of M_t stays true on the smaller feasible set. By induction crush(x) is
// feasible in the reduced model with a reduced objective not above objective(x), and 3 and the
// first half of 4 are the contract of the postsolve. What is lost in default mode is equality:
// z differs from x in the columns dual fixing moved and in their aliases. This argument needs
// dual fixing to be the only rule that uses a lock or dual argument and to fix at a finite
// bound; it has to be reread when a Stage 3 rule (dominated or free columns, stuffing, fixing
// at infinity) is added.
//
// With an optimal x the check proves that the optimum is not cut off. It sees no relaxation
// (a row deleted that is not redundant) unless that makes a later lock argument move a column
// across the deleted row; relaxations are the job of the solver checks in gprs_check.cpp.

#include <gprs/presolve.hpp>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace gprs_check {

// A point of a solution file is used when its integer columns are within point_integrality of
// integers and, with them rounded, it meets the original model within point_tolerance. The
// points derived from it are tested with check_tolerance, ten times wider, since a reduction
// can enlarge a violation (a bound from a singleton row is the row's violation divided by the
// coefficient).
constexpr double point_tolerance   = 1e-7;
constexpr double point_integrality = 1e-5;
constexpr double check_tolerance   = 1e-6;
// Two objective values of one point agree within this, relative to the sum of |cost * value|.
constexpr double identity_tolerance = 1e-9;
// Two objective values of different points are compared with this, relative to the value.
constexpr double objective_tolerance = 1e-6;

// Compensated sum of products, independent of the engine's.
struct sum_t {
  double hi = 0.;
  double lo = 0.;

  void add(double a, double b)
  {
    double p     = a * b;
    double p_err = std::fma(a, b, -p);
    double s     = hi + p;
    double z     = s - hi;
    lo += ((hi - (s - z)) + (p - z)) + p_err;
    hi = s;
  }

  double value() const { return hi + lo; }
};

// worst = max(worst, v), and NaN once a v is NaN.
inline void raise_to(double& worst, double v)
{
  if (v > worst || std::isnan(v)) { worst = v; }
}

// Largest violations of a point. A row violation is also given relative to
// max(1, largest |a * x| of the row), a bound violation relative to max(1, |bound|): one unit
// of an integer column is visible in a row with large terms, and a rounding error of a term of
// size 1e7 is not a violation.
struct point_report_t {
  double row            = 0.;
  double row_scaled     = 0.;
  int32_t worst_row     = -1;
  double bound          = 0.;
  double bound_scaled   = 0.;
  int32_t worst_col     = -1;
  double integrality    = 0.;
  double objective      = 0.;
  double objective_size = 0.;  // |offset| + sum of |cost * x|

  bool within(double tolerance) const
  {
    return row_scaled <= tolerance && bound_scaled <= tolerance && integrality <= tolerance;
  }
};

inline point_report_t check_point_scaled(const gprs::model_t& model, const std::vector<double>& x)
{
  point_report_t r;
  for (int32_t i = 0; i < model.nrows; ++i) {
    sum_t activity;
    double largest = 1.;
    for (int64_t e = model.row_start[i]; e < model.row_start[i + 1]; ++e) {
      double a = model.value[e], v = x[model.col_index[e]];
      activity.add(a, v);
      raise_to(largest, std::fabs(a * v));
    }
    double value = activity.value();
    double violation =
      std::isnan(value) ? value : std::fmax(model.row_lower[i] - value, value - model.row_upper[i]);
    if (violation <= 0.) { continue; }
    if (!(violation / largest <= r.row_scaled)) { r.worst_row = i; }
    raise_to(r.row, violation);
    raise_to(r.row_scaled, violation / largest);
  }
  sum_t objective;
  objective.add(model.objective_offset, 1.);
  r.objective_size = std::fabs(model.objective_offset);
  for (int32_t j = 0; j < model.ncols; ++j) {
    double v     = x[j];
    double below = model.col_lower[j] - v, above = v - model.col_upper[j];
    if (!(below <= 0. && above <= 0.)) {
      double violation = std::isfinite(v) ? std::fmax(below, above) : gprs::inf;
      double bound     = below > above ? model.col_lower[j] : model.col_upper[j];
      double scaled    = std::isfinite(v) ? violation / std::fmax(1., std::fabs(bound)) : gprs::inf;
      if (!(scaled <= r.bound_scaled)) { r.worst_col = j; }
      raise_to(r.bound, violation);
      raise_to(r.bound_scaled, scaled);
    }
    if (model.col_type[j]) { raise_to(r.integrality, std::fabs(v - std::round(v))); }
    objective.add(model.cost[j], v);
    r.objective_size += std::fabs(model.cost[j] * v);
  }
  r.objective = objective.value();
  return r;
}

inline bool carries_model(gprs::status_t status)
{
  return status == gprs::status_t::reduced || status == gprs::status_t::unchanged ||
         status == gprs::status_t::reduced_to_empty;
}

// Rounds the integer columns of x and reports the rounded point. True when x was within
// point_integrality of integers there and the rounded point meets the model within
// point_tolerance.
inline bool usable_point(const gprs::model_t& model,
                         std::vector<double>& x,
                         point_report_t& report,
                         double& integrality)
{
  integrality = 0.;
  for (int32_t j = 0; j < model.ncols; ++j) {
    if (!model.col_type[j]) { continue; }
    double rounded = std::round(x[j]);
    raise_to(integrality, std::fabs(x[j] - rounded));
    x[j] = rounded;
  }
  report = check_point_scaled(model, x);
  return integrality <= point_integrality && report.within(point_tolerance);
}

// The usable points of the solution files of one instance: integer columns rounded.
struct point_set_t {
  std::vector<std::vector<double>> points;
  std::vector<double> objective;
  int32_t read     = 0;
  int32_t rejected = 0;
  // Largest violations among the rejected points, for the report.
  double rejected_row         = 0.;
  double rejected_bound       = 0.;
  double rejected_integrality = 0.;

  // Takes the point when it is one of the original model.
  void add(const gprs::model_t& model, std::vector<double> x)
  {
    ++read;
    point_report_t r;
    double integrality;
    if (usable_point(model, x, r, integrality)) {
      objective.push_back(r.objective);
      points.push_back(std::move(x));
      return;
    }
    ++rejected;
    raise_to(rejected_row, r.row_scaled);
    raise_to(rejected_bound, r.bound_scaled);
    raise_to(rejected_integrality, integrality);
  }

  bool empty() const { return points.empty(); }

  double best_objective() const
  {
    double best = gprs::inf;
    for (double v : objective) {
      best = std::fmin(best, v);
    }
    return best;
  }
};

// The worst values over the points, and the number of points that fail each requirement.
struct crush_report_t {
  bool ran = false;
  std::string status;
  bool has_reduced = false;
  uint64_t hash    = 0;
  int32_t points   = 0;

  double reduced_row         = 0.;  // scaled, as in point_report_t
  double reduced_bound       = 0.;
  double reduced_integrality = 0.;
  int32_t failed_reduced     = 0;

  double postsolved_row         = 0.;
  double postsolved_bound       = 0.;
  double postsolved_integrality = 0.;
  int32_t failed_postsolved     = 0;

  // |objective(z) - reduced objective(crush(x))| relative to the sum of |cost * value|.
  double objective_identity = 0.;
  // objective(z) - objective(x) relative to max(1, |objective(x)|): its largest and its
  // smallest value over the points. Above zero the reductions made a point worse.
  double objective_rise    = -gprs::inf;
  double objective_fall    = gprs::inf;
  int32_t failed_objective = 0;

  // z against x: the largest difference in an integer column, the largest difference in a
  // continuous column relative to max(1, |x|), and the largest number of columns of one point
  // that differ by more than check_tolerance.
  double moved_integer       = 0.;
  double moved_continuous    = 0.;
  int64_t moved_columns      = 0;
  int32_t failed_round_trip  = 0;
  int32_t first_failed_point = -1;
};

inline crush_report_t crush_check(const gprs::model_t& model,
                                  const gprs::result_t& result,
                                  bool primal_only,
                                  const point_set_t& set)
{
  crush_report_t c;
  c.ran         = true;
  c.status      = gprs::status_name(result.status);
  c.has_reduced = carries_model(result.status);
  c.hash        = result.hash;
  if (!c.has_reduced) { return c; }
  std::vector<double> crushed, z;
  for (size_t k = 0; k < set.points.size(); ++k) {
    const std::vector<double>& x = set.points[k];
    ++c.points;
    result.postsolve.crush(x, crushed);
    point_report_t reduced = check_point_scaled(result.reduced, crushed);
    result.postsolve.postsolve(crushed, z);
    point_report_t back = check_point_scaled(model, z);
    bool failed         = false;

    raise_to(c.reduced_row, reduced.row_scaled);
    raise_to(c.reduced_bound, reduced.bound_scaled);
    raise_to(c.reduced_integrality, reduced.integrality);
    if (!reduced.within(check_tolerance)) {
      ++c.failed_reduced;
      failed = true;
    }

    raise_to(c.postsolved_row, back.row_scaled);
    raise_to(c.postsolved_bound, back.bound_scaled);
    raise_to(c.postsolved_integrality, back.integrality);
    if (!back.within(check_tolerance)) {
      ++c.failed_postsolved;
      failed = true;
    }

    // The two values of z agree relative to the size of their terms; the value of z against
    // that of x is a comparison of two points.
    double size     = std::fmax(1., std::fmax(back.objective_size, reduced.objective_size));
    double identity = std::fabs(back.objective - reduced.objective) / size;
    double known    = set.objective[k];
    double change   = back.objective - known;
    double allowed =
      objective_tolerance * std::fmax(1., std::fabs(known)) + identity_tolerance * size;
    double relative = change / std::fmax(1., std::fabs(known));
    raise_to(c.objective_identity, identity);
    raise_to(c.objective_rise, relative);
    if (!(relative >= c.objective_fall)) { c.objective_fall = relative; }
    if (!(identity <= identity_tolerance) || !(change <= allowed) ||
        (primal_only && !(change >= -allowed))) {
      ++c.failed_objective;
      failed = true;
    }

    double moved_integer = 0., moved_continuous = 0.;
    int64_t moved = 0;
    for (int32_t j = 0; j < model.ncols; ++j) {
      double difference = std::fabs(z[j] - x[j]);
      if (model.col_type[j]) {
        raise_to(moved_integer, difference);
        moved += !(difference <= 0.5);
      } else {
        difference /= std::fmax(1., std::fabs(x[j]));
        raise_to(moved_continuous, difference);
        moved += !(difference <= check_tolerance);
      }
    }
    raise_to(c.moved_integer, moved_integer);
    raise_to(c.moved_continuous, moved_continuous);
    if (moved > c.moved_columns) { c.moved_columns = moved; }
    if (primal_only && moved != 0) {
      ++c.failed_round_trip;
      failed = true;
    }
    if (failed && c.first_failed_point < 0) { c.first_failed_point = (int32_t)k; }
  }
  return c;
}

}  // namespace gprs_check
