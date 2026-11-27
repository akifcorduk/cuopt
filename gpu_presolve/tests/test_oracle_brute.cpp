/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "oracle.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>

using namespace gprs;
using oracle::outcome_t;
using oracle::solution_t;

namespace {

int64_t floor_multiple(int64_t v, int64_t step)
{
  int64_t rest = v % step;
  if (rest < 0) { rest += step; }
  return v - rest;
}

int64_t ceil_multiple(int64_t v, int64_t step) { return -floor_multiple(-v, step); }

// A second solver that shares nothing with oracle::solve. Every column takes every value of a
// grid: an integer column the integers, a continuous column the multiples of 1 / denominator,
// which hold every end of an interval when the data are integers and the denominator is a
// multiple of every coefficient. The arithmetic is in integers (values times the denominator)
// on a dense matrix, and every row is tested at every point.
// Columns with two finite bounds are the outer loop. An infinite bound is replaced by a box
// that is larger than every finite end an interval can have, and by a box of twice that size:
// for one outer assignment the minimum is smaller in the larger box exactly when a column can
// run to infinity in the direction of its cost.
struct brute_t {
  const model_t& m;
  int64_t denominator;
  std::vector<uint8_t> project;
  std::vector<std::vector<int64_t>> a;
  std::vector<int64_t> cost;
  std::vector<int64_t> h;
  std::vector<int32_t> outer, inner;
  int64_t box = 1;

  bool feasible = false, unbounded = false, has_best = false;
  int64_t best           = 0;
  int64_t box_mismatches = 0;
  // Values of the columns marked in project over all feasible points of the smaller box.
  std::set<std::vector<int64_t>> projections;

  bool found         = false;
  bool record        = false;
  int64_t inner_best = 0;

  int64_t scaled(double v) const
  {
    double s = v * denominator;
    GPRS_CHECK(s == std::floor(s));
    return s;
  }

  brute_t(const model_t& model, int64_t d, const std::vector<uint8_t>& project_cols)
    : m(model), denominator(d), project(project_cols)
  {
    a.assign(m.nrows, std::vector<int64_t>(m.ncols, 0));
    h.assign(m.ncols, 0);
    for (int32_t j = 0; j < m.ncols; ++j) {
      GPRS_CHECK(m.cost[j] == std::floor(m.cost[j]));
      cost.push_back(m.cost[j]);
      bool finite = std::isfinite(m.col_lower[j]) && std::isfinite(m.col_upper[j]);
      (finite ? outer : inner).push_back(j);
      for (double bound : {m.col_lower[j], m.col_upper[j]}) {
        if (std::isfinite(bound)) { box = std::max<int64_t>(box, std::ceil(std::fabs(bound)) + 1); }
      }
    }
    for (int32_t i = 0; i < m.nrows; ++i) {
      double reach = 0.;
      for (double side : {m.row_lower[i], m.row_upper[i]}) {
        if (std::isfinite(side)) { reach = std::max(reach, std::fabs(side)); }
      }
      for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
        int32_t j = m.col_index[e];
        GPRS_CHECK(m.value[e] == std::floor(m.value[e]));
        a[i][j] = m.value[e];
        if (std::isfinite(m.col_lower[j]) && std::isfinite(m.col_upper[j])) {
          reach += std::fabs(m.value[e]) *
                   std::ceil(std::max(std::fabs(m.col_lower[j]), std::fabs(m.col_upper[j])));
        }
      }
      box = std::max<int64_t>(box, reach + 1);
    }
  }

  void range(int32_t j, int64_t reach, int64_t& from, int64_t& to, int64_t& step) const
  {
    step = m.col_type[j] == 1 ? denominator : 1;
    from = m.col_lower[j] == -inf ? -reach * denominator : scaled(m.col_lower[j]);
    to   = m.col_upper[j] == inf ? reach * denominator : scaled(m.col_upper[j]);
    from = ceil_multiple(from, step);
    to   = floor_multiple(to, step);
  }

  double leaves() const
  {
    double outer_count = 1., small = 1., large = 1.;
    int64_t from, to, step;
    for (int32_t j : outer) {
      range(j, box, from, to, step);
      outer_count *= std::max<int64_t>(0, (to - from) / step + 1);
    }
    for (int32_t j : inner) {
      range(j, box, from, to, step);
      small *= (to - from) / step + 1;
      range(j, 2 * box, from, to, step);
      large *= (to - from) / step + 1;
    }
    return outer_count * (small + large);
  }

  bool rows_hold() const
  {
    for (int32_t i = 0; i < m.nrows; ++i) {
      int64_t activity = 0;
      for (int32_t j = 0; j < m.ncols; ++j) {
        activity += a[i][j] * h[j];
      }
      if (m.row_lower[i] != -inf && activity < scaled(m.row_lower[i])) { return false; }
      if (m.row_upper[i] != inf && activity > scaled(m.row_upper[i])) { return false; }
    }
    return true;
  }

  int64_t objective() const
  {
    int64_t sum = 0;
    for (int32_t j = 0; j < m.ncols; ++j) {
      sum += cost[j] * h[j];
    }
    return sum;
  }

  void scan_inner(size_t k, int64_t reach)
  {
    if (k == inner.size()) {
      if (!rows_hold()) { return; }
      int64_t value = objective();
      if (!found || value < inner_best) { inner_best = value; }
      found = true;
      if (!record) { return; }
      std::vector<int64_t> projection;
      for (int32_t j = 0; j < m.ncols; ++j) {
        if (project[j]) { projection.push_back(h[j] / denominator); }
      }
      projections.insert(projection);
      return;
    }
    int64_t from, to, step;
    range(inner[k], reach, from, to, step);
    for (int64_t v = from; v <= to; v += step) {
      h[inner[k]] = v;
      scan_inner(k + 1, reach);
    }
  }

  void scan_outer(size_t k)
  {
    if (k == outer.size()) {
      found  = false;
      record = true;
      scan_inner(0, box);
      bool found_small   = found;
      int64_t best_small = inner_best;
      found              = false;
      record             = false;
      scan_inner(0, 2 * box);
      box_mismatches += found != found_small;
      if (!found_small) { return; }
      feasible = true;
      if (inner_best < best_small) {
        unbounded = true;
      } else if (!has_best || best_small < best) {
        has_best = true;
        best     = best_small;
      }
      return;
    }
    int64_t from, to, step;
    range(outer[k], box, from, to, step);
    for (int64_t v = from; v <= to; v += step) {
      h[outer[k]] = v;
      scan_outer(k + 1);
    }
  }

  outcome_t outcome() const
  {
    if (!feasible) { return outcome_t::infeasible; }
    return unbounded ? outcome_t::unbounded : outcome_t::optimal;
  }

  // Takes a point in doubles onto the grid and tests it; value is its objective times the
  // denominator, without the offset.
  bool point_holds(const std::vector<double>& x, int64_t& value, int64_t boxes = 4)
  {
    for (int32_t j = 0; j < m.ncols; ++j) {
      double s = x[j] * denominator;
      h[j]     = std::llround(s);
      if (std::fabs(s - h[j]) > 1e-9 * std::max(1., std::fabs(s))) { return false; }
      int64_t from, to, step;
      range(j, boxes * box, from, to, step);
      if (h[j] < from || h[j] > to || (h[j] - from) % step != 0) { return false; }
    }
    value = objective();
    return rows_hold();
  }
};

struct tally_t {
  int32_t compared = 0, optimal = 0, infeasible = 0, unbounded = 0, failures = 0;
  int64_t far_moves = 0;

  ~tally_t()
  {
    if (!std::getenv("GPRS_ORACLE_CENSUS")) { return; }
    std::printf("    compared %d, optimal %d, infeasible %d, unbounded %d, far moves %lld\n",
                compared,
                optimal,
                infeasible,
                unbounded,
                (long long)far_moves);
  }
};

// An empty text when solve() and the second solver agree on the model. far_moves counts the
// line columns that the far points move.
std::string disagreement(const model_t& m,
                         const std::vector<uint8_t>& line,
                         brute_t& brute,
                         int64_t& far_moves)
{
  solution_t s = oracle::solve(m, line, 1 << 22);
  if (brute.box_mismatches != 0) { return "the two boxes of the second solver disagree"; }
  if (s.outcome != brute.outcome()) { return "outcomes differ"; }
  if ((s.outcome == outcome_t::optimal) != !s.x.empty() && m.ncols > 0) {
    return "x does not follow the outcome";
  }
  if (s.outcome == outcome_t::optimal) {
    double expected = (double)brute.best / brute.denominator + m.objective_offset;
    if (std::fabs(s.objective - expected) > 1e-12 * std::max(1., std::fabs(expected))) {
      return "optimal values differ: " + std::to_string(s.objective) + " against " +
             std::to_string(expected);
    }
    int64_t value = 0;
    if (!brute.point_holds(s.x, value)) { return "the optimal point is infeasible"; }
    if (value != brute.best) { return "the optimal point does not have the optimal value"; }
  }
  if (s.feasible.size() != brute.projections.size()) {
    return "numbers of feasible assignments differ: " + std::to_string(s.feasible.size()) +
           " against " + std::to_string(brute.projections.size());
  }
  auto expected = brute.projections.begin();
  for (const std::vector<double>& point : s.feasible) {
    int64_t value = 0;
    if (!brute.point_holds(point, value)) { return "a feasible point is infeasible"; }
    std::vector<int64_t> projection;
    for (int32_t j = 0; j < m.ncols; ++j) {
      if (!line[j]) { projection.push_back(std::llround(point[j])); }
    }
    if (projection != *expected) { return "feasible assignments differ or are out of order"; }
    ++expected;
  }
  // The far points: feasible, with the same values of the enumerated columns.
  if (s.far.size() != s.feasible.size()) { return "numbers of far points differ"; }
  for (size_t k = 0; k < s.far.size(); ++k) {
    int64_t value = 0;
    if (!brute.point_holds(s.far[k], value, 4 * (int64_t)oracle::far_step)) {
      return "a far point is infeasible";
    }
    for (int32_t j = 0; j < m.ncols; ++j) {
      bool moved = s.far[k][j] != s.feasible[k][j];
      if (!line[j] && moved) { return "a far point changes an enumerated column"; }
      far_moves += moved;
    }
  }
  return std::string();
}

void compare(const oracle::generator_options_t& options,
             uint64_t first_seed,
             int32_t count,
             int64_t denominator,
             tally_t& tally)
{
  for (uint64_t seed = first_seed; seed < first_seed + count && tally.failures < 3; ++seed) {
    oracle::case_t c = oracle::generate(seed, options);
    std::vector<uint8_t> line;
    GPRS_REQUIRE(oracle::classify(c.base, options.max_width, line));
    // The second solver enumerates the columns that solve() enumerates, and more.
    std::vector<uint8_t> enumerated(line.size());
    for (size_t j = 0; j < line.size(); ++j) {
      enumerated[j] = !line[j];
    }
    brute_t brute(c.base, denominator, enumerated);
    if (brute.leaves() > 3e6) { continue; }
    brute.scan_outer(0);
    ++tally.compared;
    tally.optimal += brute.outcome() == outcome_t::optimal;
    tally.infeasible += brute.outcome() == outcome_t::infeasible;
    tally.unbounded += brute.outcome() == outcome_t::unbounded;
    std::string error = disagreement(c.base, line, brute, tally.far_moves);
    if (error.empty()) { continue; }
    ++tally.failures;
    test::fail(__FILE__,
               __LINE__,
               "seed " + std::to_string(seed) + ": " + error + "\n" + oracle::to_code(c));
  }
}

oracle::generator_options_t small_options()
{
  oracle::generator_options_t o;
  o.max_rows      = 4;
  o.max_enum_cols = 3;
  o.max_line_cols = 1;
  o.max_side      = 6;
  o.max_width     = 2;
  o.p_huge_bound  = 0.;
  o.p_scaled      = 0.;
  return o;
}

}  // namespace

// The second solver itself, on models solved by hand.
GPRS_TEST(oracle_brute_force_solves_hand_models)
{
  model_t m;
  m.nrows     = 1;
  m.ncols     = 2;
  m.row_start = {0, 2};
  m.col_index = {0, 1};
  m.value     = {1., -2.};
  m.row_lower = {-7.};
  m.row_upper = {inf};
  m.col_lower = {0., 0.};
  m.col_upper = {1., inf};
  m.cost      = {0., -1.};
  m.col_type  = {1, 0};
  {
    // y <= (x + 7) / 2, max y: 4 at x = 1.
    brute_t brute(m, 2, {1, 0});
    brute.scan_outer(0);
    GPRS_CHECK(brute.outcome() == outcome_t::optimal);
    GPRS_CHECK_EQ(brute.best, -8);
    GPRS_CHECK_EQ(brute.projections.size(), 2u);
    GPRS_CHECK_EQ(brute.box_mismatches, 0);
  }
  m.cost[1] = 1.;
  {
    brute_t brute(m, 2, {1, 0});
    brute.scan_outer(0);
    GPRS_CHECK(brute.outcome() == outcome_t::optimal);
    GPRS_CHECK_EQ(brute.best, 0);
  }
  m.row_lower[0] = -inf;
  m.cost[1]      = -1.;
  {
    brute_t brute(m, 2, {1, 0});
    brute.scan_outer(0);
    GPRS_CHECK(brute.outcome() == outcome_t::unbounded);
  }
  m.row_upper[0] = -100.;
  m.col_upper[1] = 3.;
  {
    brute_t brute(m, 2, {1, 0});
    brute.scan_outer(0);
    GPRS_CHECK(brute.outcome() == outcome_t::infeasible);
    GPRS_CHECK(brute.projections.empty());
  }
}

GPRS_TEST(oracle_solve_agrees_with_brute_force_one_line_column)
{
  tally_t tally;
  oracle::generator_options_t o = small_options();
  compare(o, 1, 10000, 6, tally);
  o.p_continuous = 1.;
  compare(o, 100000, 4000, 6, tally);
  o.p_continuous     = 0.;
  o.p_infinite_bound = 0.8;
  compare(o, 200000, 4000, 6, tally);
  GPRS_CHECK_EQ(tally.failures, 0);
  GPRS_CHECK(tally.far_moves > 10000);
  GPRS_CHECK(tally.compared >= 17000);
  GPRS_CHECK(tally.optimal >= 5000);
  GPRS_CHECK(tally.infeasible >= 3000);
  GPRS_CHECK(tally.unbounded >= 900);
}

GPRS_TEST(oracle_solve_agrees_with_brute_force_two_line_columns)
{
  tally_t tally;
  oracle::generator_options_t o = small_options();
  o.max_enum_cols               = 2;
  o.max_line_cols               = 2;
  o.max_coefficient             = 2;
  o.max_side                    = 4;
  compare(o, 300000, 5000, 2, tally);
  GPRS_CHECK_EQ(tally.failures, 0);
  GPRS_CHECK(tally.compared >= 4500);
  GPRS_CHECK(tally.optimal >= 1500);
  GPRS_CHECK(tally.infeasible >= 800);
  GPRS_CHECK(tally.unbounded >= 400);
}

GPRS_TEST(oracle_solve_agrees_with_brute_force_integer_models)
{
  tally_t tally;
  oracle::generator_options_t o = small_options();
  o.max_rows                    = 6;
  o.max_enum_cols               = 6;
  o.max_line_cols               = 0;
  o.max_width                   = 3;
  o.density                     = 0.6;
  compare(o, 400000, 6000, 2, tally);
  GPRS_CHECK_EQ(tally.failures, 0);
  GPRS_CHECK(tally.compared >= 5800);
  GPRS_CHECK(tally.optimal >= 2000);
  GPRS_CHECK(tally.infeasible >= 1500);
  GPRS_CHECK_EQ(tally.unbounded, 0);
}
