/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Support of the tests of the row rules, bound propagation and row tightening: models written
// row by row, a fixture, small random models, and the feasible set of a state on a grid of
// points, by which a rule is checked to keep every feasible point and to create none.

#include "harness.hpp"

#include <gprs/presolve.hpp>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace rule_test {

using namespace gprs;

struct col_t {
  double lower;
  double upper;
  bool integer;
};

// Entries in increasing column order.
struct row_t {
  std::vector<std::pair<int32_t, double>> entries;
  double lower;
  double upper;
};

inline model_t make_model(const std::vector<col_t>& cols, const std::vector<row_t>& rows)
{
  model_t m;
  m.nrows = rows.size();
  m.ncols = cols.size();
  m.row_start.push_back(0);
  for (const row_t& row : rows) {
    for (const auto& entry : row.entries) {
      m.col_index.push_back(entry.first);
      m.value.push_back(entry.second);
    }
    m.row_start.push_back(m.col_index.size());
    m.row_lower.push_back(row.lower);
    m.row_upper.push_back(row.upper);
  }
  for (const col_t& col : cols) {
    m.col_lower.push_back(col.lower);
    m.col_upper.push_back(col.upper);
    m.cost.push_back(0.);
    m.col_type.push_back(col.integer);
  }
  return m;
}

constexpr int64_t unlimited = int64_t{1} << 40;

// A state with every row on every rule list, as after construction, and a budget without limit.
struct fixture_t {
  model_t model;
  state_t st;
  postsolve_t ps;
  options_t opt;
  work_budget_t budget;

  fixture_t(const std::vector<col_t>& cols, const std::vector<row_t>& rows)
    : model(make_model(cols, rows)), st(model, tolerances_t())
  {
    GPRS_CHECK(well_formed(model));
    ps.init(model);
    budget.limit       = unlimited;
    budget.stall_limit = unlimited;
  }

  // Leaves only the given rows on the given list.
  void only(int32_t list, const std::vector<int32_t>& rows)
  {
    std::vector<int32_t> taken;
    st.row_frontier.take(list, taken);
    for (int32_t row : rows) {
      st.row_frontier.push(list_bit(list), row);
    }
  }

  void check_consistent() { GPRS_CHECK_EQ(st.consistency_error(), std::string()); }
};

// Coefficient of (row, col) in the state, 0 without a live entry.
inline double coefficient(const state_t& st, int32_t row, int32_t col)
{
  int64_t e = st.find_entry(row, col);
  return e < 0 ? 0. : st.csr.value[e];
}

// Activity of the row at x less its upper side, from the stored coefficients, exact up to the
// rounding of the result: positive when x violates the upper side.
inline double upper_excess(const state_t& st, int32_t row, const std::vector<double>& x)
{
  dd_t activity;
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (st.csr.live(e)) { dd_add_product(activity, st.csr.value[e], x[st.csr.index[e]]); }
  }
  return -dd_residual(st.row_upper[row], activity);
}

struct rng_t {
  uint64_t x;
  explicit rng_t(uint64_t seed) : x(seed * 0x9e3779b97f4a7c15ull + 0x2545f4914f6cdd1dull)
  {
    if (x == 0) { x = 1; }
  }
  uint64_t next()
  {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
  }
  int32_t below(int32_t n) { return (next() >> 11) % (uint64_t)n; }
  bool chance(double p) { return (next() >> 11) * 0x1p-53 < p; }
};

struct random_options_t {
  double integer_share = 0.6;
  // Coefficients in tenths instead of halves.
  bool decimal = false;
  // Share of the columns that get an infinite bound on one side.
  double infinite_share = 0.15;
};

// 3 or 4 columns with bounds inside [-2, 4] or infinite, 1 to 5 rows with small coefficients and
// sides. Halves (or tenths) only, so that activities on the grid below are exact up to a few
// units in the last place and no test of a rule falls within its tolerance.
inline void random_model(rng_t& rng,
                         const random_options_t& o,
                         std::vector<col_t>& cols,
                         std::vector<row_t>& rows)
{
  static const double halves[] = {1., 1., 2., 3., -1., -1., -2., -3., 0.5, 1.5, -0.5, 4.};
  static const double tenths[] = {0.1, 0.2, 0.3, -0.1, -0.4, 0.5, 1.2, -0.6, 1., 2., -1., 0.9};
  cols.clear();
  rows.clear();
  int32_t ncols = 3 + rng.below(2);
  for (int32_t j = 0; j < ncols; ++j) {
    col_t col;
    col.lower   = rng.below(4) - 2;
    col.upper   = col.lower + 1 + rng.below(3);
    col.integer = rng.chance(o.integer_share);
    if (rng.chance(o.infinite_share)) { col.lower = -inf; }
    if (rng.chance(o.infinite_share)) { col.upper = inf; }
    cols.push_back(col);
  }
  int32_t nrows = 1 + rng.below(5);
  for (int32_t i = 0; i < nrows; ++i) {
    row_t row;
    double density = rng.chance(0.05) ? 0. : 0.3 + 0.2 * rng.below(4);
    for (int32_t j = 0; j < ncols; ++j) {
      if (!rng.chance(density)) { continue; }
      row.entries.push_back({j, o.decimal ? tenths[rng.below(12)] : halves[rng.below(12)]});
    }
    double base  = (rng.below(21) - 8) * 0.5;
    double width = (1 + rng.below(6)) * 0.5;
    int32_t kind = rng.below(10);
    row.lower    = base;
    row.upper    = base + width;
    // One-sided rows are loosened, or most models would have no feasible point.
    if (kind < 4) { row.lower = -inf, row.upper += 2.; }
    if (kind >= 4 && kind < 7) { row.upper = inf, row.lower -= 2.; }
    if (kind == 7) { row.upper = base; }
    if (kind == 8 && rng.chance(0.3)) { row.lower = -inf, row.upper = inf; }
    rows.push_back(row);
  }
}

// The points of a state's feasible set are looked for on a grid: integers from -4 to 5 for an
// integer column, the halves in between as well for a continuous one.
struct grid_t {
  std::vector<std::vector<double>> values;
};

inline grid_t make_grid(const state_t& st)
{
  grid_t grid;
  for (int32_t j = 0; j < st.ncols; ++j) {
    std::vector<double> values;
    for (int32_t k = -8; k <= 10; ++k) {
      if (k % 2 == 0 || !st.is_integer(j)) { values.push_back(0.5 * k); }
    }
    grid.values.push_back(values);
  }
  return grid;
}

// Bounds of live and fixed columns and the sides of the live rows, within eps.
inline bool feasible(const state_t& st, const std::vector<double>& x, double eps = 1e-9)
{
  for (int32_t j = 0; j < st.ncols; ++j) {
    if (st.col_status[j] == col_status_t::removed) { return false; }
    if (x[j] < st.col_lower[j] - eps || x[j] > st.col_upper[j] + eps) { return false; }
  }
  for (int32_t i = 0; i < st.nrows; ++i) {
    if (!st.row_alive[i]) { continue; }
    double activity = 0.;
    for (int64_t e = st.csr.begin(i); e < st.csr.end(i); ++e) {
      if (st.csr.live(e)) { activity += st.csr.value[e] * x[st.csr.index[e]]; }
    }
    if (activity < st.row_lower[i] - eps || activity > st.row_upper[i] + eps) { return false; }
  }
  return true;
}

template <typename visit_t>
void for_each_point(const grid_t& grid, visit_t visit)
{
  size_t n = grid.values.size();
  std::vector<size_t> at(n, 0);
  std::vector<double> x(n);
  while (true) {
    for (size_t j = 0; j < n; ++j) {
      x[j] = grid.values[j][at[j]];
    }
    visit(x);
    size_t j = 0;
    while (j < n && ++at[j] == grid.values[j].size()) {
      at[j++] = 0;
    }
    if (j == n) { return; }
  }
}

// A rule may move a side or a bound within the feasibility tolerance (a side of 2.9999995
// becomes 3), so a point counts as lost or created only when it is feasible on one side and
// violated by more than twice that tolerance on the other. On the grids of these tests a
// wrong reduction violates a row or a bound by 0.05 at least.
constexpr double tolerated = 2e-6;

struct set_difference_t {
  int64_t feasible_before = 0;
  int64_t lost            = 0;  // feasible before, not after
  int64_t created         = 0;  // feasible after, not before
};

inline set_difference_t compare_sets(const state_t& before, const state_t& after)
{
  set_difference_t d;
  for_each_point(make_grid(before), [&](const std::vector<double>& x) {
    bool was = feasible(before, x);
    bool is  = feasible(after, x);
    d.feasible_before += was;
    d.lost += was && !is && !feasible(after, x, tolerated);
    d.created += !was && is && !feasible(before, x, tolerated);
  });
  return d;
}

inline int64_t count_feasible(const state_t& st)
{
  int64_t count = 0;
  for_each_point(make_grid(st), [&](const std::vector<double>& x) { count += feasible(st, x); });
  return count;
}

// The check of every property test: an infeasible outcome only for a state without a feasible
// grid point, otherwise the same feasible grid points before and after and a consistent state.
inline void check_same_set(const state_t& before, state_t& after, rule_outcome_t outcome)
{
  if (outcome != rule_outcome_t::ok) {
    GPRS_CHECK_EQ(outcome, rule_outcome_t::infeasible);
    GPRS_CHECK_EQ(count_feasible(before), 0);
    return;
  }
  set_difference_t d = compare_sets(before, after);
  GPRS_CHECK_EQ(d.lost, 0);
  GPRS_CHECK_EQ(d.created, 0);
  GPRS_CHECK_EQ(after.consistency_error(), std::string());
}

}  // namespace rule_test
