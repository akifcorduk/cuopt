/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"

#include <gprs/postsolve.hpp>
#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>
#include <gprs/state.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace gprs;

namespace {

using dense_t   = std::vector<double>;
using entries_t = std::map<int32_t, double>;
using list_t    = std::vector<int32_t>;

constexpr rule_outcome_t ok         = rule_outcome_t::ok;
constexpr rule_outcome_t infeasible = rule_outcome_t::infeasible;
constexpr int64_t no_limit          = (int64_t)1 << 60;

// Rows as dense coefficient lists; zeros are no entries.
model_t make_model(const std::vector<dense_t>& rows,
                   const dense_t& row_lower,
                   const dense_t& row_upper,
                   const dense_t& col_lower,
                   const dense_t& col_upper)
{
  model_t m;
  m.nrows = rows.size();
  m.ncols = col_lower.size();
  m.row_start.push_back(0);
  for (const dense_t& row : rows) {
    for (int32_t j = 0; j < m.ncols; ++j) {
      if (row[j] == 0.) { continue; }
      m.col_index.push_back(j);
      m.value.push_back(row[j]);
    }
    m.row_start.push_back(m.col_index.size());
  }
  m.row_lower = row_lower;
  m.row_upper = row_upper;
  m.col_lower = col_lower;
  m.col_upper = col_upper;
  m.cost.assign(m.ncols, 1.);
  m.col_type.assign(m.ncols, 1);
  return m;
}

// The state after the start step of presolve(): columns with lower == upper are fixed.
struct fixture_t {
  model_t model;
  state_t st;
  postsolve_t ps;
  options_t opt;
  parallel_rows_t rule;

  explicit fixture_t(const model_t& m) : model(m), st(m, tolerances_t())
  {
    GPRS_CHECK(well_formed(m));
    ps.init(m);
    for (int32_t j = 0; j < m.ncols; ++j) {
      if (m.col_lower[j] == m.col_upper[j]) { fix_column(st, ps, j, m.col_lower[j]); }
    }
  }

  rule_result_t run(int64_t limit = no_limit)
  {
    work_budget_t budget;
    budget.limit         = limit;
    budget.stall_limit   = limit;
    rule_result_t result = rule.run(st, budget, ps, opt);
    if (result.outcome == ok) { GPRS_CHECK_EQ(st.consistency_error(), std::string()); }
    return result;
  }

  model_t reduced()
  {
    st.rebuild();
    return extract_model(st, ps.row_origin, ps.col_origin);
  }
};

entries_t row_entries(const state_t& st, int32_t row)
{
  entries_t entries;
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (st.csr.live(e)) { entries[st.csr.index[e]] = st.csr.value[e]; }
  }
  return entries;
}

bool on_list(const state_t& st, int32_t list, int32_t row)
{
  return (st.row_frontier.pending[row] >> list) & 1;
}

list_t alive_rows(const state_t& st)
{
  list_t rows;
  for (int32_t i = 0; i < st.nrows; ++i) {
    if (st.row_alive[i]) { rows.push_back(i); }
  }
  return rows;
}

void check_record(const postsolve_t& ps, size_t k, int32_t removed, int32_t kept, double ratio)
{
  GPRS_REQUIRE(k < ps.parallel_rows.size());
  GPRS_CHECK_EQ(ps.parallel_rows[k].removed_row, removed);
  GPRS_CHECK_EQ(ps.parallel_rows[k].kept_row, kept);
  GPRS_CHECK_EQ(ps.parallel_rows[k].ratio, ratio);
}

// Calls f(x) for every integer point inside the bounds of the model, which are finite.
template <typename F>
void for_each_point(const model_t& m, F f)
{
  dense_t lo(m.ncols), hi(m.ncols);
  for (int32_t j = 0; j < m.ncols; ++j) {
    lo[j] = std::ceil(m.col_lower[j]);
    hi[j] = std::floor(m.col_upper[j]);
    if (lo[j] > hi[j]) { return; }
  }
  dense_t x = lo;
  while (true) {
    f(x);
    int32_t j = 0;
    for (; j < m.ncols; ++j) {
      if (x[j] < hi[j]) {
        x[j] += 1.;
        break;
      }
      x[j] = lo[j];
    }
    if (j == m.ncols) { return; }
  }
}

bool feasible(const model_t& m, const dense_t& x)
{
  return point_feasible(check_point(m, x), tolerances_t());
}

// The rule removes no column: a point of the original box is feasible for the original model
// exactly when its crushed image is feasible for the reduced model, and the postsolve returns
// the point. Returns the number of feasible points.
int64_t check_same_feasible_set(const model_t& original,
                                const model_t& reduced,
                                const postsolve_t& ps)
{
  int64_t count    = 0;
  int32_t failures = 0;
  dense_t y, back;
  for_each_point(original, [&](const dense_t& x) {
    bool is_feasible = feasible(original, x);
    count += is_feasible;
    ps.crush(x, y);
    ps.postsolve(y, back);
    if (failures < 3 && !(feasible(reduced, y) == is_feasible && back == x)) {
      ++failures;
      GPRS_CHECK_EQ(feasible(reduced, y), is_feasible);
      GPRS_CHECK(back == x);
    }
  });
  return count;
}

int64_t feasible_points(const model_t& m)
{
  int64_t count = 0;
  for_each_point(m, [&](const dense_t& x) { count += feasible(m, x); });
  return count;
}

struct rng_t {
  uint64_t x;
  explicit rng_t(uint64_t seed) : x(seed * 0x9e3779b97f4a7c15ull + 0x2545f4914f6cdd1dull) {}
  uint64_t next()
  {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
  }
  int32_t in(int32_t lo, int32_t hi) { return lo + next() % (uint64_t)(hi - lo + 1); }
  bool chance(double p) { return (next() >> 11) * 0x1p-53 < p; }
  double coefficient() { return (chance(0.5) ? -1. : 1.) * in(1, 3); }
};

// Integer models around a planted point: a few rows drawn entry by entry and multiples of
// them, by positive and negative factors, with sides of their own and coefficients of their
// own in the fixed columns. With odd_first, a multiple often gets a coefficient of its own in
// column 0 as well, so that it is parallel to its source once that column is fixed.
model_t random_model(rng_t& rng, bool odd_first = false)
{
  static const double factor[] = {1., 1., -1., 2., -2., 0.5, -0.5, 3., -3., 1.5};
  int32_t ncols = rng.in(2, 5), nrows = rng.in(2, 9), nbase = rng.in(1, 3);
  dense_t col_lower(ncols), col_upper(ncols), planted(ncols);
  for (int32_t j = 0; j < ncols; ++j) {
    col_lower[j] = rng.in(-2, 1);
    col_upper[j] = col_lower[j] + rng.in(1, 3);
    planted[j]   = rng.in(col_lower[j], col_upper[j]);
    if (rng.chance(0.15)) { col_lower[j] = col_upper[j] = planted[j]; }
  }
  std::vector<dense_t> rows(nrows, dense_t(ncols, 0.));
  dense_t row_lower(nrows), row_upper(nrows);
  for (int32_t i = 0; i < nrows; ++i) {
    if (i < nbase || rng.chance(0.15)) {
      for (int32_t j = 0; j < ncols; ++j) {
        if (rng.chance(0.7)) { rows[i][j] = rng.coefficient(); }
      }
    } else {
      int32_t source = rng.in(0, i - 1);
      double r       = factor[rng.in(0, 9)];
      for (int32_t j = 0; j < ncols; ++j) {
        rows[i][j] = r * rows[source][j];
        bool own   = col_lower[j] == col_upper[j] ? rng.chance(0.5) : odd_first && j == 0;
        if (own && rng.chance(0.7)) { rows[i][j] = rng.coefficient(); }
      }
    }
    double activity = 0.;
    for (int32_t j = 0; j < ncols; ++j) {
      activity += rows[i][j] * planted[j];
    }
    if (rng.chance(0.05)) { activity += rng.in(-3, 3); }
    int32_t kind = rng.in(0, 4);
    row_lower[i] = kind == 1 ? -inf : activity - (kind == 0 ? 0 : rng.in(0, 3));
    row_upper[i] = kind == 2 ? inf : activity + (kind == 0 ? 0 : rng.in(0, 3));
  }
  return make_model(rows, row_lower, row_upper, col_lower, col_upper);
}

// The rule recomputed from the model, pair by pair.
struct naive_t {
  std::vector<int32_t> kept;  // smallest row of the class of each row; the row itself if alone
  dense_t lower, upper;       // sides of each class at its smallest row
  bool infeasible = false;
};

naive_t naive_parallel(const model_t& m, const tolerances_t& tol)
{
  std::vector<entries_t> free_entries(m.nrows);
  dense_t fixed(m.nrows, 0.);
  for (int32_t i = 0; i < m.nrows; ++i) {
    for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
      int32_t j = m.col_index[e];
      if (m.col_lower[j] == m.col_upper[j]) {
        fixed[i] += m.value[e] * m.col_lower[j];
      } else {
        free_entries[i][j] = m.value[e];
      }
    }
  }
  naive_t result;
  result.kept.resize(m.nrows);
  result.lower = m.row_lower;
  result.upper = m.row_upper;
  for (int32_t i = 0; i < m.nrows; ++i) {
    result.kept[i] = i;
    for (int32_t k = 0; k < i && result.kept[i] == i; ++k) {
      const entries_t &a = free_entries[k], &b = free_entries[i];
      if (result.kept[k] != k || a.size() < 2 || a.size() != b.size()) { continue; }
      double ratio  = b.begin()->second / a.begin()->second;
      bool parallel = true;
      for (auto ia = a.begin(), ib = b.begin(); ia != a.end(); ++ia, ++ib) {
        parallel = parallel && ia->first == ib->first &&
                   std::fabs(ib->second - ratio * ia->second) <= tol.zero * std::fabs(ib->second);
      }
      if (!parallel) { continue; }
      result.kept[i] = k;
      double lower   = (m.row_lower[i] - fixed[i]) / ratio + fixed[k];
      double upper   = (m.row_upper[i] - fixed[i]) / ratio + fixed[k];
      if (ratio < 0) { std::swap(lower, upper); }
      result.lower[k] = std::fmax(result.lower[k], lower);
      result.upper[k] = std::fmin(result.upper[k], upper);
      if (result.lower[k] - result.upper[k] > tol.feasibility) { result.infeasible = true; }
    }
  }
  return result;
}

}  // namespace

// Three copies of one row: the smallest index stays with the intersection of the sides.
GPRS_TEST(parallel_duplicates)
{
  model_t m = make_model({{1., 1., 0.}, {0., 1., 1.}, {1., 1., 0.}, {1., 1., 0.}},
                         {-inf, -inf, -inf, 1.},
                         {5., 4., 3., inf},
                         {0., 0., 0.},
                         {4., 4., 4.});
  fixture_t f(m);
  int64_t nnz          = f.st.live_nnz;
  rule_result_t result = f.run();
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 2);
  GPRS_CHECK(!result.stopped);
  GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1}));
  GPRS_CHECK_EQ(f.st.live_nnz, nnz - 4);
  GPRS_CHECK_EQ(f.st.row_lower[0], 1.);
  GPRS_CHECK_EQ(f.st.row_upper[0], 3.);
  GPRS_CHECK_EQ(f.st.row_lower[1], -inf);
  GPRS_CHECK_EQ(f.st.row_upper[1], 4.);
  GPRS_CHECK(row_entries(f.st, 0) == (entries_t{{0, 1.}, {1, 1.}}));
  GPRS_CHECK_EQ(f.ps.parallel_rows.size(), 2u);
  GPRS_CHECK_EQ(f.ps.order.size(), 2u);
  check_record(f.ps, 0, 2, 0, 1.);
  check_record(f.ps, 1, 3, 0, 1.);
  // A changed side is an event for the other families, not for this one.
  GPRS_CHECK(on_list(f.st, row_list::row_rules, 0) && on_list(f.st, row_list::doubleton, 0));
  GPRS_CHECK(f.st.row_frontier.empty(row_list::parallel_rows));
  // Locks of the kept row follow its new lower side.
  GPRS_CHECK_EQ(f.st.down_locks[0], 1);
  GPRS_CHECK_EQ(f.st.up_locks[0], 1);
  GPRS_CHECK_EQ(f.run().findings, 0);
  model_t reduced = f.reduced();
  GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps) > 0);
}

// Multiples: the sides are divided by the ratio and change places for a negative one.
GPRS_TEST(parallel_scaled_and_negative)
{
  model_t m = make_model({{1., 2.}, {-2., -4.}, {3., 6.}, {-0.5, -1.}},
                         {-inf, -inf, 3., -6.},
                         {10., -4., 24., inf},
                         {0., 0.},
                         {9., 9.});
  fixture_t f(m);
  rule_result_t result = f.run();
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 3);
  GPRS_CHECK(alive_rows(f.st) == (list_t{0}));
  // Row 1: x0 + 2 x1 >= 2. Row 2: 1 <= x0 + 2 x1 <= 8. Row 3: x0 + 2 x1 <= 12.
  GPRS_CHECK_EQ(f.st.row_lower[0], 2.);
  GPRS_CHECK_EQ(f.st.row_upper[0], 8.);
  check_record(f.ps, 0, 1, 0, -2.);
  check_record(f.ps, 1, 2, 0, 3.);
  check_record(f.ps, 2, 3, 0, -0.5);
  model_t reduced = f.reduced();
  GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps) > 0);
}

// The larger row is the one that goes, also when it is the one that is not scaled.
GPRS_TEST(parallel_keeps_smaller_index)
{
  model_t m = make_model({{-2., 4.}, {1., -2.}}, {-inf, -3.}, {8., inf}, {0., 0.}, {9., 9.});
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_CHECK(alive_rows(f.st) == (list_t{0}));
  // Row 1: x0 - 2 x1 >= -3 is -2 x0 + 4 x1 <= 6.
  GPRS_CHECK_EQ(f.st.row_lower[0], -inf);
  GPRS_CHECK_EQ(f.st.row_upper[0], 6.);
  check_record(f.ps, 0, 1, 0, -0.5);
}

GPRS_TEST(parallel_equations)
{
  {
    // An inequality and an equation: the kept row becomes the equation.
    model_t m = make_model({{1., -1.}, {2., -2.}}, {-inf, 6.}, {4., 6.}, {0., 0.}, {9., 9.});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK_EQ(f.st.row_lower[0], 3.);
    GPRS_CHECK_EQ(f.st.row_upper[0], 3.);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0}));
  }
  {
    // The equation is kept and the inequality that it satisfies goes without a change.
    model_t m = make_model({{2., -2.}, {1., -1.}}, {6., -inf}, {6., 4.}, {0., 0.}, {9., 9.});
    fixture_t f(m);
    int64_t side_changes = f.st.side_changes;
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK_EQ(f.st.row_lower[0], 6.);
    GPRS_CHECK_EQ(f.st.row_upper[0], 6.);
    GPRS_CHECK_EQ(f.st.side_changes, side_changes);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0}));
  }
  {
    // The equation contradicts the inequality.
    model_t m = make_model({{1., -1.}, {2., -2.}}, {-inf, 6.}, {2., 6.}, {0., 0.}, {9., 9.});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().outcome, infeasible);
    GPRS_CHECK_EQ(feasible_points(m), 0);
  }
  {
    // The inequality contradicts the equation from the other side, with a negative ratio.
    model_t m = make_model({{1., -1.}, {-2., 2.}}, {3., -inf}, {3., -8.}, {0., 0.}, {9., 9.});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().outcome, infeasible);
    GPRS_CHECK_EQ(feasible_points(m), 0);
  }
  {
    // Two equations that agree.
    model_t m = make_model({{1., 1.}, {3., 3.}}, {2., 6.}, {2., 6.}, {0., 0.}, {9., 9.});
    fixture_t f(m);
    int64_t side_changes = f.st.side_changes;
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK_EQ(f.st.row_lower[0], 2.);
    GPRS_CHECK_EQ(f.st.row_upper[0], 2.);
    GPRS_CHECK_EQ(f.st.side_changes, side_changes);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0}));
  }
  {
    // Two equations that do not.
    model_t m = make_model({{1., 1.}, {3., 3.}}, {2., 7.}, {2., 7.}, {0., 0.}, {9., 9.});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().outcome, infeasible);
  }
}

// Sides that cross by less than the feasibility tolerance meet at the kept row's side; a
// larger crossing is infeasible. Both for a new lower and a new upper side.
GPRS_TEST(parallel_crossing_tolerance)
{
  for (double crossing : {5e-7, 2e-6}) {
    bool within = crossing < 1e-6;
    {
      model_t m =
        make_model({{1., 1.}, {1., 1.}}, {-inf, 2. + crossing}, {2., inf}, {0., 0.}, {9., 9.});
      fixture_t f(m);
      rule_result_t result = f.run();
      GPRS_CHECK_EQ(result.outcome, within ? ok : infeasible);
      if (within) {
        GPRS_CHECK_EQ(f.st.row_lower[0], 2.);
        GPRS_CHECK_EQ(f.st.row_upper[0], 2.);
        GPRS_CHECK(!f.st.row_alive[1]);
      }
    }
    {
      model_t m =
        make_model({{1., 1.}, {1., 1.}}, {2., -inf}, {inf, 2. - crossing}, {0., 0.}, {9., 9.});
      fixture_t f(m);
      rule_result_t result = f.run();
      GPRS_CHECK_EQ(result.outcome, within ? ok : infeasible);
      if (within) {
        GPRS_CHECK_EQ(f.st.row_lower[0], 2.);
        GPRS_CHECK_EQ(f.st.row_upper[0], 2.);
        GPRS_CHECK(!f.st.row_alive[1]);
      }
    }
  }
}

// Not parallel: other columns, another length, one coefficient off by more than tol.zero.
// Parallel: every coefficient within tol.zero.
GPRS_TEST(parallel_exact_comparison)
{
  double off = 4. * (1. + 5e-9), near = 4. * (1. + 5e-10);
  model_t m = make_model({{1., 2., 0., 0.},
                          {1., 0., 2., 0.},
                          {1., 2., 1., 0.},
                          {2., off, 0., 0.},
                          {0., 1., 2., 0.},
                          {2., near, 0., 0.},
                          {1., -2., 0., 0.},
                          {2., 0., 0., 4.}},
                         dense_t(8, -inf),
                         {9., 8., 7., 16., 9., 14., 9., 9.},
                         dense_t(4, 0.),
                         dense_t(4, 9.));
  fixture_t f(m);
  rule_result_t result = f.run();
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 1);
  GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1, 2, 3, 4, 6, 7}));
  GPRS_CHECK_EQ(f.st.row_upper[0], 7.);
  check_record(f.ps, 0, 5, 0, 2.);
}

// The terms of fixed columns are constants: rows are compared on their free entries and the
// constants go into the sides. x2 is fixed at 2.
GPRS_TEST(parallel_fixed_terms)
{
  {
    // Row 0: x0 + x1 <= 8. Row 1: 2 x0 + 2 x1 <= 12. Row 2: x0 + x1 >= 3.
    model_t m = make_model({{1., 1., 1.}, {2., 2., 5.}, {-1., -1., 0.}},
                           {-inf, -inf, -inf},
                           {10., 22., -3.},
                           {0., 0., 2.},
                           {9., 9., 2.});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.st.row_free_size[0], 2);
    GPRS_CHECK_EQ(f.run().findings, 2);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0}));
    // The kept row still holds the term of x2: its sides are those of x0 + x1 plus 2.
    GPRS_CHECK(row_entries(f.st, 0) == (entries_t{{0, 1.}, {1, 1.}, {2, 1.}}));
    GPRS_CHECK_EQ(f.st.row_lower[0], 5.);
    GPRS_CHECK_EQ(f.st.row_upper[0], 8.);
    model_t reduced = f.reduced();
    GPRS_CHECK_EQ(reduced.row_lower[0], 3.);
    GPRS_CHECK_EQ(reduced.row_upper[0], 6.);
    GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps) > 0);
  }
  {
    // One free entry each once x2 is fixed: not a case of this rule.
    model_t m =
      make_model({{1., 0., 1.}, {1., 0., 1.}}, {0., 0.}, {5., 4.}, {0., 0., 2.}, {9., 9., 2.});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK_EQ(f.st.live_rows, 2);
  }
}

// Rows with fewer than two free entries are left to the row rules.
GPRS_TEST(parallel_short_rows)
{
  model_t m = make_model({{1., 0.}, {2., 0.}, {0., 0.}, {0., 0.}},
                         {-inf, -inf, -1., -1.},
                         {3., 4., 1., 1.},
                         {0., 0.},
                         {9., 9.});
  fixture_t f(m);
  rule_result_t result = f.run();
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK_EQ(f.st.live_rows, 4);
  GPRS_CHECK(f.st.row_frontier.empty(row_list::parallel_rows));
}

// Neither the hash nor the comparison reads the storage order: move_entry leaves row 1 as
// x1, x2, x0, twice row 0.
GPRS_TEST(parallel_storage_order)
{
  model_t m = make_model({{1., 2., 3., 0.}, {0., 4., 6., 2.}, {1., 2., 0., 3.}},
                         {-inf, -inf, -inf},
                         {5., 8., 9.},
                         dense_t(4, 0.),
                         dense_t(4, 9.));
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 0);
  f.st.move_entry(f.st.find_entry(1, 3), 0, 2.);
  GPRS_CHECK_EQ(f.st.csr.index[f.st.csr.begin(1)], 1);
  GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{1}));
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_CHECK(alive_rows(f.st) == (list_t{0, 2}));
  GPRS_CHECK_EQ(f.st.row_upper[0], 4.);
  check_record(f.ps, 0, 1, 0, 2.);
}

// A row that changed is found parallel to rows that did not, below it and above it.
GPRS_TEST(parallel_changed_against_unchanged)
{
  model_t m = make_model({{1., 2.}, {1., 3.}, {1., 4.}, {1., 5.}},
                         dense_t(4, -inf),
                         {5., 4., 3., 9.},
                         {0., 0.},
                         {9., 9.});
  {
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK(f.st.row_frontier.empty(row_list::parallel_rows));
    f.st.set_coefficient(f.st.find_entry(2, 1), 2.);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1, 3}));
    GPRS_CHECK_EQ(f.st.row_upper[0], 3.);
    check_record(f.ps, 0, 2, 0, 1.);
  }
  {
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 0);
    f.st.set_coefficient(f.st.find_entry(1, 1), 5.);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1, 2}));
    GPRS_CHECK_EQ(f.st.row_upper[1], 4.);
    check_record(f.ps, 0, 3, 1, 1.);
  }
  {
    // A column fixed between two calls: rows 0 and 2 are parallel on what is left.
    model_t wide = make_model({{1., 2., 1.}, {1., 3., 0.}, {2., 4., 5.}},
                              dense_t(3, -inf),
                              {5., 4., 30.},
                              dense_t(3, 0.),
                              dense_t(3, 9.));
    fixture_t f(wide);
    GPRS_CHECK_EQ(f.run().findings, 0);
    fix_column(f.st, f.ps, 2, 1.);
    GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{0, 2}));
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1}));
    // Row 2: 2 x0 + 4 x1 <= 25, so x0 + 2 x1 <= 12.5: no change of row 0.
    GPRS_CHECK_EQ(f.st.row_upper[0], 5.);
  }
  {
    // Only row 2 is listed when x2 is fixed at 1, and it goes into row 0 below it, which holds
    // the term of x3, fixed at 2 from the start. Row 2: 2 x0 + 4 x1 <= 6, so x0 + 2 x1 <= 3.
    model_t wide = make_model({{1., 2., 0., 1.}, {1., 3., 0., 0.}, {2., 4., 5., 0.}},
                              dense_t(3, -inf),
                              {7., 4., 11.},
                              {0., 0., 0., 2.},
                              {9., 9., 9., 2.});
    fixture_t f(wide);
    GPRS_CHECK_EQ(f.run().findings, 0);
    fix_column(f.st, f.ps, 2, 1.);
    GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{2}));
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1}));
    GPRS_CHECK_EQ(f.st.row_upper[0], 5.);
    check_record(f.ps, 0, 2, 0, 2.);
  }
}

// A row deleted by another family stays in its bucket until the rule meets it there.
GPRS_TEST(parallel_dead_row_in_bucket)
{
  model_t m = make_model({{1., 1.}, {1., 2.}, {1., -1.}, {1., 3.}},
                         dense_t(4, -inf),
                         {5., 4., 3., 9.},
                         {0., 0.},
                         {9., 9.});
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 0);
  f.st.delete_row(0);
  f.st.delete_row(3);
  f.st.set_coefficient(f.st.find_entry(2, 1), 1.);
  GPRS_CHECK_EQ(f.run().findings, 0);
  GPRS_CHECK(alive_rows(f.st) == (list_t{1, 2}));
  GPRS_CHECK_EQ(f.st.row_upper[2], 3.);
  f.st.set_coefficient(f.st.find_entry(1, 1), 1.);
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_CHECK(alive_rows(f.st) == (list_t{1}));
  GPRS_CHECK_EQ(f.st.row_upper[1], 3.);
  check_record(f.ps, 0, 2, 1, 1.);
}

// Deleted rows between two parallel rows of a bucket do not count as rows compared: with the
// limit of 4, row 5 reaches row 0 across four of them, and row 0 reaches row 5.
GPRS_TEST(parallel_dead_rows_do_not_count)
{
  for (int32_t changed : {0, 5}) {
    std::vector<dense_t> rows;
    for (int32_t k = 0; k < 6; ++k) {
      rows.push_back({1., k == changed ? 3. : 2. + (k % 5) * 1e-8});
    }
    model_t m = make_model(rows, dense_t(6, -inf), {9., 8., 8., 8., 8., 7.}, {0., 0.}, {9., 9.});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 0);
    for (int32_t row = 1; row < 5; ++row) {
      f.st.delete_row(row);
    }
    f.st.set_coefficient(f.st.find_entry(changed, 1), 2.);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0}));
    GPRS_CHECK_EQ(f.st.row_upper[0], 7.);
    check_record(f.ps, 0, 5, 0, 1.);
  }
}

// The listed rows are taken in index order, whatever the order of their events: the smallest
// row of three absorbs the other two, each with its ratio to that row.
GPRS_TEST(parallel_listed_in_index_order)
{
  model_t m = make_model(
    {{1., 2.}, {2., 6.}, {4., 16.}}, dense_t(3, -inf), {9., 16., 24.}, {0., 0.}, {9., 9.});
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 0);
  f.st.set_coefficient(f.st.find_entry(2, 1), 20.);
  f.st.set_coefficient(f.st.find_entry(1, 1), 10.);
  f.st.set_coefficient(f.st.find_entry(0, 1), 5.);
  GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{2, 1, 0}));
  GPRS_CHECK_EQ(f.run().findings, 2);
  GPRS_CHECK(alive_rows(f.st) == (list_t{0}));
  GPRS_CHECK_EQ(f.st.row_upper[0], 6.);
  check_record(f.ps, 0, 1, 0, 2.);
  check_record(f.ps, 1, 2, 0, 4.);
}

// Ten rows in one bucket: the coefficients of rows 1 to 8 differ from 2 by less than the
// rounding of the hash and by more than tol.zero. Rows 0 and 9 are equal. A row is compared
// with at most the limit of unlike rows on each side.
GPRS_TEST(parallel_bucket_limit)
{
  std::vector<dense_t> rows;
  for (int32_t k = 0; k < 10; ++k) {
    rows.push_back({1., 2. + (k % 9) * 1e-8});
  }
  model_t m = make_model(
    rows, dense_t(10, -inf), {9., 8., 8., 8., 8., 8., 8., 8., 8., 7.}, {0., 0.}, {9., 9.});
  for (int32_t limit : {4, 8, 9}) {
    fixture_t f(m);
    f.opt.parallel_bucket_limit = limit;
    work_budget_t budget;
    budget.limit = budget.stall_limit = no_limit;
    rule_result_t result              = f.rule.run(f.st, budget, f.ps, f.opt);
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, limit == 9);
    GPRS_CHECK_EQ(f.st.row_alive[9], limit != 9);
    GPRS_CHECK_EQ(f.st.row_upper[0], limit == 9 ? 7. : 9.);
    // Hashing: a visit and two passes per row. Comparing: a visit and a scatter per row, and a
    // visit and two passes per neighbour.
    int64_t hashing = 10 * (line_visit_work + 4);
    // Limit 4: 4, 4, 4, 4, 4, 4, 3, 2, 1, 0 rows above and as many below. Limit 8: 8, 8, 7, ...,
    // 1, 0 above. Limit 9: row 0 meets all nine, then rows 1 to 8 meet each other from both ends.
    int64_t pairs = limit == 4 ? 60 : limit == 8 ? 88 : 9 + 8 * 8;
    GPRS_CHECK_EQ(
      budget.used,
      hashing + (limit == 9 ? 9 : 10) * (line_visit_work + 2) + pairs * (line_visit_work + 4));
  }
}

// A stop on the budget gives the rows that were not compared back; the calls that follow end
// in the state of one call without a limit.
GPRS_TEST(parallel_budget)
{
  model_t m = make_model({{1., 2.}, {2., 4.}, {1., 3.}, {-1., -3.}, {3., 6.}, {1., 1.}},
                         {-inf, -inf, -inf, -inf, 3., 0.},
                         {10., 16., 9., 2., inf, 5.},
                         {0., 0.},
                         {9., 9.});
  fixture_t whole(m);
  GPRS_CHECK_EQ(whole.run().findings, 3);

  fixture_t f(m);
  rule_result_t first = f.run(1);
  GPRS_CHECK(first.stopped);
  GPRS_CHECK_EQ(first.findings, 0);
  GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{0, 1, 2, 3, 4, 5}));
  int64_t findings = 0;
  int32_t calls    = 1;
  while (!f.st.row_frontier.empty(row_list::parallel_rows) && calls < 100) {
    rule_result_t result = f.run(1);
    GPRS_CHECK_EQ(result.outcome, ok);
    findings += result.findings;
    ++calls;
  }
  GPRS_CHECK(calls > 6 && calls < 100);
  GPRS_CHECK_EQ(findings, 3);
  GPRS_CHECK(alive_rows(f.st) == alive_rows(whole.st));
  GPRS_CHECK(alive_rows(f.st) == (list_t{0, 2, 5}));
  for (int32_t row : {0, 2, 5}) {
    GPRS_CHECK_EQ(f.st.row_lower[row], whole.st.row_lower[row]);
    GPRS_CHECK_EQ(f.st.row_upper[row], whole.st.row_upper[row]);
  }
  GPRS_CHECK_EQ(f.st.row_lower[0], 1.);
  GPRS_CHECK_EQ(f.st.row_upper[0], 8.);
  GPRS_CHECK_EQ(f.st.row_lower[2], -2.);
  GPRS_CHECK_EQ(f.st.row_upper[2], 9.);
}

// Random models with duplicated and scaled rows: the rule removes exactly the rows that a
// pairwise comparison of the model finds, keeps the smallest row of each class with the
// intersection of the sides, reports infeasible exactly when a class has crossing sides, and
// leaves the feasible set as it was.
GPRS_TEST(parallel_random_models)
{
  tolerances_t tol;
  int64_t removed = 0, infeasible_cases = 0, nonempty = 0, with_fixed = 0;
  for (uint64_t seed = 1; seed <= 4000; ++seed) {
    rng_t rng(seed);
    model_t m     = random_model(rng);
    naive_t naive = naive_parallel(m, tol);
    fixture_t f(m);
    int32_t failures     = gprs::test::failure_count();
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.outcome == infeasible, naive.infeasible);
    if (result.outcome == ok && !naive.infeasible) {
      GPRS_CHECK(!result.stopped);
      int64_t classes_removed = 0;
      for (int32_t i = 0; i < m.nrows; ++i) {
        GPRS_CHECK_EQ(f.st.row_alive[i] != 0, naive.kept[i] == i);
        classes_removed += naive.kept[i] != i;
        if (!f.st.row_alive[i]) { continue; }
        GPRS_CHECK(std::fabs(f.st.row_lower[i] - naive.lower[i]) <= 1e-9 ||
                   f.st.row_lower[i] == naive.lower[i]);
        GPRS_CHECK(std::fabs(f.st.row_upper[i] - naive.upper[i]) <= 1e-9 ||
                   f.st.row_upper[i] == naive.upper[i]);
      }
      GPRS_CHECK_EQ(result.findings, classes_removed);
      GPRS_CHECK_EQ((int64_t)f.ps.parallel_rows.size(), classes_removed);
      for (const parallel_row_record_t& record : f.ps.parallel_rows) {
        GPRS_CHECK_EQ(naive.kept[record.removed_row], record.kept_row);
        GPRS_CHECK(record.kept_row < record.removed_row);
      }
      removed += classes_removed;
      with_fixed += classes_removed > 0 && f.st.fixed_nnz > 0;
      GPRS_CHECK_EQ(f.run().findings, 0);
      GPRS_CHECK(f.st.row_frontier.empty(row_list::parallel_rows));
      model_t reduced = f.reduced();
      nonempty += check_same_feasible_set(m, reduced, f.ps) > 0;
    } else if (naive.infeasible) {
      ++infeasible_cases;
      GPRS_CHECK_EQ(feasible_points(m), 0);
    }
    if (gprs::test::failure_count() != failures) {
      std::printf("    seed %llu\n", (unsigned long long)seed);
      return;
    }
  }
  // The test is not vacuous.
  GPRS_CHECK(removed > 5000);
  GPRS_CHECK(infeasible_cases > 100);
  GPRS_CHECK(nonempty > 2500);
  GPRS_CHECK(with_fixed > 500);
}

// The same after a second step: a column fixed between two calls makes more rows parallel,
// which are then found from the listed rows alone, in buckets whose other keys are those of
// the first call, also across a rebuild.
GPRS_TEST(parallel_random_incremental)
{
  tolerances_t tol;
  int64_t removed_later = 0, infeasible_cases = 0, nonempty = 0;
  for (uint64_t seed = 1; seed <= 4000; ++seed) {
    rng_t rng(seed);
    model_t m     = random_model(rng, true);
    naive_t naive = naive_parallel(m, tol);
    if (naive.infeasible) { continue; }
    fixture_t f(m);
    int32_t failures = gprs::test::failure_count();
    GPRS_CHECK_EQ(f.run().outcome, ok);
    if (rng.chance(0.5)) { f.st.rebuild(); }
    list_t live;
    for (int32_t j = 0; j < m.ncols; ++j) {
      if (f.st.is_live(j)) { live.push_back(j); }
    }
    if (live.empty()) { continue; }
    int32_t col  = rng.chance(0.8) ? live[0] : live[rng.in(0, live.size() - 1)];
    double value = rng.in(m.col_lower[col], m.col_upper[col]);
    fix_column(f.st, f.ps, col, value);
    model_t fixed        = m;
    fixed.col_lower[col] = fixed.col_upper[col] = value;
    naive_t naive_fixed                         = naive_parallel(fixed, tol);
    int32_t rows_before                         = f.st.live_rows;
    rule_result_t result                        = f.run();
    GPRS_CHECK_EQ(result.outcome == infeasible, naive_fixed.infeasible);
    if (result.outcome == ok) {
      for (int32_t i = 0; i < m.nrows; ++i) {
        GPRS_CHECK_EQ(f.st.row_alive[i] != 0, naive.kept[i] == i && naive_fixed.kept[i] == i);
      }
      GPRS_CHECK_EQ(result.findings, rows_before - f.st.live_rows);
      removed_later += result.findings;
      model_t reduced = f.reduced();
      nonempty += check_same_feasible_set(fixed, reduced, f.ps) > 0;
    } else {
      ++infeasible_cases;
      GPRS_CHECK_EQ(feasible_points(fixed), 0);
    }
    if (gprs::test::failure_count() != failures) {
      std::printf("    seed %llu\n", (unsigned long long)seed);
      return;
    }
  }
  // The test is not vacuous.
  GPRS_CHECK(removed_later > 800);
  GPRS_CHECK(infeasible_cases > 200);
  GPRS_CHECK(nonempty > 1000);
}
