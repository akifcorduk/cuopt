/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "oracle.hpp"

#include <gprs/postsolve.hpp>
#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>
#include <gprs/state.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
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
  // The row of the class of each row that stays, the row itself if alone: the one with the
  // largest coefficients, and of those the smallest index.
  std::vector<int32_t> kept;
  dense_t lower, upper;  // sides of each class at that row
  dense_t first;         // coefficient of each row in its smallest free column; 0 without one
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
  result.first.assign(m.nrows, 0.);
  result.lower = m.row_lower;
  result.upper = m.row_upper;
  // The first row of the class of each row.
  std::vector<int32_t> head(m.nrows);
  for (int32_t i = 0; i < m.nrows; ++i) {
    head[i]        = i;
    result.kept[i] = i;
    if (!free_entries[i].empty()) { result.first[i] = free_entries[i].begin()->second; }
    for (int32_t k = 0; k < i && head[i] == i; ++k) {
      const entries_t &a = free_entries[k], &b = free_entries[i];
      if (head[k] != k || a.size() < 2 || a.size() != b.size()) { continue; }
      double ratio  = b.begin()->second / a.begin()->second;
      bool parallel = true;
      for (auto ia = a.begin(), ib = b.begin(); ia != a.end(); ++ia, ++ib) {
        parallel = parallel && ia->first == ib->first &&
                   std::fabs(ib->second - ratio * ia->second) <= tol.zero * std::fabs(ib->second);
      }
      if (parallel) { head[i] = k; }
    }
    int32_t& stays = result.kept[head[i]];
    if (std::fabs(result.first[i]) > std::fabs(result.first[stays])) { stays = i; }
  }
  for (int32_t i = 0; i < m.nrows; ++i) {
    int32_t k      = result.kept[head[i]];
    result.kept[i] = k;
    if (k == i) { continue; }
    double ratio = result.first[i] / result.first[k];
    double lower = (m.row_lower[i] - fixed[i]) / ratio + fixed[k];
    double upper = (m.row_upper[i] - fixed[i]) / ratio + fixed[k];
    if (ratio < 0) { std::swap(lower, upper); }
    result.lower[k] = std::fmax(result.lower[k], lower);
    result.upper[k] = std::fmin(result.upper[k], upper);
  }
  for (int32_t i = 0; i < m.nrows; ++i) {
    if (result.lower[i] - result.upper[i] > tol.feasibility) { result.infeasible = true; }
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
  // A changed side is an event for the other families, not for this one. Row 0 is listed all
  // the same: x0 has no other entry left and has become a singleton column of it.
  GPRS_CHECK(on_list(f.st, row_list::row_rules, 0) && on_list(f.st, row_list::doubleton, 0));
  GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{0}));
  GPRS_CHECK(f.st.row_frontier.list[row_list::singletons] == (list_t{0}));
  // Locks of the kept row follow its new lower side.
  GPRS_CHECK_EQ(f.st.down_locks[0], 1);
  GPRS_CHECK_EQ(f.st.up_locks[0], 1);
  GPRS_CHECK_EQ(f.run().findings, 0);
  GPRS_CHECK(f.st.row_frontier.empty(row_list::parallel_rows));
  GPRS_CHECK(f.st.row_frontier.empty(row_list::singletons));
  model_t reduced = f.reduced();
  GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps) > 0);
}

// Multiples: each row goes into the next larger one, its sides times the quotient of the two
// rows, and they change places for a negative one. The row with the largest coefficients stays.
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
  GPRS_CHECK(!result.limited);
  GPRS_CHECK(alive_rows(f.st) == (list_t{2}));
  // Row 0: x0 + 2 x1 <= 10. Row 1: x0 + 2 x1 >= 2. Row 2: 1 <= x0 + 2 x1 <= 8.
  // Row 3: x0 + 2 x1 <= 12. In the units of row 2, three times those.
  GPRS_CHECK_EQ(f.st.row_lower[2], 6.);
  GPRS_CHECK_EQ(f.st.row_upper[2], 24.);
  check_record(f.ps, 0, 0, 1, -0.5);
  check_record(f.ps, 1, 1, 2, -2. / 3.);
  check_record(f.ps, 2, 3, 2, -0.5 / 3.);
  model_t reduced = f.reduced();
  GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps) > 0);
}

// The row with the larger coefficients stays, whichever of the two comes first, and of two
// rows with coefficients of one size the one with the smaller index.
GPRS_TEST(parallel_keeps_larger_row)
{
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
  {
    model_t m = make_model({{1., -2.}, {-2., 4.}}, {-3., -inf}, {inf, 8.}, {0., 0.}, {9., 9.});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{1}));
    GPRS_CHECK_EQ(f.st.row_lower[1], -inf);
    GPRS_CHECK_EQ(f.st.row_upper[1], 6.);
    check_record(f.ps, 0, 0, 1, -0.5);
  }
  {
    // Coefficients of one size, the second row the negative of the first.
    model_t m = make_model({{3., -6.}, {-3., 6.}}, {-inf, -inf}, {9., 12.}, {0., 0.}, {9., 9.});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0}));
    GPRS_CHECK_EQ(f.st.row_lower[0], -12.);
    GPRS_CHECK_EQ(f.st.row_upper[0], 9.);
    check_record(f.ps, 0, 1, 0, -1.);
  }
  {
    // With the option off the row with the smaller index stays whatever the sizes.
    // Row 1: -2 x0 + 4 x1 <= 4 is x0 - 2 x1 >= -2.
    model_t m = make_model({{1., -2.}, {-2., 4.}}, {-3., -inf}, {inf, 4.}, {0., 0.}, {9., 9.});
    fixture_t f(m);
    f.opt.parallel_keep_larger = false;
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0}));
    GPRS_CHECK_EQ(f.st.row_lower[0], -2.);
    GPRS_CHECK_EQ(f.st.row_upper[0], inf);
    check_record(f.ps, 0, 1, 0, -2.);
  }
}

GPRS_TEST(parallel_equations)
{
  {
    // An inequality and an equation that is twice as large: the equation stays as it is.
    model_t m = make_model({{1., -1.}, {2., -2.}}, {-inf, 6.}, {4., 6.}, {0., 0.}, {9., 9.});
    fixture_t f(m);
    int64_t side_changes = f.st.side_changes;
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK_EQ(f.st.row_lower[1], 6.);
    GPRS_CHECK_EQ(f.st.row_upper[1], 6.);
    GPRS_CHECK_EQ(f.st.side_changes, side_changes);
    GPRS_CHECK(alive_rows(f.st) == (list_t{1}));
  }
  {
    // An inequality and an equation of half its size: the kept row becomes the equation.
    model_t m = make_model({{2., -2.}, {1., -1.}}, {-inf, 3.}, {8., 3.}, {0., 0.}, {9., 9.});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK_EQ(f.st.row_lower[0], 6.);
    GPRS_CHECK_EQ(f.st.row_upper[0], 6.);
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
    GPRS_CHECK_EQ(f.st.row_lower[1], 6.);
    GPRS_CHECK_EQ(f.st.row_upper[1], 6.);
    GPRS_CHECK_EQ(f.st.side_changes, side_changes);
    GPRS_CHECK(alive_rows(f.st) == (list_t{1}));
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
  // Row 5 is twice row 0 and stays; x0 + 2 x1 <= 9 is weaker than its own side.
  GPRS_CHECK(alive_rows(f.st) == (list_t{1, 2, 3, 4, 5, 6, 7}));
  GPRS_CHECK_EQ(f.st.row_upper[5], 14.);
  check_record(f.ps, 0, 0, 5, 0.5);
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
    GPRS_CHECK(alive_rows(f.st) == (list_t{1}));
    // The kept row still holds the term of x2: its sides are those of 2 x0 + 2 x1 plus 10.
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{0, 2.}, {1, 2.}, {2, 5.}}));
    GPRS_CHECK_EQ(f.st.row_lower[1], 16.);
    GPRS_CHECK_EQ(f.st.row_upper[1], 22.);
    model_t reduced = f.reduced();
    GPRS_CHECK_EQ(reduced.row_lower[0], 6.);
    GPRS_CHECK_EQ(reduced.row_upper[0], 12.);
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

// A side of the kept row is stored with the fixed terms of that row in it. Fixed terms that one
// double does not hold within the tolerance next to a side leave the pair alone.
GPRS_TEST(parallel_huge_fixed_terms)
{
  {
    // Row 0: 3 x0 - 3 x1 + 2 x2 <= 12 with x2 fixed at -2e20. Row 1: x0 - x1 >= 1, which as a
    // lower side of row 0 would be 3 - 4e20: the 3 is lost and x0 == x1 becomes feasible.
    model_t m = make_model(
      {{3., -3., 2.}, {1., -1., 0.}}, {-inf, 1.}, {12., inf}, {0., 0., -2e20}, {3., 3., -2e20});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1}));
    GPRS_CHECK_EQ(f.st.row_lower[0], -inf);
    GPRS_CHECK_EQ(f.st.row_upper[0], 12.);
    GPRS_CHECK_EQ(f.ps.parallel_rows.size(), (size_t)0);
    GPRS_CHECK_EQ(check_same_feasible_set(m, f.reduced(), f.ps), 6);
  }
  {
    // The same rows in the other order: the row with the term is still the larger one, which
    // would be kept, and the pair is left alone as well.
    model_t m = make_model(
      {{1., -1., 0.}, {3., -3., 2.}}, {1., -inf}, {inf, 12.}, {0., 0., -2e20}, {3., 3., -2e20});
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.findings, 0);
    GPRS_CHECK(!result.limited);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1}));
    GPRS_CHECK_EQ(check_same_feasible_set(m, f.reduced(), f.ps), 6);
  }
  {
    // The row with the term is the smaller one, which goes. Its upper side less the term,
    // 12 + 4e20, is no double, and at that size a double does not hold the tolerance: left.
    model_t m = make_model(
      {{3., -3., 0.}, {1., -1., 2.}}, {3., -inf}, {inf, 12.}, {0., 0., -2e20}, {3., 3., -2e20});
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.findings, 0);
    GPRS_CHECK(!result.limited);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1}));
    GPRS_CHECK_EQ(check_same_feasible_set(m, f.reduced(), f.ps), 6);
  }
  {
    // A side that is the term itself leaves 0 exactly: x0 - x1 <= 0 goes into row 0 as
    // 3 x0 - 3 x1 <= 0.
    model_t m = make_model(
      {{3., -3., 0.}, {1., -1., 2.}}, {-9., -inf}, {inf, -4e20}, {0., 0., -2e20}, {3., 3., -2e20});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0}));
    GPRS_CHECK_EQ(f.st.row_lower[0], -9.);
    GPRS_CHECK_EQ(f.st.row_upper[0], 0.);
    GPRS_CHECK_EQ(check_same_feasible_set(m, f.reduced(), f.ps), 10);
  }
  {
    // The limit is |fixed terms| * tol.side_error <= tol.feasibility, 1e8 with the default
    // tolerances. Row 0: x0 + x1 + x2 <= x2 + 5, row 1: x0 + x1 >= 2.
    for (double value : {5e7, 2e8}) {
      model_t m = make_model({{1., 1., 1.}, {1., 1., 0.}},
                             {-inf, 2.},
                             {value + 5., inf},
                             {0., 0., value},
                             {4., 4., value});
      fixture_t f(m);
      bool merged = value == 5e7;
      GPRS_CHECK_EQ(f.run().findings, merged ? 1 : 0);
      GPRS_CHECK_EQ(f.st.live_rows, merged ? 1 : 2);
      if (merged) { GPRS_CHECK_EQ(f.st.row_lower[0], value + 2.); }
      GPRS_CHECK_EQ(check_same_feasible_set(m, f.reduced(), f.ps), 16);
    }
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
  // Row 2 is listed as well: x3 has its only entry there now.
  GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{2, 1}));
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_CHECK(alive_rows(f.st) == (list_t{1, 2}));
  GPRS_CHECK_EQ(f.st.row_upper[1], 8.);
  check_record(f.ps, 0, 0, 1, 0.5);
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
    GPRS_CHECK(alive_rows(f.st) == (list_t{1, 2}));
    // Row 0: x0 + 2 x1 <= 4, so 2 x0 + 4 x1 <= 8, and with the term of x2 in the side 13.
    GPRS_CHECK_EQ(f.st.row_upper[2], 13.);
    check_record(f.ps, 0, 0, 2, 0.5);
  }
  {
    // Only row 2 is listed when x2 is fixed at 1, and row 0 below it, which holds the term of
    // x3, fixed at 2 from the start, goes into it. Row 0: x0 + 2 x1 <= 5, so 2 x0 + 4 x1 <= 10;
    // row 2 has 2 x0 + 4 x1 <= 6 already.
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
    GPRS_CHECK(alive_rows(f.st) == (list_t{1, 2}));
    GPRS_CHECK_EQ(f.st.row_upper[2], 11.);
    check_record(f.ps, 0, 0, 2, 0.5);
  }
  {
    // The same with row 2 as the smaller one: it goes into row 0 below it.
    // Row 2: x0 + 2 x1 <= 6, so 2 x0 + 4 x1 <= 12, and with the term of x3 in the side 14.
    model_t wide = make_model({{2., 4., 0., 1.}, {1., 3., 0., 0.}, {1., 2., 5., 0.}},
                              dense_t(3, -inf),
                              {17., 4., 11.},
                              {0., 0., 0., 2.},
                              {9., 9., 9., 2.});
    fixture_t f(wide);
    GPRS_CHECK_EQ(f.run().findings, 0);
    fix_column(f.st, f.ps, 2, 1.);
    GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{2}));
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1}));
    GPRS_CHECK_EQ(f.st.row_upper[0], 14.);
    check_record(f.ps, 0, 2, 0, 0.5);
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

// The listed rows are taken in index order, whatever the order of their events: row 0 goes
// into row 1, the first larger row it meets, and row 1 with the sides of both into row 2.
GPRS_TEST(parallel_listed_in_index_order)
{
  model_t m = make_model(
    {{1., 2.}, {2., 6.}, {4., 16.}}, dense_t(3, -inf), {5., 16., 24.}, {0., 0.}, {9., 9.});
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 0);
  f.st.set_coefficient(f.st.find_entry(2, 1), 20.);
  f.st.set_coefficient(f.st.find_entry(1, 1), 10.);
  f.st.set_coefficient(f.st.find_entry(0, 1), 5.);
  GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{2, 1, 0}));
  GPRS_CHECK_EQ(f.run().findings, 2);
  GPRS_CHECK(alive_rows(f.st) == (list_t{2}));
  GPRS_CHECK_EQ(f.st.row_upper[2], 20.);
  check_record(f.ps, 0, 0, 1, 0.5);
  check_record(f.ps, 1, 1, 2, 0.5);
}

// Three rows, the largest first: both others go into it, each with its ratio to that row.
GPRS_TEST(parallel_largest_row_takes_the_class)
{
  model_t m = make_model(
    {{4., 20.}, {2., 10.}, {-1., -5.}}, {-inf, -inf, -5.}, {24., 16., inf}, {0., 0.}, {9., 9.});
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 2);
  GPRS_CHECK(alive_rows(f.st) == (list_t{0}));
  // Row 2: x0 + 5 x1 <= 5 is 4 x0 + 20 x1 <= 20.
  GPRS_CHECK_EQ(f.st.row_lower[0], -inf);
  GPRS_CHECK_EQ(f.st.row_upper[0], 20.);
  check_record(f.ps, 0, 1, 0, 0.5);
  check_record(f.ps, 1, 2, 0, -0.25);
  model_t reduced = f.reduced();
  GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps) > 0);
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
  GPRS_CHECK(alive_rows(f.st) == (list_t{2, 4, 5}));
  for (int32_t row : {2, 4, 5}) {
    GPRS_CHECK_EQ(f.st.row_lower[row], whole.st.row_lower[row]);
    GPRS_CHECK_EQ(f.st.row_upper[row], whole.st.row_upper[row]);
  }
  // Rows 0, 1 and 4 in the units of row 4: 1 <= x0 + 2 x1 <= 8.
  GPRS_CHECK_EQ(f.st.row_lower[4], 3.);
  GPRS_CHECK_EQ(f.st.row_upper[4], 24.);
  GPRS_CHECK_EQ(f.st.row_lower[2], -2.);
  GPRS_CHECK_EQ(f.st.row_upper[2], 9.);
}

// Random models with duplicated and scaled rows, rows that are parallel up to singleton columns
// left alone (near_parallel_rows off): the rule removes exactly the rows that a pairwise
// comparison of the model finds, keeps the largest row of each class (the smallest index among
// equals) with the intersection of the sides, reports infeasible exactly when a class has
// crossing sides, and leaves the feasible set as it was.
GPRS_TEST(parallel_random_models)
{
  tolerances_t tol;
  int64_t removed = 0, infeasible_cases = 0, nonempty = 0, with_fixed = 0;
  for (uint64_t seed = 1; seed <= 4000; ++seed) {
    rng_t rng(seed);
    model_t m     = random_model(rng);
    naive_t naive = naive_parallel(m, tol);
    fixture_t f(m);
    f.opt.near_parallel_rows = false;
    int32_t failures         = gprs::test::failure_count();
    rule_result_t result     = f.run();
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
        // A row goes into a row of its class that is not smaller, which can go in turn.
        GPRS_CHECK_EQ(naive.kept[record.removed_row], naive.kept[record.kept_row]);
        double kept_size    = std::fabs(naive.first[record.kept_row]);
        double removed_size = std::fabs(naive.first[record.removed_row]);
        GPRS_CHECK(kept_size > removed_size ||
                   (kept_size == removed_size && record.kept_row < record.removed_row));
        GPRS_CHECK_EQ(record.ratio, naive.first[record.removed_row] / naive.first[record.kept_row]);
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
    f.opt.near_parallel_rows = false;
    int32_t failures         = gprs::test::failure_count();
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

// Two rows share a bucket when their normalised coefficients agree after rounding the mantissa
// to 20 bits, a tie away from zero. Rows on the two sides of a tie are parallel within
// tol.zero and stay apart; a mantissa that rounds up to 1 meets the next power of two.
GPRS_TEST(parallel_hash_rounding)
{
  // 2^19 * tie is a whole number plus 1/2.
  double tie = 0x1.92a21p0;
  GPRS_REQUIRE(std::ldexp(tie, 19) - std::floor(std::ldexp(tie, 19)) == 0.5);
  double below = tie * (1. - 4e-10), above = tie * (1. + 4e-10), far_above = tie * (1. + 9e-10);
  for (double sign : {1., -1.}) {
    model_t m = make_model({{1., sign * below},
                            {1., sign * above},
                            {1., sign * tie},
                            {1., sign * far_above},
                            {1., sign * below}},
                           dense_t(5, -inf),
                           {9., 8., 7., 6., 5.},
                           {0., 0.},
                           {9., 9.});
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.outcome, ok);
    // Rows 1, 2 and 3 lie on the tie or beyond it, rows 0 and 4 before it.
    GPRS_CHECK_EQ(result.findings, 3);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1}));
    GPRS_CHECK_EQ(f.st.row_upper[0], 5.);
    GPRS_CHECK_EQ(f.st.row_upper[1], 6.);
    check_record(f.ps, 0, 4, 0, 1.);
    check_record(f.ps, 1, 2, 1, 1.);
    check_record(f.ps, 2, 3, 1, 1.);
  }
  {
    // 2 * (1 - 1e-10) has a mantissa that rounds up to 1: the bucket of 2 = 1/2 * 2^2.
    model_t m = make_model({{1., 2.}, {1., 2. * (1. - 1e-10)}, {-3., -6. * (1. - 1e-10)}},
                           dense_t(3, -inf),
                           {9., 8., 21.},
                           {0., 0.},
                           {9., 9.});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 2);
    GPRS_CHECK(alive_rows(f.st) == (list_t{2}));
  }
  {
    // A normalised coefficient below the normal numbers: 1e-10 / 1e300.
    model_t m = make_model({{1e300, 1e-10}, {1e300, 2e-10}, {2e300, 2e-10}},
                           dense_t(3, -inf),
                           {1e300, 1e300, 1e300},
                           {0., 0.},
                           {1., 1.});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{1, 2}));
    GPRS_CHECK_EQ(f.st.row_upper[2], 1e300);
    check_record(f.ps, 0, 0, 2, 0.5);
  }
}

// A first call with more keys than two bytes of the hash tell apart: 35,000 pairs of parallel
// rows, the second row of a pair far from the first. Every pair is found from its first row.
GPRS_TEST(parallel_many_rows)
{
  constexpr int32_t pairs = 35000, ncols = 700;
  auto partner = [](int32_t i) { return (int32_t)((i * 7919ll + 13) % pairs); };
  model_t m;
  m.nrows = 2 * pairs;
  m.ncols = ncols;
  m.row_start.push_back(0);
  for (int32_t i = 0; i < 2 * pairs; ++i) {
    int32_t source = i < pairs ? i : partner(i - pairs);
    double factor  = i < pairs ? 1. : i % 3 == 0 ? -2. : 0.5;
    // Every source row has its own pair of columns.
    int32_t first = source % ncols, second = (first + 1 + source / ncols) % ncols;
    if (first > second) { std::swap(first, second); }
    m.col_index.push_back(first);
    m.value.push_back(factor);
    m.col_index.push_back(second);
    m.value.push_back(factor * (2. + source % 97));
    m.row_start.push_back(m.col_index.size());
    m.row_lower.push_back(factor > 0 ? -inf : -1000.);
    m.row_upper.push_back(factor > 0 ? 1000. + i : inf);
  }
  m.col_lower.assign(ncols, 0.);
  m.col_upper.assign(ncols, 9.);
  m.cost.assign(ncols, 1.);
  m.col_type.assign(ncols, 1);
  fixture_t f(m);
  work_budget_t budget;
  budget.limit = budget.stall_limit = no_limit;
  rule_result_t result              = f.rule.run(f.st, budget, f.ps, f.opt);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, pairs);
  GPRS_CHECK_EQ(f.st.live_rows, pairs);
  GPRS_REQUIRE(f.ps.parallel_rows.size() == (size_t)pairs);
  list_t partner_of(pairs);
  for (int32_t i = 0; i < pairs; ++i) {
    partner_of[partner(i)] = pairs + i;
  }
  // A second row with the factor -2 is the larger one and stays; one with 0.5 goes.
  int32_t wrong = 0, second_kept = 0;
  for (int32_t k = 0; k < pairs; ++k) {
    const parallel_row_record_t& record = f.ps.parallel_rows[k];
    bool second_stays                   = partner_of[k] % 3 == 0;
    second_kept += second_stays;
    wrong += record.kept_row != (second_stays ? partner_of[k] : k) ||
             record.removed_row != (second_stays ? k : partner_of[k]) ||
             record.ratio != (second_stays ? -0.5 : 0.5);
  }
  GPRS_CHECK_EQ(wrong, 0);
  GPRS_CHECK(second_kept > 10000 && second_kept < 13000);
  // Hashing of every row; a scatter and one comparison per first row, with one more pass
  // over the second row for two rows that are not equal; a visit of every second row that
  // stayed, which finds its bucket empty.
  GPRS_CHECK_EQ(budget.used,
                2 * pairs * (line_visit_work + 4) + pairs * (line_visit_work + 2) +
                  pairs * (line_visit_work + 4) + pairs * 2 + second_kept * (line_visit_work + 2));
  GPRS_CHECK_EQ(f.st.consistency_error(), std::string());
}

namespace {

// The rule as rules.hpp (e) states it, one call at a time and without the changes of the
// state: which rows a call merges or subtracts, in which order, the work it spends and the rows
// that are listed afterwards. Buckets are keyed by the rounded normalised core entries
// themselves, not by a hash of them. Sides and fixed terms are followed in plain arithmetic,
// which is exact on the models of the tests that use it.
struct reference_t {
  using key_t = std::vector<std::tuple<int32_t, int64_t, int>>;  // column, mantissa, exponent
  struct row_t {
    key_t key;
    // Free size and version of the last reading; read is false for a row to read again.
    int32_t free     = 0;
    uint32_t version = 0;
    bool read        = false;
    bool keyed       = false;
  };
  struct reduction_t {
    bool addition;
    int32_t changed;  // the removed row of a merge, the target of an addition
    int32_t other;    // the kept row, the equation
  };
  struct call_t {
    std::vector<reduction_t> reductions;
    int64_t work        = 0;
    int64_t comparisons = 0;
    bool stopped        = false;
    bool infeasible     = false;
    bool limited        = false;
    list_t listed;      // row_list::parallel_rows after the call, ascending
    list_t singletons;  // row_list::singletons after the call, ascending
  };
  // The free entries of a row: its core and its entries in singleton columns.
  struct view_t {
    entries_t core;
    int32_t singles     = 0;
    int32_t single_col  = -1;
    double single_value = 0.;
    double fixed        = 0.;
    // Smallest and largest |a| over the free entries, and over those in singleton columns.
    double smallest = inf, largest = 0.;
    double single_smallest = inf, single_largest = 0.;
  };

  bool near = true;
  std::vector<row_t> rows;
  std::map<key_t, std::set<int32_t>> buckets;

  // size holds the live entries of every column.
  view_t view(const state_t& st, int32_t row, const std::vector<int32_t>& size) const
  {
    view_t v;
    for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
      if (!st.csr.live(e)) { continue; }
      int32_t col  = st.csr.index[e];
      double value = st.csr.value[e];
      if (!st.is_live(col)) {
        v.fixed += value * st.col_lower[col];
        continue;
      }
      v.smallest = std::fmin(v.smallest, std::fabs(value));
      v.largest  = std::fmax(v.largest, std::fabs(value));
      if (near && size[col] == 1) {
        ++v.singles;
        v.single_col      = col;
        v.single_value    = value;
        v.single_smallest = std::fmin(v.single_smallest, std::fabs(value));
        v.single_largest  = std::fmax(v.single_largest, std::fabs(value));
      } else {
        v.core[col] = value;
      }
    }
    return v;
  }

  static key_t key_of(const entries_t& entries)
  {
    constexpr int32_t mantissa_bits = 20;
    key_t key;
    for (const auto& entry : entries) {
      int exponent    = 0;
      double mantissa = std::frexp(entry.second / entries.begin()->second, &exponent);
      int64_t grid    = std::llround(std::ldexp(mantissa, mantissa_bits));
      if (grid == 1 << mantissa_bits || grid == -(1 << mantissa_bits)) {
        grid /= 2;
        ++exponent;
      }
      key.emplace_back(entry.first, grid, exponent);
    }
    return key;
  }

  // larger = ratio * smaller, entry by entry, within tol.zero.
  static bool parallel(const entries_t& smaller, const entries_t& larger)
  {
    if (smaller.size() != larger.size()) { return false; }
    double ratio = larger.begin()->second / smaller.begin()->second;
    double zero  = tolerances_t().zero;
    for (auto a = smaller.begin(), b = larger.begin(); a != smaller.end(); ++a, ++b) {
      if (a->first != b->first) { return false; }
      if (!(std::fabs(b->second - ratio * a->second) <= zero * std::fabs(b->second))) {
        return false;
      }
    }
    return true;
  }

  static bool holds(double size)
  {
    tolerances_t tol;
    return size * tol.side_error <= tol.feasibility;
  }

  // The two cores are equal entry by entry, or one is the negative of the other.
  static bool equal_cores(const view_t& kept, const view_t& gone)
  {
    double ratio = gone.core.begin()->second / kept.core.begin()->second;
    bool equal   = std::fabs(ratio) == 1.;
    for (const auto& entry : gone.core) {
      equal = equal && entry.second == ratio * kept.core.at(entry.first);
    }
    return equal;
  }

  // What the core of `gone` differs by from its multiple of the core of `kept` over the bounds
  // of the columns, each difference read from two products without rounding (rules.hpp (e)).
  static double drift(const state_t& st, const view_t& kept, const view_t& gone)
  {
    double kept_first = kept.core.begin()->second, gone_first = gone.core.begin()->second;
    double sum = 0.;
    for (const auto& entry : gone.core) {
      double a = kept.core.at(entry.first), b = entry.second;
      double b_error, a_error;
      double of_b       = two_prod(b, kept_first, b_error);
      double of_a       = two_prod(a, gone_first, a_error);
      double difference = (of_b - of_a) + (b_error - a_error);
      if (difference == 0.) { continue; }
      sum += std::fabs(difference / kept_first) *
             std::fmax(std::fabs(st.col_lower[entry.first]), std::fabs(st.col_upper[entry.first]));
    }
    return sum;
  }

  static double drift_limit()
  {
    return options_t().parallel_drift_share * tolerances_t().feasibility;
  }

  void unkey(int32_t row)
  {
    buckets[rows[row].key].erase(row);
    rows[row].keyed = false;
  }

  // listed and changed are row_list::parallel_rows and row_list::singletons before the call.
  call_t run(
    const state_t& st, list_t listed, const list_t& changed, work_budget_t budget, int32_t limit)
  {
    call_t call;
    rows.resize(st.nrows);
    std::vector<int32_t> size(st.csc.size.begin(), st.csc.size.end());
    dense_t lower = st.row_lower, upper = st.row_upper;
    std::vector<uint8_t> removed(st.nrows, 0);
    // A row that an entry or a side has changed in since the model was read.
    std::vector<uint8_t> touched(st.nrows, 0);
    for (int32_t row = 0; row < st.nrows; ++row) {
      touched[row] = st.row_version[row] != 0;
    }
    // Rows that an event of this call puts on the two lists.
    std::set<int32_t> pushed, events;
    if (near) {
      for (int32_t row : changed) {
        if (rows[row].keyed) { unkey(row); }
        rows[row].read = false;
      }
    }
    std::sort(listed.begin(), listed.end());
    auto alive  = [&](int32_t row) { return st.row_alive[row] && !removed[row]; };
    auto finish = [&](size_t given_back_from) {
      for (size_t k = given_back_from; k < listed.size(); ++k) {
        if (alive(listed[k])) { pushed.insert(listed[k]); }
      }
      call.listed.assign(pushed.begin(), pushed.end());
      call.singletons.assign(events.begin(), events.end());
      call.work = budget.used;
    };

    for (int32_t row : listed) {
      if (budget.stop()) {
        call.stopped = true;
        finish(0);
        return call;
      }
      int32_t free = st.row_alive[row] && st.row_free_size[row] >= 2 ? st.row_free_size[row] : 0;
      row_t& r     = rows[row];
      if (free != 0 && r.read && r.free == free && r.version == st.row_version[row]) { continue; }
      if (r.keyed) { unkey(row); }
      r.read = false;
      if (free == 0) { continue; }
      budget.spend(line_visit_work + 2 * (int64_t)st.csr.used[row]);
      entries_t core = view(st, row, size).core;
      r.read         = true;
      r.free         = free;
      r.version      = st.row_version[row];
      if (core.size() < 2) { continue; }
      r.key   = key_of(core);
      r.keyed = true;
      buckets[r.key].insert(row);
    }

    for (size_t k = 0; k < listed.size(); ++k) {
      if (budget.stop()) {
        call.stopped = true;
        finish(k);
        return call;
      }
      int32_t row = listed[k];
      if (!rows[row].keyed) { continue; }
      budget.spend(line_visit_work + st.csr.used[row]);
      std::set<int32_t>& bucket = buckets[rows[row].key];
      view_t mine               = view(st, row, size);

      // The reduction of the pair, if it has one.
      auto reduce = [&](int32_t other, const view_t& theirs) {
        if (mine.singles == 0 && theirs.singles == 0) {
          // The row with the larger coefficients stays, then the one with the smaller index.
          double my_size    = std::fabs(mine.core.begin()->second);
          double their_size = std::fabs(theirs.core.begin()->second);
          bool keep_mine    = my_size != their_size ? my_size > their_size : row < other;
          int32_t kept = keep_mine ? row : other, gone = keep_mine ? other : row;
          const view_t& kept_view = keep_mine ? mine : theirs;
          const view_t& gone_view = keep_mine ? theirs : mine;
          if (!holds(std::fabs(kept_view.fixed))) { return false; }
          double kept_first = kept_view.core.begin()->second;
          double gone_first = gone_view.core.begin()->second;
          double ratio      = gone_first / kept_first;
          // Two rows that are not equal up to the sign cost one more pass over the other row,
          // for what they differ by over the bounds of their columns.
          if (!equal_cores(kept_view, gone_view)) {
            budget.spend(st.csr.used[other]);
            if (!(drift(st, kept_view, gone_view) <= drift_limit())) {
              call.limited = true;
              return false;
            }
          }
          // The product first: exact on the sides and factors of these models.
          double low  = (lower[gone] - gone_view.fixed) * kept_first / gone_first + kept_view.fixed;
          double high = (upper[gone] - gone_view.fixed) * kept_first / gone_first + kept_view.fixed;
          if (ratio < 0) { std::swap(low, high); }
          // A side that changes lists the kept row.
          if (low > lower[kept] || high < upper[kept]) {
            touched[kept] = 1;
            pushed.insert(kept);
          }
          lower[kept] = std::fmax(lower[kept], low);
          upper[kept] = std::fmin(upper[kept], high);
          if (lower[kept] > upper[kept]) { call.infeasible = true; }
          call.reductions.push_back({false, gone, kept});
          removed[gone] = 1;
          unkey(gone);
          bool event = false;
          for (const auto& entry : gone_view.core) {
            event = --size[entry.first] == 1 || event;
          }
          if (event) {
            events.insert(kept);
            pushed.insert(kept);
            if (near) { unkey(kept); }
          }
          return true;
        }
        auto may = [&](int32_t equation, const view_t& of, const view_t& target) {
          return lower[equation] == upper[equation] &&
                 (of.singles == 0 || (of.singles == 1 && target.singles <= 1));
        };
        bool mine_may = may(row, mine, theirs), theirs_may = may(other, theirs, mine);
        if (!mine_may && !theirs_may) { return false; }
        bool mine_is_equation = mine_may && (!theirs_may || mine.singles < theirs.singles ||
                                             (mine.singles == theirs.singles && row < other));
        int32_t equation      = mine_is_equation ? row : other;
        int32_t target        = mine_is_equation ? other : row;
        const view_t& ev      = mine_is_equation ? mine : theirs;
        const view_t& tv      = mine_is_equation ? theirs : mine;
        double factor         = tv.core.begin()->second / ev.core.begin()->second;
        if (!std::isfinite(factor) || factor == 0.) { return false; }
        double scale = std::fabs(factor);
        if (scale > 1. || (ev.singles == 1 && scale != 1.)) { return false; }
        double grow_equation = std::fmax(1., scale / tv.single_smallest);
        double grow_target   = 1.;
        if (ev.singles == 1) { grow_target = std::fmax(1., 1. / std::fabs(ev.single_value)); }
        if (!(grow_equation <= 1e3 && grow_target <= 1e3)) { return false; }
        double rhs = upper[equation] - ev.fixed;
        if (!holds(std::fabs(upper[equation]) * grow_equation) ||
            !holds(std::fabs(ev.fixed) * grow_equation) || !holds(std::fabs(factor * rhs)) ||
            (lower[target] > -inf && !holds(std::fabs(lower[target]) * grow_target)) ||
            (upper[target] < inf && !holds(std::fabs(upper[target]) * grow_target)) ||
            (grow_target > 1. && !holds(std::fabs(tv.fixed) * grow_target))) {
          return false;
        }
        double value = -factor * ev.single_value;
        if (ev.singles == 1 && (!std::isfinite(value) || value == 0.)) { return false; }
        // The same pass and the same limit as for a merge, with the equation as the kept row.
        if (!equal_cores(ev, tv)) {
          budget.spend(st.csr.used[other]);
          if (!(drift(st, ev, tv) <= drift_limit())) {
            call.limited = true;
            return false;
          }
        }
        if ((grow_equation > 1. && touched[equation]) || (grow_target > 1. && touched[target])) {
          call.limited = true;
          return false;
        }
        budget.spend(st.csr.used[target]);
        double smallest = tv.single_smallest, largest = tv.single_largest;
        if (ev.singles == 1) {
          smallest = std::fmin(smallest, std::fabs(value));
          largest  = std::fmax(largest, std::fabs(value));
        }
        if (largest / smallest > std::fmax(tv.largest / tv.smallest, 1e6)) { return false; }
        call.reductions.push_back({true, target, equation});
        touched[target] = 1;
        lower[target] -= factor * rhs;
        upper[target] -= factor * rhs;
        unkey(target);
        pushed.insert(target);
        bool event = ev.singles == 1;
        for (const auto& entry : tv.core) {
          event = --size[entry.first] == 1 || event;
        }
        if (ev.singles == 1) { ++size[ev.single_col]; }
        if (event) {
          events.insert(equation);
          pushed.insert(equation);
          unkey(equation);
        }
        return true;
      };
      // One neighbour of the listed row. False when it counts as not parallel.
      auto visit = [&](int32_t other) {
        budget.spend(line_visit_work + 2 * (int64_t)st.csr.used[other]);
        ++call.comparisons;
        view_t theirs = view(st, other, size);
        bool is_parallel =
          row < other ? parallel(mine.core, theirs.core) : parallel(theirs.core, mine.core);
        if (!is_parallel || !reduce(other, theirs)) { return false; }
        budget.found();
        return true;
      };

      // A scan that ends on the limit with a live row of the bucket left in its direction
      // leaves pairs of the listed row uncompared.
      int32_t unlike = 0, last = row;
      while (rows[row].keyed && unlike < limit) {
        auto it = bucket.upper_bound(last);
        if (it == bucket.end()) { break; }
        int32_t other = *it;
        if (!alive(other)) {
          rows[other].keyed = false;
          bucket.erase(it);
          continue;
        }
        last = other;
        unlike += !visit(other);
      }
      if (rows[row].keyed && unlike >= limit) {
        for (auto it = bucket.upper_bound(last); it != bucket.end(); ++it) {
          call.limited = call.limited || alive(*it);
        }
      }
      unlike     = 0;
      int32_t at = row;
      while (rows[row].keyed && unlike < limit) {
        auto it = bucket.lower_bound(at);
        if (it == bucket.begin()) { break; }
        int32_t other = *std::prev(it);
        if (!alive(other)) {
          rows[other].keyed = false;
          bucket.erase(std::prev(it));
          continue;
        }
        if (!visit(other)) {
          ++unlike;
          at = other;
        }
      }
      if (rows[row].keyed && unlike >= limit) {
        for (auto it = bucket.begin(); it != bucket.end() && *it < at; ++it) {
          call.limited = call.limited || alive(*it);
        }
      }
    }
    finish(listed.size());
    return call;
  }
};

// Rows that are multiples of a few patterns: exact ones, ones within tol.zero and ones that
// are off by less than the rounding of the hash. The sides are so wide that no merge makes
// them cross, and small enough for a double to hold the tolerance next to them.
model_t bucket_model(rng_t& rng, int32_t nrows, int32_t ncols)
{
  static const double factor[] = {1., 1., 1., -1., 2., -2., 0.5, -0.5, 3., 1.5, 4., 0.25};
  int32_t npatterns            = rng.in(1, 6);
  double p_exact = rng.in(1, 9) / 10., p_unlike = rng.in(0, 9) / 10.;
  std::vector<dense_t> pattern(npatterns, dense_t(ncols, 0.));
  for (dense_t& p : pattern) {
    for (int32_t k = rng.in(2, 5); k > 0; --k) {
      p[rng.in(0, ncols - 1)] = rng.coefficient();
    }
  }
  std::vector<dense_t> rows(nrows, dense_t(ncols, 0.));
  for (dense_t& row : rows) {
    const dense_t& p = pattern[rng.in(0, npatterns - 1)];
    double f         = factor[rng.in(0, 11)];
    for (int32_t j = 0; j < ncols; ++j) {
      row[j] = f * p[j];
    }
    if (!rng.chance(p_exact)) {
      double step = rng.chance(p_unlike) ? 1e-8 : 1e-10;
      row[rng.in(0, ncols - 1)] *= 1. + step * rng.in(1, 4);
    }
    if (rng.chance(0.2)) { row[rng.in(0, ncols - 1)] = rng.in(1, 3); }
  }
  dense_t col_lower(ncols), col_upper(ncols);
  for (int32_t j = 0; j < ncols; ++j) {
    col_lower[j] = rng.in(-2, 1);
    col_upper[j] = col_lower[j] + (rng.chance(0.1) ? 0 : rng.in(1, 3));
  }
  return make_model(rows, dense_t(nrows, -inf), dense_t(nrows, 1e7), col_lower, col_upper);
}

// The same patterns over the first `shared` columns, and behind them columns of their own for
// some rows: none, one or two per row, so that rows are parallel up to singleton columns. Some
// rows are equations with the right-hand side 0, the others have sides around 0 that hold the
// tolerance, and every column has the lower bound 0, where the tests fix it: the fixed terms
// are 0, every equation keeps the right-hand side 0 and no merge makes sides cross.
model_t near_model(rng_t& rng, int32_t nrows, int32_t shared)
{
  static const double factor[] = {1., 1., 1., -1., 2., -2., 0.5, -0.5, 3., 1.5, 4., 0.25};
  constexpr double wide        = 1e7;
  int32_t npatterns            = rng.in(1, 5);
  double p_exact = rng.in(5, 10) / 10., p_single = rng.in(1, 8) / 10.,
         p_equation = rng.in(1, 7) / 10.;
  std::vector<dense_t> pattern(npatterns, dense_t(shared, 0.));
  for (dense_t& p : pattern) {
    for (int32_t k = rng.in(2, 4); k > 0; --k) {
      p[rng.in(0, shared - 1)] = rng.coefficient();
    }
  }
  std::vector<int32_t> own(nrows, 0);
  int32_t ncols = shared;
  for (int32_t& n : own) {
    n = rng.chance(p_single) ? (rng.chance(0.8) ? 1 : 2) : 0;
    ncols += n;
  }
  std::vector<dense_t> rows(nrows, dense_t(ncols, 0.));
  dense_t row_lower(nrows), row_upper(nrows);
  int32_t next = shared;
  for (int32_t i = 0; i < nrows; ++i) {
    const dense_t& p = pattern[rng.in(0, npatterns - 1)];
    double f         = factor[rng.in(0, 11)];
    for (int32_t j = 0; j < shared; ++j) {
      rows[i][j] = f * p[j];
    }
    if (!rng.chance(p_exact)) { rows[i][rng.in(0, shared - 1)] *= 1. + 1e-8 * rng.in(1, 4); }
    for (int32_t k = 0; k < own[i]; ++k) {
      rows[i][next++] = rng.coefficient();
    }
    int32_t kind = rng.chance(p_equation) ? 0 : rng.in(1, 3);
    row_lower[i] = kind == 0 ? 0. : kind == 1 ? -inf : -wide;
    row_upper[i] = kind == 0 ? 0. : kind == 2 ? inf : wide;
  }
  dense_t col_lower(ncols, 0.), col_upper(ncols);
  for (int32_t j = 0; j < ncols; ++j) {
    col_upper[j] = j < shared && rng.chance(0.1) ? 0 : rng.in(1, 3);
  }
  return make_model(rows, row_lower, row_upper, col_lower, col_upper);
}

// One call of the rule on the lists as they are, checked against the reference.
reference_t::call_t checked_call(fixture_t& f, reference_t& reference, work_budget_t budget)
{
  reference.near               = f.opt.near_parallel_rows;
  reference_t::call_t expected = reference.run(f.st,
                                               f.st.row_frontier.list[row_list::parallel_rows],
                                               f.st.row_frontier.list[row_list::singletons],
                                               budget,
                                               f.opt.parallel_bucket_limit);
  GPRS_CHECK(!expected.infeasible);
  size_t records = f.ps.order.size(), merges = f.ps.parallel_rows.size();
  size_t additions     = f.ps.row_additions.size();
  rule_result_t result = f.rule.run(f.st, budget, f.ps, f.opt);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, (int64_t)expected.reductions.size());
  GPRS_CHECK_EQ(result.stopped, expected.stopped);
  GPRS_CHECK_EQ(result.limited, expected.limited);
  GPRS_CHECK_EQ(budget.used, expected.work);
  list_t listed = f.st.row_frontier.list[row_list::parallel_rows];
  std::sort(listed.begin(), listed.end());
  GPRS_CHECK(listed == expected.listed);
  if (f.opt.near_parallel_rows) {
    list_t singletons = f.st.row_frontier.list[row_list::singletons];
    std::sort(singletons.begin(), singletons.end());
    GPRS_CHECK(singletons == expected.singletons);
  }
  GPRS_CHECK_EQ(f.ps.order.size(), records + expected.reductions.size());
  for (size_t k = 0; k < expected.reductions.size() && records + k < f.ps.order.size(); ++k) {
    const reference_t::reduction_t& r = expected.reductions[k];
    if (f.ps.order[records + k] == record_kind_t::parallel_row) {
      GPRS_CHECK(!r.addition);
      GPRS_CHECK_EQ(f.ps.parallel_rows[merges].removed_row, r.changed);
      GPRS_CHECK_EQ(f.ps.parallel_rows[merges].kept_row, r.other);
      ++merges;
    } else {
      GPRS_CHECK(r.addition);
      GPRS_CHECK_EQ(f.ps.row_additions[additions].row, r.changed);
      GPRS_CHECK_EQ(f.ps.row_additions[additions].equation_row, r.other);
      ++additions;
    }
  }
  return expected;
}

}  // namespace

namespace {

struct sequence_counts_t {
  int64_t calls     = 0;
  int64_t merges    = 0;
  int64_t additions = 0;
  int64_t stops     = 0;
  int64_t listed    = 0;  // rows that a call left on row_list::singletons
};

// Calls on one row, on a few rows and on all of them, with and without a budget, between
// changed, moved and deleted entries, fixed columns, deleted rows and rebuilds: every call
// makes the reductions that the definition makes, on the same rows and in the same order,
// spends the same work and leaves the same rows on the two lists.
sequence_counts_t run_sequences(bool singleton_columns, bool near, uint64_t seeds)
{
  sequence_counts_t counts;
  for (uint64_t seed = 1; seed <= seeds; ++seed) {
    rng_t rng(seed);
    int32_t nrows = rng.in(1, seed % 5 == 0 ? 40 : 400), shared = rng.in(2, 12);
    fixture_t f(singleton_columns ? near_model(rng, nrows, shared)
                                  : bucket_model(rng, nrows, shared));
    f.opt.near_parallel_rows = near;
    int32_t ncols            = f.st.ncols;
    reference_t reference;
    static const int32_t limits[] = {1, 2, 4, 4, 8};
    f.opt.parallel_bucket_limit   = limits[rng.in(0, 4)];
    int32_t failures              = gprs::test::failure_count();
    for (int32_t step = rng.in(3, 12); step > 0; --step) {
      work_budget_t budget;
      budget.limit                 = rng.chance(0.5) ? no_limit : rng.in(1, 30 * nrows);
      budget.stall_limit           = rng.chance(0.7) ? budget.limit : rng.in(1, 30 * nrows);
      reference_t::call_t expected = checked_call(f, reference, budget);
      GPRS_CHECK_EQ(f.st.consistency_error(), std::string());
      ++counts.calls;
      for (const reference_t::reduction_t& r : expected.reductions) {
        ++(r.addition ? counts.additions : counts.merges);
      }
      counts.stops += expected.stopped;
      counts.listed += expected.singletons.size();
      if (gprs::test::failure_count() != failures) {
        std::printf("    seed %llu\n", (unsigned long long)seed);
        return counts;
      }
      // One row changes, or a few, or many.
      int32_t changes = rng.chance(0.3) ? rng.in(4, 40) : rng.in(0, 3);
      for (; changes > 0; --changes) {
        int32_t row  = rng.in(0, nrows - 1);
        int32_t col  = rng.chance(0.7) ? rng.in(0, shared - 1) : rng.in(0, ncols - 1);
        int32_t kind = rng.in(0, singleton_columns ? 12 : 9);
        if (kind == 0) {
          if (f.st.is_live(col)) { fix_column(f.st, f.ps, col, f.st.col_lower[col]); }
        } else if (kind == 1) {
          if (f.st.row_alive[row]) { f.st.delete_row(row); }
        } else if (kind == 2) {
          f.st.rebuild();
        } else if (f.st.row_alive[row] && f.st.is_live(col)) {
          int64_t e = f.st.find_entry(row, col);
          if (e < 0) { continue; }
          double old = f.st.csr.value[e];
          if (kind == 10) {
            f.st.delete_entry(e);
          } else if (kind > 10) {
            // Often into a column behind the shared ones, which then has a second entry.
            int32_t target = rng.chance(0.5) ? rng.in(shared, ncols - 1) : rng.in(0, ncols - 1);
            if (!f.st.is_live(target) || f.st.find_entry(row, target) >= 0) { continue; }
            f.st.move_entry(e, target, rng.coefficient());
          } else {
            double value = kind < 6 ? rng.coefficient() : old * (1. + 1e-8 * rng.in(1, 3));
            if (value != old) { f.st.set_coefficient(e, value); }
          }
        }
      }
    }
  }
  return counts;
}

}  // namespace

// Rows without singleton columns to begin with, and the rule as it was before rows parallel up
// to singleton columns: every reduction is a merge.
GPRS_TEST(parallel_random_sequences)
{
  sequence_counts_t counts = run_sequences(false, false, 250);
  // The test is not vacuous.
  GPRS_CHECK(counts.calls > 1500);
  GPRS_CHECK(counts.merges > 15000);
  GPRS_CHECK_EQ(counts.additions, 0);
  GPRS_CHECK(counts.stops > 120);
}

// The same models with the cores as keys: columns become singleton columns as rows go, which
// lists the rows that hold them. No row is an equation, so every reduction is still a merge.
GPRS_TEST(parallel_random_sequences_on_cores)
{
  sequence_counts_t counts = run_sequences(false, true, 250);
  GPRS_CHECK(counts.calls > 1500);
  GPRS_CHECK(counts.merges > 15000);
  GPRS_CHECK_EQ(counts.additions, 0);
  GPRS_CHECK(counts.stops > 120);
  GPRS_CHECK(counts.listed > 10);
}

// Rows parallel up to singleton columns, equations among them: merges and subtractions.
GPRS_TEST(near_random_sequences)
{
  sequence_counts_t counts = run_sequences(true, true, 300);
  GPRS_CHECK(counts.calls > 1800);
  GPRS_CHECK(counts.merges > 10000);
  GPRS_CHECK(counts.additions > 7000);
  GPRS_CHECK(counts.stops > 120);
  GPRS_CHECK(counts.listed > 400);
}

// 2,000 rows x0 + v xj over 8 columns j, each alone in its bucket at first. Then one to three
// rows move per call, so that a call lists few of the rows: into the bucket of a row of their
// column, as a parallel row or as one that is off by less than the rounding of the hash, or
// into a bucket of their own. Rows that moved lately move again and are joined more often than
// the others, so that buckets form, change and dissolve among the keys of few calls.
GPRS_TEST(parallel_few_rows_among_many)
{
  constexpr int32_t nrows = 2000, classes = 8;
  std::vector<dense_t> rows(nrows, dense_t(classes + 1, 0.));
  for (int32_t i = 0; i < nrows; ++i) {
    rows[i][0]               = 1.;
    rows[i][1 + i % classes] = 3. + i / classes;
  }
  fixture_t f(make_model(rows,
                         dense_t(nrows, -inf),
                         dense_t(nrows, 1e9),
                         dense_t(classes + 1, 0.),
                         dense_t(classes + 1, 9.)));
  reference_t reference;
  rng_t rng(7);
  auto slot = [&](int32_t row) { return f.st.find_entry(row, 1 + row % classes); };
  list_t moved;
  double unused  = 1000.5;
  int64_t merges = 0, stops = 0, comparisons = 0;
  int32_t failures = gprs::test::failure_count();
  for (int32_t step = 0; step < 1500 && gprs::test::failure_count() == failures; ++step) {
    work_budget_t budget;
    budget.limit = budget.stall_limit = rng.chance(0.9) ? no_limit : rng.in(1, 200);
    reference_t::call_t expected      = checked_call(f, reference, budget);
    merges += expected.reductions.size();
    stops += expected.stopped;
    comparisons += expected.comparisons;
    if (step % 150 == 149) {
      f.st.rebuild();
      GPRS_CHECK_EQ(f.st.consistency_error(), std::string());
    }
    for (int32_t changes = rng.in(1, 3); changes > 0; --changes) {
      int32_t row = rng.in(0, nrows - 1);
      if (!moved.empty() && rng.chance(0.4)) { row = moved[rng.in(0, moved.size() - 1)]; }
      if (!f.st.row_alive[row]) { continue; }
      int32_t kind = rng.in(0, 9);
      double value = unused;
      if (kind == 0) {
        f.st.delete_row(row);
        continue;
      } else if (kind < 3) {
        unused += 1.;
      } else {
        int32_t target = classes * rng.in(0, nrows / classes - 1) + row % classes;
        for (int32_t tries = kind < 8 ? 6 : 0; tries > 0 && !moved.empty(); --tries) {
          int32_t other = moved[rng.in(0, moved.size() - 1)];
          if (other % classes == row % classes && other != row) {
            target = other;
            break;
          }
        }
        if (target == row || !f.st.row_alive[target]) { continue; }
        value = f.st.csr.value[slot(target)] * (1. + 1e-8 * rng.in(0, 3));
      }
      if (value != f.st.csr.value[slot(row)]) { f.st.set_coefficient(slot(row), value); }
      moved.push_back(row);
      if (moved.size() > 30) { moved.erase(moved.begin()); }
    }
  }
  GPRS_CHECK_EQ(f.st.consistency_error(), std::string());
  // The test is not vacuous.
  GPRS_CHECK(merges > 250);
  GPRS_CHECK(comparisons > 1200);
  GPRS_CHECK(stops > 8);
}

// A row that joined a bucket in a call on one row, stayed there through a call on enough rows
// to order all keys afresh, and then left: a row that joins the bucket afterwards meets the
// rows that are in it and not the one that left.
GPRS_TEST(parallel_bucket_left_after_many_rows_changed)
{
  constexpr int32_t nrows = 100;
  std::vector<dense_t> rows;
  for (int32_t i = 0; i < nrows; ++i) {
    rows.push_back({1., 3. + i});
  }
  fixture_t f(make_model(rows, dense_t(nrows, -inf), dense_t(nrows, 1e9), {0., 0.}, {9., 9.}));
  reference_t reference;
  work_budget_t budget;
  budget.limit = budget.stall_limit = no_limit;
  auto set                          = [&](int32_t row, double value) {
    f.st.set_coefficient(f.st.find_entry(row, 1), value);
  };
  auto call = [&]() { return checked_call(f, reference, budget).comparisons; };
  GPRS_CHECK_EQ(call(), 0);
  // Rows 50, 30 and 70 join row 10, each off by less than the rounding of the hash.
  double joined = f.st.csr.value[f.st.find_entry(10, 1)];
  set(50, joined * (1. + 1e-8));
  GPRS_CHECK_EQ(call(), 1);
  for (int32_t row = 90; row < 94; ++row) {
    set(row, 500. + row);
  }
  GPRS_CHECK_EQ(call(), 0);
  set(50, 700.);
  GPRS_CHECK_EQ(call(), 0);
  set(30, joined * (1. + 2e-8));
  GPRS_CHECK_EQ(call(), 1);
  set(70, joined * (1. + 3e-8));
  GPRS_CHECK_EQ(call(), 2);
  GPRS_CHECK_EQ(f.st.live_rows, nrows);
  GPRS_CHECK_EQ(f.st.consistency_error(), std::string());
}

// ---- Rows parallel up to singleton columns (opt.near_parallel_rows) ----

namespace {

void check_addition(const postsolve_t& ps, size_t k, int32_t row, int32_t equation, double factor)
{
  GPRS_REQUIRE(k < ps.row_additions.size());
  GPRS_CHECK_EQ(ps.row_additions[k].row, row);
  GPRS_CHECK_EQ(ps.row_additions[k].equation_row, equation);
  GPRS_CHECK_EQ(ps.row_additions[k].factor, factor);
}

// Findings of the same model with the option off.
int64_t findings_without(const model_t& m)
{
  fixture_t f(m);
  f.opt.near_parallel_rows = false;
  return f.run().findings;
}

}  // namespace

// A set-packing row and the equation that defines the sum of its columns as a column of its
// own (brazil3): the equation is subtracted and the row is left as a bound of that column.
GPRS_TEST(near_inequality_and_equation_with_a_singleton_column)
{
  // Row 0: x0 + x1 + x2 <= 1. Row 1: x0 + x1 + x2 - y = 0.
  model_t m = make_model(
    {{1., 1., 1., 0.}, {1., 1., 1., -1.}}, {-inf, 0.}, {1., 0.}, dense_t(4, 0.), dense_t(4, 1.));
  GPRS_CHECK_EQ(findings_without(m), 0);
  fixture_t f(m);
  int64_t nnz          = f.st.live_nnz;
  rule_result_t result = f.run();
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 1);
  GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1}));
  GPRS_CHECK(row_entries(f.st, 0) == (entries_t{{3, 1.}}));
  GPRS_CHECK_EQ(f.st.row_lower[0], -inf);
  GPRS_CHECK_EQ(f.st.row_upper[0], 1.);
  GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{0, 1.}, {1, 1.}, {2, 1.}, {3, -1.}}));
  GPRS_CHECK_EQ(f.st.row_upper[1], 0.);
  GPRS_CHECK_EQ(f.st.live_nnz, nnz - 2);
  GPRS_CHECK_EQ(f.ps.order.size(), 1u);
  GPRS_CHECK(f.ps.parallel_rows.empty());
  check_addition(f.ps, 0, 0, 1, 1.);
  // Row 0 is left to the row rules as a singleton row. The equation is listed again with the
  // entries it had: y has a second entry and x0, x1 and x2 have one each.
  GPRS_CHECK_EQ(f.st.row_version[1], 0u);
  GPRS_CHECK(f.st.row_frontier.list[row_list::singletons] == (list_t{1}));
  GPRS_CHECK(on_list(f.st, row_list::parallel_rows, 0) &&
             on_list(f.st, row_list::parallel_rows, 1));
  GPRS_CHECK_EQ(f.run().findings, 0);
  GPRS_CHECK(f.st.row_frontier.empty(row_list::parallel_rows));
  GPRS_CHECK(f.st.row_frontier.empty(row_list::singletons));
  model_t reduced = f.reduced();
  GPRS_CHECK_EQ(reduced.nnz(), 5);
  GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps) > 0);
}

// Two equations, one with a singleton column (piperout-d27): the one without it is subtracted,
// whatever the order of the rows, and the other is left as a singleton row that fixes the column.
GPRS_TEST(near_two_equations_one_singleton_column)
{
  for (bool swapped : {false, true}) {
    // x0 + x1 + x2 = 1 and y - x0 - x1 - x2 = 0.
    dense_t plain = {1., 1., 1., 0.}, with_y = {-1., -1., -1., 1.};
    model_t m        = swapped
                         ? make_model({with_y, plain}, {0., 1.}, {0., 1.}, dense_t(4, 0.), dense_t(4, 1.))
                         : make_model({plain, with_y}, {1., 0.}, {1., 0.}, dense_t(4, 0.), dense_t(4, 1.));
    int32_t equation = swapped ? 1 : 0, target = swapped ? 0 : 1;
    GPRS_CHECK_EQ(findings_without(m), 0);
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(row_entries(f.st, target) == (entries_t{{3, 1.}}));
    GPRS_CHECK_EQ(f.st.row_lower[target], 1.);
    GPRS_CHECK_EQ(f.st.row_upper[target], 1.);
    GPRS_CHECK(row_entries(f.st, equation) == (entries_t{{0, 1.}, {1, 1.}, {2, 1.}}));
    check_addition(f.ps, 0, target, equation, -1.);
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK_EQ(check_same_feasible_set(m, f.reduced(), f.ps), 3);
  }
}

// Two equations with a singleton column each (t11nonreg): the smaller index is subtracted and
// the other row is left as a doubleton equation in the two columns.
GPRS_TEST(near_two_equations_a_singleton_column_each)
{
  // Row 0: x0 + x1 + s0 = 2. Row 1: -x0 - x1 + s1 = 1.
  model_t m = make_model(
    {{1., 1., 1., 0.}, {-1., -1., 0., 1.}}, {2., 1.}, {2., 1.}, dense_t(4, 0.), dense_t(4, 3.));
  GPRS_CHECK_EQ(findings_without(m), 0);
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_CHECK(row_entries(f.st, 0) == (entries_t{{0, 1.}, {1, 1.}, {2, 1.}}));
  GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, 1.}, {3, 1.}}));
  GPRS_CHECK_EQ(f.st.row_lower[1], 3.);
  GPRS_CHECK_EQ(f.st.row_upper[1], 3.);
  check_addition(f.ps, 0, 1, 0, -1.);
  GPRS_CHECK(on_list(f.st, row_list::doubleton, 1));
  GPRS_CHECK_EQ(f.run().findings, 0);
  GPRS_CHECK(check_same_feasible_set(m, f.reduced(), f.ps) > 0);
}

// The equation with three singleton columns is the target of the one without: one step, after
// which neither row has a core. Subtracting the first from the second and back would never end.
GPRS_TEST(near_equation_with_several_singleton_columns)
{
  // Row 0: x0 + x1 + x2 = 1. Row 1: x0 + x1 + x2 + s0 + s1 + s2 = 4.
  for (bool swapped : {false, true}) {
    dense_t plain = {1., 1., 1., 0., 0., 0.}, wide = {1., 1., 1., 1., 1., 1.};
    model_t m        = swapped
                         ? make_model({wide, plain}, {4., 1.}, {4., 1.}, dense_t(6, 0.), dense_t(6, 2.))
                         : make_model({plain, wide}, {1., 4.}, {1., 4.}, dense_t(6, 0.), dense_t(6, 2.));
    int32_t equation = swapped ? 1 : 0, target = swapped ? 0 : 1;
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(row_entries(f.st, target) == (entries_t{{3, 1.}, {4, 1.}, {5, 1.}}));
    GPRS_CHECK_EQ(f.st.row_lower[target], 3.);
    GPRS_CHECK_EQ(f.st.row_upper[target], 3.);
    GPRS_CHECK(row_entries(f.st, equation) == (entries_t{{0, 1.}, {1, 1.}, {2, 1.}}));
    check_addition(f.ps, 0, target, equation, 1.);
    for (int32_t call = 0; call < 5; ++call) {
      GPRS_CHECK_EQ(f.run().findings, 0);
    }
    GPRS_CHECK(check_same_feasible_set(m, f.reduced(), f.ps) > 0);
  }
  {
    // The row without singleton columns is no equation: nothing to subtract from the other.
    model_t m = make_model({{1., 1., 1., 0., 0., 0.}, {1., 1., 1., 1., 1., 1.}},
                           {-inf, 4.},
                           {1., 4.},
                           dense_t(6, 0.),
                           dense_t(6, 2.));
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK_EQ(f.st.live_nnz, 9);
  }
  {
    // An equation with one singleton column and a row with two: left alone, either way round.
    model_t m = make_model({{1., 1., 1., 0., 0.}, {1., 1., 0., 1., 1.}},
                           {2., -inf},
                           {2., 5.},
                           dense_t(5, 0.),
                           dense_t(5, 2.));
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 0);
    model_t both      = m;
    both.row_lower[1] = 5.;
    fixture_t g(both);
    GPRS_CHECK_EQ(g.run().findings, 0);
  }
}

// One equation and two rows with its core. Without a singleton column the equation serves both
// in one call. With one it serves one row per call: its column has a second entry then, until
// the row rules have removed the first row.
GPRS_TEST(near_one_equation_and_two_rows)
{
  {
    // Row 0: 2 x0 + 2 x1 = 2. Row 1: 2 x0 + 2 x1 + s <= 4. Row 2: x0 + x1 - t >= 0.
    model_t m = make_model({{2., 2., 0., 0.}, {2., 2., 1., 0.}, {1., 1., 0., -1.}},
                           {2., -inf, 0.},
                           {2., 4., inf},
                           dense_t(4, 0.),
                           dense_t(4, 3.));
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 2);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, 1.}}));
    GPRS_CHECK_EQ(f.st.row_upper[1], 2.);
    GPRS_CHECK(row_entries(f.st, 2) == (entries_t{{3, -1.}}));
    GPRS_CHECK_EQ(f.st.row_lower[2], -1.);
    GPRS_CHECK_EQ(f.st.row_upper[2], inf);
    check_addition(f.ps, 0, 1, 0, 1.);
    check_addition(f.ps, 1, 2, 0, 0.5);
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK(check_same_feasible_set(m, f.reduced(), f.ps) > 0);
  }
  {
    // Row 0: x0 + x1 - y = 0. Row 1: x0 + x1 <= 1. Row 2: x0 + x1 + t <= 3.
    model_t m = make_model({{1., 1., -1., 0.}, {1., 1., 0., 0.}, {1., 1., 0., 1.}},
                           {0., -inf, -inf},
                           {0., 1., 3.},
                           dense_t(4, 0.),
                           dense_t(4, 2.));
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, 1.}}));
    GPRS_CHECK_EQ(f.st.row_upper[1], 1.);
    GPRS_CHECK(row_entries(f.st, 2) == (entries_t{{0, 1.}, {1, 1.}, {3, 1.}}));
    GPRS_CHECK_EQ(f.run().findings, 0);
    // What the row rules do with y <= 1: the bound, and the row goes. y has one entry again,
    // which lists the equation without an event of its own.
    f.st.set_col_upper(2, 1.);
    f.st.delete_row(1);
    GPRS_CHECK(f.st.row_frontier.list[row_list::singletons] == (list_t{0}));
    GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{0}));
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(row_entries(f.st, 2) == (entries_t{{2, 1.}, {3, 1.}}));
    GPRS_CHECK_EQ(f.st.row_upper[2], 3.);
    check_addition(f.ps, 1, 2, 0, 1.);
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK(check_same_feasible_set(m, f.reduced(), f.ps) > 0);
  }
}

// The target as a ranged row, as an inequality of either kind and as a row without sides, with
// a negative factor: both finite sides move by the same amount and keep their places.
GPRS_TEST(near_sides_of_the_target)
{
  struct sides_t {
    double lower, upper, new_lower, new_upper;
  };
  // Row 0: x0 - x1 + s = 2. Row 1: -x0 + x1, which is s - 2.
  for (sides_t sides : {sides_t{-4., 6., -2., 8.},
                        sides_t{-inf, 6., -inf, 8.},
                        sides_t{-4., inf, -2., inf},
                        sides_t{-inf, inf, -inf, inf}}) {
    model_t m = make_model({{1., -1., 1.}, {-1., 1., 0.}},
                           {2., sides.lower},
                           {2., sides.upper},
                           dense_t(3, 0.),
                           dense_t(3, 4.));
    fixture_t f(m);
    int64_t side_changes = f.st.side_changes;
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, 1.}}));
    GPRS_CHECK_EQ(f.st.row_lower[1], sides.new_lower);
    GPRS_CHECK_EQ(f.st.row_upper[1], sides.new_upper);
    GPRS_CHECK_EQ(f.st.side_changes - side_changes,
                  (sides.lower > -inf ? 1 : 0) + (sides.upper < inf ? 1 : 0));
    check_addition(f.ps, 0, 1, 0, -1.);
    GPRS_CHECK(check_same_feasible_set(m, f.reduced(), f.ps) > 0);
  }
  {
    // A right-hand side of 0 moves no side.
    model_t m = make_model(
      {{1., -1., 1.}, {-1., 1., 0.}}, {0., -4.}, {0., 6.}, dense_t(3, 0.), dense_t(3, 4.));
    fixture_t f(m);
    int64_t side_changes = f.st.side_changes;
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK_EQ(f.st.side_changes, side_changes);
    GPRS_CHECK_EQ(f.st.row_lower[1], -4.);
    GPRS_CHECK_EQ(f.st.row_upper[1], 6.);
  }
  {
    // With row 1 an equation as well, it is the one without a singleton column and row 0 is
    // the target: s = 1 + 2 / 2.
    model_t m = make_model(
      {{1., -1., 1.}, {-2., 2., 0.}}, {1., 2.}, {1., 2.}, dense_t(3, 0.), dense_t(3, 4.));
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(row_entries(f.st, 0) == (entries_t{{2, 1.}}));
    GPRS_CHECK_EQ(f.st.row_lower[0], 2.);
    GPRS_CHECK_EQ(f.st.row_upper[0], 2.);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{0, -2.}, {1, 2.}}));
    check_addition(f.ps, 0, 0, 1, -0.5);
    GPRS_CHECK(check_same_feasible_set(m, f.reduced(), f.ps) > 0);
  }
}

// The rule does not read the type of a column: an integer and a continuous singleton column
// give the same row, in the equation and in the target.
GPRS_TEST(near_integer_and_continuous_singleton_columns)
{
  for (int32_t continuous = 0; continuous < 4; ++continuous) {
    // Row 0: 4 x0 + 8 x1 + 2 s = 12. Row 1: 4 x0 + 8 x1 + t <= 20, which is t - 2 s <= 8.
    model_t m     = make_model({{4., 8., 2., 0.}, {4., 8., 0., 1.}},
                               {12., -inf},
                               {12., 20.},
                           dense_t(4, 0.),
                               {2., 2., 6., 5.});
    m.col_type[2] = continuous & 1 ? 0 : 1;
    m.col_type[3] = continuous & 2 ? 0 : 1;
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, -2.}, {3, 1.}}));
    GPRS_CHECK_EQ(f.st.row_upper[1], 8.);
    check_addition(f.ps, 0, 1, 0, 1.);
    GPRS_CHECK(check_same_feasible_set(m, f.reduced(), f.ps) > 0);
  }
}

// The terms of fixed columns are constants of their rows: those of the equation go into the
// shift, those of the target stay in the target.
GPRS_TEST(near_fixed_columns_in_both_rows)
{
  // Row 0: x0 + x1 + y + 2 z = 5 with z fixed at 1. Row 1: x0 + x1 + 3 w <= 7 with w fixed at 2.
  model_t m = make_model({{1., 1., 1., 2., 0.}, {1., 1., 0., 0., 3.}},
                         {5., -inf},
                         {5., 7.},
                         {0., 0., 0., 1., 2.},
                         {2., 2., 4., 1., 2.});
  fixture_t f(m);
  GPRS_CHECK_EQ(f.st.row_free_size[0], 3);
  GPRS_CHECK_EQ(f.run().findings, 1);
  // The target: -y + 3 w <= 7 - (5 - 2).
  GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, -1.}, {4, 3.}}));
  GPRS_CHECK_EQ(f.st.row_upper[1], 4.);
  GPRS_CHECK(row_entries(f.st, 0) == (entries_t{{0, 1.}, {1, 1.}, {2, 1.}, {3, 2.}}));
  check_addition(f.ps, 0, 1, 0, 1.);
  model_t reduced = f.reduced();
  GPRS_REQUIRE(reduced.nrows == 2 && reduced.nnz() == 4);
  GPRS_CHECK_EQ(reduced.row_upper[1], -2.);
  GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps) > 0);
}

// The shift is the right-hand side of the equation less its fixed terms, times the factor, and
// goes into a side of the target: a pair is left alone when one of them is too large to hold
// the tolerance next to the others. A fixed term of the target is not read and may be huge.
GPRS_TEST(near_huge_fixed_terms_and_sides)
{
  {
    // The equation holds 2 z with z fixed at 1e20: its right-hand side less that term would
    // lose the 3, and with it the side of the target.
    model_t m = make_model({{1., 1., 1., 2.}, {1., 1., 0., 0.}},
                           {3., -inf},
                           {3., 5.},
                           {0., 0., -1e21, 1e20},
                           {3., 3., 1e21, 1e20});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{0, 1.}, {1, 1.}}));
    GPRS_CHECK_EQ(f.st.row_upper[1], 5.);
    GPRS_CHECK(f.ps.row_additions.empty());
  }
  {
    // The target holds that term: its stored side moves by 3 and the term stays an entry.
    model_t m = make_model({{1., 1., 1., 0.}, {1., 1., 0., 2.}},
                           {3., -inf},
                           {3., 5.},
                           {0., 0., 0., -2e20},
                           {3., 3., 3., -2e20});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, -1.}, {3, 2.}}));
    GPRS_CHECK_EQ(f.st.row_upper[1], 2.);
    GPRS_CHECK(check_same_feasible_set(m, f.reduced(), f.ps) > 0);
  }
  // The limit is |value| * tol.side_error <= tol.feasibility, 1e8 with the default tolerances.
  for (double value : {5e7, 2e8}) {
    bool taken = value == 5e7;
    {
      // The right-hand side of the equation. x0 + x1 + y = value, x0 + x1 <= 5.
      model_t m = make_model({{1., 1., 1.}, {1., 1., 0.}},
                             {value, -inf},
                             {value, 5.},
                             {0., 0., value - 6.},
                             {3., 3., value});
      fixture_t f(m);
      GPRS_CHECK_EQ(f.run().findings, taken ? 1 : 0);
      if (taken) { GPRS_CHECK_EQ(f.st.row_upper[1], 5. - value); }
    }
    {
      // The fixed terms of the equation, next to a small right-hand side less them.
      model_t m = make_model({{1., 1., 1., 1.}, {1., 1., 0., 0.}},
                             {value + 3., -inf},
                             {value + 3., 5.},
                             {0., 0., 0., value},
                             {3., 3., 3., value});
      fixture_t f(m);
      GPRS_CHECK_EQ(f.run().findings, taken ? 1 : 0);
      if (taken) { GPRS_CHECK_EQ(f.st.row_upper[1], 2.); }
    }
    {
      // A side of the target.
      model_t m = make_model(
        {{1., 1., 1.}, {1., 1., 0.}}, {3., -inf}, {3., value}, dense_t(3, 0.), {3., 3., 3.});
      fixture_t f(m);
      GPRS_CHECK_EQ(f.run().findings, taken ? 1 : 0);
      if (taken) { GPRS_CHECK_EQ(f.st.row_upper[1], value - 3.); }
    }
  }
  // The shift itself: the right-hand side of the equation less its fixed terms can be twice as
  // large as each of them. x0 + x1 + y + w = half with w fixed at -half, x0 + x1 <= 5.
  for (double half : {4e7, 6e7}) {
    model_t m = make_model({{1., 1., 1., 1.}, {1., 1., 0., 0.}},
                           {half, -inf},
                           {half, 5.},
                           {0., 0., 2. * half - 6., -half},
                           {3., 3., 2. * half, -half});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, half == 4e7 ? 1 : 0);
    if (half == 4e7) {
      GPRS_CHECK_EQ(f.st.row_upper[1], 5. - 2. * half);
      GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, -1.}}));
    }
  }
}

// Tiny and huge coefficients: a factor that is not finite and a target whose coefficients
// would spread further than before and than 1e6 leave the pair alone.
GPRS_TEST(near_coefficient_guards)
{
  {
    // Row 1 = 2^-1070 * row 0 on the core, a ratio that a double holds; the factor of the
    // other direction, 2^1070, is not finite.
    double large = std::ldexp(1., 600), small = std::ldexp(1., -470);
    model_t m = make_model({{large, large, 0.}, {small, small, 1.}},
                           {-inf, 1.},
                           {1., 1.},
                           dense_t(3, 0.),
                           dense_t(3, 1.));
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK_EQ(f.st.live_nnz, 5);
  }
  for (double large : {1e2, 1e4}) {
    // Row 0: x0 + x1 + s / 512 = 1. Row 1: x0 + x1 + large * t <= 3, which would be
    // large * t - s / 512 <= 2.
    double tiny = 1. / 512.;
    model_t m   = make_model({{1., 1., tiny, 0.}, {1., 1., 0., large}},
                             {1., -inf},
                             {1., 3.},
                           dense_t(4, 0.),
                           dense_t(4, 2.));
    fixture_t f(m);
    bool taken = large == 1e2;
    GPRS_CHECK_EQ(f.run().findings, taken ? 1 : 0);
    if (taken) { GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, -tiny}, {3, large}})); }
  }
  {
    // Row 1: (x0 + x1) / 1024 + 4096 t + u / 8192 <= 3 spreads over 3e7 already, and
    // 4096 t + u / 8192 is no worse.
    double c = 1. / 1024., d = 1. / 8192.;
    model_t m = make_model(
      {{1., 1., 0., 0.}, {c, c, 4096., d}}, {1., -inf}, {1., 3.}, dense_t(4, 0.), dense_t(4, 2.));
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, 4096.}, {3, d}}));
    GPRS_CHECK_EQ(f.st.row_upper[1], 3. - c);
  }
}

// The step multiplies the error of the equation by |f| / |c| and that of the target by 1 / |a|
// (c a coefficient of a singleton column of the target, a that of the equation): above 1000
// the pair is left alone, whatever the size of its constants, and a presolve of the same rows
// leaves it too.
GPRS_TEST(near_growth_limit)
{
  for (double growth : {512., 1024., 4096.}) {
    bool taken = growth <= 1000.;
    {
      // Row 0: x0 + x1 = 1. Row 1: x0 + x1 + y / growth <= 3.
      model_t m = make_model({{1., 1., 0.}, {1., 1., 1. / growth}},
                             {1., -inf},
                             {1., 3.},
                             dense_t(3, 0.),
                             dense_t(3, 2.));
      fixture_t f(m);
      rule_result_t result = f.run();
      GPRS_CHECK_EQ(result.findings, taken ? 1 : 0);
      GPRS_CHECK(!result.limited);
      if (taken) { GPRS_CHECK_EQ(f.st.row_upper[1], 2.); }
    }
    {
      // Row 0: x0 + x1 + s / growth = 1. Row 1: x0 + x1 + y <= 3.
      model_t m = make_model({{1., 1., 1. / growth, 0.}, {1., 1., 0., 1.}},
                             {1., -inf},
                             {1., 3.},
                             dense_t(4, 0.),
                             dense_t(4, 2.));
      fixture_t f(m);
      rule_result_t result = f.run();
      GPRS_CHECK_EQ(result.findings, taken ? 1 : 0);
      GPRS_CHECK(!result.limited);
      if (taken) { GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, -1. / growth}, {3, 1.}})); }
    }
  }
}

// The factor f of the target over the equation. What is done to the equation later reaches
// the old target row times |f|, and what is done to the target reaches the equation divided by
// |f| once the singleton column of the equation has gone through the target: the subtraction is
// made with |f| <= 1, and with |f| = 1 when the equation has a singleton column.
GPRS_TEST(near_factor)
{
  struct pair_t {
    double equation, target;  // coefficients of the cores
    double a;                 // of the singleton column of the equation, 0 without one
    bool taken;
  };
  for (pair_t p : {pair_t{1., 1., 1., true},
                   pair_t{1., -1., 1., true},
                   pair_t{-2., 2., 0., true},
                   pair_t{1., 4., 1., false},
                   pair_t{1., 4., 0., false},
                   pair_t{4., 1., 1., false},
                   pair_t{4., 1., 0., true},
                   pair_t{1024., -1., 0., true},
                   pair_t{1., 1. + 0x1p-52, 0., false},
                   pair_t{1., 1. - 0x1p-53, 1., false},
                   pair_t{1., 1. - 0x1p-53, 0., true}}) {
    double factor = p.target / p.equation;
    // Row 0: e (x0 + x1) + a s = 2 e. Row 1: t (x0 + x1) + y <= 9, which is
    // y - f a s <= 9 - 2 t.
    model_t m = make_model({{p.equation, p.equation, p.a, 0.}, {p.target, p.target, 0., 1.}},
                           {2. * p.equation, -inf},
                           {2. * p.equation, 9.},
                           dense_t(4, 0.),
                           dense_t(4, 3.));
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, p.taken ? 1 : 0);
    if (p.taken) {
      entries_t left = {{3, 1.}};
      if (p.a != 0.) { left[2] = -factor * p.a; }
      GPRS_CHECK(row_entries(f.st, 1) == left);
      GPRS_CHECK_EQ(f.st.row_upper[1], 9. - factor * (2. * p.equation));
      check_addition(f.ps, 0, 1, 0, factor);
    } else {
      GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{0, p.target}, {1, p.target}, {3, 1.}}));
      GPRS_CHECK_EQ(f.st.row_upper[1], 9.);
      GPRS_CHECK(f.ps.order.empty());
    }
  }
}

// The errors with which the constants of the two rows are stored reach the singleton columns:
// that of the equation a singleton column of the target with the coefficient c, times
// |f| / |c|, and that of the target the singleton column of the equation, with the coefficient
// a there, divided by |a|. A row whose error would grow must be as the model gave it, and its
// constants must hold the tolerance times that growth.
GPRS_TEST(near_rows_that_the_presolve_changed)
{
  // What an edit does to the version of a row, with the row as it was.
  auto change = [](fixture_t& f, int32_t row) { f.st.scale_row(row, 1., false); };
  struct pair_t {
    double equation, target;  // coefficients of the cores
    double a, c;              // of the singleton column of the equation (0: none), of the target
    bool grows_equation, grows_target;
  };
  for (pair_t p : {pair_t{1., -1., 1., 1., false, false},
                   pair_t{1., 1., 1., 0.25, true, false},  // |f| / |c| = 4
                   pair_t{1., 1., 0.25, 1., false, true},  // 1 / |a| = 4
                   pair_t{1., 1., 4., 4., false, false},
                   pair_t{4., 1., 0., 1., false, false},    // |f| / |c| = 1 / 4
                   pair_t{4., 1., 0., 0.125, true, false},  // |f| / |c| = 2
                   pair_t{1., 1., 0.25, 0.25, true, true}}) {
    double factor = p.target / p.equation;
    for (int32_t changed = 0; changed < 4; ++changed) {
      // Row 0: e (x0 + x1) + a s = 2 e. Row 1: t (x0 + x1) + c y <= 9, which is
      // c y - f a s <= 9 - 2 t.
      model_t m = make_model({{p.equation, p.equation, p.a, 0.}, {p.target, p.target, 0., p.c}},
                             {2. * p.equation, -inf},
                             {2. * p.equation, 9.},
                             dense_t(4, 0.),
                             dense_t(4, 3.));
      fixture_t f(m);
      if (changed & 1) { change(f, 0); }
      if (changed & 2) { change(f, 1); }
      bool taken = !(p.grows_equation && (changed & 1)) && !(p.grows_target && (changed & 2));
      rule_result_t result = f.run();
      GPRS_CHECK_EQ(result.findings, taken ? 1 : 0);
      // A pair that only the history of a row holds back: the same rows, as a model, are
      // subtracted.
      GPRS_CHECK_EQ(result.limited, !taken);
      if (taken) {
        entries_t left = {{3, p.c}};
        if (p.a != 0.) { left[2] = -factor * p.a; }
        GPRS_CHECK(row_entries(f.st, 1) == left);
        GPRS_CHECK_EQ(f.st.row_upper[1], 9. - 2. * p.target);
        check_addition(f.ps, 0, 1, 0, factor);
        GPRS_CHECK(check_same_feasible_set(m, f.reduced(), f.ps) > 0);
      } else {
        GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{0, p.target}, {1, p.target}, {3, p.c}}));
        GPRS_CHECK_EQ(f.st.row_upper[1], 9.);
        GPRS_CHECK(f.ps.order.empty());
      }
    }
  }
  // Rows as the model gave them: the limit of 1e8 is on the size of a constant times the
  // growth of its error.
  for (double value : {5e4, 2e5}) {
    bool taken = value == 5e4;
    {
      // The right-hand side and the fixed terms of the equation, |f| / |c| = 512.
      // Row 0: x0 + x1 + s + w = value + 3 with w fixed at value. Row 1:
      // x0 + x1 + y / 512 <= 5.
      model_t m = make_model({{1., 1., 1., 0., 1.}, {1., 1., 0., 1. / 512., 0.}},
                             {value + 3., -inf},
                             {value + 3., 5.},
                             {0., 0., 0., 0., value},
                             {3., 3., 3., 3., value});
      fixture_t f(m);
      rule_result_t result = f.run();
      GPRS_CHECK_EQ(result.findings, taken ? 1 : 0);
      // The size of a constant is read from the rows as they are: no reduction is left that
      // another presolve of them would make.
      GPRS_CHECK(!result.limited);
      if (taken) { GPRS_CHECK_EQ(f.st.row_upper[1], 2.); }
    }
    for (double a : {1. / 512., 0.}) {
      {
        // A side of the target, 1 / |a| = 512, and no growth without a singleton column of
        // the equation. Row 0: x0 + x1 + a s = 1. Row 1: x0 + x1 + y <= value.
        model_t m = make_model({{1., 1., a, 0.}, {1., 1., 0., 1.}},
                               {1., -inf},
                               {1., value},
                               dense_t(4, 0.),
                               dense_t(4, 3.));
        fixture_t f(m);
        GPRS_CHECK_EQ(f.run().findings, taken || a == 0. ? 1 : 0);
        if (taken) { GPRS_CHECK_EQ(f.st.row_upper[1], value - 1.); }
      }
      {
        // The fixed terms of the target. Row 1: x0 + x1 + y + w <= 5 with w fixed at value.
        model_t m = make_model({{1., 1., a, 0., 0.}, {1., 1., 0., 1., 1.}},
                               {1., -inf},
                               {1., 5.},
                               {0., 0., 0., 0., value},
                               {3., 3., 3., 3., value});
        fixture_t f(m);
        GPRS_CHECK_EQ(f.run().findings, taken || a == 0. ? 1 : 0);
        if (taken) { GPRS_CHECK_EQ(f.st.row_upper[1], 4.); }
      }
    }
  }
}

// A row is found parallel up to a singleton column when the column becomes one between two
// calls, and no longer when the column gets a second entry: the rows themselves do not change.
GPRS_TEST(near_columns_that_change_between_calls)
{
  // Row 0: x0 + x1 - y = 0. Row 1: x0 + x1 <= 1. Row 2: y + z <= 3.
  model_t m = make_model({{1., 1., -1., 0.}, {1., 1., 0., 0.}, {0., 0., 1., 1.}},
                         {0., -inf, -inf},
                         {0., 1., 3.},
                         dense_t(4, 0.),
                         dense_t(4, 2.));
  {
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK(f.st.row_frontier.empty(row_list::parallel_rows));
    uint32_t version = f.st.row_version[0];
    f.st.delete_row(2);
    GPRS_CHECK_EQ(f.st.row_version[0], version);
    GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{0}));
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, 1.}}));
    check_addition(f.ps, 0, 1, 0, 1.);
  }
  {
    // The other way round. Without row 2, y is a singleton column of row 0, which a call would
    // subtract from row 1. Before that call the entry of x0 in row 1 becomes an entry of y:
    // row 0 has the core x1, y and the singleton column x0, row 1 is x1 + y.
    fixture_t f(m);
    f.st.delete_row(2);
    f.st.move_entry(f.st.find_entry(1, 0), 2, 1.);
    GPRS_CHECK(f.st.row_frontier.list[row_list::singletons] == (list_t{0}));
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK_EQ(f.st.live_nnz, 5);
    GPRS_CHECK(f.ps.order.empty());
  }
}

// The sides decide what is done with a pair, so a row is listed when one of them changes: a
// row that becomes an equation between two calls is subtracted from the row it was left alone
// with, and a row whose side a merge has changed is compared again.
GPRS_TEST(near_sides_that_change_between_calls)
{
  {
    // Row 0: x0 + x1 <= 1. Row 1: x0 + x1 + y <= 3.
    model_t m = make_model(
      {{1., 1., 0.}, {1., 1., 1.}}, {-inf, -inf}, {1., 3.}, dense_t(3, 0.), dense_t(3, 2.));
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK(f.st.row_frontier.empty(row_list::parallel_rows));
    f.st.set_row_lower(0, 1.);
    GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{0}));
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.findings, 1);
    GPRS_CHECK(!result.limited);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, 1.}}));
    GPRS_CHECK_EQ(f.st.row_upper[1], 2.);
    check_addition(f.ps, 0, 1, 0, 1.);
  }
  {
    // Row 0: x0 + x1 >= 1. Row 1: x0 + x1 + y <= 3. Row 2: x0 + 2 x1 <= 1, which a change of
    // its coefficient makes x0 + x1 <= 1. Only row 2 is listed then: it is compared with row 1
    // first, which is left alone, and then merged into row 0, which becomes an equation. Row 0
    // is listed for its side, and the next call subtracts it from row 1.
    model_t m = make_model({{1., 1., 0.}, {1., 1., 1.}, {1., 2., 0.}},
                           {1., -inf, -inf},
                           {inf, 3., 1.},
                           dense_t(3, 0.),
                           dense_t(3, 2.));
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 0);
    f.st.set_coefficient(f.st.find_entry(2, 1), 1.);
    GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{2}));
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1}));
    GPRS_CHECK_EQ(f.st.row_lower[0], 1.);
    GPRS_CHECK_EQ(f.st.row_upper[0], 1.);
    GPRS_CHECK(f.ps.row_additions.empty());
    GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] == (list_t{0}));
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, 1.}}));
    GPRS_CHECK_EQ(f.st.row_upper[1], 2.);
    check_addition(f.ps, 0, 1, 0, 1.);
  }
}

// A row is compared with the rows of its bucket until it has been left alone with
// opt.parallel_bucket_limit of them. A scan that ends so with rows of the bucket left reports
// limited: a pair of the row was not compared, and it stays so only while the rows between the
// two stay where they are.
GPRS_TEST(near_scan_that_ends_on_the_bucket_limit)
{
  // Rows 0 and 3: x0 + x1 <= 5 and <= 4. Rows 1 and 2: x0 + x1 + y <= 7, x0 + x1 + z <= 8,
  // which no row of the bucket is subtracted from.
  model_t m = make_model({{1., 1., 0., 0.}, {1., 1., 1., 0.}, {1., 1., 0., 1.}, {1., 1., 0., 0.}},
                         dense_t(4, -inf),
                         {5., 7., 8., 4.},
                         dense_t(4, 0.),
                         dense_t(4, 9.));
  {
    fixture_t f(m);
    f.opt.parallel_bucket_limit = 2;
    rule_result_t result        = f.run();
    GPRS_CHECK_EQ(result.findings, 0);
    GPRS_CHECK(result.limited);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1, 2, 3}));
    // With row 1 gone, the two rows are within reach, and nothing lists them.
    f.st.delete_row(1);
    GPRS_CHECK(f.st.row_frontier.empty(row_list::parallel_rows));
  }
  {
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.findings, 1);
    GPRS_CHECK(!result.limited);
    GPRS_CHECK(alive_rows(f.st) == (list_t{0, 1, 2}));
    GPRS_CHECK_EQ(f.st.row_upper[0], 4.);
  }
  {
    // The scan reaches the limit at the last row of the bucket: every pair was compared.
    fixture_t f(m);
    f.opt.parallel_bucket_limit = 2;
    f.st.delete_row(3);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.findings, 0);
    GPRS_CHECK(!result.limited);
  }
}

// A stop on the budget gives the rows that were not compared back; the calls that follow end
// in the state of one call without a limit.
GPRS_TEST(near_budget)
{
  // Three groups of three rows, each an equation with a singleton column, the inequality on
  // its core and a row of its own.
  std::vector<dense_t> rows;
  dense_t row_lower, row_upper;
  for (int32_t g = 0; g < 3; ++g) {
    dense_t equation(9, 0.), packing(9, 0.), other(9, 0.);
    equation[3 * g] = packing[3 * g] = 1.;
    equation[3 * g + 1] = packing[3 * g + 1] = 2. + g;
    equation[3 * g + 2]                      = -1.;
    other[3 * g]                             = 1.;
    other[3 * g + 1]                         = 5.;
    rows.insert(rows.end(), {g == 1 ? packing : equation, g == 1 ? equation : packing, other});
    row_lower.insert(row_lower.end(), {g == 1 ? -inf : 0., g == 1 ? 0. : -inf, -inf});
    row_upper.insert(row_upper.end(), {g == 1 ? 3. : 0., g == 1 ? 0. : 3., 9.});
  }
  model_t m = make_model(rows, row_lower, row_upper, dense_t(9, 0.), dense_t(9, 2.));
  fixture_t whole(m);
  GPRS_CHECK_EQ(whole.run().findings, 3);
  GPRS_CHECK_EQ(whole.run().findings, 0);

  fixture_t f(m);
  rule_result_t first = f.run(1);
  GPRS_CHECK(first.stopped);
  GPRS_CHECK_EQ(first.findings, 0);
  GPRS_CHECK(f.st.row_frontier.list[row_list::parallel_rows] ==
             (list_t{0, 1, 2, 3, 4, 5, 6, 7, 8}));
  int64_t findings = 0;
  int32_t calls    = 1;
  while (!f.st.row_frontier.empty(row_list::parallel_rows) && calls < 200) {
    rule_result_t result = f.run(1);
    GPRS_CHECK_EQ(result.outcome, ok);
    findings += result.findings;
    ++calls;
  }
  GPRS_CHECK(calls > 9 && calls < 200);
  GPRS_CHECK_EQ(findings, 3);
  GPRS_CHECK(f.st.row_frontier.empty(row_list::singletons));
  for (int32_t row = 0; row < 9; ++row) {
    GPRS_CHECK(row_entries(f.st, row) == row_entries(whole.st, row));
    GPRS_CHECK_EQ(f.st.row_lower[row], whole.st.row_lower[row]);
    GPRS_CHECK_EQ(f.st.row_upper[row], whole.st.row_upper[row]);
  }
  GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, 1.}}));
  GPRS_CHECK(row_entries(f.st, 3) == (entries_t{{5, 1.}}));
  GPRS_CHECK(postsolve_hash(f.ps) == postsolve_hash(whole.ps));
}

namespace {

// Integer models around a planted point: rows that are multiples of a few patterns over the
// shared columns, each with none, one or two columns of its own behind them; equations,
// inequalities and ranged rows that the planted point meets; some columns fixed.
model_t near_random_model(rng_t& rng)
{
  static const double factor[] = {1., 1., -1., 2., -2., 3., 0.5, -0.5};
  int32_t shared = rng.in(2, 4), nrows = rng.in(2, 7), npatterns = rng.in(1, 2);
  std::vector<dense_t> pattern(npatterns, dense_t(shared, 0.));
  for (dense_t& p : pattern) {
    for (int32_t j = 0; j < shared; ++j) {
      if (rng.chance(0.75)) { p[j] = 2. * rng.coefficient(); }
    }
  }
  std::vector<int32_t> own(nrows, 0);
  int32_t ncols = shared;
  for (int32_t& n : own) {
    n =
      ncols < shared + 4 && rng.chance(0.55) ? (ncols < shared + 3 && rng.chance(0.2) ? 2 : 1) : 0;
    ncols += n;
  }
  dense_t col_lower(ncols), col_upper(ncols), planted(ncols);
  for (int32_t j = 0; j < ncols; ++j) {
    col_lower[j] = rng.in(-1, 1);
    col_upper[j] = col_lower[j] + rng.in(1, j < shared ? 3 : 2);
    planted[j]   = rng.in(col_lower[j], col_upper[j]);
    if (rng.chance(0.1)) { col_lower[j] = col_upper[j] = planted[j]; }
  }
  std::vector<dense_t> rows(nrows, dense_t(ncols, 0.));
  dense_t row_lower(nrows), row_upper(nrows);
  int32_t next = shared;
  for (int32_t i = 0; i < nrows; ++i) {
    const dense_t& p = pattern[rng.in(0, npatterns - 1)];
    double f         = factor[rng.in(0, 7)];
    double activity  = 0.;
    for (int32_t j = 0; j < shared; ++j) {
      rows[i][j] = f * p[j];
    }
    for (int32_t k = 0; k < own[i]; ++k) {
      rows[i][next++] = rng.coefficient();
    }
    for (int32_t j = 0; j < ncols; ++j) {
      activity += rows[i][j] * planted[j];
    }
    int32_t kind = rng.chance(0.45) ? 0 : rng.in(1, 3);
    row_lower[i] = kind == 1 ? -inf : activity - (kind == 0 ? 0 : rng.in(0, 3));
    row_upper[i] = kind == 2 ? inf : activity + (kind == 0 ? 0 : rng.in(0, 3));
  }
  return make_model(rows, row_lower, row_upper, col_lower, col_upper);
}

}  // namespace

// Random models with rows parallel up to singleton columns: the calls end, every state is
// consistent, and the model after them has the feasible points of the model before, whatever
// the budget of a call.
GPRS_TEST(near_random_models)
{
  int64_t additions = 0, merges = 0, nonempty = 0, with_fixed = 0, stops = 0, models = 0;
  int64_t two_entries = 0, ranged = 0;
  for (uint64_t seed = 1; seed <= 3000; ++seed) {
    rng_t rng(seed);
    model_t m = near_random_model(rng);
    if (feasible_points(m) == 0) { continue; }
    ++models;
    fixture_t f(m);
    int32_t failures = gprs::test::failure_count();
    int64_t limit    = seed % 3 == 0 ? rng.in(1, 60) : no_limit;
    int32_t calls    = 0;
    while (!f.st.row_frontier.empty(row_list::parallel_rows) && calls < 400) {
      rule_result_t result = f.run(limit);
      GPRS_CHECK_EQ(result.outcome, ok);
      stops += result.stopped;
      ++calls;
    }
    GPRS_CHECK(calls < 400);
    GPRS_CHECK(f.st.row_frontier.empty(row_list::singletons));
    additions += f.ps.row_additions.size();
    merges += f.ps.parallel_rows.size();
    with_fixed += !f.ps.row_additions.empty() && f.st.fixed_nnz > 0;
    for (const row_addition_record_t& record : f.ps.row_additions) {
      two_entries += f.st.row_alive[record.row] && f.st.row_free_size[record.row] == 2;
      ranged += m.row_lower[record.row] > -inf && m.row_upper[record.row] < inf &&
                m.row_lower[record.row] < m.row_upper[record.row];
    }
    model_t reduced = f.reduced();
    GPRS_CHECK(reduced.nnz() + (int64_t)f.ps.row_additions.size() <= m.nnz());
    nonempty += check_same_feasible_set(m, reduced, f.ps) > 0;
    if (gprs::test::failure_count() != failures) {
      std::printf("    seed %llu\n", (unsigned long long)seed);
      return;
    }
  }
  // The test is not vacuous.
  GPRS_CHECK(models > 2500);
  GPRS_CHECK(additions > 1500);
  GPRS_CHECK(merges > 500);
  GPRS_CHECK(nonempty == models);
  GPRS_CHECK(with_fixed > 200);
  GPRS_CHECK(stops > 2000);
  GPRS_CHECK(two_entries > 200);
  GPRS_CHECK(ranged > 100);
}

// ---- Which of two parallel rows is kept, and what its sides can take ----

namespace {

model_t continuous(model_t m)
{
  m.col_type.assign(m.ncols, 0);
  return m;
}

options_t only(std::initializer_list<int32_t> rules)
{
  options_t options;
  options.enabled.fill(false);
  for (int32_t rule : rules) {
    options.enabled[rule] = true;
  }
  return options;
}

// The contract of a merge and of what follows it: a point that satisfies the reduced model
// with violation 0 postsolves to a point that the model accepts within the feasibility
// tolerance (check_point). Tested at the points with one reduced row at one of its sides: for
// every row, finite side and continuous column of the row, with the other columns at their
// lower bounds and then at their upper bounds (0 for an infinite one), the column takes what
// the side leaves it, moved by single doubles until the reduced point violates nothing.
// Returns the number of points tested.
int32_t check_side_points(const model_t& m, const result_t& result)
{
  const model_t& q = result.reduced;
  tolerances_t tol;
  int32_t points = 0, failures = 0;
  dense_t x(q.ncols), back;
  auto clean = [&]() {
    point_check_t c = check_point(q, x);
    return c.row_violation == 0. && c.bound_violation == 0. && c.integrality_violation == 0.;
  };
  for (bool at_upper : {false, true}) {
    for (int32_t i = 0; i < q.nrows; ++i) {
      for (double side : {q.row_lower[i], q.row_upper[i]}) {
        if (!std::isfinite(side)) { continue; }
        for (int64_t e = q.row_start[i]; e < q.row_start[i + 1]; ++e) {
          int32_t col = q.col_index[e];
          if (q.col_type[col] != 0) { continue; }
          for (int32_t j = 0; j < q.ncols; ++j) {
            double bound = at_upper ? q.col_upper[j] : q.col_lower[j];
            x[j]         = std::isfinite(bound) ? bound : 0.;
          }
          dd_t rest;
          for (int64_t k = q.row_start[i]; k < q.row_start[i + 1]; ++k) {
            if (k != e) { dd_add_product(rest, q.value[k], x[q.col_index[k]]); }
          }
          double value = dd_residual(side, rest) / q.value[e];
          bool found   = false;
          for (double towards : {inf, -inf}) {
            x[col] = value;
            for (int32_t step = 0; step < 4 && !(found = clean()); ++step) {
              x[col] = std::nextafter(x[col], towards);
            }
            if (found) { break; }
          }
          if (!found) { continue; }
          ++points;
          result.postsolve.postsolve(x, back);
          point_check_t c = check_point(m, back);
          if (!point_feasible(c, tol) && ++failures <= 3) {
            GPRS_CHECK_EQ(c.row_violation, 0.);
            GPRS_CHECK_EQ(c.bound_violation, 0.);
          }
        }
      }
    }
  }
  GPRS_CHECK_EQ(failures, 0);
  return points;
}

}  // namespace

// A row and the same row times 2^18, in either order, and a continuous column without cost
// that only the two hold, with a range of 9e7. Once one row has gone into the other the column
// is a singleton column, and the rule of those columns takes it out of the kept row when its
// term fits the sides of that row. In the smaller row it fits, and the new side 0.1 - 9e7, one
// double, is off by 6e-9, which the larger row reads as 1.6e-3. The larger row is kept, where
// the term does not fit.
GPRS_TEST(parallel_larger_row_is_kept_before_a_singleton_column_leaves)
{
  const double factor = 262144.;
  dense_t small = {1., 1., 1.}, large = {factor, factor, factor};
  for (bool larger_first : {false, true}) {
    double first = larger_first ? factor : 1., second = larger_first ? 1. : factor;
    model_t m       = continuous(make_model(
      {larger_first ? large : small, larger_first ? small : large, {0., 1., 1.}, {0., 1., -1.}},
      {first * 0.1, second * 0.1, -inf, -2e8},
      {first * 5.3, second * 5.3, 100., inf},
      {0., -1e8, 0.},
      {9e7, 10., 1.}));
    m.cost          = {0., 1., 1.};
    result_t result = presolve(m, only({rule::parallel_rows, rule::singleton_cols}));
    GPRS_REQUIRE(result.status == status_t::reduced);
    int32_t kept = larger_first ? 0 : 1;
    check_record(result.postsolve, 0, 1 - kept, kept, 1. / factor);
    GPRS_CHECK_EQ(result.postsolve.parallel_rows.size(), 1u);
    GPRS_CHECK_EQ(result.sub_findings[sub_rule::zero_cost_singleton], 0);
    GPRS_CHECK(check_side_points(m, result) >= 4);
  }
}

// Sides that cross are measured in the units of the row that goes, which is the one that the
// new side leaves off: x0 + x1 <= 5 and 2^20 (x0 + x1) >= 2^20 (5 + crossing). Within the
// tolerance the larger row becomes an equation at its own side; in its own units the two
// sides are half a unit apart.
GPRS_TEST(parallel_crossing_in_the_units_of_the_removed_row)
{
  const double factor = 1048576.;
  dense_t small = {1., 1.}, large = {factor, factor};
  for (bool larger_first : {false, true}) {
    for (double crossing : {5e-7, 2e-6}) {
      bool within = crossing < 1e-6;
      double from = factor * (5. + crossing);
      model_t m =
        continuous(make_model({larger_first ? large : small, larger_first ? small : large},
                              {larger_first ? from : -inf, larger_first ? -inf : from},
                              {larger_first ? inf : 5., larger_first ? 5. : inf},
                              {0., 0.},
                              {9., 9.}));
      int32_t kept = larger_first ? 0 : 1;
      fixture_t f(m);
      rule_result_t result = f.run();
      GPRS_CHECK_EQ(result.outcome, within ? ok : infeasible);
      if (!within) { continue; }
      GPRS_CHECK(!result.limited);
      GPRS_CHECK(alive_rows(f.st) == (list_t{kept}));
      GPRS_CHECK_EQ(f.st.row_lower[kept], from);
      GPRS_CHECK_EQ(f.st.row_upper[kept], from);
      result_t whole = presolve(m, only({rule::parallel_rows}));
      GPRS_CHECK(check_side_points(m, whole) >= 2);
    }
  }
}

// A crossing that the rounding of the image does not decide. Row 0: x0 + x1 + x2 <= 5e7 + 2
// with x2 fixed at 5e7; row 1: x0 + x1 >= 2 + crossing, which as a side of row 0 is rounded at
// 5e7. Clearly within the tolerance the sides meet, clearly beyond it the rows are infeasible,
// and between the two the pair is left alone and the call reports limited.
GPRS_TEST(parallel_crossing_next_to_the_tolerance)
{
  for (double crossing : {8e-7, 1e-6, 1.2e-6}) {
    model_t m = continuous(make_model({{1., 1., 1.}, {1., 1., 0.}},
                                      {-inf, 2. + crossing},
                                      {5e7 + 2., inf},
                                      {0., 0., 5e7},
                                      {4., 4., 5e7}));
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.outcome, crossing > 1e-6 ? infeasible : ok);
    if (result.outcome != ok) { continue; }
    GPRS_CHECK_EQ(result.findings, crossing < 1e-6 ? 1 : 0);
    GPRS_CHECK_EQ(result.limited, crossing == 1e-6);
    GPRS_CHECK_EQ(f.st.live_rows, crossing < 1e-6 ? 1 : 2);
    if (crossing < 1e-6) { GPRS_CHECK_EQ(f.st.row_lower[0], 5e7 + 2.); }
  }
}

// A ranged row of half the size and the other sign: its sides change places and double, in
// either order of the two rows.
GPRS_TEST(parallel_ranged_row_into_a_larger_one)
{
  for (bool larger_first : {false, true}) {
    dense_t small = {-1., 2.}, large = {2., -4.};
    model_t m    = make_model({larger_first ? large : small, larger_first ? small : large},
                              {larger_first ? -inf : -3., larger_first ? -3. : -inf},
                              {larger_first ? 7. : 4., larger_first ? 4. : 7.},
                              {0., 0.},
                              {9., 9.});
    int32_t kept = larger_first ? 0 : 1;
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{kept}));
    GPRS_CHECK_EQ(f.st.row_lower[kept], -8.);
    GPRS_CHECK_EQ(f.st.row_upper[kept], 6.);
    check_record(f.ps, 0, 1 - kept, kept, -0.5);
    model_t reduced = f.reduced();
    GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps) > 0);
  }
}

// Rows that are parallel within tol.zero and not exactly: the row that goes differs from its
// multiple of the kept row by the differences of the coefficients times the values of the
// columns. Over wide bounds that is more than the tolerance, and the pair is left alone.
GPRS_TEST(parallel_rows_that_are_not_exact_multiples)
{
  for (double range : {1e7, 50.}) {
    bool merged = range == 50.;
    model_t m   = continuous(make_model(
      {{1., 1.}, {1., 1. + 9e-10}}, {-inf, -inf}, {0., 0.}, {-range, -range}, {range, range}));
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, merged ? 1 : 0);
    GPRS_CHECK_EQ(result.limited, !merged);
    GPRS_CHECK_EQ(f.st.live_rows, merged ? 1 : 2);
    result_t whole = presolve(m, only({rule::parallel_rows}));
    GPRS_CHECK_EQ(whole.fixed_point, merged);
    GPRS_CHECK(check_side_points(m, whole) >= 2);
    // Without the limit the rows are merged whatever the bounds are.
    fixture_t loose(m);
    loose.opt.parallel_drift_share = inf;
    GPRS_CHECK_EQ(loose.run().findings, 1);
  }
  // Exact multiples are merged over any bounds, columns without a bound included; rows that
  // differ in the last bit of one coefficient are not, there.
  for (double second : {6., std::nextafter(6., 7.)}) {
    model_t m = continuous(
      make_model({{1., 2.}, {3., second}}, {-inf, -inf}, {4., 9.}, {-inf, 0.}, {inf, inf}));
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.findings, second == 6. ? 1 : 0);
    GPRS_CHECK_EQ(result.limited, second != 6.);
  }
}

// The same for an equation that is subtracted. Row 0: x0 + (1 + 9e-10) x1 + t <= 5 with the
// singleton column t; row 1: x0 + x1 = 3; rows 2 and 3 hold x0 and x1. With row 0 left as
// t <= 2, the row of the model is off by 9e-10 x1 where the reduced model holds: 9e-3 at
// x1 = 1e7, which leaves the pair alone, and 4.5e-8 at x1 = 50.
GPRS_TEST(near_core_that_is_not_an_exact_multiple)
{
  for (double range : {1e7, 50.}) {
    bool subtracted = range == 50.;
    model_t m =
      continuous(make_model({{1., 1. + 9e-10, 1.}, {1., 1., 0.}, {1., -1., 0.}, {1., 2., 0.}},
                            {-inf, 3., -3. * range, -3. * range},
                            {5., 3., 3. * range, 3. * range},
                            {-range, -range, 0.},
                            {range, range, 10.}));
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, subtracted ? 1 : 0);
    GPRS_CHECK_EQ(result.limited, !subtracted);
    GPRS_CHECK_EQ(f.ps.row_additions.size(), subtracted ? 1u : 0u);
    // Without the limit the equation is subtracted whatever the bounds are.
    fixture_t loose(m);
    loose.opt.parallel_drift_share = inf;
    GPRS_CHECK_EQ(loose.run().findings, 1);
    if (!subtracted) { continue; }
    GPRS_CHECK_EQ(f.st.row_upper[0], 2.);
    model_t reduced = f.reduced();
    dense_t point   = {3. - range, range, 2.}, back;
    point_check_t c = check_point(reduced, point);
    GPRS_CHECK(c.row_violation == 0. && c.bound_violation == 0.);
    f.ps.postsolve(point, back);
    GPRS_CHECK(feasible(m, back));
  }
}

// The image of a side is exact wherever it is a double, and a rounded one is used only at a
// size at which a double holds the tolerance of the row that goes.
GPRS_TEST(parallel_image_of_a_side)
{
  {
    // 3 (x0 + x1) = 3e10 + 3 and 7 (x0 + x1) = 7e10 + 7 say the same: no side moves.
    model_t m = make_model(
      {{3., 3.}, {7., 7.}}, {3e10 + 3., 7e10 + 7.}, {3e10 + 3., 7e10 + 7.}, {0., 0.}, {1e11, 1e11});
    fixture_t f(m);
    int64_t side_changes = f.st.side_changes;
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, 1);
    GPRS_CHECK(alive_rows(f.st) == (list_t{1}));
    GPRS_CHECK_EQ(f.st.side_changes, side_changes);
    check_record(f.ps, 0, 0, 1, 3. / 7.);
  }
  {
    // x0 + x1 <= 3e8 + 2^-24 in units of 3 has no double, and at 3e8 a double does not hold
    // the tolerance: left alone, for good. With a side that has an image the rows are merged.
    model_t m =
      make_model({{1., 1.}, {3., 3.}}, {-inf, -inf}, {3e8 + 0x1p-24, 1e12}, {0., 0.}, {1e11, 1e11});
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.findings, 0);
    GPRS_CHECK(!result.limited);
    GPRS_CHECK_EQ(f.st.live_rows, 2);
    m.row_upper[0] = 3e8 + 0.5;
    fixture_t exact(m);
    GPRS_CHECK_EQ(exact.run().findings, 1);
    GPRS_CHECK(alive_rows(exact.st) == (list_t{1}));
    GPRS_CHECK_EQ(exact.st.row_upper[1], 9e8 + 1.5);
  }
  {
    // x0 + x1 <= 3e8 is 9e8 in units of 3, but the kept row holds the term 0.1 of a fixed
    // column, and 9e8 + 0.1 has no double. The term can leave the row without an event of the
    // two rows: the call reports limited.
    model_t m = continuous(make_model(
      {{1., 1., 0.}, {3., 3., 1.}}, {-inf, -inf}, {3e8, 1e12}, {0., 0., 0.1}, {1e11, 1e11, 0.1}));
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.findings, 0);
    GPRS_CHECK(result.limited);
    GPRS_CHECK_EQ(f.st.live_rows, 2);
  }
  {
    // An image beyond the doubles: left alone.
    model_t m =
      make_model({{1., 1.}, {1e200, 1e200}}, {-inf, -inf}, {1e150, 1e300}, {0., 0.}, {1., 1.});
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, 0);
    GPRS_CHECK(!result.limited);
    GPRS_CHECK_EQ(f.st.row_upper[1], 1e300);
  }
}

// The generator knob for parallel rows of very different scale (oracle.hpp): the base model is
// that of the same seed without it, a row factor is an integer from 1 to 2^20, powers of two
// and others, and the oracle accepts the presolve of every case, with the larger row kept and
// with the smaller index kept, which is as right while every later step is exact.
GPRS_TEST(parallel_scale_generated_models)
{
  oracle::generator_options_t plain;
  plain.p_parallel_row               = 0.5;
  plain.p_scaled                     = 0.;
  plain.p_equation                   = 0.3;
  plain.max_rows                     = 7;
  oracle::generator_options_t scaled = plain;
  scaled.p_parallel_scale            = 0.8;
  int64_t powers = 0, others = 0, kept_larger_index = 0, far_ratios = 0;
  for (uint64_t seed = 1; seed <= 1500; ++seed) {
    oracle::case_t c    = oracle::generate(seed, scaled);
    oracle::case_t same = oracle::generate(seed, plain);
    int32_t failures    = gprs::test::failure_count();
    GPRS_CHECK(c.base.value == same.base.value && c.base.row_lower == same.base.row_lower &&
               c.base.row_upper == same.base.row_upper && c.base.col_index == same.base.col_index);
    for (double factor : c.row_scale) {
      int exponent  = 0;
      bool power    = std::frexp(factor, &exponent) == 0.5;
      bool in_range = factor >= 1. && factor <= 1048576. && factor == std::floor(factor);
      powers += power && factor > 1.;
      others += !power;
      GPRS_CHECK(in_range);
    }
    // All rules with the larger row kept, all rules with the smaller index kept, and the
    // parallel rows alone, which then meet the rows that the other rules would take first.
    for (int32_t set = 0; set < 3; ++set) {
      options_t options            = set == 2 ? only({rule::parallel_rows}) : options_t();
      options.parallel_keep_larger = set != 1;
      result_t result              = presolve(c.model, options);
      oracle::verdict_t verdict    = oracle::check_result(c, options, result);
      if (!verdict.ok && verdict.message.find("outside the oracle class") == std::string::npos) {
        gprs::test::fail(__FILE__, __LINE__, verdict.message);
      }
      if (set == 1) { continue; }
      for (const parallel_row_record_t& record : result.postsolve.parallel_rows) {
        GPRS_CHECK(std::fabs(record.ratio) <= 1.);
        kept_larger_index += record.kept_row > record.removed_row;
        far_ratios += std::fabs(record.ratio) <= 1. / 1024.;
      }
    }
    if (gprs::test::failure_count() != failures) {
      std::printf("    seed %llu\n", (unsigned long long)seed);
      return;
    }
  }
  // The test is not vacuous.
  GPRS_CHECK(powers > 300);
  GPRS_CHECK(others > 300);
  GPRS_CHECK(kept_larger_index > 100);
  GPRS_CHECK(far_ratios > 100);
}
