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
                   const dense_t& col_upper,
                   const dense_t& cost,
                   const std::vector<uint8_t>& col_type)
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
  m.cost      = cost;
  m.col_type  = col_type;
  return m;
}

// The state after the start step of presolve(): columns with lower == upper are fixed.
struct fixture_t {
  model_t model;
  state_t st;
  postsolve_t ps;
  options_t opt;
  doubleton_t rule;

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

  // Until the rule has nothing left on its list.
  rule_result_t run_to_end()
  {
    rule_result_t total;
    for (int32_t call = 0; call < 50; ++call) {
      rule_result_t result = run();
      total.findings += result.findings;
      total.outcome = result.outcome;
      if (result.outcome != ok || st.row_frontier.empty(row_list::doubleton)) { return total; }
    }
    GPRS_CHECK(!"the doubleton list does not run empty");
    return total;
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

bool feasible(const model_t& m, const dense_t& x, double& objective)
{
  point_check_t check = check_point(m, x);
  objective           = check.objective;
  return point_feasible(check, tolerances_t());
}

bool same_value(double a, double b) { return std::fabs(a - b) <= 1e-9 * (1. + std::fabs(a)); }

struct box_counts_t {
  int64_t original = 0;
  int64_t reduced  = 0;
};

// The feasible set is unchanged, on the integer points of both boxes: a feasible point of the
// original model crushes to a feasible point of the reduced model with the same objective and
// comes back from the postsolve as it was; a feasible point of the reduced model postsolves to
// a feasible point of the original model with the same objective.
box_counts_t check_same_feasible_set(const model_t& original,
                                     const model_t& reduced,
                                     const postsolve_t& ps)
{
  box_counts_t counts;
  dense_t y, back;
  int32_t failures = 0;
  for_each_point(original, [&](const dense_t& x) {
    double objective, reduced_objective;
    if (!feasible(original, x, objective)) { return; }
    ++counts.original;
    ps.crush(x, y);
    bool crushed_ok = feasible(reduced, y, reduced_objective);
    ps.postsolve(y, back);
    bool same_point = true;
    for (int32_t j = 0; j < original.ncols; ++j) {
      same_point = same_point && same_value(x[j], back[j]);
    }
    if (failures < 3 && !(crushed_ok && same_value(objective, reduced_objective) && same_point)) {
      ++failures;
      GPRS_CHECK(crushed_ok);
      GPRS_CHECK(same_value(objective, reduced_objective));
      GPRS_CHECK(same_point);
    }
  });
  for_each_point(reduced, [&](const dense_t& point) {
    double objective, original_objective;
    if (!feasible(reduced, point, objective)) { return; }
    ++counts.reduced;
    ps.postsolve(point, back);
    bool postsolved_ok = feasible(original, back, original_objective);
    if (failures < 3 && !(postsolved_ok && same_value(objective, original_objective))) {
      ++failures;
      GPRS_CHECK(postsolved_ok);
      GPRS_CHECK(same_value(objective, original_objective));
    }
  });
  return counts;
}

int64_t feasible_points(const model_t& m)
{
  int64_t count = 0;
  for_each_point(m, [&](const dense_t& x) {
    double objective;
    count += feasible(m, x, objective);
  });
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

// Small models with integer data around a planted point, many of their rows doubleton
// equations (chains, cycles and repeated pairs occur), some with a fixed column.
model_t random_model(rng_t& rng, bool all_integer)
{
  int32_t ncols = rng.in(2, 6), nrows = rng.in(1, 6);
  dense_t col_lower(ncols), col_upper(ncols), cost(ncols), planted(ncols);
  std::vector<uint8_t> col_type(ncols);
  for (int32_t j = 0; j < ncols; ++j) {
    col_lower[j] = rng.in(-2, 1);
    col_upper[j] = col_lower[j] + rng.in(1, 3);
    col_type[j]  = all_integer || rng.chance(0.6);
    planted[j]   = rng.in(col_lower[j], col_upper[j]);
    cost[j]      = rng.chance(0.3) ? 0. : rng.in(-3, 3);
    if (rng.chance(0.08)) { col_lower[j] = col_upper[j] = planted[j]; }
  }
  std::vector<dense_t> rows(nrows, dense_t(ncols, 0.));
  dense_t row_lower(nrows), row_upper(nrows);
  for (int32_t i = 0; i < nrows; ++i) {
    bool doubleton = rng.chance(0.5);
    if (doubleton) {
      int32_t a = rng.in(0, ncols - 1), b = rng.in(0, ncols - 2);
      if (b >= a) { ++b; }
      rows[i][a] = rng.coefficient();
      rows[i][b] = rng.chance(0.5) ? -rows[i][a] : rng.coefficient();
      if (rng.chance(0.3)) { rows[i][rng.in(0, ncols - 1)] = rng.coefficient(); }
    } else {
      for (int32_t j = 0; j < ncols; ++j) {
        if (rng.chance(0.5)) { rows[i][j] = rng.coefficient(); }
      }
      rows[i][rng.in(0, ncols - 1)] = rng.coefficient();
    }
    double activity = 0.;
    for (int32_t j = 0; j < ncols; ++j) {
      activity += rows[i][j] * planted[j];
    }
    if (rng.chance(0.1)) { activity += rng.in(-2, 2); }
    int32_t kind = doubleton ? 0 : rng.in(0, 3);
    row_lower[i] = kind == 1 ? -inf : activity - (kind == 0 ? 0 : rng.in(0, 2));
    row_upper[i] = kind == 2 ? inf : activity + (kind == 0 ? 0 : rng.in(0, 2));
  }
  model_t m          = make_model(rows, row_lower, row_upper, col_lower, col_upper, cost, col_type);
  m.objective_offset = rng.in(-3, 3);
  return m;
}

}  // namespace

// x1 = -x0 / 2 + 2 from row 0. Row 1 has both columns (set_coefficient), row 2 the substituted
// one only (move_entry), row 3 the root only (untouched).
GPRS_TEST(doubleton_substitution_by_hand)
{
  model_t m          = make_model({{1., 2., 0.}, {3., 1., 1.}, {0., 1., -1.}, {1., 0., 1.}},
                                  {4., -inf, -5., -inf},
                                  {4., 10., inf, 7.},
                                  {0., 0., -3.},
                                  {10., 3., 3.},
                                  {1., 4., 0.},
                                  {0, 0, 0});
  m.objective_offset = 1.;
  fixture_t f(m);
  int64_t nnz          = f.st.live_nnz;
  rule_result_t result = f.run();
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 1);
  GPRS_CHECK(!result.stopped);

  GPRS_CHECK(!f.st.row_alive[0]);
  GPRS_CHECK_EQ(f.st.live_rows, 3);
  GPRS_CHECK(f.st.col_status[1] == col_status_t::removed);
  GPRS_CHECK(f.st.is_live(0) && f.st.is_live(2));
  GPRS_CHECK_EQ(f.st.live_nnz, nnz - 3);

  GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{0, 2.5}, {2, 1.}}));
  GPRS_CHECK_EQ(f.st.row_lower[1], -inf);
  GPRS_CHECK_EQ(f.st.row_upper[1], 8.);
  GPRS_CHECK(row_entries(f.st, 2) == (entries_t{{0, -0.5}, {2, -1.}}));
  GPRS_CHECK_EQ(f.st.row_lower[2], -7.);
  GPRS_CHECK_EQ(f.st.row_upper[2], inf);
  GPRS_CHECK(row_entries(f.st, 3) == (entries_t{{0, 1.}, {2, 1.}}));
  GPRS_CHECK_EQ(f.st.row_upper[3], 7.);

  // 0 <= x1 <= 3 gives -2 <= x0 <= 4: with the negative scale the bounds change sides.
  GPRS_CHECK_EQ(f.st.col_lower[0], 0.);
  GPRS_CHECK_EQ(f.st.col_upper[0], 4.);
  GPRS_CHECK_EQ(f.st.cost[0], -1.);
  GPRS_CHECK_EQ(f.st.cost[1], 0.);
  GPRS_CHECK_EQ(dd_value(f.st.objective_offset), 9.);

  GPRS_REQUIRE(f.ps.aliases.size() == 1 && f.ps.order.size() == 1);
  GPRS_CHECK_EQ(f.ps.aliases[0].col, 1);
  GPRS_CHECK_EQ(f.ps.aliases[0].rep, 0);
  GPRS_CHECK_EQ(f.ps.aliases[0].row, 0);
  GPRS_CHECK_EQ(f.ps.aliases[0].scale, -0.5);
  GPRS_CHECK_EQ(f.ps.aliases[0].offset, 2.);

  // The changed rows are back on every rule list, the untouched row is not.
  for (int32_t list = 0; list < row_list::compaction; ++list) {
    GPRS_CHECK(on_list(f.st, list, 1) && on_list(f.st, list, 2));
  }
  GPRS_CHECK(!on_list(f.st, row_list::doubleton, 3));

  model_t reduced     = f.reduced();
  box_counts_t counts = check_same_feasible_set(m, reduced, f.ps);
  GPRS_CHECK(counts.original > 0 && counts.reduced > 0);
}

// The postsolve reproduces the eliminated column on a point of the reduced model.
GPRS_TEST(doubleton_postsolve_reproduces_column)
{
  // 2 x0 - 4 x1 = 6, x0 - x1 + x2 >= 0: with as many entries as x1 and the smaller index, x0 is
  // the root, and x1 = x0 / 2 - 1.5.
  model_t m = make_model({{2., -4., 0.}, {1., -1., 1.}},
                         {6., 0.},
                         {6., inf},
                         {-10., -10., -10.},
                         {10., 10., 10.},
                         {0., 1., 0.},
                         {0, 0, 0});
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 1);
  model_t reduced = f.reduced();
  GPRS_REQUIRE(reduced.ncols == 2 && f.ps.col_origin == (list_t{0, 2}));
  dense_t x;
  f.ps.postsolve({7., 4.}, x);
  GPRS_REQUIRE(x.size() == 3);
  GPRS_CHECK_EQ(x[0], 7.);
  GPRS_CHECK_EQ(x[1], 2.);
  GPRS_CHECK_EQ(x[2], 4.);
  double objective;
  GPRS_CHECK(feasible(m, x, objective));
  GPRS_CHECK_EQ(objective, 2.);
  GPRS_CHECK_EQ(check_point(reduced, {7., 4.}).objective, 2.);
}

// Two integers: only an integral map is used, whatever index and length say; no map, no change.
GPRS_TEST(doubleton_integer_guards)
{
  {
    // 2 x0 + 3 x1 = 6: x0 = 3 - 1.5 x1 and x1 = 2 - x0 / 1.5, neither is integral.
    model_t m =
      make_model({{2., 3.}, {1., 1.}}, {6., -inf}, {6., 9.}, {0., 0.}, {5., 5.}, {1., 1.}, {1, 1});
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, 0);
    GPRS_CHECK(f.st.row_alive[0] && f.st.is_live(0) && f.st.is_live(1));
    GPRS_CHECK(f.ps.order.empty());
    GPRS_CHECK(f.st.row_frontier.empty(row_list::doubleton));
    GPRS_CHECK(row_entries(f.st, 0) == (entries_t{{0, 2.}, {1, 3.}}));
  }
  {
    // x0 - 2 x1 = 1: x0 = 2 x1 + 1 is integral, x1 = x0 / 2 - 1 / 2 is not. x0 has the smaller
    // index and more entries and is still the one that leaves.
    model_t m = make_model({{1., -2.}, {1., 1.}, {1., 0.}},
                           {1., -inf, -inf},
                           {1., 9., 8.},
                           {-9., -9.},
                           {9., 9.},
                           {1., 0.},
                           {1, 1});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_REQUIRE(f.ps.aliases.size() == 1);
    GPRS_CHECK_EQ(f.ps.aliases[0].col, 0);
    GPRS_CHECK_EQ(f.ps.aliases[0].rep, 1);
    GPRS_CHECK_EQ(f.ps.aliases[0].scale, 2.);
    GPRS_CHECK_EQ(f.ps.aliases[0].offset, 1.);
    // -9 <= 2 x1 + 1 <= 9 on an integer: -5 <= x1 <= 4.
    GPRS_CHECK_EQ(f.st.col_lower[1], -5.);
    GPRS_CHECK_EQ(f.st.col_upper[1], 4.);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{1, 3.}}));
    GPRS_CHECK_EQ(f.st.row_upper[1], 8.);
    GPRS_CHECK(row_entries(f.st, 2) == (entries_t{{1, 2.}}));
    GPRS_CHECK_EQ(f.st.row_upper[2], 7.);
    model_t reduced     = f.reduced();
    box_counts_t counts = check_same_feasible_set(m, reduced, f.ps);
    GPRS_CHECK(counts.original > 0);
    GPRS_CHECK_EQ(counts.original, counts.reduced);
  }
  for (int32_t integer_col = 0; integer_col < 2; ++integer_col) {
    // One integer, one continuous column: the continuous one leaves, with a fractional scale,
    // also when it has the smaller index or more entries.
    int32_t continuous_col = 1 - integer_col;
    dense_t extra(2, 0.);
    extra[continuous_col] = 1.;
    std::vector<uint8_t> type(2, 0);
    type[integer_col] = 1;
    model_t m         = make_model(
      {{2., 2.}, extra, extra}, {6., -inf, -inf}, {6., 9., 8.}, {0., 0.}, {5., 5.}, {1., 1.}, type);
    m.value[integer_col] = 3.;
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_REQUIRE(f.ps.aliases.size() == 1);
    GPRS_CHECK_EQ(f.ps.aliases[0].col, continuous_col);
    GPRS_CHECK_EQ(f.ps.aliases[0].rep, integer_col);
    GPRS_CHECK_EQ(f.ps.aliases[0].scale, -1.5);
    GPRS_CHECK_EQ(f.ps.aliases[0].offset, 3.);
    // 0 <= 3 - 1.5 x <= 5 on an integer: 0 <= x <= 2.
    GPRS_CHECK_EQ(f.st.col_lower[integer_col], 0.);
    GPRS_CHECK_EQ(f.st.col_upper[integer_col], 2.);
    model_t reduced = f.reduced();
    GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps).reduced > 0);
  }
}

// x1 = x0. In row 1 the root's coefficient cancels and its entry goes; in row 2 it does not.
GPRS_TEST(doubleton_cancellation)
{
  model_t m = make_model({{1., -1., 0.}, {1., -1., 1.}, {2., 1., 0.}, {0.1, 0.2, 1.}},
                         {0., -inf, -inf, -inf},
                         {0., 5., 7., 9.},
                         {0., 0., 0.},
                         {4., 4., 4.},
                         {0.3, -0.3, 0.},
                         {0, 0, 0});
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_REQUIRE(f.ps.aliases.size() == 1 && f.ps.aliases[0].col == 1);
  GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, 1.}}));
  GPRS_CHECK_EQ(f.st.row_free_size[1], 1);
  GPRS_CHECK(row_entries(f.st, 2) == (entries_t{{0, 3.}}));
  GPRS_CHECK(row_entries(f.st, 3) == (entries_t{{0, 0.1 + 0.2}, {2, 1.}}));
  GPRS_CHECK_EQ(f.st.cost[0], 0.);
  GPRS_CHECK_EQ(f.st.csc.size[0], 2);
  model_t reduced = f.reduced();
  GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps).original > 0);
}

// A sum that is zero up to rounding is zero, in a row and in the cost: 0.3 - 0.1 * 3 is -5.6e-17
// in doubles.
GPRS_TEST(doubleton_cancellation_by_rounding)
{
  model_t m = make_model({{3., -1., 0.}, {0.3, -0.1, 1.}},
                         {0., -inf},
                         {0., 5.},
                         {0., 0., 0.},
                         {4., 12., 9.},
                         {0.3, -0.1, 0.},
                         {0, 0, 0});
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_REQUIRE(f.ps.aliases.size() == 1 && f.ps.aliases[0].col == 1);
  GPRS_CHECK_EQ(f.ps.aliases[0].scale, 3.);
  GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{2, 1.}}));
  GPRS_CHECK_EQ(f.st.csc.size[0], 0);
  GPRS_CHECK_EQ(f.st.cost[0], 0.);
  GPRS_CHECK_EQ(f.st.cost[1], 0.);
}

// Integers. x1 = 2 x0 from row 0 is the one map of the first batch: row 1 has no integral map
// and row 2 is walked from x2, where x3 = x2 / 2 is not integral. Row 2 has no event, since
// neither of its columns left, and still comes back: the second batch takes it as x2 = 2 x3.
GPRS_TEST(doubleton_cut_edge_comes_back)
{
  model_t m = make_model({{-2., 1., 0., 0.}, {0., 2., 3., 0.}, {0., 0., 1., -2.}},
                         {0., 6., 0.},
                         {0., 6., 0.},
                         dense_t(4, 0.),
                         dense_t(4, 9.),
                         dense_t(4, 1.),
                         {1, 1, 1, 1});
  fixture_t f(m);
  rule_result_t first = f.run();
  GPRS_CHECK_EQ(first.findings, 1);
  GPRS_REQUIRE(f.ps.aliases.size() == 1);
  GPRS_CHECK_EQ(f.ps.aliases[0].col, 1);
  GPRS_CHECK_EQ(f.ps.aliases[0].rep, 0);
  GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{0, 4.}, {2, 3.}}));
  GPRS_CHECK(row_entries(f.st, 2) == (entries_t{{2, 1.}, {3, -2.}}));
  GPRS_CHECK(f.st.row_frontier.list[row_list::doubleton] == (list_t{1, 2}));
  rule_result_t second = f.run();
  GPRS_CHECK_EQ(second.findings, 1);
  GPRS_REQUIRE(f.ps.aliases.size() == 2);
  GPRS_CHECK_EQ(f.ps.aliases[1].col, 2);
  GPRS_CHECK_EQ(f.ps.aliases[1].rep, 3);
  GPRS_CHECK_EQ(f.ps.aliases[1].scale, 2.);
  GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{0, 4.}, {3, 6.}}));
  // Row 1 is cut in a tree that lost a column, comes back once more and is then left alone.
  GPRS_CHECK_EQ(f.run().findings, 0);
  GPRS_CHECK(f.st.row_frontier.empty(row_list::doubleton));
  model_t reduced     = f.reduced();
  box_counts_t counts = check_same_feasible_set(m, reduced, f.ps);
  GPRS_CHECK_EQ(counts.original, 1);
  GPRS_CHECK_EQ(counts.reduced, 1);
}

// The bounds of the column that leaves fix the root: both are fixed and recorded, no alias.
GPRS_TEST(doubleton_root_fixed_by_transfer)
{
  {
    model_t m = make_model(
      {{1., -1.}, {1., 1.}}, {0., -inf}, {0., 20.}, {0., 5.}, {5., 9.}, {1., 2.}, {0, 0});
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, 1);
    GPRS_CHECK(!f.st.row_alive[0] && f.st.row_alive[1]);
    GPRS_CHECK(f.st.col_status[0] == col_status_t::fixed);
    GPRS_CHECK(f.st.col_status[1] == col_status_t::fixed);
    GPRS_CHECK(f.ps.aliases.empty());
    GPRS_REQUIRE(f.ps.fixed_cols.size() == 2);
    GPRS_CHECK_EQ(f.ps.fixed_cols[0].col, 0);
    GPRS_CHECK_EQ(f.ps.fixed_cols[0].value, 5.);
    GPRS_CHECK_EQ(f.ps.fixed_cols[1].col, 1);
    GPRS_CHECK_EQ(f.ps.fixed_cols[1].value, 5.);
    GPRS_CHECK_EQ(dd_value(f.st.objective_offset), 15.);
    GPRS_CHECK_EQ(f.st.row_free_size[1], 0);
  }
  {
    // The bounds cross by less than the tolerance: the root stays at its bound and the image of
    // the column, 5, is moved into the column's bounds.
    double lower = 5. + 5e-7;
    model_t m    = make_model({{1., -1.}}, {0.}, {0.}, {0., lower}, {5., 9.}, {0., 0.}, {0, 0});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().outcome, ok);
    GPRS_REQUIRE(f.ps.fixed_cols.size() == 2);
    GPRS_CHECK_EQ(f.ps.fixed_cols[0].value, 5.);
    GPRS_CHECK_EQ(f.ps.fixed_cols[1].col, 1);
    GPRS_CHECK_EQ(f.ps.fixed_cols[1].value, lower);
  }
  {
    // Three columns on one root; the second map fixes the root, the first column is already an
    // alias and the third is fixed at its image.
    model_t m =
      make_model({{1., -1., 0., 0.}, {1., 0., -1., 0.}, {1., 0., 0., -2.}, {1., 0., 0., 0.}},
                 {0., 0., 0., -inf},
                 {0., 0., 0., 50.},
                 {0., -1., 3., 0.},
                 {3., 9., 9., 9.},
                 {0., 1., 0., 1.},
                 {0, 0, 0, 0});
    fixture_t f(m);
    rule_result_t result = f.run();
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, 3);
    GPRS_CHECK(f.st.col_status[0] == col_status_t::fixed);
    GPRS_CHECK(f.st.col_status[1] == col_status_t::removed);
    GPRS_CHECK(f.st.col_status[2] == col_status_t::fixed);
    GPRS_CHECK(f.st.col_status[3] == col_status_t::fixed);
    GPRS_CHECK_EQ(f.st.col_lower[3], 1.5);
    GPRS_CHECK_EQ(f.ps.aliases.size(), 1u);
    GPRS_CHECK_EQ(f.ps.fixed_cols.size(), 3u);
    model_t reduced = f.reduced();
    dense_t x;
    f.ps.postsolve(dense_t(reduced.ncols, 0.), x);
    GPRS_CHECK(x == (dense_t{3., 3., 3., 1.5}));
  }
}

GPRS_TEST(doubleton_infeasible)
{
  {
    // x1 = x0 with disjoint bounds.
    model_t m = make_model({{1., -1.}}, {0.}, {0.}, {0., 3.}, {1., 4.}, {0., 0.}, {0, 0});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().outcome, infeasible);
  }
  {
    // x1 = x0, x0 integer, x1 in [0.3, 0.6]: no integer in the transferred bounds.
    model_t m = make_model({{1., -1.}}, {0.}, {0.}, {0., 0.3}, {3., 0.6}, {0., 0.}, {1, 0});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().outcome, infeasible);
  }
  {
    // The first column fixes the root at 3, the bounds of the second exclude it.
    model_t m = make_model({{1., -1., 0.}, {1., 0., -1.}},
                           {0., 0.},
                           {0., 0.},
                           {0., 3., 0.},
                           {3., 9., 2.},
                           {0., 0., 0.},
                           {0, 0, 0});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().outcome, infeasible);
    GPRS_CHECK_EQ(feasible_points(m), 0);
  }
}

// The term of a fixed column belongs to the right-hand side: x0 + x1 + 3 * 2 = 10.
GPRS_TEST(doubleton_fixed_terms)
{
  model_t m = make_model({{1., 1., 3.}, {1., 2., 1.}},
                         {10., -inf},
                         {10., 30.},
                         {0., 0., 2.},
                         {9., 9., 2.},
                         {0., 1., 0.},
                         {0, 0, 0});
  fixture_t f(m);
  GPRS_CHECK(f.st.col_status[2] == col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.row_free_size[0], 2);
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_REQUIRE(f.ps.aliases.size() == 1);
  GPRS_CHECK_EQ(f.ps.aliases[0].col, 1);
  GPRS_CHECK_EQ(f.ps.aliases[0].scale, -1.);
  GPRS_CHECK_EQ(f.ps.aliases[0].offset, 4.);
  GPRS_CHECK_EQ(f.st.col_upper[0], 4.);
  // x0 + 2 (4 - x0) + 2 <= 30.
  GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{0, -1.}, {2, 1.}}));
  GPRS_CHECK_EQ(f.st.row_upper[1], 22.);
  model_t reduced = f.reduced();
  GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps).original > 0);
}

// x0 - x1 = 1 and x1 - x2 = 2 in one batch: both rows go, x2 gets the composed map x0 - 3.
GPRS_TEST(doubleton_chain)
{
  model_t m = make_model({{1., -1., 0.}, {0., 1., -1.}, {1., 0., 1.}, {1., 1., 1.}, {1., 0., 0.}},
                         {1., 2., -inf, -inf, -inf},
                         {1., 2., 9., 20., 50.},
                         {-9., -9., -9.},
                         {9., 9., 5.},
                         {0., 1., 1.},
                         {0, 0, 0});
  fixture_t f(m);
  rule_result_t result = f.run();
  GPRS_CHECK_EQ(result.findings, 2);
  GPRS_CHECK(!f.st.row_alive[0] && !f.st.row_alive[1]);
  GPRS_REQUIRE(f.ps.aliases.size() == 2);
  GPRS_CHECK_EQ(f.ps.aliases[0].col, 1);
  GPRS_CHECK_EQ(f.ps.aliases[0].rep, 0);
  GPRS_CHECK_EQ(f.ps.aliases[0].row, 0);
  GPRS_CHECK_EQ(f.ps.aliases[0].scale, 1.);
  GPRS_CHECK_EQ(f.ps.aliases[0].offset, -1.);
  GPRS_CHECK_EQ(f.ps.aliases[1].col, 2);
  GPRS_CHECK_EQ(f.ps.aliases[1].rep, 0);
  GPRS_CHECK_EQ(f.ps.aliases[1].row, 1);
  GPRS_CHECK_EQ(f.ps.aliases[1].scale, 1.);
  GPRS_CHECK_EQ(f.ps.aliases[1].offset, -3.);
  GPRS_CHECK(row_entries(f.st, 2) == (entries_t{{0, 2.}}));
  GPRS_CHECK_EQ(f.st.row_upper[2], 12.);
  GPRS_CHECK(row_entries(f.st, 3) == (entries_t{{0, 3.}}));
  GPRS_CHECK_EQ(f.st.row_upper[3], 24.);
  // x1 in [-9, 9] gives x0 in [-8, 10], x2 in [-9, 5] gives x0 in [-6, 8].
  GPRS_CHECK_EQ(f.st.col_lower[0], -6.);
  GPRS_CHECK_EQ(f.st.col_upper[0], 8.);
  GPRS_CHECK_EQ(f.st.cost[0], 2.);
  GPRS_CHECK_EQ(dd_value(f.st.objective_offset), -4.);
  model_t reduced = f.reduced();
  GPRS_CHECK(check_same_feasible_set(m, reduced, f.ps).original > 0);
}

// A row that closes a cycle stays in the model with the root in the place of its columns.
GPRS_TEST(doubleton_cycle)
{
  {
    model_t m =
      make_model({{1., -1.}, {1., 1.}}, {0., 4.}, {0., 4.}, {0., 0.}, {9., 9.}, {0., 0.}, {0, 0});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK(!f.st.row_alive[0] && f.st.row_alive[1]);
    GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{0, 2.}}));
    GPRS_CHECK(on_list(f.st, row_list::doubleton, 1) && on_list(f.st, row_list::row_rules, 1));
    GPRS_CHECK_EQ(f.run().findings, 0);
    GPRS_CHECK(f.st.row_frontier.empty(row_list::doubleton));
  }
  {
    // Integers. Row 0 has no integral map and is cut; row 1 closes the cycle in the first
    // batch, comes back and is the tree edge of the second.
    model_t m =
      make_model({{2., 3.}, {1., -1.}}, {5., 0.}, {5., 0.}, {0., 0.}, {9., 9.}, {0., 0.}, {1, 1});
    fixture_t f(m);
    rule_result_t first = f.run();
    GPRS_CHECK_EQ(first.outcome, ok);
    GPRS_CHECK_EQ(first.findings, 0);
    GPRS_CHECK(f.st.row_frontier.list[row_list::doubleton] == (list_t{1}));
    rule_result_t second = f.run();
    GPRS_CHECK_EQ(second.findings, 1);
    GPRS_CHECK(f.st.row_alive[0] && !f.st.row_alive[1]);
    GPRS_CHECK(row_entries(f.st, 0) == (entries_t{{0, 5.}}));
    model_t reduced     = f.reduced();
    box_counts_t counts = check_same_feasible_set(m, reduced, f.ps);
    GPRS_CHECK_EQ(counts.original, 1);
    GPRS_CHECK_EQ(counts.reduced, 1);
  }
}

// The budget is tested between two edges: the collected edge is applied in full, the rows that
// were not visited are back on the list.
GPRS_TEST(doubleton_budget)
{
  model_t m =
    make_model({{1., -1., 0., 0., 0., 0.}, {0., 0., 1., -1., 0., 0.}, {0., 0., 0., 0., 1., -1.}},
               {0., 0., 0.},
               {0., 0., 0.},
               dense_t(6, 0.),
               dense_t(6, 5.),
               dense_t(6, 1.),
               std::vector<uint8_t>(6, 0));
  fixture_t f(m);
  rule_result_t first = f.run(1);
  GPRS_CHECK_EQ(first.outcome, ok);
  GPRS_CHECK(first.stopped);
  GPRS_CHECK_EQ(first.findings, 1);
  GPRS_CHECK(!f.st.row_alive[0] && f.st.row_alive[1] && f.st.row_alive[2]);
  GPRS_CHECK(f.st.col_status[1] == col_status_t::removed);
  GPRS_CHECK_EQ(f.st.cost[0], 2.);
  GPRS_CHECK(f.st.row_frontier.list[row_list::doubleton] == (list_t{1, 2}));
  rule_result_t second = f.run();
  GPRS_CHECK(!second.stopped);
  GPRS_CHECK_EQ(second.findings, 2);
  GPRS_CHECK_EQ(f.st.live_rows, 0);
  GPRS_CHECK_EQ(f.st.live_cols, 3);
  GPRS_CHECK(f.st.row_frontier.empty(row_list::doubleton));

  // The work of an edge is spent and a finding is noted on the budget.
  fixture_t g(m);
  work_budget_t budget;
  budget.limit = budget.stall_limit = no_limit;
  g.rule.run(g.st, budget, g.ps, g.opt);
  // Per row: the visit, its two entries, one entry in each of its columns.
  GPRS_CHECK_EQ(budget.used, 3 * (line_visit_work + 2 + 2));
  GPRS_CHECK_EQ(budget.used_at_finding, budget.used);
}

// Not collected: an inequality, an equation with three free entries, a deleted row, a row with
// one free entry.
GPRS_TEST(doubleton_rows_not_collected)
{
  model_t m = make_model({{1., -1., 0., 0.}, {1., 1., 1., 0.}, {0., 0., 1., -1.}, {1., 0., 0., 2.}},
                         {0., 3., 0., 4.},
                         {1., 3., 0., 4.},
                         {0., 0., 0., 2.},
                         {5., 5., 5., 2.},
                         {1., 1., 1., 1.},
                         {0, 0, 0, 0});
  fixture_t f(m);
  f.st.delete_row(2);
  rule_result_t result = f.run();
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK(!result.stopped);
  GPRS_CHECK_EQ(f.st.live_rows, 3);
  GPRS_CHECK_EQ(f.st.live_cols, 3);
  GPRS_CHECK(f.ps.aliases.empty());
  GPRS_CHECK(f.st.row_frontier.empty(row_list::doubleton));
  // An event brings a row back: with its lower side raised row 0 is an equation.
  f.st.set_row_lower(0, 1.);
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_CHECK(!f.st.row_alive[0]);
  // x1 = x0 - 1 in row 1 leaves two free entries: collected by the next call.
  GPRS_CHECK(on_list(f.st, row_list::doubleton, 1));
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_CHECK_EQ(f.st.live_rows, 1);
}

// x1 = 1e7 x0 with x1 >= 1: x0 >= 1e-7 means x0 >= 1 on the integers. A rounding with the
// feasibility tolerance in the units of x0 would keep x0 = 0, where x1 = 0 is below its bound.
GPRS_TEST(doubleton_integer_root_large_scale)
{
  model_t m = make_model({{-1e7, 1.}}, {0.}, {0.}, {0., 1.}, {10., 5e7}, {0., 0.}, {1, 1});
  fixture_t f(m);
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_REQUIRE(f.ps.aliases.size() == 1 && f.ps.aliases[0].col == 1);
  GPRS_CHECK_EQ(f.ps.aliases[0].scale, 1e7);
  GPRS_CHECK_EQ(f.st.col_lower[0], 1.);
  GPRS_CHECK_EQ(f.st.col_upper[0], 5.);
}

// The same with a scale that is no integer, 1 / 7e-11: the bound of x0 that comes out of
// (k / 7e-11) / (1 / 7e-11) is k up to rounding, and where it lies next to k it is still k.
GPRS_TEST(doubleton_integer_root_rounding_error)
{
  double a = 7e-11, scale = 1. / a;
  int32_t above = 0, below = 0;
  for (int32_t k = 3; k < 200 && (above == 0 || below == 0); ++k) {
    double quotient = (k / a) / scale;
    if (quotient > k && above == 0) { above = k; }
    if (quotient < k && below == 0) { below = k; }
  }
  GPRS_REQUIRE(above > 0 && below > 0);
  {
    model_t m = make_model({{-1., a}}, {0.}, {0.}, {0., above / a}, {250., 1e13}, {0., 0.}, {1, 0});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_REQUIRE(f.ps.aliases.size() == 1 && f.ps.aliases[0].col == 1);
    GPRS_CHECK_EQ(f.ps.aliases[0].scale, scale);
    GPRS_CHECK_EQ(f.st.col_lower[0], above);
    GPRS_CHECK_EQ(f.st.col_upper[0], 250.);
  }
  {
    model_t m = make_model({{-1., a}}, {0.}, {0.}, {0., 0.}, {250., below / a}, {0., 0.}, {1, 0});
    fixture_t f(m);
    GPRS_CHECK_EQ(f.run().findings, 1);
    GPRS_CHECK_EQ(f.st.col_lower[0], 0.);
    GPRS_CHECK_EQ(f.st.col_upper[0], below);
  }
}

// After the first substitution the lines are out of order and columns have moved in the store;
// a second batch on the same root works on them.
GPRS_TEST(doubleton_repeated_batches)
{
  model_t m = make_model({{1., -1., 0., 0.},
                          {1., 0., 1., 1.},
                          {0., 1., 1., 0.},
                          {0., 2., 0., 1.},
                          {0., 1., 0., -1.},
                          {0., 0., 1., 1.}},
                         {0., -inf, -inf, -inf, 0., -inf},
                         {0., 9., 8., 7., inf, 6.},
                         {0., 0., 0., 0.},
                         {3., 3., 3., 3.},
                         {1., 1., 1., 1.},
                         {1, 1, 1, 1});
  fixture_t f(m);
  // x1 has the most entries and is the root; the entry it gets in row 1 does not fit into its
  // slots, so the column moves.
  int64_t start = f.st.csc.start[1];
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_CHECK(f.st.col_status[0] == col_status_t::removed);
  GPRS_CHECK_EQ(f.st.csc.size[1], 4);
  GPRS_CHECK(f.st.csc.start[1] != start);
  GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{1, 1.}, {2, 1.}, {3, 1.}}));
  // Row 4 becomes an equation between the root and x3.
  f.st.set_row_upper(4, 0.);
  GPRS_CHECK_EQ(f.run().findings, 1);
  GPRS_CHECK_EQ(f.st.live_cols, 2);
  GPRS_CHECK(row_entries(f.st, 1) == (entries_t{{1, 2.}, {2, 1.}}));
  GPRS_CHECK(row_entries(f.st, 3) == (entries_t{{1, 3.}}));
  GPRS_CHECK(row_entries(f.st, 5) == (entries_t{{1, 1.}, {2, 1.}}));
  model_t reduced     = f.reduced();
  m.row_upper[4]      = 0.;
  box_counts_t counts = check_same_feasible_set(m, reduced, f.ps);
  GPRS_CHECK(counts.original > 0);
  GPRS_CHECK_EQ(counts.original, counts.reduced);
}

// Random models, the rule run to the end: the state stays consistent, the feasible set and the
// objective are unchanged, with equal numbers of feasible points when every column is integer,
// and an infeasible verdict comes with an empty feasible set.
GPRS_TEST(doubleton_random_feasible_set)
{
  int64_t reductions = 0, infeasible_cases = 0, nonempty = 0, fixed = 0, cut = 0;
  for (uint64_t seed = 1; seed <= 4000; ++seed) {
    rng_t rng(seed);
    bool all_integer = seed % 2 == 0;
    model_t m        = random_model(rng, all_integer);
    fixture_t f(m);
    size_t fixed_at_start = f.ps.fixed_cols.size();
    rule_result_t result  = f.run_to_end();
    if (result.outcome != ok) {
      ++infeasible_cases;
      GPRS_CHECK_EQ(feasible_points(m), 0);
      if (feasible_points(m) != 0) { std::printf("    seed %llu\n", (unsigned long long)seed); }
      continue;
    }
    reductions += result.findings;
    GPRS_CHECK((int64_t)f.ps.aliases.size() <= result.findings);
    fixed += f.ps.fixed_cols.size() - fixed_at_start;
    for (int32_t i = 0; i < m.nrows; ++i) {
      cut +=
        f.st.row_alive[i] && f.st.row_free_size[i] == 2 && f.st.row_lower[i] == f.st.row_upper[i];
    }
    int32_t failures    = gprs::test::failure_count();
    model_t reduced     = f.reduced();
    box_counts_t counts = check_same_feasible_set(m, reduced, f.ps);
    if (all_integer) { GPRS_CHECK_EQ(counts.original, counts.reduced); }
    nonempty += counts.original > 0;
    GPRS_CHECK(well_formed(reduced));
    GPRS_CHECK_EQ(reduced.ncols + (int32_t)f.ps.aliases.size() + (int32_t)f.ps.fixed_cols.size(),
                  m.ncols);
    if (gprs::test::failure_count() != failures) {
      std::printf("    seed %llu\n", (unsigned long long)seed);
      return;
    }
  }
  // The test is not vacuous.
  GPRS_CHECK(reductions > 3000);
  GPRS_CHECK(infeasible_cases > 50);
  GPRS_CHECK(nonempty > 2500);
  GPRS_CHECK(fixed > 1000);
  GPRS_CHECK(cut > 100);
}
