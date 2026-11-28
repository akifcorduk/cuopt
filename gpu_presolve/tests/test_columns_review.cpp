/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The singleton-column rules (rules.hpp (h)) on the models of the design reviews and on models
// built to break them: fixed terms in a pivot row, sides that become readable, rows without a
// side next to a forcing column, a cost at the end of the number range, and what a second
// presolve finds.

#include "test_columns_support.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace gprs;
using columns_test::call_t;
using columns_test::free_col;
using columns_test::silent;
using columns_test::unit;
using gprs_check::structure_report_t;
using rule_test::col_t;
using rule_test::row_t;

namespace {

// Rows that keep columns y and z in the model.
row_t sum_row(int32_t y, int32_t z) { return {{{y, 1.}, {z, 1.}}, -inf, 2.}; }
row_t difference_row(int32_t y, int32_t z) { return {{{y, 1.}, {z, -1.}}, -1., inf}; }

}  // namespace

// Both ends of the range that the pivot row gives the column are read from the stored side and
// the stored activities, which hold the terms of the fixed columns (review item 9 a): with
// k + y + f = 6 and f fixed at 2, k = 4 - y lies in [3, 4]. Taking the fixed term off twice
// would test [1, 2].
GPRS_TEST(columns_review_fixed_terms_enter_the_range_once)
{
  const std::vector<row_t> equation = {
    {{{0, 1.}, {1, 1.}, {3, 1.}}, 6., 6.}, sum_row(1, 2), difference_row(1, 2)};
  const std::vector<row_t> inequality = {
    {{{0, 1.}, {1, 1.}, {3, 1.}}, 6., inf}, sum_row(1, 2), difference_row(1, 2)};
  struct bounds_t {
    double lower;
    double upper;
    bool taken;
  };
  for (bounds_t b : {bounds_t{3., 4., true}, bounds_t{1., 2., false}, bounds_t{1., 3.5, false}}) {
    for (bool is_equation : {true, false}) {
      call_t c({{b.lower, b.upper, false}, unit, unit, {2., 3., false}},
               is_equation ? equation : inequality,
               {1., 0., 0., 0.});
      fix_column(c.st, c.ps, 3, 2.);
      c.run();
      GPRS_CHECK_EQ(c.result.findings, b.taken ? 1 : 0);
      if (!b.taken) { continue; }
      GPRS_CHECK_EQ(c.result.dual_findings, is_equation ? 0 : 1);
      GPRS_CHECK_EQ(c.ps.saved_rows[0].upper, 4.);
      GPRS_CHECK(c.postsolved({{1, 0.5}, {2, 0.}}) == std::vector<double>({3.5, 0.5, 0., 2.}));
      GPRS_CHECK_EQ(c.reduced.objective_offset, 4.);
      structure_report_t s = c.structure();
      GPRS_CHECK(silent(s));
      GPRS_CHECK_EQ(s.box_tested, 2);
    }
  }
}

// The zero-cost singleton columns and the forcing column next to a fixed column: the row that
// stays keeps the term in its sides, and the record holds the sides without it.
GPRS_TEST(columns_review_fixed_terms_in_the_other_records)
{
  // 6 <= s + y + z + f <= 7 with s in [0, 1] and f fixed at 2.
  call_t interval({unit, {0., 4., false}, {0., 4., false}, {2., 3., false}},
                  {{{{0, 1.}, {1, 1.}, {2, 1.}, {3, 1.}}, 6., 7.},
                   {{{1, 1.}, {2, 1.}}, -inf, 7.},
                   {{{1, 1.}, {2, -1.}}, -3., inf}});
  fix_column(interval.st, interval.ps, 3, 2.);
  interval.run();
  GPRS_CHECK_EQ(interval.result.sub_findings[sub_rule::zero_cost_singleton], 1);
  GPRS_CHECK_EQ(interval.st.row_lower[0], 5.);
  GPRS_CHECK_EQ(interval.st.row_upper[0], 7.);
  GPRS_REQUIRE(interval.ps.saved_rows.size() == 1);
  GPRS_CHECK_EQ(interval.ps.saved_rows[0].lower, 4.);
  GPRS_CHECK_EQ(interval.ps.saved_rows[0].upper, 5.);
  // y + z = 3 is one short of the row: s = 1.
  GPRS_CHECK(interval.postsolved({{1, 1.}, {2, 2.}}) == std::vector<double>({1., 1., 2., 2.}));
  GPRS_CHECK(silent(interval.structure()));

  // k + y + f <= 6 with k free to go down.
  call_t forcing({{-inf, 9., false}, {0., 4., false}, {0., 4., false}, {2., 3., false}},
                 {{{{0, 1.}, {1, 1.}, {3, 1.}}, -inf, 6.},
                  {{{1, 1.}, {2, 1.}}, -inf, 7.},
                  {{{1, 1.}, {2, -1.}}, -3., inf}});
  fix_column(forcing.st, forcing.ps, 3, 2.);
  forcing.run();
  GPRS_CHECK_EQ(forcing.result.sub_findings[sub_rule::forcing_column], 1);
  GPRS_REQUIRE(forcing.ps.saved_rows.size() == 1);
  GPRS_CHECK_EQ(forcing.ps.saved_rows[0].upper, 4.);
  GPRS_CHECK_EQ(forcing.postsolved({{1, 1.5}, {2, 0.}})[0], 2.5);
  GPRS_CHECK(silent(forcing.structure()));
}

// A term of a fixed column that swallows the side: no reduction reads the side less the terms.
// With the column fixed at 1e20 a side of 3 is gone from the difference.
GPRS_TEST(columns_review_side_swallowed_by_a_fixed_term)
{
  const col_t y{0., 4., false}, fixed{1e20, 2e20, false};
  const std::vector<row_t> keep = {{{{1, 1.}, {2, 1.}}, -inf, 7.}, {{{1, 1.}, {2, -1.}}, -3., inf}};
  auto rows                     = [&](double lower, double upper) {
    std::vector<row_t> all = {{{{0, 1.}, {1, 1.}, {3, 1.}}, lower, upper}};
    all.insert(all.end(), keep.begin(), keep.end());
    return all;
  };
  struct shape_t {
    col_t k;
    double lower;
    double upper;
    double cost;
  };
  for (const shape_t& t : {shape_t{free_col, 3., 3., 1.},   // the free singleton column
                           shape_t{free_col, 3., inf, 1.},  // on an inequality
                           shape_t{unit, 3., 5., 0.},       // a zero-cost singleton column
                           shape_t{{-inf, 9., false}, -inf, 3., 0.}})  // a forcing column
  {
    call_t c({t.k, y, y, fixed}, rows(t.lower, t.upper), {t.cost, 0., 0., 0.});
    fix_column(c.st, c.ps, 3, 1e20);
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 0);
    GPRS_CHECK(c.ps.col_from_rows.empty() && c.ps.saved_entries.col.empty());
    GPRS_CHECK(c.st.row_alive[0] && c.st.is_live(0));
  }
}

// The undo of a forcing column is one rule (review item 9 b): the smallest value that its rows
// and its bound allow for a column that is free to go down, also when it has no bound at all
// and the value nearest to zero would be another one.
GPRS_TEST(columns_review_forcing_column_without_a_bound)
{
  // k + y <= 4 and -2 k + z >= -6.
  const col_t y{0., 4., false};
  call_t c({free_col, y, y},
           {{{{0, 1.}, {1, 1.}}, -inf, 4.},
            {{{0, -2.}, {2, 1.}}, -6., inf},
            {{{1, 1.}, {2, 1.}}, -inf, 7.},
            {{{1, 1.}, {2, -1.}}, -3., inf}});
  c.run();
  GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::forcing_column], 1);
  GPRS_REQUIRE(c.ps.record_cols.size() == 1);
  GPRS_CHECK_EQ(c.ps.record_cols[0].prefer, 0.);
  // min(4 - y, 3 + z / 2).
  GPRS_CHECK_EQ(c.postsolved({{1, 0.}, {2, 4.}})[0], 4.);
  GPRS_CHECK_EQ(c.postsolved({{1, 1.}, {2, 0.}})[0], 3.);
  GPRS_CHECK_EQ(c.postsolved({{1, 4.}, {2, 4.}})[0], 0.);
  GPRS_CHECK(silent(c.structure()));
}

// A row of a forcing column without a finite side is deleted and not saved, wherever it stands
// among the rows of the column: the coefficients of the record are those of the saved rows.
GPRS_TEST(columns_review_forcing_column_with_a_free_row_first)
{
  const col_t y{0., 4., false};
  call_t c({{-inf, 9., false}, y, y},
           {{{{0, 5.}, {1, 1.}}, -inf, inf},
            {{{0, 1.}, {1, 1.}}, -inf, 4.},
            {{{0, 7.}, {2, 1.}}, -inf, inf},
            {{{0, -2.}, {2, 1.}}, -6., inf},
            {{{1, 1.}, {2, 1.}}, -inf, 7.},
            {{{1, 1.}, {2, -1.}}, -3., inf}});
  c.run();
  GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::forcing_column], 1);
  GPRS_CHECK(!c.st.row_alive[0] && !c.st.row_alive[1] && !c.st.row_alive[2] && !c.st.row_alive[3]);
  GPRS_REQUIRE(c.ps.saved_rows.size() == 2);
  GPRS_CHECK_EQ(c.ps.saved_rows[0].row, 1);
  GPRS_CHECK_EQ(c.ps.saved_rows[1].row, 3);
  GPRS_CHECK(c.ps.record_coefficients == std::vector<double>({1., -2.}));
  GPRS_CHECK_EQ(c.postsolved({{1, 2.5}, {2, 4.}})[0], 1.5);
  GPRS_CHECK_EQ(c.postsolved({{1, 0.}, {2, 0.}})[0], 3.);
  GPRS_CHECK(silent(c.structure()));
}

// A cost so small that cost / a is zero cannot be moved: the column keeps it and stays.
GPRS_TEST(columns_review_cost_that_underflows)
{
  const double tiny = std::numeric_limits<double>::denorm_min();
  GPRS_REQUIRE(tiny / 4. == 0.);
  for (double upper : {6., inf}) {
    call_t c({free_col, unit, unit},
             {{{{0, 4.}, {1, 2.}, {2, 3.}}, 6., upper}, sum_row(1, 2), difference_row(1, 2)},
             {tiny, 0., 0.});
    c.run();
    GPRS_CHECK_EQ(c.result.findings, 0);
    GPRS_CHECK(c.st.is_live(0) && c.st.row_alive[0]);
    GPRS_CHECK_EQ(c.st.cost[0], tiny);
  }
  // A cost that is merely small moves.
  call_t small({free_col, unit, unit},
               {{{{0, 4.}, {1, 2.}, {2, 3.}}, 6., 6.}, sum_row(1, 2), difference_row(1, 2)},
               {1e-300, 0., 0.});
  small.run();
  GPRS_CHECK_EQ(small.result.sub_findings[sub_rule::free_singleton], 1);
  GPRS_CHECK(small.reduced.cost == std::vector<double>({-0.5e-300, -0.75e-300}));
  GPRS_CHECK(silent(small.structure()));
}

// The side that is taken as integral and the drift of the ratios move the same row, so they
// share its tolerance: 1000 k + (1000 - 9e-10) y = 1000 + 9.9e-7 with an integer k and y in
// [0, 100] would give k = 1 - y, which leaves the row off by 9.9e-7 + 9e-8 at y = 100.
GPRS_TEST(columns_review_integral_side_and_drift_share_the_tolerance)
{
  const std::vector<col_t> cols = {{-inf, inf, true}, {0., 100., true}, {0., 1., true}};
  auto rows                     = [](double side) {
    return std::vector<row_t>{{{{0, 1000.}, {1, 1000. - 9e-10}}, side, side},
                              {{{1, 1.}, {2, 1.}}, -inf, 200.},
                              {{{1, 1.}, {2, -1.}}, -200., inf}};
  };
  call_t beyond(cols, rows(1000. + 9.9e-7));
  beyond.run();
  GPRS_CHECK_EQ(beyond.result.findings, 0);
  // With a side that leaves room for the drift the column is taken, and the row holds within
  // the tolerance at every point of the reduced model.
  call_t within(cols, rows(1000. + 5e-7));
  within.run();
  GPRS_CHECK_EQ(within.result.sub_findings[sub_rule::free_singleton], 1);
  for (double y : {0., 100.}) {
    std::vector<double> x = within.postsolved({{1, y}, {2, 0.}});
    GPRS_CHECK_EQ(x[0], 1. - y);
    point_check_t check = check_point(within.model, x);
    GPRS_CHECK(check.row_violation <= within.opt.tol.feasibility);
  }
  GPRS_CHECK(silent(within.structure()));
}

// A family none of whose reductions is active takes its lists and spends nothing, as a family
// without a reduction does: it cannot stop on its budget and keep a presolve from its fixed
// point.
GPRS_TEST(columns_review_family_without_an_active_reduction)
{
  for (bool primal_only : {false, true}) {
    call_t c({free_col, unit, unit},
             {{{{0, 1.}, {1, 2.}, {2, 3.}}, 6., 6.}, sum_row(1, 2), difference_row(1, 2)},
             {2., 5., 6.});
    c.opt.primal_only                           = primal_only;
    c.opt.sub_enabled[sub_rule::free_singleton] = false;
    if (!primal_only) {
      c.opt.sub_enabled[sub_rule::free_singleton_dual] = false;
      c.opt.sub_enabled[sub_rule::zero_cost_singleton] = false;
      c.opt.sub_enabled[sub_rule::forcing_column]      = false;
    }
    c.budget.limit       = 1;
    c.budget.stall_limit = 1;
    c.run();
    GPRS_CHECK(!c.result.stopped && !c.result.limited);
    GPRS_CHECK_EQ(c.result.findings, 0);
    GPRS_CHECK_EQ(c.budget.used, 0);
    GPRS_CHECK(c.st.col_frontier.empty(col_list::singleton_cols));
  }
  // With one reduction active the same budget stops the call after its first column.
  call_t active({free_col, unit, unit},
                {{{{0, 1.}, {1, 2.}, {2, 3.}}, 6., inf}, sum_row(1, 2), difference_row(1, 2)},
                {0., 5., 6.});
  active.opt.sub_enabled[sub_rule::free_singleton]      = false;
  active.opt.sub_enabled[sub_rule::free_singleton_dual] = false;
  active.opt.sub_enabled[sub_rule::forcing_column]      = false;
  active.budget.limit                                   = 1;
  active.run();
  GPRS_CHECK(active.result.stopped);
}

// The forcing column reads the sides of its rows, and a side changes without an event of the
// column: a row whose side was not readable when the column was left lists its columns when it
// changes.
GPRS_TEST(columns_review_forcing_column_follows_the_sides_of_its_rows)
{
  // k + y <= 3e9 with k free to go down: the side is beyond the size that a rule reads.
  const col_t y{0., 4., false};
  call_t c({{-inf, 9., false}, y, y},
           {{{{0, 1.}, {1, 1.}}, -inf, 3e9},
            {{{1, 1.}, {2, 1.}}, -inf, 7.},
            {{{1, 1.}, {2, -1.}}, -3., inf}});
  c.run();
  GPRS_CHECK_EQ(c.result.findings, 0);
  // The side becomes 4: an event of the row and none of the column.
  c.st.set_row_upper(0, 4.);
  GPRS_CHECK((c.st.col_frontier.pending[0] & list_bit(col_list::singleton_cols)) == 0);
  c.run();
  GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::forcing_column], 1);
  GPRS_CHECK(c.st.col_status[0] == col_status_t::removed);
  GPRS_CHECK(silent(c.structure()));

  // A row that no forcing column was left for costs one visit when it is listed, as before.
  call_t plain({{-inf, 9., false}, y, y},
               {{{{0, 1.}, {1, 1.}}, -inf, 5.},
                {{{0, 1.}, {2, 1.}}, 0., inf},
                {{{1, 1.}, {2, 1.}}, -inf, 7.},
                {{{1, 1.}, {2, -1.}}, -3., inf}});
  plain.run();
  plain.st.set_row_upper(0, 4.);
  plain.budget.used = 0;
  plain.run();
  GPRS_CHECK_EQ(plain.result.findings, 0);
  GPRS_CHECK_EQ(plain.budget.used, line_visit_work);
}

// The same in the engine: k + y + f <= 1e9 + 4 with f fixed at 1e9 by its bounds. The stored
// side is not readable while the row holds the term of f and is 4 once the term has left it,
// which lists the row and not k. The presolve takes k then, and its result is a fixed point.
GPRS_TEST(columns_review_fixed_point_with_a_side_that_becomes_readable)
{
  std::vector<col_t> cols = {free_col, {0., 10., true}, {1e9, 1e9, false}, {0., 10., true}};
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1.}, {2, 1.}}, -inf, 1e9 + 4.},
                             {{{1, 1.}, {3, 1.}}, 3., inf},
                             {{{1, 1.}, {3, -1.}}, -inf, 2.}};
  model_t m               = rule_test::make_model(cols, rows);
  m.cost                  = {0., 1., 0., 2.};
  options_t options;
  result_t first = presolve(m, options);
  GPRS_REQUIRE(first.status == status_t::reduced);
  GPRS_CHECK(first.fixed_point);
  GPRS_CHECK_EQ(first.sub_findings[sub_rule::forcing_column], 1);
  GPRS_CHECK_EQ(first.reduced.nrows, 2);
  GPRS_CHECK_EQ(presolve(first.reduced, options).status, status_t::unchanged);
  GPRS_CHECK(silent(gprs_check::structure_check(m, first, false)));
}
