/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "test_rows_support.hpp"

#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <vector>

using namespace gprs;
using namespace rule_test;

namespace {

constexpr rule_outcome_t ok         = rule_outcome_t::ok;
constexpr rule_outcome_t infeasible = rule_outcome_t::infeasible;

rule_result_t run_rows(fixture_t& f)
{
  row_rules_t rule;
  return rule.run(f.st, f.budget, f.ps, f.opt);
}

// Outcome of the row rules on one row over the columns, the first `fixed.size()` of them fixed.
rule_result_t one_row(const std::vector<col_t>& cols,
                      const row_t& row,
                      const std::vector<double>& fixed = {})
{
  fixture_t f(cols, {row});
  for (size_t j = 0; j < fixed.size(); ++j) {
    fix_column(f.st, f.ps, j, fixed[j]);
  }
  rule_result_t result = run_rows(f);
  if (result.outcome == ok) {
    f.check_consistent();
    // An alive row is reported as no finding, a deleted one as at least one.
    GPRS_CHECK_EQ(f.st.row_alive[0] == 0, result.findings > 0);
  }
  return result;
}

const std::vector<col_t> two_int = {{0., 2., true}, {0., 2., true}};

}  // namespace

GPRS_TEST(rows_empty_row)
{
  std::vector<col_t> cols = {{0., 5., true}, {0., 5., false}};
  // No entry at all.
  GPRS_CHECK_EQ(one_row(cols, {{}, -1., 1.}).findings, 1);
  GPRS_CHECK_EQ(one_row(cols, {{}, 1., 2.}).outcome, infeasible);
  GPRS_CHECK_EQ(one_row(cols, {{}, -2., -1.}).outcome, infeasible);
  // Fixed columns only: the constant is 1 + 2 * 2 = 5.
  row_t row = {{{0, 1.}, {1, 2.}}, -inf, 5.};
  GPRS_CHECK_EQ(one_row(cols, row, {1., 2.}).findings, 1);
  row.upper = 5. - 5e-7;
  GPRS_CHECK_EQ(one_row(cols, row, {1., 2.}).findings, 1);
  row.upper = 5. - 2e-6;
  GPRS_CHECK_EQ(one_row(cols, row, {1., 2.}).outcome, infeasible);
  row = {{{0, 1.}, {1, 2.}}, 5. + 5e-7, inf};
  GPRS_CHECK_EQ(one_row(cols, row, {1., 2.}).findings, 1);
  row.lower = 5. + 2e-6;
  GPRS_CHECK_EQ(one_row(cols, row, {1., 2.}).outcome, infeasible);
  row = {{{0, 1.}, {1, 2.}}, 4., 6.};
  GPRS_CHECK_EQ(one_row(cols, row, {1., 2.}).findings, 1);
}

GPRS_TEST(rows_infeasible_row)
{
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, 1.}}, -inf, -1.}).outcome, infeasible);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, 1.}}, 5., inf}).outcome, infeasible);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, -1.}}, -inf, -3.}).outcome, infeasible);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, -1.}}, 2.5, 8.}).outcome, infeasible);
  // Beyond the side by less than the tolerance, and by more.
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, 1.}}, -inf, -5e-7}).outcome, ok);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, 1.}}, -inf, -2e-6}).outcome, infeasible);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, 1.}}, 4. + 5e-7, inf}).outcome, ok);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, 1.}}, 4. + 2e-6, inf}).outcome, infeasible);
  // An infinite term in the activity that the side needs: nothing is known.
  std::vector<col_t> open_below = {{-inf, 2., true}, {0., 2., true}};
  rule_result_t result          = one_row(open_below, {{{0, 1.}, {1, 1.}}, -inf, -100.});
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 0);
  std::vector<col_t> open_above = {{0., inf, true}, {0., 2., true}};
  result                        = one_row(open_above, {{{0, 1.}, {1, 1.}}, 100., inf});
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 0);
  // -x + y <= -3 with x open below: only the maximum activity has the infinite term.
  GPRS_CHECK_EQ(one_row(open_below, {{{0, -1.}, {1, 1.}}, -inf, -3.}).outcome, infeasible);
}

GPRS_TEST(rows_redundant_row)
{
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, 1.}}, -inf, 4.}).findings, 1);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, 1.}}, -inf, 4. - 5e-7}).findings, 1);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, -1.}}, -2., inf}).findings, 1);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, -1.}}, -2., 2.}).findings, 1);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, 1.}}, -inf, inf}).findings, 1);
  // One side redundant, the other not.
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, -1.}}, -2., 1.}).findings, 0);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, -1.}}, -1., 2.}).findings, 0);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, 1.}}, -inf, 3.5}).findings, 0);
  // The maximum activity has an infinite term.
  std::vector<col_t> open_above = {{0., inf, true}, {0., 2., true}};
  GPRS_CHECK_EQ(one_row(open_above, {{{0, 1.}, {1, 1.}}, -inf, 1000.}).findings, 0);
  GPRS_CHECK_EQ(one_row(open_above, {{{0, 1.}, {1, 1.}}, 0., inf}).findings, 1);

  // A redundant row with one entry is deleted before it is read as bounds: x <= 2 - 5e-7 is
  // redundant within the tolerance and leaves the upper bound 2 of the continuous x.
  std::vector<col_t> x = {{0., 2., false}};
  fixture_t f(x,
              {{{{0, 1.}}, -3., 10.}, {{{0, 1.}}, -inf, 2. - 5e-7}, {{{0, -2.}}, -4. + 5e-7, inf}});
  rule_result_t result = run_rows(f);
  GPRS_CHECK_EQ(result.findings, 3);
  GPRS_CHECK_EQ(f.st.live_rows, 0);
  GPRS_CHECK_EQ(f.st.bound_changes, 0);
  GPRS_CHECK_EQ(f.st.col_upper[0], 2.);
}

GPRS_TEST(rows_singleton_row)
{
  std::vector<col_t> cols = {
    {0., 10., false}, {-5., 5., false}, {0., 10., true}, {0., 10., true}, {0., 10., false}};
  fixture_t f(cols,
              {{{{0, 2.}}, -inf, 6.},
               {{{1, -2.}}, -4., 6.},
               {{{2, 3.}}, 4., 20.},
               {{{3, 1.}, {4, 2.}}, 7., 7.}});
  fix_column(f.st, f.ps, 4, 2.);
  rule_result_t result = run_rows(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK(!result.stopped);
  GPRS_CHECK_EQ(result.findings, 11);
  GPRS_CHECK_EQ(f.st.live_rows, 0);
  GPRS_CHECK_EQ(f.st.col_lower[0], 0.);
  GPRS_CHECK_EQ(f.st.col_upper[0], 3.);
  GPRS_CHECK_EQ(f.st.col_lower[1], -3.);
  GPRS_CHECK_EQ(f.st.col_upper[1], 2.);
  GPRS_CHECK_EQ(f.st.col_lower[2], 2.);
  GPRS_CHECK_EQ(f.st.col_upper[2], 6.);
  GPRS_CHECK(f.st.col_status[3] == col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.col_lower[3], 3.);
  GPRS_REQUIRE(f.ps.fixed_cols.size() == 2);
  GPRS_CHECK_EQ(f.ps.fixed_cols[1].col, 3);
  GPRS_CHECK_EQ(f.ps.fixed_cols[1].value, 3.);
  f.check_consistent();
}

GPRS_TEST(rows_singleton_row_integer)
{
  std::vector<col_t> z = {{0., 3., true}};
  // No integer between the bounds of the row, although the lower one alone fixes z at 3.
  GPRS_CHECK_EQ(one_row(z, {{{0, 1.}}, 2.5, 2.7}).outcome, infeasible);
  GPRS_CHECK_EQ(one_row(z, {{{0, 2.}}, 3., 3.}).outcome, infeasible);
  GPRS_CHECK_EQ(one_row(z, {{{0, -2.}}, -5.4, -5.}).outcome, infeasible);

  fixture_t f(z, {{{{0, 1.}}, 2.5, inf}});
  GPRS_CHECK_EQ(run_rows(f).findings, 2);
  GPRS_CHECK(f.st.col_status[0] == col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.col_lower[0], 3.);

  fixture_t g(z, {{{{0, 2.}}, 1.5, 4.5}});
  GPRS_CHECK_EQ(run_rows(g).findings, 3);
  GPRS_CHECK_EQ(g.st.col_lower[0], 1.);
  GPRS_CHECK_EQ(g.st.col_upper[0], 2.);
}

// 0.01 x >= 1 with x <= 99.99995 is violated by 5e-7 in the units of the row, which is feasible,
// and by 5e-5 in the units of x, which tighten_lower would reject.
GPRS_TEST(rows_singleton_row_small_coefficient)
{
  std::vector<col_t> x = {{0., 99.99995, false}};
  fixture_t f(x, {{{{0, 0.01}}, 1., inf}});
  rule_result_t result = run_rows(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(f.st.live_rows, 0);
  GPRS_CHECK(f.st.col_status[0] == col_status_t::fixed);
  GPRS_CHECK_EQ(f.st.col_lower[0], 99.99995);

  fixture_t g(x, {{{{0, -0.01}}, -inf, -1.}});
  GPRS_CHECK_EQ(run_rows(g).outcome, ok);
  GPRS_CHECK_EQ(g.st.col_lower[0], 99.99995);

  std::vector<col_t> y = {{-99.99995, 0., false}};
  fixture_t h(y, {{{{0, 0.01}}, -inf, -1.}});
  GPRS_CHECK_EQ(run_rows(h).outcome, ok);
  GPRS_CHECK(h.st.col_status[0] == col_status_t::fixed);
  GPRS_CHECK_EQ(h.st.col_upper[0], -99.99995);

  GPRS_CHECK_EQ(one_row(x, {{{0, 0.01}}, 1.01, inf}).outcome, infeasible);
}

// The bound of an integer column from a singleton row: 0.01 x0 - 0.9 x1 <= 96.92 with x1 fixed
// at 633000170 is met by x0 = 56970024992. The quotient is that number less a rounding error of
// 1e-5, ten times the feasibility tolerance, and rounded as it is would cut the point off.
GPRS_TEST(rows_singleton_row_large_bound)
{
  std::vector<col_t> cols = {{0., inf, true}, {0., 700000000., true}};
  fixture_t f(cols, {{{{0, 0.01}, {1, -0.9}}, -inf, 96.92}});
  fix_column(f.st, f.ps, 1, 633000170.);
  GPRS_CHECK(upper_excess(f.st, 0, {56970024992., 633000170.}) <= 0.);
  GPRS_CHECK(upper_excess(f.st, 0, {56970024993., 633000170.}) > 1e-3);
  rule_result_t result = run_rows(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(f.st.live_rows, 0);
  GPRS_CHECK_EQ(f.st.col_upper[0], 56970024992.);

  // The same row in -x0: a lower bound.
  std::vector<col_t> mirrored = {{-inf, 0., true}, {0., 700000000., true}};
  fixture_t g(mirrored, {{{{0, -0.01}, {1, -0.9}}, -inf, 96.92}});
  fix_column(g.st, g.ps, 1, 633000170.);
  GPRS_CHECK_EQ(run_rows(g).outcome, ok);
  GPRS_CHECK_EQ(g.st.col_lower[0], -56970024992.);
}

// a x + w1 - w2 with w1 <= 1e20 and w2 >= 1e20: the two large terms cancel in the maximum
// activity and leave it with an error bound of 3.7e-12, while the minimum activity is infinite.
// A test that holds only without that allowance is not applied. The row times -1 does the same
// for the minimum activity.
GPRS_TEST(rows_activity_error_allowance)
{
  std::vector<col_t> cols = {{0., 2., true}, {0., 1e20, false}, {1e20, 2e20, false}};
  for (double sign : {1., -1.}) {
    // The row sign * (a x + w1 - w2), from the sides of the row with sign 1.
    auto make_row = [&](double a, double lower, double upper) {
      row_t row = {{{0, sign * a}, {1, sign}, {2, -sign}}, lower, upper};
      if (sign < 0) { row.lower = -upper, row.upper = -lower; }
      return row;
    };
    {
      fixture_t f(cols, {make_row(1., -inf, inf)});
      const activity_t& finite = sign > 0 ? f.st.max_act : f.st.min_act;
      const activity_t& other  = sign > 0 ? f.st.min_act : f.st.max_act;
      GPRS_CHECK_EQ(finite.inf_count[0], 0);
      GPRS_CHECK_EQ(other.inf_count[0], 1);
      GPRS_CHECK_EQ(finite.hi[0] + finite.lo[0], sign * 2.);
      double error = sign > 0 ? f.st.max_activity_error(0) : f.st.min_activity_error(0);
      GPRS_CHECK(error > 3e-12 && error < 5e-12);
    }
    // Redundant: the activity 2 is above the upper side by 1e-6 at most.
    GPRS_CHECK_EQ(one_row(cols, make_row(1., -inf, 2. - 1e-6 + 1e-11)).findings, 1);
    GPRS_CHECK_EQ(one_row(cols, make_row(1., -inf, 2. - 1e-6 + 2e-12)).findings, 0);
    // Infeasible: the activity is below the lower side by more than 1e-6.
    GPRS_CHECK_EQ(one_row(cols, make_row(1., 2. + 1e-6 + 1e-11, inf)).outcome, infeasible);
    rule_result_t result = one_row(cols, make_row(1., 2. + 1e-6 + 2e-12, inf));
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, 0);
    // Forcing: with the coefficient 0.5 the tolerance is 5e-7 and the activity is 1.
    GPRS_CHECK_EQ(one_row(cols, make_row(0.5, 1. + 5e-7 - 1e-11, inf)).findings, 4);
    GPRS_CHECK_EQ(one_row(cols, make_row(0.5, 1. + 5e-7 - 2e-12, inf)).findings, 0);
  }
}

GPRS_TEST(rows_forcing_row)
{
  std::vector<col_t> cols = {{0., 2., true}, {1., 3., false}, {-1., 4., true}};
  {
    // The minimum activity 0 + 1 - 4 meets the upper side.
    fixture_t f(cols, {{{{0, 1.}, {1, 1.}, {2, -1.}}, -inf, -3.}});
    rule_result_t result = run_rows(f);
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, 4);
    GPRS_CHECK_EQ(f.st.live_rows, 0);
    GPRS_CHECK_EQ(f.st.live_cols, 0);
    GPRS_REQUIRE(f.ps.fixed_cols.size() == 3);
    GPRS_CHECK_EQ(f.ps.fixed_cols[0].value, 0.);
    GPRS_CHECK_EQ(f.ps.fixed_cols[1].value, 1.);
    GPRS_CHECK_EQ(f.ps.fixed_cols[2].value, 4.);
    GPRS_CHECK_EQ(f.ps.fixed_cols[2].col, 2);
    f.check_consistent();
  }
  {
    // The maximum activity 2 + 3 + 1 meets the lower side; the upper side is not in the way.
    fixture_t f(cols, {{{{0, 1.}, {1, 1.}, {2, -1.}}, 6., 9.}});
    rule_result_t result = run_rows(f);
    GPRS_CHECK_EQ(result.findings, 4);
    GPRS_REQUIRE(f.ps.fixed_cols.size() == 3);
    GPRS_CHECK_EQ(f.ps.fixed_cols[0].value, 2.);
    GPRS_CHECK_EQ(f.ps.fixed_cols[1].value, 3.);
    GPRS_CHECK_EQ(f.ps.fixed_cols[2].value, -1.);
    f.check_consistent();
  }
  {
    // A fixed column is a constant of the row: x fixed at 1, then 1 + 1 - 4 = -2.
    fixture_t f(cols, {{{{0, 1.}, {1, 1.}, {2, -1.}}, -inf, -2.}});
    fix_column(f.st, f.ps, 0, 1.);
    GPRS_CHECK_EQ(run_rows(f).findings, 3);
    GPRS_CHECK_EQ(f.st.live_cols, 0);
    GPRS_CHECK_EQ(f.st.col_lower[1], 1.);
    GPRS_CHECK_EQ(f.st.col_lower[2], 4.);
  }
  // Half a unit away from the side.
  GPRS_CHECK_EQ(one_row(cols, {{{0, 1.}, {1, 1.}, {2, -1.}}, -inf, -2.5}).findings, 0);
  GPRS_CHECK_EQ(one_row(cols, {{{0, 1.}, {1, 1.}, {2, -1.}}, 5.5, inf}).findings, 0);
  // Within the tolerance of the side, with coefficients of one size.
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1.}, {1, 1.}}, 4. + 5e-7, inf}).findings, 3);
  GPRS_CHECK_EQ(one_row(two_int, {{{0, 1000.}, {1, 1000.}}, -inf, 0.}).findings, 3);
}

// x1 + 1e-7 x2 <= 1e-7 leaves x2 free in [0, 1] although the minimum activity is within the
// feasibility tolerance of the side: the tolerance is divided by the dynamism of the row.
GPRS_TEST(rows_forcing_row_dynamism)
{
  std::vector<col_t> unit = {{0., 1., false}, {0., 1., false}};
  rule_result_t result    = one_row(unit, {{{0, 1.}, {1, 1e-7}}, -inf, 1e-7});
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 0);
  result = one_row(unit, {{{0, -1.}, {1, -1e-7}}, -1e-7, inf});
  GPRS_CHECK_EQ(result.findings, 0);
  // The same row with the side at the activity is forcing.
  GPRS_CHECK_EQ(one_row(unit, {{{0, 1.}, {1, 1e-7}}, -inf, 0.}).findings, 3);
}

// 1e-7 x1 + 1e-7 x2 <= 1e-7 has dynamism 1 and leaves (0, 1) feasible: the tolerance is also
// divided by the size of the coefficients when they are below 1.
GPRS_TEST(rows_forcing_row_small_coefficients)
{
  std::vector<col_t> cols = {{0., 100., false}, {0., 1., false}};
  rule_result_t result    = one_row(cols, {{{0, 1e-7}, {1, 1e-7}}, -inf, 1e-7});
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK_EQ(one_row(cols, {{{0, 1e-7}, {1, 1e-7}}, -inf, 0.}).findings, 3);
}

// The bound from a singleton row makes the row before it redundant, and that one is on the list
// again within the call.
GPRS_TEST(rows_cascade)
{
  std::vector<col_t> cols = {{0., 10., false}, {0., 2., false}, {0., 9., true}};
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1.}}, -inf, 5.},
                             {{{0, 1.}}, -inf, 3.},
                             {{{0, 1.}, {1, 1.}, {2, 1.}}, 13., inf},
                             {{{2, 1.}}, -inf, 8.5}};
  // First pass: row 1 gives x <= 3 and row 3 z <= 8. Second pass: row 0 is redundant and row 2,
  // with the maximum activity 3 + 2 + 8, is forcing.
  fixture_t f(cols, rows);
  rule_result_t result = run_rows(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK(!result.stopped);
  GPRS_CHECK_EQ(result.findings, 9);
  GPRS_CHECK_EQ(f.st.live_rows, 0);
  GPRS_CHECK_EQ(f.st.live_cols, 0);
  GPRS_CHECK_EQ(f.st.col_lower[0], 3.);
  GPRS_CHECK_EQ(f.st.col_lower[1], 2.);
  GPRS_CHECK_EQ(f.st.col_lower[2], 8.);
  GPRS_CHECK(f.st.row_frontier.empty(row_list::row_rules));
  f.check_consistent();

  // With 14 for 13 the second pass finds row 2 infeasible.
  rows[2].lower = 14.;
  fixture_t g(cols, rows);
  GPRS_CHECK_EQ(run_rows(g).outcome, infeasible);
}

GPRS_TEST(rows_budget_stop)
{
  std::vector<row_t> rows(6, row_t{{{0, 1.}, {1, 1.}}, -inf, 4.});
  fixture_t f(two_int, rows);
  // A redundant row costs line_visit_work: the fourth row is the first one not visited.
  f.budget.limit       = 2 * line_visit_work + 4;
  rule_result_t result = run_rows(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK(result.stopped);
  GPRS_CHECK_EQ(result.findings, 3);
  GPRS_CHECK_EQ(f.budget.used, 3 * line_visit_work);
  GPRS_CHECK_EQ(f.budget.used_at_finding, 3 * line_visit_work);
  GPRS_CHECK_EQ(f.st.live_rows, 3);
  GPRS_CHECK(f.st.row_frontier.list[row_list::row_rules] == std::vector<int32_t>({3, 4, 5}));
  f.budget = work_budget_t{unlimited, unlimited, 0, 0};
  result   = run_rows(f);
  GPRS_CHECK(!result.stopped);
  GPRS_CHECK_EQ(result.findings, 3);
  GPRS_CHECK_EQ(f.st.live_rows, 0);

  // Rows without a reduction: the stall limit ends the call.
  std::vector<row_t> stay(5, row_t{{{0, 1.}, {1, 1.}}, -inf, 3.});
  fixture_t g(two_int, stay);
  g.budget.stall_limit = line_visit_work + 1;
  result               = run_rows(g);
  GPRS_CHECK(result.stopped);
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK_EQ(g.budget.used, 2 * line_visit_work);
  GPRS_CHECK(g.st.row_frontier.list[row_list::row_rules] == std::vector<int32_t>({2, 3, 4}));
}

GPRS_TEST(rows_list_only)
{
  std::vector<row_t> rows(4, row_t{{{0, 1.}, {1, 1.}}, -inf, 4.});
  fixture_t f(two_int, rows);
  f.st.delete_row(2);
  f.only(row_list::row_rules, {2, 3});
  rule_result_t result = run_rows(f);
  // Row 2 is dead and skipped, rows 0 and 1 are not on the list.
  GPRS_CHECK_EQ(result.findings, 1);
  GPRS_CHECK_EQ(f.st.live_rows, 2);
  GPRS_CHECK(f.st.row_alive[0] && f.st.row_alive[1] && !f.st.row_alive[3]);
  f.check_consistent();
}

GPRS_TEST(rows_property_feasible_set)
{
  int64_t findings = 0, infeasible_models = 0, fixed_cols = 0, deleted_rows = 0;
  std::vector<col_t> cols;
  std::vector<row_t> rows;
  for (uint64_t seed = 1; seed <= 1000; ++seed) {
    rng_t rng(seed);
    random_model(rng, random_options_t(), cols, rows);
    fixture_t f(cols, rows);
    for (int32_t j = 0; j < f.st.ncols; ++j) {
      if (cols[j].lower > -inf && rng.chance(0.2)) { fix_column(f.st, f.ps, j, cols[j].lower); }
    }
    state_t before       = f.st;
    rule_result_t result = run_rows(f);
    check_same_set(before, f.st, result.outcome);
    if (result.outcome != ok) {
      ++infeasible_models;
      continue;
    }
    GPRS_CHECK(!result.stopped);
    GPRS_CHECK(f.st.row_frontier.empty(row_list::row_rules));
    findings += result.findings;
    fixed_cols += before.live_cols - f.st.live_cols;
    deleted_rows += before.live_rows - f.st.live_rows;
  }
  // The models reach every outcome.
  GPRS_CHECK(findings > 500);
  GPRS_CHECK(infeasible_models > 100);
  GPRS_CHECK(fixed_cols > 30);
  GPRS_CHECK(deleted_rows > 400);
}

// The three row families in the order of the cascade, repeated until none finds anything, then
// the recorded bounds made explicit: the feasible set is the same at the end.
GPRS_TEST(rows_property_cascade)
{
  int64_t infeasible_models = 0, emptied = 0, passes = 0;
  std::vector<col_t> cols;
  std::vector<row_t> rows;
  for (uint64_t seed = 1; seed <= 1000; ++seed) {
    rng_t rng(seed);
    random_options_t options;
    options.integer_share = seed % 3 == 0 ? 1. : 0.7;
    options.decimal       = seed % 2 == 0;
    random_model(rng, options, cols, rows);
    fixture_t f(cols, rows);
    state_t before = f.st;
    row_rules_t row_rules;
    bound_propagation_t propagation;
    row_tightening_t tightening;
    rule_result_t result;
    bool busy = true;
    for (int32_t pass = 0; pass < 50 && busy && result.outcome == ok; ++pass) {
      busy = false;
      for (int32_t family = 0; family < 3 && result.outcome == ok; ++family) {
        // A budget per call: propagation between integer columns without bounds on one side
        // can go on for as many steps as the budget allows.
        work_budget_t budget{200000, unlimited, 0, 0};
        result = family == 0   ? row_rules.run(f.st, budget, f.ps, f.opt)
                 : family == 1 ? propagation.run(f.st, budget, f.ps, f.opt)
                               : tightening.run(f.st, budget, f.ps, f.opt);
        busy   = busy || result.findings > 0;
      }
      ++passes;
    }
    if (result.outcome == ok) { result = propagation.make_explicit(f.st, f.ps, f.opt); }
    check_same_set(before, f.st, result.outcome);
    infeasible_models += result.outcome != ok;
    emptied += result.outcome == ok && f.st.live_rows == 0;
  }
  GPRS_CHECK(infeasible_models > 200);
  GPRS_CHECK(emptied > 100);
  GPRS_CHECK(passes > 1200);
}

// Every live row without a free entry is deleted, whatever the list holds and without a
// budget; a row whose fixed terms violate a side gives infeasible.
GPRS_TEST(rows_remove_empty_rows)
{
  std::vector<col_t> cols = {{0., 5., true}, {0., 5., false}, {0., 5., true}};
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1.}}, -inf, 7.},
                             {{{1, 2.}, {2, 1.}}, 1., inf},
                             {{}, -1., 1.},
                             {{{0, 1.}}, 2., 3.}};
  {
    fixture_t f(cols, rows);
    fix_column(f.st, f.ps, 0, 3.);
    fix_column(f.st, f.ps, 1, 2.);
    f.only(row_list::row_rules, {});
    f.budget.limit = 0;
    row_rules_t rule;
    state_t before       = f.st;
    rule_result_t result = rule.remove_empty_rows(f.st, f.ps, f.opt);
    GPRS_CHECK_EQ(result.outcome, ok);
    // Rows 0, 2 and 3; row 1 has the free column x2.
    GPRS_CHECK_EQ(result.findings, 3);
    GPRS_CHECK(!result.stopped);
    GPRS_CHECK(f.st.row_alive == std::vector<uint8_t>({0, 1, 0, 0}));
    GPRS_CHECK_EQ(f.st.bound_changes, before.bound_changes);
    check_same_set(before, f.st, result.outcome);
    // A second call finds none.
    GPRS_CHECK_EQ(rule.remove_empty_rows(f.st, f.ps, f.opt).findings, 0);
  }
  {
    // x0 = 4 violates the upper side of row 3.
    fixture_t f(cols, rows);
    fix_column(f.st, f.ps, 0, 4.);
    row_rules_t rule;
    GPRS_CHECK_EQ(rule.remove_empty_rows(f.st, f.ps, f.opt).outcome, infeasible);
  }
}

// 1e7 x0 + x1 + x2 <= 0 with all lower bounds 0 is forcing within 1e-6 * 1 / 1e7. x2 had the
// lower bound -1e19, a term that the minimum activity no longer holds but whose size was still
// in its error bound, 6e-13: the row was not taken as forcing here and was by a presolve that
// starts from these bounds.
GPRS_TEST(rows_forcing_row_after_a_huge_term_left)
{
  std::vector<col_t> cols = {{0., 1., true}, {0., 1., true}, {-1e19, 5., false}};
  fixture_t f(cols, {{{{0, 1e7}, {1, 1.}, {2, 1.}}, -inf, 0.}});
  f.st.set_col_lower(2, 0.);
  GPRS_CHECK(f.st.min_activity_error(0) > 1e-13);
  GPRS_CHECK(!activity_fresh(f.st, f.st.min_act, 0));
  state_t before       = f.st;
  rule_result_t result = run_rows(f);
  GPRS_CHECK_EQ(result.outcome, ok);
  GPRS_CHECK_EQ(result.findings, 4);
  GPRS_CHECK(!f.st.row_alive[0]);
  GPRS_CHECK_EQ(f.st.live_cols, 0);
  check_same_set(before, f.st, result.outcome);
}
