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

#include <cmath>
#include <vector>

using namespace gprs;
using namespace rule_test;

namespace {

constexpr rule_outcome_t ok         = rule_outcome_t::ok;
constexpr rule_outcome_t infeasible = rule_outcome_t::infeasible;

rule_result_t run_tightening(fixture_t& f)
{
  row_tightening_t rule;
  return rule.run(f.st, f.budget, f.ps, f.opt);
}

const std::vector<col_t> two_int  = {{0., 10., true}, {0., 10., true}};
const std::vector<col_t> two_bin  = {{0., 1., true}, {0., 1., true}};
const std::vector<col_t> with_one = {{0., 10., true}, {0., 10., true}, {0., 5., false}};

// The row after the rule: coefficients by column, sides, findings.
struct edited_t {
  rule_outcome_t outcome;
  int64_t findings;
  std::vector<double> a;
  double lower;
  double upper;
};

// Runs the rule on one row; the columns from `first_fixed` on are fixed at 1 before.
edited_t edit(const std::vector<col_t>& cols, const row_t& row, size_t first_fixed = 1000)
{
  fixture_t f(cols, {row});
  for (size_t j = first_fixed; j < cols.size(); ++j) {
    fix_column(f.st, f.ps, j, 1.);
  }
  // A rule that edits its row at every visit ends on this budget and fails the check below.
  f.budget.limit       = 100000;
  state_t before       = f.st;
  rule_result_t result = run_tightening(f);
  edited_t edited{result.outcome, result.findings, {}, f.st.row_lower[0], f.st.row_upper[0]};
  for (int32_t j = 0; j < f.st.ncols; ++j) {
    edited.a.push_back(coefficient(f.st, 0, j));
  }
  if (result.outcome == ok) {
    GPRS_CHECK(!result.stopped);
    GPRS_CHECK(f.st.row_alive[0]);
    GPRS_CHECK(f.st.row_frontier.empty(row_list::tightening));
    GPRS_CHECK_EQ(f.st.bound_changes, before.bound_changes);
  }
  // Infeasible outcomes of the cases below are checked on their grid as well.
  check_same_set(before, f.st, result.outcome);
  return edited;
}

void check_row(
  const edited_t& e, int64_t findings, const std::vector<double>& a, double lower, double upper)
{
  GPRS_CHECK_EQ(e.outcome, ok);
  GPRS_CHECK_EQ(e.findings, findings);
  GPRS_CHECK(e.a == a);
  GPRS_CHECK_EQ(e.lower, lower);
  GPRS_CHECK_EQ(e.upper, upper);
}

// Lower side of the free part of row 0, whose columns from `first_fixed` on are fixed, in units
// of the coefficient of column 0.
double free_lower(const state_t& st, int32_t first_fixed)
{
  dd_t fixed;
  for (int32_t j = first_fixed; j < st.ncols; ++j) {
    dd_add_product(fixed, coefficient(st, 0, j), st.col_lower[j]);
  }
  return dd_residual(st.row_lower[0], fixed) / coefficient(st, 0, 0);
}

}  // namespace

GPRS_TEST(tightening_integral_scale)
{
  // Halves: the factor 2, then the side 5.5 is rounded.
  check_row(edit(two_int, {{{0, 0.5}, {1, 1.5}}, -inf, 2.75}), 2, {1., 3.}, -inf, 5.);
  // The gcd 2, then Chvatal-Gomory rounding of 7.5.
  check_row(edit(two_int, {{{0, 4.}, {1, 6.}}, -inf, 15.}), 2, {2., 3.}, -inf, 7.);
  check_row(edit(two_int, {{{0, 4.}, {1, -6.}}, -15., inf}), 2, {2., -3.}, -7., inf);
  // Tenths: the factor 10.
  check_row(edit(two_int, {{{0, 0.1}, {1, 0.3}}, 0.25, inf}), 2, {1., 3.}, 3., inf);
  // Scale and gcd together: 1.5 and 4.5 become 3 and 9, then 1 and 3.
  check_row(edit(two_int, {{{0, 1.5}, {1, 4.5}}, -inf, 10.}), 2, {1., 3.}, -inf, 6.);
  // Sevenths and elevenths: the factor 77, the products are integers only after rounding.
  check_row(edit(two_int, {{{0, 1. / 7.}, {1, 1. / 11.}}, -inf, 1.}), 1, {11., 7.}, -inf, 77.);
  // Nothing to do.
  check_row(edit(two_int, {{{0, 1.}, {1, 2.}}, -inf, 4.}), 0, {1., 2.}, -inf, 4.);
  check_row(edit(two_int, {{{0, 3.}, {1, -2.}}, -4., 7.}), 0, {3., -2.}, -4., 7.);
  // An equation divided by its gcd keeps integral sides: one finding, the scaling.
  check_row(edit(two_int, {{{0, 2.}, {1, 2.}}, 4., 4.}), 1, {1., 1.}, 2., 2.);
}

GPRS_TEST(tightening_infeasible)
{
  GPRS_CHECK_EQ(edit(two_int, {{{0, 2.}, {1, 2.}}, 3., 3.}).outcome, infeasible);
  GPRS_CHECK_EQ(edit(two_int, {{{0, 1.}, {1, 1.}}, 1.2, 1.8}).outcome, infeasible);
  GPRS_CHECK_EQ(edit(two_int, {{{0, 0.5}, {1, 0.5}}, 0.6, 0.9}).outcome, infeasible);
  GPRS_CHECK_EQ(edit(two_int, {{{0, 3.}, {1, 6.}}, 4., 5.}).outcome, infeasible);
  // With a fixed column: x + y + 0.5 = 2.
  GPRS_CHECK_EQ(edit(with_one, {{{0, 1.}, {1, 1.}, {2, 0.5}}, 2., 2.}, 2).outcome, infeasible);
  check_row(
    edit(with_one, {{{0, 1.}, {1, 1.}, {2, 0.5}}, 2.5, 2.5}, 2), 0, {1., 1., 0.5}, 2.5, 2.5);
}

// Both sides move: the order of the two calls must keep lower <= upper in between, which the
// state asserts.
GPRS_TEST(tightening_both_sides)
{
  check_row(edit(two_int, {{{0, 2.}, {1, 2.}}, 3., 5.}), 3, {1., 1.}, 2., 2.);
  check_row(edit(two_int, {{{0, 1.}, {1, 1.}}, 2.9999995, 2.9999995}), 2, {1., 1.}, 3., 3.);
  check_row(edit(two_int, {{{0, 1.}, {1, 1.}}, 3.0000004, 3.0000004}), 2, {1., 1.}, 3., 3.);
  check_row(edit(two_int, {{{0, 1.}, {1, 1.}}, 0.5, 3.5}), 2, {1., 1.}, 1., 3.);
  check_row(edit(two_int, {{{0, 1.}, {1, 1.}}, 2.2, 3.1}), 2, {1., 1.}, 3., 3.);
}

GPRS_TEST(tightening_side_tolerance)
{
  // Within the feasibility tolerance of an integer.
  check_row(edit(two_int, {{{0, 1.}, {1, 1.}}, -inf, 2.9999995}), 1, {1., 1.}, -inf, 3.);
  check_row(edit(two_int, {{{0, 1.}, {1, 1.}}, -inf, 2.999}), 1, {1., 1.}, -inf, 2.);
  check_row(edit(two_int, {{{0, 1.}, {1, 1.}}, 2.0000004, inf}), 1, {1., 1.}, 2., inf);
  check_row(edit(two_int, {{{0, 1.}, {1, 1.}}, 2.001, inf}), 1, {1., 1.}, 3., inf);
  // The tolerance belongs to the row with integral coefficients, before the division by the
  // gcd: 2999999.5 is not 3000000, although 2.9999995 is 3 within the tolerance.
  check_row(edit(two_int, {{{0, 1e6}, {1, 2e6}}, -inf, 2999999.5}), 2, {1., 2.}, -inf, 2.);
  check_row(edit(two_int, {{{0, 1e6}, {1, 2e6}}, -inf, 2999999.9999995}), 2, {1., 2.}, -inf, 3.);
  check_row(edit(two_int, {{{0, 1e6}, {1, 2e6}}, 2000000.5, inf}), 2, {1., 2.}, 3., inf);
}

// The terms of fixed columns are taken off the sides before the rounding and put back after;
// their coefficients are scaled, not rounded.
GPRS_TEST(tightening_fixed_terms)
{
  check_row(
    edit(with_one, {{{0, 1.}, {1, 1.}, {2, 0.5}}, -inf, 3.2}, 2), 1, {1., 1., 0.5}, -inf, 2.5);
  check_row(
    edit(with_one, {{{0, 1.}, {1, 1.}, {2, 0.5}}, 1.7, inf}, 2), 1, {1., 1., 0.5}, 2.5, inf);
  // Within the tolerance of an integer the side stays as it is.
  check_row(edit(with_one, {{{0, 1.}, {1, 1.}, {2, 0.5}}, -inf, 2.5000004}, 2),
            0,
            {1., 1., 0.5},
            -inf,
            2.5000004);
  // One side moves: both are rewritten, or 2.5000004 would be above the new upper side 2.5.
  check_row(
    edit(with_one, {{{0, 1.}, {1, 1.}, {2, 0.5}}, 2.5000004, 3.2}, 2), 2, {1., 1., 0.5}, 2.5, 2.5);
  check_row(edit(with_one, {{{0, 0.5}, {1, 0.5}, {2, 0.3}}, -inf, 2.}, 2),
            2,
            {1., 1., 0.6},
            -inf,
            3. + 0.6);
  // A row whose only continuous column is fixed counts as an integer row; one with a free
  // continuous column does not.
  check_row(
    edit(with_one, {{{0, 0.5}, {1, 1.5}, {2, 1.}}, -inf, 2.75}), 0, {0.5, 1.5, 1.}, -inf, 2.75);
}

GPRS_TEST(tightening_no_integral_scale)
{
  double root = std::sqrt(2.);
  check_row(edit(two_int, {{{0, 1.}, {1, root}}, -inf, 3.5}), 0, {1., root}, -inf, 3.5);
  // The factor 1024 is above the limit.
  check_row(edit(two_int, {{{0, 1. / 1024.}, {1, 1.}}, -inf, 3.5}), 0, {1. / 1024., 1.}, -inf, 3.5);
  // So is 7 * 11 * 13.
  std::vector<col_t> three(3, col_t{0., 10., true});
  check_row(edit(three, {{{0, 1. / 7.}, {1, 1. / 11.}, {2, 1. / 13.}}, -inf, 1.5}),
            0,
            {1. / 7., 1. / 11., 1. / 13.},
            -inf,
            1.5);
  // 1e-8 away from an integer is not integral.
  check_row(edit(two_int, {{{0, 2.00000001}, {1, 1.}}, -inf, 7.5}), 0, {2.00000001, 1.}, -inf, 7.5);
  // A coefficient that rounds to zero.
  check_row(edit(two_int, {{{0, 1e-10}, {1, 1.}}, -inf, 7.5}), 0, {1e-10, 1.}, -inf, 7.5);
}

// A coefficient 5e-10 away from 2 is rounded only if that moves the activity by at most a
// tenth of the feasibility tolerance over the bounds of its column.
GPRS_TEST(tightening_rounding_guard)
{
  double a = 2. + 5e-10;
  check_row(edit(two_int, {{{0, a}, {1, 1.}}, -inf, 7.5}), 2, {2., 1.}, -inf, 7.);
  std::vector<col_t> wide = {{0., 1000., true}, {0., 10., true}};
  check_row(edit(wide, {{{0, a}, {1, 1.}}, -inf, 7.5}), 0, {a, 1.}, -inf, 7.5);
  std::vector<col_t> open = {{0., inf, true}, {0., 10., true}};
  check_row(edit(open, {{{0, a}, {1, 1.}}, -inf, 7.5}), 0, {a, 1.}, -inf, 7.5);
  // An exact integer needs no bound.
  check_row(edit(open, {{{0, 2.}, {1, 1.}}, -inf, 7.5}), 1, {2., 1.}, -inf, 7.);
}

GPRS_TEST(tightening_strengthening)
{
  // 3x + 2y <= 4 over binaries: M = 5, d = 1.
  check_row(edit(two_bin, {{{0, 3.}, {1, 2.}}, -inf, 4.}), 3, {1., 1.}, -inf, 1.);
  // >= with lower bounds 0: the side stays.
  check_row(edit(two_bin, {{{0, 3.}, {1, 2.}}, 1., inf}), 2, {1., 1.}, 1., inf);
  // A big-M row with a continuous column: -3x + y <= 0 with y <= 2.
  std::vector<col_t> mixed = {{0., 1., true}, {0., 2., false}};
  check_row(edit(mixed, {{{0, -3.}, {1, 1.}}, -inf, 0.}), 1, {-2., 1.}, -inf, 0.);
  // Bounds other than 0. -3x + y <= -2 with x in [1, 2]: M = -1, d = 1, side -2 + 3 - 1.
  std::vector<col_t> shifted = {{1., 2., true}, {0., 2., false}};
  check_row(edit(shifted, {{{0, -3.}, {1, 1.}}, -inf, -2.}), 2, {-1., 1.}, -inf, 0.);
  // 3x + y >= 5 with x in [1, 3], y in [0, 3]: m = 3, d = 2, side 5 - 3 + 2.
  std::vector<col_t> lifted = {{1., 3., true}, {0., 3., false}};
  check_row(edit(lifted, {{{0, 3.}, {1, 1.}}, 5., inf}), 2, {2., 1.}, 4., inf);
  // -4x - y >= -13: the upper bound 3 enters for the negative coefficient, m = -15, d = 2,
  // side -13 + 4 * 3 - 2 * 3.
  check_row(edit(lifted, {{{0, -4.}, {1, -1.}}, -13., inf}), 2, {-2., -1.}, -7., inf);
}

GPRS_TEST(tightening_strengthening_guards)
{
  // A continuous column keeps its coefficient.
  std::vector<col_t> cont_first = {{0., 1., false}, {0., 2., true}};
  check_row(edit(cont_first, {{{0, -3.}, {1, 1.}}, -inf, 0.}), 0, {-3., 1.}, -inf, 0.);
  // Two finite sides.
  check_row(edit(two_bin, {{{0, 3.}, {1, 2.}}, -5., 4.}), 0, {3., 2.}, -5., 4.);
  // An infinite maximum activity.
  std::vector<col_t> open = {{0., 1., true}, {0., inf, true}};
  check_row(edit(open, {{{0, 3.}, {1, 2.}}, -inf, 4.}), 0, {3., 2.}, -inf, 4.);
  // M - b = 0 and M - b = 5e-7: the row is redundant, not strengthened.
  std::vector<col_t> mixed = {{0., 1., true}, {0., 1., true}, {0., 1., false}};
  check_row(edit(mixed, {{{0, 3.}, {1, 2.}, {2, 1.}}, -inf, 6.}), 0, {3., 2., 1.}, -inf, 6.);
  check_row(
    edit(mixed, {{{0, 3.}, {1, 2.}, {2, 1.}}, -inf, 6. - 5e-7}), 0, {3., 2., 1.}, -inf, 6. - 5e-7);
  // |a| = d, and |a| above d by less than the tolerance.
  check_row(edit(two_bin, {{{0, 3.}, {1, 2.}}, -inf, 2.}), 0, {3., 2.}, -inf, 2.);
  check_row(
    edit(mixed, {{{0, 3.}, {1, 2.}, {2, 1.}}, -inf, 3. + 5e-7}), 0, {3., 2., 1.}, -inf, 3. + 5e-7);
  // Above it by more: d = 3 - 2e-6 for the coefficient 3.
  edited_t e = edit(mixed, {{{0, 3.}, {1, 2.}, {2, 1.}}, -inf, 3. + 2e-6});
  GPRS_CHECK_EQ(e.findings, 2);
  GPRS_CHECK(std::fabs(e.a[0] - (3. - 2e-6)) < 1e-12);
  GPRS_CHECK_EQ(e.a[1], 2.);
  GPRS_CHECK(std::fabs(e.upper - 3.) < 1e-12);
}

// A side is one double. Next to fixed terms of 1e11 it holds the side of the free part only up
// to 1e-5, ten times the tolerance: a row edited there reads back with another integer side at
// the next visit, one unit further each time, until no point is left. A row with fixed terms
// whose sides or fixed terms are that large is not edited.
GPRS_TEST(tightening_large_fixed_terms)
{
  std::vector<col_t> cols = {{0., 10., true}, {0., 10., true}, {0., 4e13, false}, {0., 10., false}};
  for (double size : {1e11, 1e12, 1e13}) {
    for (int32_t k = 0; k < 40; ++k) {
      // 3x + 3y + z >= z + 9 with z fixed: the division by 3 rounds the side and the term of z.
      {
        double z = size + 1. + 7. * k;
        fixture_t f(cols, {{{{0, 3.}, {1, 3.}, {2, 1.}}, -inf, inf}});
        fix_column(f.st, f.ps, 2, z);
        f.st.set_row_lower(0, z + 9.);
        f.budget.limit       = 100000;
        rule_result_t result = run_tightening(f);
        GPRS_CHECK_EQ(result.outcome, ok);
        GPRS_CHECK(!result.stopped);
        GPRS_CHECK_EQ(result.findings, 0);
        GPRS_CHECK_EQ(coefficient(f.st, 0, 0), 3.);
        GPRS_CHECK_EQ(f.st.row_lower[0], z + 9.);
      }
      // x + y + 0.3 z + 0.7 w >= 2.5 + the fixed terms, which are no double: no scaling, the
      // side would be rewritten as 3 + the fixed terms.
      {
        double z = size + 1. + k, w = 0.1 * (k + 1);
        fixture_t f(cols, {{{{0, 1.}, {1, 1.}, {2, 0.3}, {3, 0.7}}, -inf, inf}});
        fix_column(f.st, f.ps, 2, z);
        fix_column(f.st, f.ps, 3, w);
        dd_t fixed;
        dd_add_product(fixed, 0.3, z);
        dd_add_product(fixed, 0.7, w);
        double lower = dd_value(fixed) + 2.5;
        f.st.set_row_lower(0, lower);
        f.budget.limit       = 100000;
        rule_result_t result = run_tightening(f);
        GPRS_CHECK_EQ(result.outcome, ok);
        GPRS_CHECK(!result.stopped);
        GPRS_CHECK_EQ(result.findings, 0);
        GPRS_CHECK_EQ(f.st.row_lower[0], lower);
      }
    }
  }
  std::vector<col_t> wide = {{0., 1e13, true}, {0., 1e13, true}, {0., 4e13, false}};
  for (int32_t k = 0; k < 40; ++k) {
    // Large sides next to a small fixed term: x + y + 0.7 w >= 1e11 + 2.5 + 0.7 w. The new
    // side 1e11 + 3 + 0.7 w is no double; the nearest one is up to 7.6e-6 away.
    {
      double w = 0.1 * (k + 1);
      fixture_t f(wide, {{{{0, 1.}, {1, 1.}, {2, 0.7}}, -inf, inf}});
      fix_column(f.st, f.ps, 2, w);
      double lower = 1e11 + 2.5 + 0.7 * w;
      f.st.set_row_lower(0, lower);
      f.budget.limit       = 100000;
      rule_result_t result = run_tightening(f);
      GPRS_CHECK(!result.stopped);
      GPRS_CHECK_EQ(result.findings, 0);
      GPRS_CHECK_EQ(f.st.row_lower[0], lower);
    }
    // A small side next to a large fixed term: x + y - 0.7 z >= 2.5, where the side of the
    // free part is the small side less a term of 7e10.
    {
      fixture_t f(wide, {{{{0, 1.}, {1, 1.}, {2, -0.7}}, 2.5, inf}});
      fix_column(f.st, f.ps, 2, 1e11 + 1. + k);
      f.budget.limit       = 100000;
      rule_result_t result = run_tightening(f);
      GPRS_CHECK(!result.stopped);
      GPRS_CHECK_EQ(result.findings, 0);
      GPRS_CHECK_EQ(f.st.row_lower[0], 2.5);
    }
  }
  // Without fixed terms the sides become the integers themselves, exactly at any size.
  check_row(edit(wide, {{{0, 3.}, {1, 3.}}, -inf, 3e11 + 7.5}), 2, {1., 1., 0.}, -inf, 1e11 + 2.);
  check_row(edit(wide, {{{0, 1.}, {1, 1.}}, 1e11 + 2.5, inf}), 1, {1., 1., 0.}, 1e11 + 3., inf);

  // Fixed terms of 1e7 leave the side exact to 1e-9: the rows are edited, once.
  for (int32_t k = 0; k < 40; ++k) {
    double z = 1e7 + 1. + 7. * k;
    fixture_t f(cols, {{{{0, 3.}, {1, 3.}, {2, 1.}}, -inf, inf}});
    fix_column(f.st, f.ps, 2, z);
    f.st.set_row_lower(0, z + 7.5);
    rule_result_t result = run_tightening(f);
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(coefficient(f.st, 0, 0), 1.);
    GPRS_CHECK(std::fabs(free_lower(f.st, 2) - 3.) < 1e-7);
    GPRS_CHECK(result.findings >= 1 && result.findings <= 3);

    double w = 0.1 * (k + 1);
    fixture_t g(cols, {{{{0, 1.}, {1, 1.}, {2, 0.3}, {3, 0.7}}, -inf, inf}});
    fix_column(g.st, g.ps, 2, z);
    fix_column(g.st, g.ps, 3, w);
    dd_t fixed;
    dd_add_product(fixed, 0.3, z);
    dd_add_product(fixed, 0.7, w);
    g.st.set_row_lower(0, dd_value(fixed) + 2.5);
    result = run_tightening(g);
    GPRS_CHECK_EQ(result.outcome, ok);
    GPRS_CHECK_EQ(result.findings, 1);
    GPRS_CHECK(std::fabs(free_lower(g.st, 2) - 3.) < 1e-7);
  }
}

// 0.3 x + 0.7 y + z <= b with x, y in [0, u] and z in [0, 1]: M - b is 0.5 and the coefficient
// 0.7 becomes 0.5 with the side b - 0.2 u. Strengthening leaves M - b as it is, but only as far
// as the new side is stored: at 8e10 a double is exact to 8e-6, more than the tolerance, and the
// row would cut feasible points off or be strengthened again and again. No side of that size is
// written.
GPRS_TEST(tightening_strengthening_large_side)
{
  for (double u : {1e4, 1e6, 1e8, 1e10, 1e11, 1e12, 1e13}) {
    for (int32_t k = 0; k < 3; ++k) {
      std::vector<col_t> cols = {{0., u, true}, {0., u + 3. * k, true}, {0., 1., false}};
      double b                = u + 0.5 + 0.7 * 3. * k;
      fixture_t f(cols, {{{{0, 0.3}, {1, 0.7}, {2, 1.}}, -inf, b}});
      f.budget.limit       = 100000;
      double d_before      = -dd_residual(b, dd_t{f.st.max_act.hi[0], f.st.max_act.lo[0]});
      rule_result_t result = run_tightening(f);
      GPRS_CHECK_EQ(result.outcome, ok);
      GPRS_CHECK(!result.stopped);
      double d_after =
        -dd_residual(f.st.row_upper[0], dd_t{f.st.max_act.hi[0], f.st.max_act.lo[0]});
      GPRS_CHECK(std::fabs(d_after - d_before) <= 0.1 * f.opt.tol.feasibility);
      GPRS_CHECK(std::fabs(d_before - 0.5) < 1e-3);
      if (u <= 1e8) {
        // The new side is about 0.8 u.
        GPRS_CHECK_EQ(result.findings, 2);
        GPRS_CHECK(std::fabs(coefficient(f.st, 0, 1) - d_before) < 1e-12);
        GPRS_CHECK(f.st.row_upper[0] < 0.81 * u + 3.);
      } else {
        GPRS_CHECK_EQ(result.findings, 0);
        GPRS_CHECK_EQ(coefficient(f.st, 0, 1), 0.7);
        GPRS_CHECK_EQ(f.st.row_upper[0], b);
      }
      f.check_consistent();
    }
  }
}

GPRS_TEST(tightening_budget_stop)
{
  // No finding; a visit reads the two entries four times.
  std::vector<row_t> rows(4, row_t{{{0, 1.}, {1, 1.}}, -inf, 4.});
  fixture_t f(two_int, rows);
  int64_t visit        = line_visit_work + 4 * 2;
  f.budget.limit       = visit + 1;
  rule_result_t result = run_tightening(f);
  GPRS_CHECK(result.stopped);
  GPRS_CHECK_EQ(result.findings, 0);
  GPRS_CHECK_EQ(f.budget.used, 2 * visit);
  GPRS_CHECK(f.st.row_frontier.list[row_list::tightening] == std::vector<int32_t>({2, 3}));

  f.budget = work_budget_t{unlimited, unlimited, 0, 0};
  f.st.delete_row(2);
  result = run_tightening(f);
  GPRS_CHECK(!result.stopped);
  GPRS_CHECK_EQ(f.budget.used, visit);
  GPRS_CHECK(f.st.row_frontier.empty(row_list::tightening));
}

GPRS_TEST(tightening_property_feasible_set)
{
  int64_t findings = 0, infeasible_models = 0, sides = 0, coefficients = 0;
  std::vector<col_t> cols;
  std::vector<row_t> rows;
  for (uint64_t seed = 1; seed <= 1500; ++seed) {
    rng_t rng(seed);
    random_options_t options;
    options.integer_share = 0.8;
    options.decimal       = seed % 2 == 0;
    random_model(rng, options, cols, rows);
    fixture_t f(cols, rows);
    for (int32_t j = 0; j < f.st.ncols; ++j) {
      if (cols[j].lower > -inf && rng.chance(0.15)) { fix_column(f.st, f.ps, j, cols[j].lower); }
    }
    f.budget.limit       = 1000000;
    state_t before       = f.st;
    rule_result_t result = run_tightening(f);
    GPRS_REQUIRE(!result.stopped);
    check_same_set(before, f.st, result.outcome);
    if (result.outcome != ok) {
      ++infeasible_models;
      continue;
    }
    GPRS_CHECK(f.st.row_frontier.empty(row_list::tightening));
    GPRS_CHECK_EQ(f.st.live_rows, before.live_rows);
    GPRS_CHECK_EQ(f.st.bound_changes, before.bound_changes);
    findings += result.findings;
    sides += f.st.side_changes - before.side_changes;
    coefficients += f.st.coefficient_changes - before.coefficient_changes;
  }
  GPRS_CHECK(findings > 1500);
  GPRS_CHECK(infeasible_models > 40);
  GPRS_CHECK(sides > 1500);
  GPRS_CHECK(coefficients > 2000);
}
