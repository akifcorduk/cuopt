/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// src/integral.hpp: when a row gives one of its columns an integer value, and how far the
// rounding of its ratios can move the row.

#include "../src/integral.hpp"
#include "harness.hpp"
#include "test_rows_support.hpp"

#include <cmath>

using namespace gprs;
using rule_test::col_t;
using rule_test::row_t;

namespace {

integral_form_t term(double a, double b, double lower, double upper)
{
  integral_form_t f;
  integral_term(tolerances_t(), a, b, lower, upper, f);
  return f;
}

integral_form_t side(double a, double value)
{
  integral_form_t f;
  integral_side(tolerances_t(), a, value, f);
  return f;
}

}  // namespace

GPRS_TEST(columns_integral_term)
{
  // A ratio that is an integer exactly has no drift, whatever the bounds are.
  GPRS_CHECK(term(2., 6., -inf, inf).integral);
  GPRS_CHECK_EQ(term(2., 6., -inf, inf).drift, 0.);
  GPRS_CHECK(term(-0.5, 1.5, 0., 10.).integral);
  GPRS_CHECK(!term(2., 5., 0., 1.).integral);
  GPRS_CHECK(!term(3., 1., 0., 1.).integral);
  // Within tol.zero of an integer: the drift is the deviation times the reach of the column.
  integral_form_t near = term(1., 3. + 5e-10, -100., 50.);
  GPRS_CHECK(near.integral);
  GPRS_CHECK(std::fabs(near.drift - 5e-8) < 1e-12);
  GPRS_CHECK(term(1., 3. + 5e-10, 0., inf).integral);
  GPRS_CHECK_EQ(term(1., 3. + 5e-10, 0., inf).drift, inf);
  GPRS_CHECK(!term(1., 3. + 5e-9, 0., 1.).integral);
  // The ratio is tested in the units of the row as well: 1e6 x and (2e6 + 1 / 2048) y have a
  // ratio within 4.9e-10 of 2, and 2 leaves 1 / 2048 of y in the row.
  GPRS_CHECK(!term(1e6, 2e6 + 0x1p-11, 0., 1.).integral);
  // The terms of a row add up.
  integral_form_t sum;
  integral_term(tolerances_t(), 1., 3. + 5e-10, 0., 10., sum);
  integral_term(tolerances_t(), 1., 2. - 5e-10, -20., 10., sum);
  integral_term(tolerances_t(), 1., 7., -inf, inf, sum);
  GPRS_CHECK(sum.integral);
  GPRS_CHECK(std::fabs(sum.drift - 1.5e-8) < 1e-12);
}

GPRS_TEST(columns_integral_side)
{
  GPRS_CHECK(side(3., 6.).integral);
  GPRS_CHECK(side(-0.5, 2.).integral);
  GPRS_CHECK(!side(2., 3.).integral);
  GPRS_CHECK(side(1., 4. + 5e-10).integral);
  GPRS_CHECK(!side(1., 4. + 5e-9).integral);
  // In the units of the row the side has the feasibility tolerance: 1e3 x = 1e3 + 5e-7 is x = 1
  // within it, 1e4 x = 1e4 + 5e-6 is not.
  GPRS_CHECK(side(1e3, 1e3 + 5e-7).integral);
  GPRS_CHECK(!side(1e4, 1e4 + 5e-6).integral);
}

GPRS_TEST(columns_integral_drift_units)
{
  tolerances_t tol;
  integral_form_t f;
  f.drift = 5e-8;
  // The share 0.1 of the tolerance 1e-6: the row is off by |a| times the drift.
  GPRS_CHECK(drift_within(tol, 0.1, 1., f, drift_units_t::row));
  GPRS_CHECK(drift_within(tol, 0.1, -1.5, f, drift_units_t::row));
  GPRS_CHECK(!drift_within(tol, 0.1, 3., f, drift_units_t::row));
  GPRS_CHECK(drift_within(tol, 1., 3., f, drift_units_t::row));
  GPRS_CHECK(drift_within(tol, 0.1, 0.01, f, drift_units_t::row));
  // In the units of the column the drift itself counts as well, which matters below 1: the
  // model of the review has a = 0.01 and a column that is off an integer by 5e-6.
  GPRS_CHECK(drift_within(tol, 0.1, 1., f, drift_units_t::column));
  GPRS_CHECK(!drift_within(tol, 0.1, 3., f, drift_units_t::column));
  GPRS_CHECK(drift_within(tol, 0.1, 0.5, f, drift_units_t::column));
  f.drift = 5e-6;
  GPRS_CHECK(drift_within(tol, 0.1, 0.01, f, drift_units_t::row));
  GPRS_CHECK(!drift_within(tol, 0.1, 0.01, f, drift_units_t::column));
  f.drift = inf;
  GPRS_CHECK(!drift_within(tol, 0.1, 1e-300, f, drift_units_t::row));
  f.drift = 0.;
  GPRS_CHECK(drift_within(tol, 0.1, 1e300, f, drift_units_t::column));
}

// integral_for reads the free entries of the row other than the column's own, and wants every
// one of them integer.
GPRS_TEST(columns_integral_for_a_row)
{
  // 2 k + 4 y - 6 z + w = 8 with w fixed.
  std::vector<col_t> cols = {{0., 9., true}, {0., 9., true}, {0., 9., true}, {5., 6., false}};
  std::vector<row_t> rows = {{{{0, 2.}, {1, 4.}, {2, -6.}, {3, 1.}}, 8., 8.}};
  rule_test::fixture_t f(cols, rows);
  fix_column(f.st, f.ps, 3, 5.);
  int64_t k = f.st.find_entry(0, 0), y = f.st.find_entry(0, 1);
  GPRS_REQUIRE(k >= 0 && y >= 0);
  // For k the side less the fixed term is 3, which is no multiple of 2; 4 is.
  GPRS_CHECK(!integral_for(f.st, f.st.tol, 0, k, 3.).integral);
  integral_form_t form = integral_for(f.st, f.st.tol, 0, k, 4.);
  GPRS_CHECK(form.integral);
  GPRS_CHECK_EQ(form.drift, 0.);
  // For y the ratios 2 / 4 and -6 / 4 are no integers.
  GPRS_CHECK(!integral_for(f.st, f.st.tol, 0, y, 4.).integral);

  // A continuous column next to it.
  cols[2].integer = false;
  rule_test::fixture_t mixed(cols, rows);
  fix_column(mixed.st, mixed.ps, 3, 5.);
  GPRS_CHECK(!integral_for(mixed.st, mixed.st.tol, 0, mixed.st.find_entry(0, 0), 4.).integral);

  // The drift over the bounds of the other columns.
  std::vector<col_t> wide = {{0., 9., true}, {-3., 2000000., true}, {0., 9., true}};
  rule_test::fixture_t drifting(wide, {{{{0, 1.}, {1, -1.0000000005}, {2, 2.}}, 0., 0.}});
  form = integral_for(drifting.st, drifting.st.tol, 0, drifting.st.find_entry(0, 0), 0.);
  GPRS_CHECK(form.integral);
  GPRS_CHECK(std::fabs(form.drift - 1e-3) < 1e-9);
  GPRS_CHECK(!drift_within(
    drifting.st.tol, drifting.st.tol.integral_drift_share, 1., form, drift_units_t::row));
}
