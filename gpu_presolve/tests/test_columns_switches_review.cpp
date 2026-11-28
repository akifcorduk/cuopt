/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Review of the choices of the singleton-column rules (options_t, from dual_integer_pivots
// on): the values that are not options, each with the model that says why, and the defaults
// next to tolerances that are not the default ones.

#include "test_columns_support.hpp"

#include <cmath>
#include <vector>

using namespace gprs;
using columns_test::binary;
using columns_test::call_t;
using columns_test::unit;
using rule_test::row_t;

namespace {

bool feasible(const call_t& c, const std::vector<double>& x)
{
  return point_feasible(check_point(c.model, x), c.opt.tol);
}

}  // namespace

// The default choices are the rule as it is with every tolerance: a feasibility tolerance of
// 1e-10 is below what side_guard_scale needs of it for a factor other than 1.
GPRS_TEST(switches_review_defaults_take_every_tolerance)
{
  options_t o;
  o.tol.feasibility = 1e-10;
  GPRS_CHECK(choices_in_range(o));
  model_t m = rule_test::make_model(
    {unit, unit, unit}, {{{{0, 1.}, {1, 2.}, {2, 3.}}, 2., 4.}, {{{1, 1.}, {2, 1.}}, -inf, 2.}});
  result_t r = presolve(m, o);
  GPRS_CHECK(r.status == status_t::reduced || r.status == status_t::reduced_to_empty);
  o.side_guard_scale = 0.5;
  GPRS_CHECK(!choices_in_range(o));
  o                = options_t();
  o.tol.side_error = 1e-16;
  GPRS_CHECK(choices_in_range(o));
  o.side_guard_scale = 2.;
  GPRS_CHECK(!choices_in_range(o));
}

// k + y <= 1e8 with an integer k that is free to go down. At y = 49999999.00000105 the row
// leaves k the value 50000000.99999895. Within 8 rounding units of the side and the term that
// is no integer, and k = 50000000. A band of 64 units takes it as 50000001, which misses the
// row by 1.05e-6: a band above that of the rule is no value of forcing_rounding_units.
GPRS_TEST(switches_review_forcing_band_above_the_rule_misses_the_row)
{
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1.}}, -inf, 1e8}, {{{1, 1.}, {2, 1.}}, 3., inf}};
  call_t c({{-inf, 1e9, true}, {0., 6e7, false}, binary}, rows, {0., 1., 1.});
  c.opt.sub_enabled[sub_rule::free_singleton]      = false;
  c.opt.sub_enabled[sub_rule::free_singleton_dual] = false;
  c.opt.sub_enabled[sub_rule::zero_cost_singleton] = false;
  c.run();
  GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::forcing_column], 1);
  double y = 49999999.00000105;
  for (double units : {0., 4., 8.}) {
    c.ps.forcing_rounding_units = units;
    std::vector<double> x       = c.postsolved({{1, y}, {2, 1.}});
    GPRS_CHECK_EQ(x[0], 50000000.);
    GPRS_CHECK(feasible(c, x));
  }
  c.ps.forcing_rounding_units = 64.;
  std::vector<double> x       = c.postsolved({{1, y}, {2, 1.}});
  GPRS_CHECK_EQ(x[0], 50000001.);
  GPRS_CHECK(!feasible(c, x));
  GPRS_CHECK(columns_test::activity(c.model, 0, x) - 1e8 > 1e-6);
  options_t o;
  o.forcing_rounding_units = 64.;
  GPRS_CHECK(!choices_in_range(o));
  o.forcing_rounding_units = 8.;
  GPRS_CHECK(choices_in_range(o));
}

// x + 1000 y + w = 500 and y + 0.5 z + 0.5 v = 1 with binaries z and v. The second row gives
// y the range [0, 1], and the lower bound of y is 9.5e-7: not within the band of the implied
// test, so y is no free singleton column once x has left with the first row. It leaves as a
// zero-cost singleton column, and at z = v = 1 the point is inside the tolerance. With five
// times the band, which is no option for this reason, y left as a free singleton column: then
// z = v = 1 gives y = 0 and x = 500, 9.5e-4 above its bound.
GPRS_TEST(switches_review_bound_beyond_the_band_is_not_implied)
{
  std::vector<row_t> rows = {{{{0, 1.}, {1, 1000.}, {2, 1.}}, 500., 500.},
                             {{{1, 1.}, {3, 0.5}, {4, 0.5}}, 1., 1.}};
  call_t c({{-501., 499.99905, false}, {9.5e-7, 1., false}, unit, binary, binary},
           rows,
           {0., 0., 1., -1., -1.});
  c.run();
  GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::free_singleton], 1);
  GPRS_CHECK_EQ(c.result.sub_findings[sub_rule::zero_cost_singleton], 1);
  GPRS_CHECK(c.result.columns.counts[column_count::refused_bound] >= 1);
  GPRS_CHECK(c.ps.col_from_rows[0].kind == col_from_rows_kind_t::defined);
  GPRS_CHECK_EQ(c.ps.record_cols[c.ps.col_from_rows[0].cols_begin].col, 0);
  GPRS_CHECK(c.ps.col_from_rows[1].kind == col_from_rows_kind_t::interval);
  std::vector<double> x = c.postsolved({{2, 0.}, {3, 1.}, {4, 1.}});
  GPRS_CHECK_EQ(x[1], 9.5e-7);
  GPRS_CHECK(x[0] <= 499.99905);
  GPRS_CHECK(feasible(c, x));
  x = c.postsolved({{2, 0.}, {3, 0.}, {4, 1.}});
  GPRS_CHECK_EQ(x[1], 0.5);
  GPRS_CHECK(feasible(c, x));
}
