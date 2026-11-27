/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The internal code that the column rules share with bound propagation (src/implied.hpp) and
// with the doubleton equations (src/saved_row.hpp).

#include "../src/implied.hpp"
#include "../src/saved_row.hpp"
#include "harness.hpp"
#include "test_engine_support.hpp"

#include <gprs/state.hpp>

#include <cmath>
#include <vector>

using namespace gprs;
using namespace gprs::engine_test;

namespace {

//   r0:  x0 + 2 x1 - x2        <= 10
//   r1:  x0        + x2 + 4 x3 >= -3
//   x0 in [0, 4], x1 in [0, 10], x2 in [1, inf), x3 in [-2, 2].
model_t two_rows()
{
  return make_model({{1., 2., -1., 0.}, {1., 0., 1., 4.}},
                    {-inf, -3.},
                    {10., inf},
                    {0., 0., 1., -2.},
                    {4., 10., inf, 2.},
                    {0., 0., 0., 0.},
                    {0, 0, 0, 0});
}

}  // namespace

// The rest of an activity is the stored activity less the term of the column, or the stored
// activity when that term is its one infinite term; an infinite term elsewhere gives none.
GPRS_TEST(columns_row_rest)
{
  state_t st(two_rows(), tolerances_t());
  row_rest_t rest;
  // Minimum activity of r0: 0 + 0 - inf. Its one infinite term is that of x2 at slot 2.
  GPRS_CHECK_EQ(st.min_act.inf_count[0], 1);
  GPRS_CHECK(!rest_finite(st.min_act, 0, 0));
  GPRS_CHECK(rest_finite(st.min_act, 0, 2));
  GPRS_CHECK(!row_rest(st, 0, 0, true, rest));
  GPRS_CHECK(!row_rest(st, 0, 1, true, rest));
  GPRS_REQUIRE(row_rest(st, 0, 2, true, rest));
  GPRS_CHECK_EQ(dd_value(rest.sum), 0.);
  GPRS_CHECK_EQ(rest.error, activity_error(st.min_act.updates[0], st.min_act.max_term[0]));
  // Maximum activity of r0: 4 + 20 - 1 = 23, no infinite term.
  GPRS_REQUIRE(row_rest(st, 0, 0, false, rest));
  GPRS_CHECK_EQ(dd_value(rest.sum), 19.);
  GPRS_REQUIRE(row_rest(st, 0, 1, false, rest));
  GPRS_CHECK_EQ(dd_value(rest.sum), 3.);
  GPRS_REQUIRE(row_rest(st, 0, 2, false, rest));
  GPRS_CHECK_EQ(dd_value(rest.sum), 24.);
  GPRS_CHECK_EQ(rest.error, activity_error(st.max_act.updates[0] + 1, st.max_act.max_term[0]));
  // Two infinite terms: none for any entry.
  st.set_col_upper(1, inf);
  st.set_col_lower(2, -inf);
  for (int64_t e = st.csr.begin(0); e < st.csr.end(0); ++e) {
    GPRS_CHECK(!row_rest(st, 0, e, false, rest));
  }
  // The term of a fixed column stays in the rest: r1 without x2, which is free by now, is
  // x0 + 4 * 2 with x0 at a bound.
  st.fix_col(3, 2.);
  GPRS_REQUIRE(row_rest(st, 1, st.csr.begin(1) + 1, true, rest));
  GPRS_CHECK_EQ(dd_value(rest.sum), 8.);
  GPRS_REQUIRE(row_rest(st, 1, st.csr.begin(1) + 1, false, rest));
  GPRS_CHECK_EQ(dd_value(rest.sum), 12.);
}

// candidate() is the bound of bound propagation: side less rest over the coefficient.
GPRS_TEST(columns_candidate)
{
  state_t st(two_rows(), tolerances_t());
  candidate_t k;
  bool guarded = false;
  // The upper side of r0 with its minimum activity gives a candidate to x2 alone, whose term
  // is the infinite one: -x2 <= 10 - 0, so x2 >= -10.
  GPRS_CHECK(!candidate(st, st.tol, 0, 0, true, k, guarded));
  GPRS_CHECK(!guarded);
  GPRS_REQUIRE(candidate(st, st.tol, 0, 2, true, k, guarded));
  GPRS_CHECK_EQ(k.bound, -10.);
  GPRS_CHECK_EQ(k.a, -1.);
  GPRS_CHECK_EQ(k.side, 10.);
  GPRS_CHECK_EQ(candidate_slack(k), 0.);
  // The lower side of r1 with its maximum activity, which x2 makes infinite: only x2.
  GPRS_CHECK(!candidate(st, st.tol, 1, st.csr.begin(1), false, k, guarded));
  GPRS_REQUIRE(candidate(st, st.tol, 1, st.csr.begin(1) + 1, false, k, guarded));
  GPRS_CHECK_EQ(k.bound, -3. - 4. - 8.);
  // With x2 <= 5 every column of r1 has one: x3 >= (-3 - 4 - 5) / 4.
  st.set_col_upper(2, 5.);
  GPRS_REQUIRE(candidate(st, st.tol, 1, st.csr.begin(1) + 2, false, k, guarded));
  GPRS_CHECK_EQ(k.bound, -3.);
  GPRS_CHECK_EQ(candidate_slack(k), 0.);
  GPRS_CHECK(!guarded);

  // A candidate that the division rounds is moved outward by its slack.
  model_t thirds = make_model({{3., 1.}}, {-inf}, {1.}, {0., 0.}, {1., 1.}, {0., 0.}, {0, 0});
  state_t rounded(thirds, tolerances_t());
  GPRS_REQUIRE(candidate(rounded, rounded.tol, 0, 0, true, k, guarded));
  GPRS_CHECK_EQ(k.bound, 1. / 3.);
  GPRS_CHECK(candidate_slack(k) > 0. && candidate_slack(k) < 1e-12);
  tolerances_t tol;
  GPRS_CHECK(candidate_slack(k) < implied_tolerance(tol, k.bound));
  GPRS_CHECK_EQ(implied_tolerance(tol, 0.), 2. * tol.activity_error_share * tol.feasibility);
  GPRS_CHECK_EQ(implied_tolerance(tol, -4.),
                2. * (bound_rounding(4.) + tol.activity_error_share * tol.feasibility));
}

// The error guard: a coefficient so small that the error bound of the rest, divided by it,
// would use more than its share of the tolerance. Next to a term of 1e15 that is 1e-12.
GPRS_TEST(columns_candidate_guard)
{
  model_t m = make_model({{1e-12, 1.}}, {-5.}, {inf}, {0., 0.}, {1., 1e15}, {0., 0.}, {0, 0});
  state_t st(m, tolerances_t());
  candidate_t k;
  bool guarded = false;
  GPRS_CHECK(!candidate(st, st.tol, 0, 0, false, k, guarded));
  GPRS_CHECK(guarded);
  guarded = false;
  GPRS_REQUIRE(candidate(st, st.tol, 0, 1, false, k, guarded));
  GPRS_CHECK(!guarded);
  GPRS_CHECK_EQ(k.bound, -5. - 1e-12);
}

// Where the rest of an activity is finite, candidate() is candidate_of_finite_rest(), which
// bound propagation calls, and both compute on the rest and the error bound of row_rest().
GPRS_TEST(columns_candidate_agrees_with_row_rest)
{
  state_t st(two_rows(), tolerances_t());
  int32_t compared = 0, refused = 0;
  for (int32_t step = 0; step < 2; ++step) {
    for (int32_t row = 0; row < 2; ++row) {
      for (bool upper_side : {true, false}) {
        if (std::isinf(upper_side ? st.row_upper[row] : st.row_lower[row])) { continue; }
        const activity_t& act = upper_side ? st.min_act : st.max_act;
        for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
          candidate_t a, b;
          bool guarded_a = false, guarded_b = false;
          row_rest_t rest;
          bool finite = rest_finite(act, row, e);
          GPRS_CHECK_EQ(row_rest(st, row, e, upper_side, rest), finite);
          bool has = candidate(st, st.tol, row, e, upper_side, a, guarded_a);
          if (!finite) {
            GPRS_CHECK(!has && !guarded_a);
            ++refused;
            continue;
          }
          GPRS_CHECK_EQ(candidate_of_finite_rest(st, st.tol, row, e, upper_side, b, guarded_b),
                        has);
          GPRS_CHECK_EQ(guarded_a, guarded_b);
          GPRS_CHECK_EQ(a.rest.hi, rest.sum.hi);
          GPRS_CHECK_EQ(a.rest.lo, rest.sum.lo);
          GPRS_CHECK_EQ(a.error, rest.error);
          GPRS_CHECK_EQ(b.rest.hi, rest.sum.hi);
          GPRS_CHECK_EQ(b.error, rest.error);
          if (has) { GPRS_CHECK_EQ(a.bound, b.bound); }
          ++compared;
        }
      }
    }
    // The second time without an infinite term.
    if (step == 0) { st.set_col_upper(2, 5.); }
  }
  GPRS_CHECK_EQ(compared, 2 + 6);
  GPRS_CHECK_EQ(refused, 4);
}

// scan_row visits the free entries in storage order and sums the terms of the fixed columns;
// save_row keeps what its predicate accepts, with the sides less those terms.
GPRS_TEST(columns_saved_row)
{
  //   r0:  2 x0 + 3 x1 - x2 + 5 x3 in [-4, 20]
  //   r1:    x0             +   x3 <= 7
  model_t m = make_model({{2., 3., -1., 5.}, {1., 0., 0., 1.}},
                         {-4., -inf},
                         {20., 7.},
                         {0., 0., 0., 0.},
                         {4., 4., 4., 4.},
                         {0., 0., 0., 0.},
                         {0, 0, 0, 0});
  state_t st(m, tolerances_t());
  std::vector<int32_t> cols;
  std::vector<double> values;
  auto collect = [&](int32_t col, double value) {
    cols.push_back(col);
    values.push_back(value);
  };
  dd_t fixed = scan_row(st, 0, collect);
  GPRS_CHECK(cols == std::vector<int32_t>({0, 1, 2, 3}));
  GPRS_CHECK(values == std::vector<double>({2., 3., -1., 5.}));
  GPRS_CHECK_EQ(dd_value(fixed), 0.);

  st.fix_col(1, 2.);
  st.fix_col(2, 0.5);
  cols.clear();
  values.clear();
  fixed = scan_row(st, 0, collect);
  GPRS_CHECK(cols == std::vector<int32_t>({0, 3}));
  GPRS_CHECK(values == std::vector<double>({2., 5.}));
  GPRS_CHECK_EQ(dd_value(fixed), 3. * 2. - 0.5);
  GPRS_CHECK_EQ(free_side(20., fixed), 14.5);
  GPRS_CHECK_EQ(free_side(-4., fixed), -9.5);
  GPRS_CHECK_EQ(free_side(inf, fixed), inf);
  GPRS_CHECK_EQ(free_side(-inf, fixed), -inf);

  row_pool_t pool;
  saved_row_t all = save_row(st, 0, pool, [](int32_t) { return true; });
  GPRS_CHECK_EQ(all.row, 0);
  GPRS_CHECK_EQ(all.lower, -9.5);
  GPRS_CHECK_EQ(all.upper, 14.5);
  GPRS_CHECK_EQ(all.begin, 0);
  GPRS_CHECK_EQ(all.end, 2);
  // A second row goes behind the first, here without its column 3.
  saved_row_t part = save_row(st, 1, pool, [](int32_t col) { return col != 3; });
  GPRS_CHECK_EQ(part.row, 1);
  GPRS_CHECK_EQ(part.lower, -inf);
  GPRS_CHECK_EQ(part.upper, 7.);
  GPRS_CHECK_EQ(part.begin, 2);
  GPRS_CHECK_EQ(part.end, 3);
  GPRS_CHECK(pool.col == std::vector<int32_t>({0, 3, 0}));
  GPRS_CHECK(pool.value == std::vector<double>({2., 5., 1.}));

  // Storage order, not column order: the entry of x3 in r0 becomes one of x1, in its slot.
  state_t moved(m, tolerances_t());
  moved.delete_entry(moved.find_entry(0, 1));
  moved.move_entry(moved.find_entry(0, 3), 1, 7.);
  cols.clear();
  values.clear();
  scan_row(moved, 0, collect);
  GPRS_CHECK(cols == std::vector<int32_t>({0, 2, 1}));
  GPRS_CHECK(values == std::vector<double>({2., -1., 7.}));

  // A column fixed at a huge value: the side of the free part is one double, and holds
  // nothing of the side it was.
  model_t huge = make_model({{1., 1.}}, {3.}, {3.}, {0., 0.}, {1., 1e20}, {0., 0.}, {0, 0});
  state_t swallowed(huge, tolerances_t());
  swallowed.fix_col(1, 1e20);
  pool          = row_pool_t();
  saved_row_t s = save_row(swallowed, 0, pool, [](int32_t) { return true; });
  GPRS_CHECK_EQ(s.lower, -1e20);
  GPRS_CHECK_EQ(s.upper, -1e20);
  GPRS_CHECK(std::fabs(s.upper) * swallowed.tol.side_error > swallowed.tol.feasibility);
}
