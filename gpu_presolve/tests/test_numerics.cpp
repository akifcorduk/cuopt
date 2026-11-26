/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"

#include <gprs/numerics.hpp>

#include <cmath>

using namespace gprs;

GPRS_TEST(numerics_tolerance_defaults)
{
  tolerances_t tol;
  GPRS_CHECK_EQ(tol.feasibility, 1e-6);
  GPRS_CHECK_EQ(tol.dual, 1e-7);
  GPRS_CHECK_EQ(tol.zero, 1e-9);
  GPRS_CHECK_EQ(tol.integrality, 1e-5);
  GPRS_CHECK_EQ(tol.infinite_term, 1e20);
  GPRS_CHECK_EQ(tol.activity_error_share, 0.1);
  GPRS_CHECK_EQ(tol.implied_bound_step, 1000.);
  GPRS_CHECK_EQ(tol.side_error, 1e-14);
  GPRS_CHECK_EQ(tol.cancellation, 1e-12);
  GPRS_CHECK_EQ(rounding_unit, std::ldexp(1., -53));
}

GPRS_TEST(numerics_two_sum_is_error_free)
{
  double error;
  GPRS_CHECK_EQ(two_sum(1e16, 1., error), 1e16);
  GPRS_CHECK_EQ(error, 1.);
  GPRS_CHECK_EQ(two_sum(1., 1e16, error), 1e16);
  GPRS_CHECK_EQ(error, 1.);
  GPRS_CHECK_EQ(two_sum(0.5, 0.25, error), 0.75);
  GPRS_CHECK_EQ(error, 0.);
  GPRS_CHECK_EQ(two_sum(1., -std::ldexp(1., -60), error), 1.);
  GPRS_CHECK_EQ(error, -std::ldexp(1., -60));
}

GPRS_TEST(numerics_two_prod_is_error_free)
{
  double error;
  double a = 1. + std::ldexp(1., -30);
  // a * a = 1 + 2^-29 + 2^-60; the last term does not fit.
  GPRS_CHECK_EQ(two_prod(a, a, error), 1. + std::ldexp(1., -29));
  GPRS_CHECK_EQ(error, std::ldexp(1., -60));
  GPRS_CHECK_EQ(two_prod(3., 0.5, error), 1.5);
  GPRS_CHECK_EQ(error, 0.);
}

GPRS_TEST(numerics_dd_cancellation_2p54)
{
  double big = std::ldexp(1., 54);
  GPRS_CHECK_EQ((big + 1.) - big, 0.);
  dd_t s;
  dd_add_product(s, 1., big);
  dd_add_product(s, 1., 1.);
  dd_add_product(s, -1., big);
  GPRS_CHECK_EQ(dd_value(s), 1.);

  dd_t t;
  dd_add_product(t, 1., big);
  dd_add_product(t, 1., 1.);
  dd_sub_product(t, 1., big);
  GPRS_CHECK_EQ(dd_value(t), 1.);
}

GPRS_TEST(numerics_dd_rounding_513_terms)
{
  double tiny  = std::ldexp(1., -53);
  double plain = 1.;
  dd_t s;
  dd_add_product(s, 1., 1.);
  for (int k = 0; k < 512; ++k) {
    plain += tiny;
    dd_add_product(s, 1., tiny);
  }
  GPRS_CHECK_EQ(plain, 1.);
  GPRS_CHECK_EQ(s.hi, 1.);
  GPRS_CHECK_EQ(s.lo, std::ldexp(1., -44));
  GPRS_CHECK_EQ(dd_value(s), 1. + std::ldexp(1., -44));
}

GPRS_TEST(numerics_dd_product_error_is_kept)
{
  double a = 1. + std::ldexp(1., -30);
  dd_t s;
  dd_add_product(s, a, a);
  dd_add_product(s, -1., 1. + std::ldexp(1., -29));
  GPRS_CHECK_EQ(dd_value(s), std::ldexp(1., -60));

  dd_t t;
  dd_add_product(t, a, a);
  dd_sub_product(t, a, a);
  GPRS_CHECK_EQ(t.hi, 0.);
  GPRS_CHECK_EQ(t.lo, 0.);
}

// 400 products of integers below 2^40, each beyond the 53 bits of a double, added and then
// removed in another order: what is left is exactly the one term that was not removed. Every
// intermediate sum and every error is an integer that fits, so the pair is exact throughout.
GPRS_TEST(numerics_dd_exact_after_removing_every_term)
{
  uint64_t x = 99;
  auto draw  = [&x]() {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    return (x >> 24) - std::ldexp(1., 39);
  };
  const int n = 400;
  double a[n], b[n];
  dd_t s;
  double plain = 0.;
  dd_add_product(s, 7., 3.);
  plain += 7. * 3.;
  for (int k = 0; k < n; ++k) {
    a[k] = draw();
    b[k] = draw();
    dd_add_product(s, a[k], b[k]);
    plain += a[k] * b[k];
  }
  for (int k = 0; k < n; ++k) {
    int j = (k * 137) % n;
    if (k % 2) {
      dd_sub_product(s, a[j], b[j]);
    } else {
      dd_add_product(s, -a[j], b[j]);
    }
    plain -= a[j] * b[j];
  }
  GPRS_CHECK_EQ(dd_value(s), 21.);
  GPRS_CHECK(plain != 21.);
}

// An infinite side has no residual: the helper is for finite sides only.
GPRS_TEST(numerics_residual_needs_a_finite_side)
{
  GPRS_CHECK(std::isnan(dd_residual(inf, dd_t{1., 0.})));
  GPRS_CHECK(std::isnan(dd_residual(-inf, dd_t{1., 0.})));
  GPRS_CHECK_EQ(dd_residual(1e300, dd_t{-1e300, 0.}), 2e300);
}

GPRS_TEST(numerics_dd_add_merges_two_sums)
{
  dd_t a{std::ldexp(1., 54), 1.};
  dd_t b{-std::ldexp(1., 54), 0.25};
  dd_t c = dd_add(a, b);
  GPRS_CHECK_EQ(c.hi, 0.);
  GPRS_CHECK_EQ(c.lo, 1.25);

  dd_t d = dd_add(dd_t{1., 0.}, dd_t{std::ldexp(1., -60), 0.});
  GPRS_CHECK_EQ(d.hi, 1.);
  GPRS_CHECK_EQ(d.lo, std::ldexp(1., -60));

  dd_t zero;
  dd_t e = dd_add(a, zero);
  GPRS_CHECK_EQ(e.hi, a.hi);
  GPRS_CHECK_EQ(e.lo, a.lo);
  dd_t f = dd_add(zero, a);
  GPRS_CHECK_EQ(f.hi, a.hi);
  GPRS_CHECK_EQ(f.lo, a.lo);
}

// x1 = 2 and x2 = 3 fixed, x3 free, x1 + x2 + x3 <= 10: the minimum activity has the finite part 5
// and one infinite term, and x3 <= 5.
GPRS_TEST(numerics_residual_single_infinite_term)
{
  tolerances_t tol;
  dd_t finite_part;
  dd_add_product(finite_part, 1., 2.);
  dd_add_product(finite_part, 1., 3.);
  GPRS_CHECK(is_infinite_term(1., -inf, tol));
  GPRS_CHECK_EQ(dd_residual(10., finite_part) / 1., 5.);
}

// The own term leaves the activity before the side is used: with the terms 2^54 and 1 and the
// side 3, the column of the term 2^54 gets (3 - 1) / 1. (3 - activity) + 2^54 gives 4.
GPRS_TEST(numerics_residual_before_side)
{
  double big = std::ldexp(1., 54);
  dd_t activity;
  dd_add_product(activity, 1., big);
  dd_add_product(activity, 1., 1.);
  GPRS_CHECK_EQ((3. - (big + 1.)) + big, 4.);
  dd_t residual = activity;
  dd_sub_product(residual, 1., big);
  GPRS_CHECK_EQ(dd_residual(3., residual), 2.);
  GPRS_CHECK_EQ(dd_residual(3., dd_t{big, 1.}), 2. - big);
}

GPRS_TEST(numerics_infinite_term_rule)
{
  tolerances_t tol;
  GPRS_CHECK(!is_infinite_term(1., 1e20, tol));
  GPRS_CHECK(!is_infinite_term(-2., 5e19, tol));
  GPRS_CHECK(is_infinite_term(2., 5.0000001e19, tol));
  GPRS_CHECK(is_infinite_term(1e-3, 1e24, tol));
  GPRS_CHECK(is_infinite_term(1., inf, tol));
  GPRS_CHECK(is_infinite_term(-1., inf, tol));
  GPRS_CHECK(is_infinite_term(1e-300, -inf, tol));
  GPRS_CHECK(!is_infinite_term(1e-3, 1e22, tol));
  // The limit itself is finite, the next double is not; a product that overflows is infinite.
  GPRS_CHECK(is_infinite_term(1., std::nextafter(1e20, inf), tol));
  GPRS_CHECK(is_infinite_term(1., -std::nextafter(1e20, inf), tol));
  GPRS_CHECK(is_infinite_term(1e200, 1e200, tol));
  GPRS_CHECK(is_infinite_term(-1e200, 1e200, tol));
}

GPRS_TEST(numerics_activity_bounds_by_sign)
{
  GPRS_CHECK_EQ(min_activity_bound(2., -1., 4.), -1.);
  GPRS_CHECK_EQ(min_activity_bound(-2., -1., 4.), 4.);
  GPRS_CHECK_EQ(max_activity_bound(2., -1., 4.), 4.);
  GPRS_CHECK_EQ(max_activity_bound(-2., -1., 4.), -1.);
}

GPRS_TEST(numerics_activity_error_bound)
{
  GPRS_CHECK_EQ(activity_error(0, 1e20), 0.);
  GPRS_CHECK_EQ(activity_error(4, 1.), std::ldexp(1., -104));
  tolerances_t tol;
  // 8e6 updates of terms up to 1e18 reach a tenth of the feasibility tolerance.
  GPRS_CHECK(activity_error(8000000, 1e18) < tol.activity_error_share * tol.feasibility);
  GPRS_CHECK(activity_error(9000000, 1e18) > tol.activity_error_share * tol.feasibility);
}

GPRS_TEST(numerics_integer_rounding)
{
  tolerances_t tol;
  GPRS_CHECK_EQ(round_lower_integer(2.9999999, tol), 3.);
  GPRS_CHECK_EQ(round_lower_integer(2.0000005, tol), 2.);
  GPRS_CHECK_EQ(round_lower_integer(2.000002, tol), 3.);
  GPRS_CHECK_EQ(round_lower_integer(-0.5, tol), 0.);
  GPRS_CHECK_EQ(round_lower_integer(-inf, tol), -inf);
  GPRS_CHECK_EQ(round_upper_integer(3.0000001, tol), 3.);
  GPRS_CHECK_EQ(round_upper_integer(2.9999995, tol), 3.);
  GPRS_CHECK_EQ(round_upper_integer(2.999998, tol), 2.);
  GPRS_CHECK_EQ(round_upper_integer(-0.5, tol), -1.);
  GPRS_CHECK_EQ(round_upper_integer(inf, tol), inf);
}

GPRS_TEST(numerics_activity_lanes)
{
  int32_t expected[][2] = {{0, 1},
                           {1, 1},
                           {2, 1},
                           {3, 2},
                           {4, 2},
                           {5, 4},
                           {8, 4},
                           {9, 8},
                           {16, 8},
                           {17, 16},
                           {32, 16},
                           {33, 32},
                           {64, 32},
                           {65, 256},
                           {300, 256},
                           {100000, 256}};
  for (auto& pair : expected) {
    GPRS_CHECK_EQ(activity_lanes(pair[0]), pair[1]);
  }
  for (int32_t n = 1; n <= max_light_row; ++n) {
    GPRS_CHECK(2 * activity_lanes(n) >= n);
  }
}
