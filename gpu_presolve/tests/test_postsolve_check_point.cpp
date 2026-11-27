/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"

#include <gprs/postsolve.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace gprs;

namespace {

// splitmix64.
struct rng_t {
  uint64_t state;

  uint64_t next()
  {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  int32_t uniform(int32_t lo, int32_t hi)
  {
    return lo + (int32_t)(next() % (uint64_t)(hi - lo + 1));
  }
  bool chance(double probability) { return (next() >> 11) * 0x1p-53 < probability; }
};

//   r0:  x0 + 2 x1        <= 4
//   r1:  x0 -   x1        >= 1
//   r2:         x1 + x2    = 3
//   minimise x0 - 2 x1 + x2 / 2 + 10,  x0 in [0, 2], x1 >= -1, x2 integer and free.
model_t hand_model()
{
  model_t m;
  m.nrows            = 3;
  m.ncols            = 3;
  m.row_start        = {0, 2, 4, 6};
  m.col_index        = {0, 1, 0, 1, 1, 2};
  m.value            = {1., 2., 1., -1., 1., 1.};
  m.row_lower        = {-inf, 1., 3.};
  m.row_upper        = {4., inf, 3.};
  m.col_lower        = {0., -1., -inf};
  m.col_upper        = {2., inf, inf};
  m.cost             = {1., -2., 0.5};
  m.col_type         = {0, 0, 1};
  m.objective_offset = 10.;
  return m;
}

// One row  x0 + x1 - x2  with the given sides, objective x0 + x1 - x2 + 1 / 4, free columns.
model_t cancelling_model(double lower, double upper)
{
  model_t m;
  m.nrows            = 1;
  m.ncols            = 3;
  m.row_start        = {0, 3};
  m.col_index        = {0, 1, 2};
  m.value            = {1., 1., -1.};
  m.row_lower        = {lower};
  m.row_upper        = {upper};
  m.col_lower        = {-inf, -inf, -inf};
  m.col_upper        = {inf, inf, inf};
  m.cost             = {1., 1., -1.};
  m.col_type         = {0, 0, 0};
  m.objective_offset = 0.25;
  return m;
}

void check_values(
  const point_check_t& check, double row, double bound, double integrality, double objective)
{
  GPRS_CHECK_EQ(check.row_violation, row);
  GPRS_CHECK_EQ(check.bound_violation, bound);
  GPRS_CHECK_EQ(check.integrality_violation, integrality);
  GPRS_CHECK_EQ(check.objective, objective);
}

}  // namespace

GPRS_TEST(postsolve_check_point_by_hand)
{
  model_t m = hand_model();
  tolerances_t tol;

  point_check_t check = check_point(m, {2., 1., 2.});
  check_values(check, 0., 0., 0., 11.);
  GPRS_CHECK(point_feasible(check, tol));

  // r0 is 2 above its upper side, r2 is 1 / 4 below its lower side, x0 is 1 above its bound.
  check = check_point(m, {3., 1.5, 1.25});
  check_values(check, 2., 1., 0.25, 10.625);
  GPRS_CHECK(!point_feasible(check, tol));

  // The largest violation is on a lower side: r1 is 3 / 2 below, r2 is 1 / 2 below.
  check = check_point(m, {0., 0.5, 2.});
  check_values(check, 1.5, 0., 0., 10.);

  // Bounds violated from below: x0 by 1 / 4, x1 by 1 / 2; r2 is 1 / 2 above.
  check = check_point(m, {-0.25, -1.5, 5.});
  check_values(check, 0.5, 0.5, 0., 15.25);

  // Integrality is measured to the nearest integer on either side, on integer columns only.
  check = check_point(m, {1.5, 0.5, 2.75});
  check_values(check, 0.25, 0., 0.25, 1.5 - 1. + 1.375 + 10.);
  check = check_point(m, {1.5, 0.5, -2.25});
  GPRS_CHECK_EQ(check.integrality_violation, 0.25);
}

// 1e21 + 1 / 2 - 1e21: a plain sum in storage order loses the 1 / 2.
GPRS_TEST(postsolve_check_point_sums_are_compensated)
{
  std::vector<double> x = {1e21, 0.5, 1e21};
  point_check_t check   = check_point(cancelling_model(0.5, 0.5), x);
  check_values(check, 0., 0., 0., 0.75);
  check = check_point(cancelling_model(0., 0.), x);
  GPRS_CHECK_EQ(check.row_violation, 0.5);
  check = check_point(cancelling_model(1., 2.), x);
  GPRS_CHECK_EQ(check.row_violation, 0.5);
  check = check_point(cancelling_model(-inf, inf), x);
  GPRS_CHECK_EQ(check.row_violation, 0.);
}

GPRS_TEST(postsolve_point_feasible_uses_its_tolerances)
{
  tolerances_t tol;
  point_check_t check;
  GPRS_CHECK(point_feasible(check, tol));
  check.row_violation = tol.feasibility;
  GPRS_CHECK(point_feasible(check, tol));
  check.row_violation = 1.01 * tol.feasibility;
  GPRS_CHECK(!point_feasible(check, tol));

  check                 = point_check_t{};
  check.bound_violation = tol.feasibility;
  GPRS_CHECK(point_feasible(check, tol));
  check.bound_violation = 1.01 * tol.feasibility;
  GPRS_CHECK(!point_feasible(check, tol));

  // Between the feasibility and the integrality tolerance.
  check                       = point_check_t{};
  check.integrality_violation = 5. * tol.feasibility;
  GPRS_CHECK(tol.integrality > 5. * tol.feasibility);
  GPRS_CHECK(point_feasible(check, tol));
  check.integrality_violation = 1.01 * tol.integrality;
  GPRS_CHECK(!point_feasible(check, tol));

  check           = point_check_t{};
  check.objective = 1e30;
  GPRS_CHECK(point_feasible(check, tol));

  tol.feasibility = 0.5;
  tol.integrality = 0.25;
  check           = point_check_t{0.4, 0.5, 0.25, 0.};
  GPRS_CHECK(point_feasible(check, tol));
  check = point_check_t{0.4, 0.5, 0.3, 0.};
  GPRS_CHECK(!point_feasible(check, tol));
}

// A value that is not finite is never a feasible point, also on a free column and in a free
// row, and neither is a finite point whose activity overflows.
GPRS_TEST(postsolve_check_point_rejects_values_that_are_not_finite)
{
  tolerances_t tol;
  double nan = std::numeric_limits<double>::quiet_NaN();
  for (double bad : {nan, inf, -inf}) {
    for (int32_t j = 0; j < 3; ++j) {
      std::vector<double> x = {2., 1., 2.};
      x[j]                  = bad;
      GPRS_CHECK(!point_feasible(check_point(hand_model(), x), tol));
      GPRS_CHECK(!point_feasible(check_point(cancelling_model(-inf, inf), x), tol));
      GPRS_CHECK(!point_feasible(check_point(cancelling_model(0., 5.), x), tol));
    }
  }
  // A finite point whose activity overflows in one row; the other row is satisfied and must
  // not hide it, whichever comes first.
  for (int32_t bad_row = 0; bad_row < 2; ++bad_row) {
    model_t m;
    m.nrows               = 2;
    m.ncols               = 2;
    m.row_start           = {0, 1, 2};
    m.col_index           = {0, 1};
    m.value               = {1., 1.};
    m.row_lower           = {-5., -5.};
    m.row_upper           = {5., 5.};
    m.col_lower           = {-inf, -inf};
    m.col_upper           = {inf, inf};
    m.cost                = {0., 0.};
    m.col_type            = {0, 0};
    std::vector<double> x = {1., 1.};
    GPRS_CHECK(point_feasible(check_point(m, x), tol));
    m.value[bad_row]    = 1e200;
    x[bad_row]          = 1e200;
    point_check_t check = check_point(m, x);
    GPRS_CHECK_EQ(check.bound_violation, 0.);
    GPRS_CHECK(!(check.row_violation <= tol.feasibility));
    GPRS_CHECK(!point_feasible(check, tol));
  }
}

GPRS_TEST(postsolve_check_point_empty_models)
{
  model_t m;
  m.row_start         = {0};
  m.objective_offset  = -2.5;
  point_check_t check = check_point(m, {});
  check_values(check, 0., 0., 0., -2.5);

  // Rows without entries: the activity is 0.
  m.nrows     = 2;
  m.row_start = {0, 0, 0};
  m.row_lower = {-1., 0.5};
  m.row_upper = {1., inf};
  check       = check_point(m, {});
  check_values(check, 0.5, 0., 0., -2.5);
}

// Small integers and quarters: every sum is exact in double arithmetic, so the reference needs
// no care.
GPRS_TEST(postsolve_check_point_random_against_plain_sums)
{
  int64_t violated_rows = 0, violated_bounds = 0, fractional = 0, clean = 0;
  for (uint64_t seed = 1; seed <= 2000; ++seed) {
    rng_t rng{seed};
    model_t m;
    m.nrows = rng.uniform(0, 6);
    m.ncols = rng.uniform(0, 6);
    m.row_start.push_back(0);
    for (int32_t i = 0; i < m.nrows; ++i) {
      for (int32_t j = 0; j < m.ncols; ++j) {
        if (!rng.chance(0.5)) { continue; }
        m.col_index.push_back(j);
        m.value.push_back(rng.chance(0.5) ? rng.uniform(1, 4) : -rng.uniform(1, 4));
      }
      m.row_start.push_back(m.col_index.size());
      double a = rng.uniform(-12, 12), b = rng.uniform(-12, 12);
      m.row_lower.push_back(rng.chance(0.3) ? -inf : std::min(a, b));
      m.row_upper.push_back(rng.chance(0.3) ? inf : (rng.chance(0.1) ? a : std::max(a, b)));
    }
    std::vector<double> x(m.ncols);
    for (int32_t j = 0; j < m.ncols; ++j) {
      double a = rng.uniform(-6, 6), b = rng.uniform(-6, 6);
      m.col_lower.push_back(rng.chance(0.3) ? -inf : std::min(a, b));
      m.col_upper.push_back(rng.chance(0.3) ? inf : (rng.chance(0.1) ? a : std::max(a, b)));
      m.cost.push_back(rng.uniform(-3, 3));
      m.col_type.push_back(rng.chance(0.5));
      x[j] = rng.chance(0.5) ? rng.uniform(-6, 6) : rng.uniform(-24, 24) / 4.;
    }
    m.objective_offset = rng.uniform(-8, 8) / 4.;
    GPRS_REQUIRE(well_formed(m));

    double row = 0., bound = 0., integrality = 0., objective = m.objective_offset;
    for (int32_t i = 0; i < m.nrows; ++i) {
      double activity = 0.;
      for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
        activity += m.value[e] * x[m.col_index[e]];
      }
      if (activity < m.row_lower[i]) { row = std::max(row, m.row_lower[i] - activity); }
      if (activity > m.row_upper[i]) { row = std::max(row, activity - m.row_upper[i]); }
    }
    for (int32_t j = 0; j < m.ncols; ++j) {
      if (x[j] < m.col_lower[j]) { bound = std::max(bound, m.col_lower[j] - x[j]); }
      if (x[j] > m.col_upper[j]) { bound = std::max(bound, x[j] - m.col_upper[j]); }
      if (m.col_type[j]) {
        double down = x[j] - std::floor(x[j]), up = std::ceil(x[j]) - x[j];
        integrality = std::max(integrality, std::min(down, up));
      }
      objective += m.cost[j] * x[j];
    }

    point_check_t check = check_point(m, x);
    check_values(check, row, bound, integrality, objective);
    tolerances_t tol;
    GPRS_CHECK_EQ(point_feasible(check, tol), row == 0. && bound == 0. && integrality == 0.);
    violated_rows += row > 0.;
    violated_bounds += bound > 0.;
    fractional += integrality > 0.;
    clean += row == 0. && bound == 0. && integrality == 0.;
  }
  GPRS_CHECK(violated_rows > 200);
  GPRS_CHECK(violated_bounds > 200);
  GPRS_CHECK(fractional > 200);
  GPRS_CHECK(clean > 50);
}
