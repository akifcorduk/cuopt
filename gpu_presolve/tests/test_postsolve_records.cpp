/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"

#include <gprs/postsolve.hpp>

#include <cmath>
#include <cstdint>
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

// Columns without entries, bounds or cost, and nrows free empty rows.
model_t bare_model(const std::vector<uint8_t>& col_type, int32_t nrows)
{
  model_t m;
  m.nrows = nrows;
  m.ncols = col_type.size();
  m.row_start.assign(nrows + 1, 0);
  m.row_lower.assign(nrows, -inf);
  m.row_upper.assign(nrows, inf);
  m.col_lower.assign(m.ncols, -inf);
  m.col_upper.assign(m.ncols, inf);
  m.cost.assign(m.ncols, 0.);
  m.col_type = col_type;
  return m;
}

void check_equal(const std::vector<double>& a, const std::vector<double>& b)
{
  GPRS_REQUIRE(a.size() == b.size());
  for (size_t k = 0; k < a.size(); ++k) {
    GPRS_CHECK_EQ(a[k], b[k]);
  }
}

// How a column of a random stack gets its value.
struct definition_t {
  record_kind_t kind = record_kind_t::parallel_row;  // stands for: left in the reduced model
  int32_t rep        = -1;
  double scale       = 0.;
  double offset      = 0.;  // the value of a fixed column
};

// The value of a column from its definition alone, whatever the order of the records.
double value_of(const std::vector<definition_t>& def,
                const std::vector<double>& x_left,
                int32_t col)
{
  if (def[col].kind == record_kind_t::fixed_col) { return def[col].offset; }
  if (def[col].kind == record_kind_t::alias) {
    return def[col].scale * value_of(def, x_left, def[col].rep) + def[col].offset;
  }
  return x_left[col];
}

}  // namespace

GPRS_TEST(postsolve_init_empties_the_stack)
{
  postsolve_t ps;
  ps.init(bare_model({1, 0, 0, 1}, 3));
  GPRS_CHECK_EQ(ps.nrows, 3);
  GPRS_CHECK_EQ(ps.ncols, 4);
  GPRS_CHECK(ps.col_type == std::vector<uint8_t>({1, 0, 0, 1}));
  GPRS_CHECK(ps.col_removed == std::vector<uint8_t>(4, 0));
  GPRS_CHECK(ps.order.empty() && ps.fixed_cols.empty() && ps.aliases.empty());
  GPRS_CHECK(ps.parallel_rows.empty() && ps.row_origin.empty() && ps.col_origin.empty());

  ps.push_fixed_col(0, 2.);
  ps.push_alias(1, 2, 0, 3., 1.);
  ps.push_parallel_row(2, 1, -1.);
  ps.col_origin = {2, 3};
  ps.row_origin = {0};
  ps.init(bare_model({0, 1}, 1));
  GPRS_CHECK_EQ(ps.nrows, 1);
  GPRS_CHECK_EQ(ps.ncols, 2);
  GPRS_CHECK(ps.col_type == std::vector<uint8_t>({0, 1}));
  GPRS_CHECK(ps.col_removed == std::vector<uint8_t>(2, 0));
  GPRS_CHECK(ps.order.empty() && ps.fixed_cols.empty() && ps.aliases.empty());
  GPRS_CHECK(ps.parallel_rows.empty() && ps.row_origin.empty() && ps.col_origin.empty());
}

GPRS_TEST(postsolve_push_keeps_kinds_in_order)
{
  postsolve_t ps;
  ps.init(bare_model({0, 0, 0, 0, 0}, 4));
  ps.push_fixed_col(3, 1.5);
  ps.push_alias(0, 1, 2, -2., 0.25);
  ps.push_parallel_row(1, 0, 3.);
  ps.push_fixed_col(2, -4.);
  ps.push_alias(1, 4, 3, 0.5, -1.);
  ps.push_parallel_row(3, 2, -0.5);

  std::vector<record_kind_t> order = {record_kind_t::fixed_col,
                                      record_kind_t::alias,
                                      record_kind_t::parallel_row,
                                      record_kind_t::fixed_col,
                                      record_kind_t::alias,
                                      record_kind_t::parallel_row};
  GPRS_CHECK(ps.order == order);
  GPRS_REQUIRE(ps.fixed_cols.size() == 2 && ps.aliases.size() == 2);
  GPRS_REQUIRE(ps.parallel_rows.size() == 2);
  GPRS_CHECK_EQ(ps.fixed_cols[0].col, 3);
  GPRS_CHECK_EQ(ps.fixed_cols[0].value, 1.5);
  GPRS_CHECK_EQ(ps.fixed_cols[1].col, 2);
  GPRS_CHECK_EQ(ps.fixed_cols[1].value, -4.);
  GPRS_CHECK_EQ(ps.aliases[0].col, 0);
  GPRS_CHECK_EQ(ps.aliases[0].rep, 1);
  GPRS_CHECK_EQ(ps.aliases[0].row, 2);
  GPRS_CHECK_EQ(ps.aliases[0].scale, -2.);
  GPRS_CHECK_EQ(ps.aliases[0].offset, 0.25);
  GPRS_CHECK_EQ(ps.aliases[1].col, 1);
  GPRS_CHECK_EQ(ps.aliases[1].rep, 4);
  GPRS_CHECK_EQ(ps.aliases[1].row, 3);
  GPRS_CHECK_EQ(ps.aliases[1].scale, 0.5);
  GPRS_CHECK_EQ(ps.aliases[1].offset, -1.);
  GPRS_CHECK_EQ(ps.parallel_rows[0].removed_row, 1);
  GPRS_CHECK_EQ(ps.parallel_rows[0].kept_row, 0);
  GPRS_CHECK_EQ(ps.parallel_rows[0].ratio, 3.);
  GPRS_CHECK_EQ(ps.parallel_rows[1].removed_row, 3);
  GPRS_CHECK_EQ(ps.parallel_rows[1].kept_row, 2);
  GPRS_CHECK_EQ(ps.parallel_rows[1].ratio, -0.5);
  // Column 4 is only a representative.
  GPRS_CHECK(ps.col_removed == std::vector<uint8_t>({1, 1, 1, 1, 0}));
}

// Kinds interleaved, so that element k of a kind's array is not record k of the stack.
GPRS_TEST(postsolve_hand_sequence)
{
  postsolve_t ps;
  ps.init(bare_model({0, 0, 0, 0, 1, 0, 1}, 3));
  ps.push_fixed_col(5, 3.5);
  ps.push_alias(0, 3, 0, 2., 1.);  // x0 = 2 x3 + 1
  ps.push_parallel_row(2, 1, 4.);
  ps.push_fixed_col(6, -7.);
  ps.push_alias(1, 4, 1, -0.5, 4.);  // x1 = -x4 / 2 + 4
  ps.push_fixed_col(2, 0.125);
  ps.col_origin = {3, 4};

  std::vector<double> x;
  ps.postsolve({7.25, 6.}, x);
  check_equal(x, {15.5, 1., 0.125, 7.25, 6., 3.5, -7.});

  std::vector<double> crushed;
  ps.crush(x, crushed);
  check_equal(crushed, {7.25, 6.});
  // crush reads the columns that are left and nothing else.
  ps.crush({-1., -2., -3., 0.75, 9., -4., -5.}, crushed);
  check_equal(crushed, {0.75, 9.});
}

// x0 = 2 x1 + 1, then x1 = -x2 / 2 + 4, then x2 = 6: every record needs the value that a later
// record sets, so a walk from the first record to the last gives x0 = 1 and x1 = 4.
GPRS_TEST(postsolve_undoes_in_reverse_order)
{
  postsolve_t ps;
  ps.init(bare_model({0, 0, 0, 0}, 2));
  ps.push_alias(0, 1, 0, 2., 1.);
  ps.push_alias(1, 2, 1, -0.5, 4.);
  ps.push_fixed_col(2, 6.);
  ps.col_origin = {3};

  std::vector<double> x;
  ps.postsolve({7.25}, x);
  check_equal(x, {3., 1., 6., 7.25});

  // The same with a representative that leaves through an alias of its own later on.
  ps.init(bare_model({0, 0, 0, 0}, 3));
  ps.push_alias(0, 1, 0, 2., 1.);    // x0 = 2 x1 + 1
  ps.push_alias(1, 2, 1, -0.5, 4.);  // x1 = -x2 / 2 + 4
  ps.push_alias(2, 3, 2, 4., -2.);   // x2 = 4 x3 - 2
  ps.col_origin = {3};
  ps.postsolve({1.5}, x);
  check_equal(x, {5., 2., 4., 1.5});
}

GPRS_TEST(postsolve_rounds_integers_before_the_undo)
{
  postsolve_t ps;
  ps.init(bare_model({1, 0, 1, 0, 1}, 2));
  ps.push_alias(1, 0, 0, 0.5, 0.25);  // continuous x1 = x0 / 2 + 1 / 4, x0 integer
  ps.push_alias(2, 4, 1, -3., 2.);    // integer x2 = -3 x4 + 2
  ps.col_origin = {0, 3, 4};

  std::vector<double> x;
  ps.postsolve({2.9999997, 2.9999997, -4.0000002}, x);
  // The aliases see the rounded representatives; the continuous column 3 is not rounded.
  check_equal(x, {3., 1.75, 14., 2.9999997, -4.});

  ps.postsolve({-0.4, 0.6, 0.6}, x);
  check_equal(x, {-0., 0.25, -1., 0.6, 1.});
}

GPRS_TEST(postsolve_fixed_columns_at_huge_values)
{
  postsolve_t ps;
  ps.init(bare_model({0, 1, 0, 0, 0, 1, 0}, 3));
  ps.push_alias(3, 0, 0, 2., 1.);        // x3 = 2 x0 + 1 with x0 fixed at 1e21 below
  ps.push_alias(4, 2, 1, 1e-290, -0.5);  // x4 = 1e-290 x2 - 1 / 2 with x2 fixed at -1e300
  ps.push_fixed_col(0, 1e21);
  ps.push_fixed_col(1, 0x1p53);
  ps.push_fixed_col(2, -1e300);
  ps.push_fixed_col(5, -1e21);
  ps.col_origin = {6};

  std::vector<double> x;
  ps.postsolve({0.5}, x);
  GPRS_REQUIRE(x.size() == 7);
  GPRS_CHECK_EQ(x[0], 1e21);
  GPRS_CHECK_EQ(x[1], 0x1p53);
  GPRS_CHECK_EQ(x[2], -1e300);
  GPRS_CHECK_EQ(x[3], 2. * 1e21 + 1.);
  GPRS_CHECK_EQ(x[3], 2e21);
  GPRS_CHECK_EQ(x[4], 1e-290 * -1e300 - 0.5);
  GPRS_CHECK(std::fabs(x[4] + 1e10) < 1.);
  GPRS_CHECK_EQ(x[5], -1e21);
  GPRS_CHECK_EQ(x[6], 0.5);

  std::vector<double> crushed;
  ps.crush(x, crushed);
  check_equal(crushed, {0.5});
}

GPRS_TEST(postsolve_empty_stack_and_empty_models)
{
  postsolve_t ps;
  ps.init(bare_model({0, 1, 0}, 0));
  ps.col_origin = {0, 1, 2};
  std::vector<double> x, crushed;
  ps.postsolve({1.5, 2.4, -3.5}, x);
  check_equal(x, {1.5, 2., -3.5});
  ps.crush(x, crushed);
  check_equal(crushed, x);

  // Every column removed: the reduced model has no column.
  ps.init(bare_model({0, 1}, 1));
  ps.push_alias(0, 1, 0, 0.5, 0.);
  ps.push_fixed_col(1, 4.);
  ps.postsolve({}, x);
  check_equal(x, {2., 4.});
  ps.crush(x, crushed);
  GPRS_CHECK(crushed.empty());

  ps.init(bare_model({}, 0));
  x = {1.};
  ps.postsolve({}, x);
  GPRS_CHECK(x.empty());
  crushed = {1.};
  ps.crush(x, crushed);
  GPRS_CHECK(crushed.empty());
}

// A stack as a presolve would build it: every record removes a column that is still there, and
// an alias names a representative that is still there. Compared with the value each column
// has by its own definition.
GPRS_TEST(postsolve_random_stacks_against_definitions)
{
  const double continuous_scale[] = {0.5, -0.25, 3., -1., 1e-3, 1e6, -7.5};
  const double huge[]             = {1e21, -1e21, 1e300, -3e18};
  int64_t records = 0, removed_by_alias = 0;
  for (uint64_t seed = 1; seed <= 400; ++seed) {
    rng_t rng{seed};
    int32_t ncols = rng.uniform(1, 40), nrows = rng.uniform(2, 6);
    std::vector<uint8_t> col_type(ncols);
    for (uint8_t& t : col_type) {
      t = rng.chance(0.5);
    }
    postsolve_t ps;
    ps.init(bare_model(col_type, nrows));

    std::vector<definition_t> def(ncols);
    std::vector<int32_t> left(ncols);
    for (int32_t j = 0; j < ncols; ++j) {
      left[j] = j;
    }
    int32_t steps = rng.uniform(0, ncols + 5);
    for (int32_t step = 0; step < steps; ++step) {
      int32_t pick = rng.uniform(0, 9);
      if (pick < 2) {
        int32_t removed = rng.uniform(0, nrows - 1), kept = rng.uniform(0, nrows - 2);
        ps.push_parallel_row(removed, kept + (kept >= removed), rng.chance(0.5) ? 2. : -0.5);
        continue;
      }
      if (left.empty()) { continue; }
      int32_t at  = rng.uniform(0, left.size() - 1);
      int32_t col = left[at];
      int32_t rep = left[rng.uniform(0, left.size() - 1)];
      if (pick < 6 && rep != col && (!col_type[col] || col_type[rep])) {
        def[col].kind = record_kind_t::alias;
        def[col].rep  = rep;
        if (col_type[col]) {
          def[col].scale  = rng.chance(0.5) ? rng.uniform(1, 4) : -rng.uniform(1, 4);
          def[col].offset = rng.uniform(-9, 9);
        } else {
          def[col].scale  = continuous_scale[rng.uniform(0, 6)];
          def[col].offset = rng.uniform(-40, 40) / 8.;
        }
        ps.push_alias(col, rep, rng.uniform(0, nrows - 1), def[col].scale, def[col].offset);
        ++removed_by_alias;
      } else {
        def[col].kind = record_kind_t::fixed_col;
        if (rng.chance(0.2)) {
          def[col].offset = huge[rng.uniform(0, 3)];
        } else {
          def[col].offset = col_type[col] ? rng.uniform(-50, 50) : rng.uniform(-400, 400) / 8.;
        }
        ps.push_fixed_col(col, def[col].offset);
      }
      left.erase(left.begin() + at);
      ++records;
    }
    ps.col_origin = left;

    // Integer columns of the reduced point are off an integer by less than a half.
    std::vector<double> x_reduced(left.size()), x_left(ncols, 0.), rounded(left.size());
    for (size_t k = 0; k < left.size(); ++k) {
      if (col_type[left[k]]) {
        rounded[k]   = rng.uniform(-20, 20);
        x_reduced[k] = rounded[k] + rng.uniform(-3, 3) / 8.;
      } else {
        rounded[k] = x_reduced[k] = rng.uniform(-320, 320) / 16.;
      }
      x_left[left[k]] = rounded[k];
    }
    std::vector<double> expected(ncols);
    for (int32_t j = 0; j < ncols; ++j) {
      expected[j] = value_of(def, x_left, j);
    }

    std::vector<double> x, crushed, again;
    ps.postsolve(x_reduced, x);
    check_equal(x, expected);
    ps.crush(x, crushed);
    check_equal(crushed, rounded);
    ps.postsolve(crushed, again);
    check_equal(again, x);

    // The removed columns of the point do not matter to crush, and postsolve reproduces them.
    std::vector<double> spoiled = x;
    for (int32_t j = 0; j < ncols; ++j) {
      if (ps.col_removed[j]) { spoiled[j] = 12345.5; }
    }
    ps.crush(spoiled, crushed);
    check_equal(crushed, rounded);
    ps.postsolve(crushed, again);
    check_equal(again, x);
  }
  GPRS_CHECK(records > 2000);
  GPRS_CHECK(removed_by_alias > 500);
}

// minimise x0 + 2 x1 + 3 x2 + 1 / 2 with  r0: x0 - 2 x1 = 1,  r1: x0 + x1 + x2 <= 10,
// x0 in [0, 20], x1 in [0, 5], x2 integer in [2, 2], reduced by hand: x2 = 2, x0 = 2 x1 + 1.
// What is left is  minimise 4 x1 + 15 / 2  with  3 x1 <= 7,  x1 in [0, 5].
GPRS_TEST(postsolve_crush_keeps_feasibility_and_objective)
{
  model_t original;
  original.nrows            = 2;
  original.ncols            = 3;
  original.row_start        = {0, 2, 5};
  original.col_index        = {0, 1, 0, 1, 2};
  original.value            = {1., -2., 1., 1., 1.};
  original.row_lower        = {1., -inf};
  original.row_upper        = {1., 10.};
  original.col_lower        = {0., 0., 2.};
  original.col_upper        = {20., 5., 2.};
  original.cost             = {1., 2., 3.};
  original.col_type         = {0, 0, 1};
  original.objective_offset = 0.5;

  model_t reduced;
  reduced.nrows            = 1;
  reduced.ncols            = 1;
  reduced.row_start        = {0, 1};
  reduced.col_index        = {0};
  reduced.value            = {3.};
  reduced.row_lower        = {-inf};
  reduced.row_upper        = {7.};
  reduced.col_lower        = {0.};
  reduced.col_upper        = {5.};
  reduced.cost             = {4.};
  reduced.col_type         = {0};
  reduced.objective_offset = 7.5;

  postsolve_t ps;
  ps.init(original);
  ps.push_fixed_col(2, 2.);
  ps.push_alias(0, 1, 0, 2., 1.);
  ps.col_origin = {1};
  ps.row_origin = {1};

  tolerances_t tol;
  std::vector<double> x = {5., 2., 2.}, crushed, back;
  point_check_t check   = check_point(original, x);
  GPRS_CHECK(point_feasible(check, tol));
  GPRS_CHECK_EQ(check.objective, 15.5);
  ps.crush(x, crushed);
  check_equal(crushed, {2.});
  check = check_point(reduced, crushed);
  GPRS_CHECK(point_feasible(check, tol));
  GPRS_CHECK_EQ(check.objective, 15.5);
  ps.postsolve(crushed, back);
  check_equal(back, x);

  // Every feasible point of the reduced model comes back feasible with its objective value,
  // and a point outside comes back outside.
  for (double x1 : {0., 0.3, 7. / 3.}) {
    ps.postsolve({x1}, back);
    point_check_t there = check_point(reduced, {x1}), here = check_point(original, back);
    GPRS_CHECK(point_feasible(there, tol));
    GPRS_CHECK(point_feasible(here, tol));
    GPRS_CHECK(std::fabs(there.objective - here.objective) <= 1e-12);
  }
  ps.postsolve({2.5}, back);
  GPRS_CHECK(!point_feasible(check_point(reduced, {2.5}), tol));
  GPRS_CHECK(!point_feasible(check_point(original, back), tol));
}
