/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "oracle.hpp"

#include <cmath>
#include <utility>

using namespace gprs;
using oracle::outcome_t;
using oracle::solution_t;

namespace {

struct col_t {
  double lower;
  double upper;
  double cost;
  uint8_t type;
};

struct row_t {
  double lower;
  std::vector<std::pair<int32_t, double>> entries;
  double upper;
};

model_t make_model(const std::vector<col_t>& cols, const std::vector<row_t>& rows, double offset)
{
  model_t m;
  m.nrows = rows.size();
  m.ncols = cols.size();
  for (const col_t& c : cols) {
    m.col_lower.push_back(c.lower);
    m.col_upper.push_back(c.upper);
    m.cost.push_back(c.cost);
    m.col_type.push_back(c.type);
  }
  m.row_start.push_back(0);
  for (const row_t& r : rows) {
    for (const std::pair<int32_t, double>& entry : r.entries) {
      m.col_index.push_back(entry.first);
      m.value.push_back(entry.second);
    }
    m.row_start.push_back(m.value.size());
    m.row_lower.push_back(r.lower);
    m.row_upper.push_back(r.upper);
  }
  m.objective_offset = offset;
  return m;
}

// Classifies with the width limit and solves; the model must be in the class.
solution_t solve_model(const model_t& m, int32_t max_width, int32_t keep_points = 0)
{
  std::vector<uint8_t> line;
  bool in_class = oracle::classify(m, max_width, line);
  GPRS_CHECK(well_formed(m));
  GPRS_CHECK(in_class);
  if (!in_class) { return solution_t(); }
  return oracle::solve(m, line, keep_points);
}

using point_t  = std::vector<double>;
using points_t = std::vector<std::vector<double>>;

}  // namespace

GPRS_TEST(oracle_classify_marks_line_columns)
{
  // Width 3, width 4, one infinite bound, continuous, crossing, huge.
  model_t m =
    make_model({{0., 3., 0., 1},
                {0., 4., 0., 1},
                {0., inf, 0., 1},
                {0., 1., 0., 0},
                {2., 1., 0., 1},
                {-1e22, 1e22, 0., 1},
                {0.5, 3.5, 0., 1}},
               {{-inf, {{0, 1.}, {1, 1.}, {4, 1.}}, 1.}, {0., {{0, 1.}, {2, 1.}, {6, 1.}}, 5.}},
               0.);
  GPRS_REQUIRE(well_formed(m));
  std::vector<uint8_t> line;
  GPRS_CHECK(oracle::classify(m, 3, line));
  GPRS_CHECK(line == std::vector<uint8_t>({0, 1, 1, 1, 0, 1, 0}));
  // At width 2, columns 0 and 1 share row 0 as line columns.
  GPRS_CHECK(!oracle::classify(m, 2, line));
  GPRS_CHECK(line == std::vector<uint8_t>({1, 1, 1, 1, 0, 1, 1}));
  GPRS_CHECK(oracle::classify(m, 4, line));
  GPRS_CHECK(line == std::vector<uint8_t>({0, 0, 1, 1, 0, 1, 0}));
  GPRS_CHECK_EQ(oracle::assignments(m, line), 0.);
  m.col_lower[4] = 1.;
  GPRS_CHECK_EQ(oracle::assignments(m, line), 4. * 5. * 1. * 3.);
}

GPRS_TEST(oracle_classify_widest_respects_the_limit)
{
  // Columns of width 1, 2 and 9; column 2 is alone in its row.
  model_t m = make_model({{0., 1., 0., 1}, {0., 2., 0., 1}, {0., 9., 0., 1}},
                         {{-inf, {{0, 1.}, {1, 1.}}, 2.}, {-inf, {{2, 1.}}, 4.}},
                         0.);
  std::vector<uint8_t> line;
  GPRS_CHECK(oracle::classify_widest(m, 1000., line));
  GPRS_CHECK(line == std::vector<uint8_t>({0, 0, 0}));
  GPRS_CHECK_EQ(oracle::assignments(m, line), 60.);
  GPRS_CHECK(oracle::classify_widest(m, 59., line));
  GPRS_CHECK(line == std::vector<uint8_t>({0, 0, 1}));
  GPRS_CHECK(oracle::classify_widest(m, 6., line));
  GPRS_CHECK(line == std::vector<uint8_t>({0, 0, 1}));
  // With column 1 on the line as well the model is in the class at 2 assignments; with all
  // three on the line row 0 holds two line columns.
  GPRS_CHECK(oracle::classify_widest(m, 5., line));
  GPRS_CHECK(line == std::vector<uint8_t>({0, 1, 1}));
  GPRS_CHECK(oracle::classify_widest(m, 2., line));
  GPRS_CHECK(!oracle::classify_widest(m, 1., line));
  // Two continuous columns in one row are outside the class for every limit.
  model_t bad = make_model({{0., 1., 0., 0}, {0., 1., 0., 0}}, {{0., {{0, 1.}, {1, 1.}}, 1.}}, 0.);
  GPRS_CHECK(!oracle::classify_widest(bad, 1e9, line));
}

GPRS_TEST(oracle_solve_models_without_rows_or_columns)
{
  model_t none = make_model({}, {}, 2.5);
  solution_t s = solve_model(none, 3, 4);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK_EQ(s.objective, 2.5);
  GPRS_CHECK(s.x.empty());
  GPRS_CHECK_EQ(s.feasible.size(), 1u);

  model_t rows_only = make_model({}, {{-1., {}, 1.}, {-inf, {}, 0.}}, -1.);
  s                 = solve_model(rows_only, 3);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK_EQ(s.objective, -1.);

  model_t bad_row = make_model({}, {{-1., {}, 1.}, {1., {}, 2.}}, 0.);
  GPRS_CHECK(solve_model(bad_row, 3, 4).outcome == outcome_t::infeasible);
  GPRS_CHECK(solve_model(bad_row, 3, 4).feasible.empty());

  model_t cols_only = make_model({{-2., 1., 3., 1}, {0., 4., -1., 0}, {1., 1., 2., 1}}, {}, 10.);
  s                 = solve_model(cols_only, 3);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK_EQ(s.objective, 10. - 6. - 4. + 2.);
  GPRS_CHECK(s.x == point_t({-2., 4., 1.}));
}

GPRS_TEST(oracle_solve_binary_knapsack)
{
  // max 3 x0 + 2 x1 + 4 x2 with 2 x0 + x1 + 3 x2 <= 4: x = (0, 1, 1), value 6.
  model_t m    = make_model({{0., 1., -3., 1}, {0., 1., -2., 1}, {0., 1., -4., 1}},
                            {{-inf, {{0, 2.}, {1, 1.}, {2, 3.}}, 4.}},
                         1.);
  solution_t s = solve_model(m, 1, 100);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK_EQ(s.objective, -5.);
  GPRS_CHECK(s.x == point_t({0., 1., 1.}));
  GPRS_CHECK(
    s.feasible ==
    points_t({{0., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}, {0., 1., 1.}, {1., 0., 0.}, {1., 1., 0.}}));
  // The >= form of the same row gives the complement.
  m.row_lower[0] = 5.;
  m.row_upper[0] = inf;
  s              = solve_model(m, 1, 100);
  GPRS_CHECK(s.feasible == points_t({{1., 0., 1.}, {1., 1., 1.}}));
  GPRS_CHECK_EQ(s.objective, 1. - 9.);
  m.row_lower[0] = 7.;
  GPRS_CHECK(solve_model(m, 1).outcome == outcome_t::infeasible);
}

GPRS_TEST(oracle_solve_continuous_line_column)
{
  // y >= 2 x - 1, y >= 1 - x, x in {0, 1, 2}: y >= 1, 1, 3.
  model_t m    = make_model({{0., 2., 0., 1}, {-inf, inf, 1., 0}},
                            {{-1., {{0, -2.}, {1, 1.}}, inf}, {1., {{0, 1.}, {1, 1.}}, inf}},
                         0.);
  solution_t s = solve_model(m, 2, 100);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK_EQ(s.objective, 1.);
  GPRS_CHECK(s.x == point_t({0., 1.}));
  GPRS_CHECK(s.feasible == points_t({{0., 1.}, {1., 1.}, {2., 3.}}));
  // min y - x: 1, 0, 1.
  m.cost[0] = -1.;
  s         = solve_model(m, 2);
  GPRS_CHECK_EQ(s.objective, 0.);
  GPRS_CHECK(s.x == point_t({1., 1.}));
  // min -y is unbounded; with y <= 2 the assignment x = 2 is infeasible and y = 2.
  m.cost[1] = -1.;
  GPRS_CHECK(solve_model(m, 2).outcome == outcome_t::unbounded);
  m.col_upper[1] = 2.;
  s              = solve_model(m, 2, 100);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK_EQ(s.objective, -3.);
  GPRS_CHECK(s.x == point_t({1., 2.}));
  GPRS_CHECK(s.feasible == points_t({{0., 2.}, {1., 2.}}));
}

GPRS_TEST(oracle_solve_negative_coefficient_of_the_line_column)
{
  // -2 y + x >= -7, that is y <= (x + 7) / 2; max y.
  model_t m =
    make_model({{0., 1., 0., 1}, {0., inf, -1., 0}}, {{-7., {{0, 1.}, {1, -2.}}, inf}}, 0.);
  solution_t s = solve_model(m, 1, 100);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK_EQ(s.objective, -4.);
  GPRS_CHECK(s.x == point_t({1., 4.}));
  GPRS_CHECK(s.feasible == points_t({{0., 3.5}, {1., 4.}}));
  // As an integer column y is rounded inwards.
  m.col_type[1] = 1;
  s             = solve_model(m, 1, 100);
  GPRS_CHECK(s.feasible == points_t({{0., 3.}, {1., 4.}}));
  // min y sits at its lower bound, and the row as an upper side makes y >= (x + 7) / 2.
  m.cost[1] = 1.;
  GPRS_CHECK(solve_model(m, 1).x == point_t({0., 0.}));
  m.row_lower[0] = -inf;
  m.row_upper[0] = -7.;
  s              = solve_model(m, 1, 100);
  GPRS_CHECK(s.feasible == points_t({{0., 4.}, {1., 4.}}));
  // Ranged: -9 <= x - 2 y <= -7 gives (x + 7) / 2 <= y <= (x + 9) / 2.
  m.row_lower[0] = -9.;
  m.col_type[1]  = 0;
  m.cost[1]      = -1.;
  s              = solve_model(m, 1, 100);
  GPRS_CHECK(s.feasible == points_t({{0., 4.5}, {1., 5.}}));
  GPRS_CHECK_EQ(s.objective, -5.);
}

GPRS_TEST(oracle_solve_integer_line_column_rounds_inwards)
{
  // 2 y >= 3: y >= 2 as an integer, 1.5 as a continuous column.
  model_t m    = make_model({{-inf, inf, 1., 1}}, {{3., {{0, 2.}}, inf}}, 0.);
  solution_t s = solve_model(m, 3);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK_EQ(s.objective, 2.);
  m.col_type[0] = 0;
  GPRS_CHECK_EQ(solve_model(m, 3).objective, 1.5);
  // 3 y <= 7, max y: 2. And -3 y >= -7 the same.
  m = make_model({{-inf, inf, -1., 1}}, {{-inf, {{0, 3.}}, 7.}}, 0.);
  GPRS_CHECK_EQ(solve_model(m, 3).objective, -2.);
  m = make_model({{-inf, inf, -1., 1}}, {{-7., {{0, -3.}}, inf}}, 0.);
  GPRS_CHECK_EQ(solve_model(m, 3).objective, -2.);
  // 2 y = 3 has no integer solution; 2 y = 4 has one; 3 <= 2 y <= 3.5 none; 3 <= 2 y <= 4 one.
  m = make_model({{-inf, inf, 0., 1}}, {{3., {{0, 2.}}, 3.}}, 0.);
  GPRS_CHECK(solve_model(m, 3).outcome == outcome_t::infeasible);
  m.row_lower[0] = m.row_upper[0] = 4.;
  GPRS_CHECK(solve_model(m, 3, 1).feasible == points_t({{2.}}));
  m.row_lower[0] = 3.;
  m.row_upper[0] = 3.5;
  GPRS_CHECK(solve_model(m, 3).outcome == outcome_t::infeasible);
  m.row_upper[0] = 4.;
  GPRS_CHECK(solve_model(m, 3, 1).feasible == points_t({{2.}}));
  // Negative values: -7 <= 2 y <= -5 leaves y = -3.
  m.row_lower[0] = -7.;
  m.row_upper[0] = -5.;
  GPRS_CHECK(solve_model(m, 3, 1).feasible == points_t({{-3.}}));
  // Fractional bounds of an integer line column: [-2.5, inf) starts at -2.
  m = make_model({{-2.5, inf, 1., 1}}, {}, 0.);
  GPRS_CHECK_EQ(solve_model(m, 3).objective, -2.);
}

GPRS_TEST(oracle_solve_unbounded_needs_a_feasible_point)
{
  // min -y, y >= 0: unbounded. A row that no x satisfies makes it infeasible instead.
  model_t m = make_model({{0., 1., 0., 1}, {0., inf, -1., 0}}, {{-inf, {{0, 1.}}, 5.}}, 0.);
  GPRS_CHECK(solve_model(m, 1).outcome == outcome_t::unbounded);
  m.row_upper[0] = -1.;
  GPRS_CHECK(solve_model(m, 1).outcome == outcome_t::infeasible);
  // Feasible for x = 1 only: unbounded, with the one feasible point at the finite end.
  m.row_lower[0] = 1.;
  m.row_upper[0] = inf;
  solution_t s   = solve_model(m, 1, 10);
  GPRS_CHECK(s.outcome == outcome_t::unbounded);
  GPRS_CHECK(s.feasible == points_t({{1., 0.}}));
  GPRS_CHECK(s.x.empty());
  // Bounded in the direction of the cost: optimal, whatever the other end is.
  m.cost[1] = 1.;
  s         = solve_model(m, 1);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK_EQ(s.objective, 0.);
  // Zero cost on a free column: optimal at 0. An integer column behaves like a continuous one.
  m = make_model({{-inf, inf, 0., 0}, {-inf, 4., 2., 1}}, {}, 0.);
  s = solve_model(m, 1, 1);
  GPRS_CHECK(s.outcome == outcome_t::unbounded);
  GPRS_CHECK(s.feasible == points_t({{0., 4.}}));
  m.cost[1] = -2.;
  s         = solve_model(m, 1);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK(s.x == point_t({0., 4.}));
  GPRS_CHECK_EQ(s.objective, -8.);
  // An empty column with a cost and an infinite bound, and a row that confines it.
  m = make_model({{-inf, 3., 1., 0}}, {}, 0.);
  GPRS_CHECK(solve_model(m, 1).outcome == outcome_t::unbounded);
  m = make_model({{-inf, 3., 1., 0}}, {{-6., {{0, 3.}}, inf}}, 0.);
  GPRS_CHECK_EQ(solve_model(m, 1).objective, -2.);
}

GPRS_TEST(oracle_solve_huge_bounds_are_finite)
{
  model_t m    = make_model({{-1e22, 5., 1., 0}}, {}, 0.);
  solution_t s = solve_model(m, 3);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK_EQ(s.objective, -1e22);
  m.col_type[0] = 1;
  GPRS_CHECK_EQ(solve_model(m, 3).objective, -1e22);
  // y + x >= -3 with x in {0, 1}: y >= -3 - x.
  m = make_model({{0., 1., 0., 1}, {-1e22, 1e21, 1., 0}}, {{-3., {{0, 1.}, {1, 1.}}, inf}}, 0.);
  s = solve_model(m, 3, 10);
  GPRS_CHECK_EQ(s.objective, -4.);
  GPRS_CHECK(s.x == point_t({1., -4.}));
  m.cost[1] = -2.;
  s         = solve_model(m, 3);
  GPRS_CHECK_EQ(s.objective, -2e21);
  GPRS_CHECK(s.x == point_t({0., 1e21}));
  // y <= 1e21 by its bound and y <= -2e21 by a row.
  m = make_model({{-inf, 1e21, -1., 0}}, {{-inf, {{0, 1.}}, -2e21}}, 0.);
  GPRS_CHECK_EQ(solve_model(m, 3).objective, 2e21);
}

GPRS_TEST(oracle_solve_crossing_and_fractional_bounds)
{
  model_t m = make_model({{2., 1., 0., 1}, {0., 1., 0., 1}}, {}, 0.);
  GPRS_CHECK(solve_model(m, 3, 5).outcome == outcome_t::infeasible);
  GPRS_CHECK(solve_model(m, 3, 5).feasible.empty());
  m = make_model({{1., 0., 0., 0}}, {}, 0.);
  GPRS_CHECK(solve_model(m, 3).outcome == outcome_t::infeasible);
  m = make_model({{0.5, 0.75, 0., 1}}, {}, 0.);
  GPRS_CHECK(solve_model(m, 3).outcome == outcome_t::infeasible);
  m            = make_model({{0.5, 2.5, 1., 1}, {-1.5, 0.5, -1., 1}}, {}, 0.);
  solution_t s = solve_model(m, 3, 10);
  GPRS_CHECK(s.feasible == points_t({{1., -1.}, {1., 0.}, {2., -1.}, {2., 0.}}));
  GPRS_CHECK_EQ(s.objective, 1.);
  // A row with crossing sides.
  m = make_model({{0., 3., 0., 1}}, {{2., {{0, 1.}}, 1.}}, 0.);
  GPRS_CHECK(solve_model(m, 3).outcome == outcome_t::infeasible);
  // A continuous column confined to a point by two rows, and to nothing by a third.
  m = make_model({{-inf, inf, 0., 0}},
                 {{-inf, {{0, 2.}}, 3.}, {1.5, {{0, 1.}}, inf}, {-inf, {{0, 1.}}, inf}},
                 0.);
  GPRS_CHECK(solve_model(m, 3, 1).feasible == points_t({{1.5}}));
  m.row_lower[1] = 1.75;
  GPRS_CHECK(solve_model(m, 3).outcome == outcome_t::infeasible);
}

GPRS_TEST(oracle_solve_places_line_columns_as_documented)
{
  // y in [x0, 10] by a row and a bound; x0 + x1 <= 2.
  model_t m    = make_model({{0., 1., 0., 1}, {0., 2., 0., 1}, {-inf, 10., 0., 0}},
                            {{-inf, {{0, 1.}, {1, 1.}}, 2.}, {0., {{0, -1.}, {2, 1.}}, inf}},
                         0.);
  solution_t s = solve_model(m, 2, 100);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK(s.x == point_t({0., 0., 0.}));
  GPRS_CHECK(s.feasible ==
             points_t({{0., 0., 0.}, {0., 1., 0.}, {0., 2., 0.}, {1., 0., 1.}, {1., 1., 1.}}));
  GPRS_CHECK(solve_model(m, 2, 2).feasible == points_t({{0., 0., 0.}, {0., 1., 0.}}));
  GPRS_CHECK(solve_model(m, 2).feasible.empty());
  m.cost[2] = -1.;
  s         = solve_model(m, 2, 100);
  GPRS_CHECK(s.feasible[4] == point_t({1., 1., 10.}));
  GPRS_CHECK_EQ(s.objective, -10.);
  // Cost towards an infinite end: the other end. No finite end: 0.
  m.cost[2]      = 1.;
  m.row_lower[1] = -inf;
  s              = solve_model(m, 2, 100);
  GPRS_CHECK(s.outcome == outcome_t::unbounded);
  GPRS_CHECK(s.feasible[0] == point_t({0., 0., 10.}));
  m.col_upper[2] = inf;
  GPRS_CHECK(solve_model(m, 2, 1).feasible[0] == point_t({0., 0., 0.}));
  m.cost[2] = 0.;
  GPRS_CHECK(solve_model(m, 2).outcome == outcome_t::optimal);
}

GPRS_TEST(oracle_solve_far_points_are_the_other_ends)
{
  // y in [x, 10] by a row and a bound, x in {0, 1}; z is an integer with 2 z <= 7 and z >= -3.
  model_t m    = make_model({{0., 1., 0., 1}, {-inf, 10., 0., 0}, {-3., inf, 1., 1}},
                            {{0., {{0, -1.}, {1, 1.}}, inf}, {-inf, {{2, 2.}}, 7.}},
                         0.);
  solution_t s = solve_model(m, 1, 100);
  GPRS_CHECK(s.feasible == points_t({{0., 0., -3.}, {1., 1., -3.}}));
  GPRS_CHECK(s.far == points_t({{0., 10., 3.}, {1., 10., 3.}}));
  // The costs turned round: the points change places.
  m.cost[1] = -1.;
  m.cost[2] = -1.;
  s         = solve_model(m, 1, 100);
  GPRS_CHECK(s.feasible == points_t({{0., 10., 3.}, {1., 10., 3.}}));
  GPRS_CHECK(s.far == points_t({{0., 0., -3.}, {1., 1., -3.}}));
  GPRS_CHECK(solve_model(m, 1, 1).far == points_t({{0., 0., -3.}}));
  GPRS_CHECK(solve_model(m, 1).far.empty());
  // An infinite other end: far_step * max(1, |value|) into the interval, on either side.
  m.cost[1]      = 0.;
  m.cost[2]      = 1.;
  m.col_upper[1] = inf;
  m.row_upper[1] = inf;
  s              = solve_model(m, 1, 100);
  GPRS_CHECK(s.feasible == points_t({{0., 0., -3.}, {1., 1., -3.}}));
  GPRS_CHECK(s.far == points_t({{0., 1024., -3. + 3. * 1024.}, {1., 1025., -3. + 3. * 1024.}}));
  m.row_lower[0] = -inf;
  m.col_upper[1] = -10.;
  m.col_lower[2] = -inf;
  m.col_upper[2] = 0.5;
  m.cost[2]      = 0.;
  s              = solve_model(m, 1, 1);
  GPRS_CHECK(s.feasible == points_t({{0., -10., 0.}}));
  GPRS_CHECK(s.far == points_t({{0., -10. - 10. * 1024., -1024.}}));
  // No finite end: 0 and far_step. One point: itself.
  m = make_model({{-inf, inf, 0., 0}, {2., 2., 0., 0}, {0.5, 1.5, 3., 1}}, {}, 0.);
  s = solve_model(m, 1, 1);
  GPRS_CHECK(s.feasible == points_t({{0., 2., 1.}}));
  GPRS_CHECK(s.far == points_t({{1024., 2., 1.}}));
  GPRS_CHECK_EQ(oracle::far_step, 1024.);
}

GPRS_TEST(oracle_solve_equation_with_line_column_and_two_line_columns)
{
  // x + y = 3, y >= 2, min 3 x + y = 3 + 2 x.
  model_t m = make_model({{0., 3., 3., 1}, {2., inf, 1., 0}}, {{3., {{0, 1.}, {1, 1.}}, 3.}}, 0.5);
  solution_t s = solve_model(m, 3, 100);
  GPRS_CHECK_EQ(s.objective, 3.5);
  GPRS_CHECK(s.feasible == points_t({{0., 3.}, {1., 2.}}));
  // y1 >= x, y2 >= 2 - x, min y1 + 2 y2 = 4 - x.
  m = make_model({{-inf, inf, 1., 0}, {0., 2., 0., 1}, {-inf, inf, 2., 1}},
                 {{0., {{0, 1.}, {1, -1.}}, inf}, {2., {{1, 1.}, {2, 1.}}, inf}},
                 0.);
  s = solve_model(m, 3, 100);
  GPRS_CHECK_EQ(s.objective, 2.);
  GPRS_CHECK(s.x == point_t({2., 2., 0.}));
  GPRS_CHECK(s.feasible == points_t({{0., 0., 2.}, {1., 1., 1.}, {2., 2., 0.}}));
}

GPRS_TEST(oracle_solve_ties_keep_the_first_assignment)
{
  model_t m = make_model(
    {{0., 1., 0., 1}, {0., 1., 1., 1}, {0., 1., 1., 1}}, {{1., {{1, 1.}, {2, 1.}}, inf}}, 0.);
  solution_t s = solve_model(m, 1);
  GPRS_CHECK_EQ(s.objective, 1.);
  GPRS_CHECK(s.x == point_t({0., 0., 1.}));
}

GPRS_TEST(oracle_solve_uses_the_tolerance_on_data_that_are_not_integer)
{
  // 0.1 + 0.2 is not 0.3 in double arithmetic.
  model_t m =
    make_model({{0., 1., -1., 1}, {0., 1., -1., 1}}, {{0.3, {{0, 0.1}, {1, 0.2}}, 0.3}}, 0.);
  solution_t s = solve_model(m, 1, 10);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK(s.feasible == points_t({{1., 1.}}));
  m.row_lower[0] = m.row_upper[0] = 0.300001;
  GPRS_CHECK(solve_model(m, 1).outcome == outcome_t::infeasible);
  // The same through a line column: 0.1 x0 + 0.2 x1 + y = 0.3 and y = 0 by another row.
  m = make_model({{0., 1., -1., 1}, {0., 1., -1., 1}, {-inf, inf, 0., 0}},
                 {{0.3, {{0, 0.1}, {1, 0.2}, {2, 1.}}, 0.3}, {0., {{2, 3.}}, 0.}},
                 0.);
  s = solve_model(m, 1, 10);
  GPRS_CHECK(s.outcome == outcome_t::optimal);
  GPRS_CHECK_EQ(s.objective, -2.);
  // An integer line column: 0.1 y = 0.3 has the solution 3 although 0.3 / 0.1 is not 3.
  GPRS_CHECK(0.3 / 0.1 != 3.);
  m = make_model({{-inf, inf, 1., 1}}, {{0.3, {{0, 0.1}}, 0.3}}, 0.);
  s = solve_model(m, 1, 10);
  GPRS_CHECK(s.feasible == points_t({{3.}}));
  // A side that is not an integer gives its row the tolerance: 1 <= 3 y <= 1 - 1e-12 is
  // accepted, 1 <= 3 y <= 1 - 1e-6 is not.
  m = make_model({{-inf, inf, 0., 0}}, {{1., {{0, 3.}}, inf}, {-inf, {{0, 3.}}, 1. - 1e-12}}, 0.);
  GPRS_CHECK(solve_model(m, 1).outcome == outcome_t::optimal);
  m.row_upper[1] = 1. - 1e-6;
  GPRS_CHECK(solve_model(m, 1).outcome == outcome_t::infeasible);
  m = make_model({{0., 1., 0., 1}, {0., 1., 0., 1}}, {{1., {{0, 1.}, {1, 1.}}, 1.}}, 0.);
  GPRS_CHECK_EQ(solve_model(m, 1, 10).feasible.size(), 2u);
}

GPRS_TEST(oracle_solve_tolerance_of_a_side_does_not_grow_with_the_other_side)
{
  // 0.25 <= 0.25 x with a huge finite upper side, as a fixed column at a huge bound leaves it
  // in a scaled row: x = 0 stays infeasible. The same for the mirrored row.
  model_t m    = make_model({{0., 1., 1., 1}}, {{0.25, {{0, 0.25}}, 7.5e19}}, 0.);
  solution_t s = solve_model(m, 1, 10);
  GPRS_CHECK(s.feasible == points_t({{1.}}));
  GPRS_CHECK_EQ(s.objective, 1.);
  m = make_model({{0., 1., -1., 1}}, {{-7.5e19, {{0, 0.25}}, 0.125}}, 0.);
  s = solve_model(m, 1, 10);
  GPRS_CHECK(s.feasible == points_t({{0.}}));
  // Two enumerated columns and a side that is not huge but far above the other one.
  m = make_model({{0., 1., 0., 1}, {0., 1., 0., 1}}, {{0.5, {{0, 0.5}, {1, 0.5}}, 1e12}}, 0.);
  GPRS_CHECK_EQ(solve_model(m, 1, 10).feasible.size(), 3u);
  // Through a line column each end has had its own tolerance: 0.25 y >= 0.25, y <= 7.5e19.
  m = make_model({{-inf, inf, 1., 0}}, {{0.25, {{0, 0.25}}, 7.5e19}}, 0.);
  GPRS_CHECK_EQ(solve_model(m, 1).objective, 1.);
}
