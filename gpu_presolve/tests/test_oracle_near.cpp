/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The parts of the oracle for rows parallel up to singleton columns: the singleton columns of
// the generator, and into_class(), which takes a row with two line columns back into the class.

#include "harness.hpp"
#include "oracle.hpp"

#include <cmath>
#include <vector>

using namespace gprs;
using oracle::case_t;
using oracle::generator_options_t;

namespace {

bool same_model(const model_t& a, const model_t& b)
{
  return a.nrows == b.nrows && a.ncols == b.ncols && a.row_start == b.row_start &&
         a.col_index == b.col_index && a.value == b.value && a.row_lower == b.row_lower &&
         a.row_upper == b.row_upper && a.col_lower == b.col_lower && a.col_upper == b.col_upper &&
         a.cost == b.cost && a.col_type == b.col_type && a.objective_offset == b.objective_offset;
}

generator_options_t near_options()
{
  generator_options_t o;
  o.p_parallel_row  = 0.5;
  o.p_equation      = 0.45;
  o.p_singleton_col = 0.6;
  o.max_enum_cols   = 4;
  o.max_width       = 2;
  return o;
}

// Entries of every column of the model.
std::vector<int32_t> column_sizes(const model_t& m)
{
  std::vector<int32_t> size(m.ncols, 0);
  for (int32_t col : m.col_index) {
    ++size[col];
  }
  return size;
}

// Rows i < k that are proportional on their entries outside columns with one entry, on two
// such entries at least, with such a column in one of them and an equation among the two.
int32_t near_parallel_pairs(const model_t& m)
{
  std::vector<int32_t> size = column_sizes(m);
  auto core                 = [&](int32_t i) {
    std::vector<std::pair<int32_t, double>> entries;
    for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
      if (size[m.col_index[e]] > 1) { entries.emplace_back(m.col_index[e], m.value[e]); }
    }
    return entries;
  };
  int32_t pairs = 0;
  for (int32_t i = 0; i < m.nrows; ++i) {
    for (int32_t k = i + 1; k < m.nrows; ++k) {
      auto a = core(i), b = core(k);
      if (a.size() < 2 || a.size() != b.size()) { continue; }
      bool parallel = true;
      for (size_t n = 0; n < a.size(); ++n) {
        parallel = parallel && a[n].first == b[n].first &&
                   a[n].second * b[0].second == b[n].second * a[0].second;
      }
      bool singleton = (int64_t)a.size() < m.row_start[i + 1] - m.row_start[i] ||
                       (int64_t)b.size() < m.row_start[k + 1] - m.row_start[k];
      bool equation = m.row_lower[i] == m.row_upper[i] || m.row_lower[k] == m.row_upper[k];
      pairs += parallel && singleton && equation;
    }
  }
  return pairs;
}

}  // namespace

// A singleton column is behind the columns of the model without them, holds one entry, and
// leaves the case in the oracle class; without p_singleton_col the case is the one of before.
GPRS_TEST(oracle_generate_singleton_columns)
{
  generator_options_t o    = near_options();
  generator_options_t none = o;
  none.p_singleton_col     = 0.;
  int64_t columns = 0, line_columns = 0, continuous = 0, pairs = 0, cases_with = 0;
  for (uint64_t seed = 0; seed < 2000; ++seed) {
    case_t c = oracle::generate(seed, o);
    case_t d = oracle::generate(seed, o);
    GPRS_REQUIRE(same_model(c.base, d.base) && same_model(c.model, d.model));
    GPRS_REQUIRE(well_formed(c.base) && well_formed(c.model));
    std::vector<uint8_t> line;
    GPRS_REQUIRE(oracle::classify(c.base, o.max_width, line));
    GPRS_REQUIRE(oracle::classify_widest(c.base, oracle::check_max_assignments, line));
    oracle::classify(c.base, o.max_width, line);
    // The columns of the case without singleton columns come first.
    case_t plain = oracle::generate(seed, none);
    int32_t own  = c.base.ncols - plain.base.ncols;
    GPRS_REQUIRE(own >= 0 && own <= o.max_singleton_cols);
    std::vector<int32_t> size = column_sizes(c.base);
    for (int32_t j = plain.base.ncols; j < c.base.ncols; ++j) {
      GPRS_CHECK_EQ(size[j], 1);
      GPRS_CHECK(c.col_scale[j] == 1. || c.base.col_type[j] == 0);
      ++columns;
      line_columns += line[j];
      continuous += c.base.col_type[j] == 0;
      if (!line[j]) { GPRS_CHECK(c.base.col_upper[j] - c.base.col_lower[j] <= 1.); }
    }
    cases_with += own > 0;
    pairs += near_parallel_pairs(c.base);
  }
  // The test is not vacuous.
  GPRS_CHECK(cases_with > 1200);
  GPRS_CHECK(columns > 2500);
  GPRS_CHECK(line_columns > 300 && line_columns < columns / 2);
  GPRS_CHECK(continuous > 100);
  GPRS_CHECK(pairs > 600);

  // With the default the draws are those of before.
  generator_options_t defaults;
  GPRS_CHECK_EQ(defaults.p_singleton_col, 0.);
}

// Row 1 has the line columns x2 and x3. Row 0 is an equation with x2 as its only line column:
// subtracting it brings x0 and x1 into row 1 in the place of x2.
GPRS_TEST(oracle_into_class_takes_a_line_column_out_of_a_row)
{
  model_t m;
  m.nrows = 2;
  m.ncols = 4;
  // Row 0: x0 + x1 + x2 = 2. Row 1: x3 - 2 x2 <= 1, which is 2 x0 + 2 x1 + x3 <= 5.
  m.row_start        = {0, 3, 5};
  m.col_index        = {0, 1, 2, 2, 3};
  m.value            = {1., 1., 1., -2., 1.};
  m.row_lower        = {2., -inf};
  m.row_upper        = {2., 1.};
  m.col_lower        = {0., 0., -inf, -inf};
  m.col_upper        = {1., 2., inf, inf};
  m.cost             = {1., 1., 0., -1.};
  m.col_type         = {1, 1, 0, 0};
  m.objective_offset = 0.;
  GPRS_REQUIRE(well_formed(m));
  std::vector<uint8_t> line;
  GPRS_CHECK(!oracle::classify_widest(m, oracle::check_max_assignments, line));
  model_t in_class;
  GPRS_REQUIRE(oracle::into_class(m, oracle::check_max_assignments, in_class, line));
  GPRS_CHECK(line == std::vector<uint8_t>({0, 0, 1, 1}));
  GPRS_CHECK(in_class.row_start == std::vector<int64_t>({0, 3, 6}));
  GPRS_CHECK(in_class.col_index == std::vector<int32_t>({0, 1, 2, 0, 1, 3}));
  GPRS_CHECK(in_class.value == std::vector<double>({1., 1., 1., 2., 2., 1.}));
  GPRS_CHECK(in_class.row_lower == std::vector<double>({2., -inf}));
  GPRS_CHECK(in_class.row_upper == std::vector<double>({2., 5.}));
  GPRS_CHECK(in_class.col_lower == m.col_lower && in_class.cost == m.cost);
  // x3 as large as row 1 lets it be: 5 - 2 (x0 + x1), so both columns at 0 and the optimum -5.
  oracle::solution_t solution = oracle::solve(in_class, line);
  GPRS_REQUIRE(solution.outcome == oracle::outcome_t::optimal);
  GPRS_CHECK_EQ(solution.objective, -5.);
  GPRS_CHECK(solution.x == std::vector<double>({0., 0., 2., 5.}));
  GPRS_CHECK(point_feasible(check_point(m, solution.x), tolerances_t()));

  // A model of the class comes back as it is.
  model_t same;
  GPRS_REQUIRE(oracle::into_class(in_class, oracle::check_max_assignments, same, line));
  GPRS_CHECK(same_model(same, in_class));

  // Without an equation for one of the two columns the row stays outside the class: row 0 as
  // an inequality, and row 0 with both line columns.
  model_t inequality      = m;
  inequality.row_lower[0] = -inf;
  GPRS_CHECK(!oracle::into_class(inequality, oracle::check_max_assignments, in_class, line));
  model_t both   = m;
  both.row_start = {0, 4, 6};
  both.col_index = {0, 1, 2, 3, 2, 3};
  both.value     = {1., 1., 1., 1., -2., 1.};
  GPRS_REQUIRE(well_formed(both));
  GPRS_CHECK(!oracle::into_class(both, oracle::check_max_assignments, in_class, line));
}

// Two rows share the line column of one equation, a third has three line columns and needs
// two equations; a coefficient that cancels leaves the row.
GPRS_TEST(oracle_into_class_several_rows_and_columns)
{
  model_t m;
  m.nrows = 5;
  m.ncols = 6;
  // Row 0: x0 + x3 = 1. Row 1: x1 - x4 = 0. Rows 2 and 3 hold x3 and x5; row 4 holds x3, x4
  // and x5, and x0 with the coefficient that the subtraction of row 0 cancels.
  m.row_start = {0, 2, 4, 6, 8, 12};
  m.col_index = {0, 3, 1, 4, 3, 5, 3, 5, 0, 3, 4, 5};
  m.value     = {1., 1., 1., -1., 1., 1., 2., -1., 3., 3., 1., 1.};
  m.row_lower = {1., 0., -inf, 0., -inf};
  m.row_upper = {1., 0., 4., inf, 7.};
  m.col_lower = {0., 0., 0., -inf, -inf, 0.};
  m.col_upper = {1., 1., 1., inf, inf, 9.};
  m.cost      = {0., 0., 0., 0., 0., -1.};
  m.col_type  = {1, 1, 1, 0, 0, 0};
  GPRS_REQUIRE(well_formed(m));
  std::vector<uint8_t> line;
  model_t in_class;
  GPRS_REQUIRE(oracle::into_class(m, oracle::check_max_assignments, in_class, line));
  GPRS_CHECK(line == std::vector<uint8_t>({0, 0, 0, 1, 1, 1}));
  // Row 2: x5 - x0 <= 3. Row 3: -x5 - 2 x0 >= -2. Row 4: x5 + x1 <= 4.
  GPRS_CHECK(in_class.row_start == std::vector<int64_t>({0, 2, 4, 6, 8, 10}));
  GPRS_CHECK(in_class.col_index == std::vector<int32_t>({0, 3, 1, 4, 0, 5, 0, 5, 1, 5}));
  GPRS_CHECK(in_class.value == std::vector<double>({1., 1., 1., -1., -1., 1., -2., -1., 1., 1.}));
  GPRS_CHECK(in_class.row_lower == std::vector<double>({1., 0., -inf, -2., -inf}));
  GPRS_CHECK(in_class.row_upper == std::vector<double>({1., 0., 3., inf, 4.}));
  // x5 <= 3 + x0, x5 <= 2 - 2 x0 and x5 <= 4 - x1: at most 2, with x0 = 0.
  oracle::solution_t solution = oracle::solve(in_class, line);
  GPRS_REQUIRE(solution.outcome == oracle::outcome_t::optimal);
  GPRS_CHECK_EQ(solution.objective, -2.);
  GPRS_CHECK(point_feasible(check_point(m, solution.x), tolerances_t()));
}
