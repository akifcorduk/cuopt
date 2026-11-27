/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The structure check (check/structure_check.hpp) on results with row addition records: the
// contract of the stack, and the test of a row that had equations subtracted, on results
// written by hand where the proof must pass and where it must fail, and on the engine.

#include "../check/structure_check.hpp"
#include "harness.hpp"
#include "oracle.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <utility>
#include <vector>

using namespace gprs;
using gprs_check::structure_check;
using gprs_check::structure_report_t;

namespace {

struct row_t {
  std::vector<std::pair<int32_t, double>> entries;
  double lower;
  double upper;
};

// Continuous columns with zero cost; the entries of a row in any order.
model_t build(const std::vector<row_t>& rows,
              const std::vector<double>& col_lower,
              const std::vector<double>& col_upper)
{
  model_t m;
  m.nrows     = rows.size();
  m.ncols     = col_lower.size();
  m.col_lower = col_lower;
  m.col_upper = col_upper;
  m.cost.assign(m.ncols, 0.);
  m.col_type.assign(m.ncols, 0);
  m.row_start.push_back(0);
  for (const row_t& row : rows) {
    std::vector<std::pair<int32_t, double>> entries = row.entries;
    std::sort(entries.begin(), entries.end());
    for (const std::pair<int32_t, double>& entry : entries) {
      m.col_index.push_back(entry.first);
      m.value.push_back(entry.second);
    }
    m.row_start.push_back(m.value.size());
    m.row_lower.push_back(row.lower);
    m.row_upper.push_back(row.upper);
  }
  return m;
}

// A result with every column of the model left and the given rows of it, in the reduced model
// as `rows`; the records are pushed by the test.
result_t reduced_to(const model_t& model,
                    const std::vector<row_t>& rows,
                    const std::vector<int32_t>& row_origin)
{
  result_t result;
  result.status  = status_t::reduced;
  result.reduced = build(rows, model.col_lower, model.col_upper);
  result.postsolve.init(model);
  result.postsolve.row_origin = row_origin;
  result.postsolve.col_origin.resize(model.ncols);
  std::iota(result.postsolve.col_origin.begin(), result.postsolve.col_origin.end(), 0);
  return result;
}

void add(result_t& result, int32_t row, int32_t equation_row, double factor)
{
  result.postsolve.order.push_back(record_kind_t::row_addition);
  result.postsolve.row_additions.push_back(row_addition_record_t{row, equation_row, factor});
}

void merge(result_t& result, int32_t removed_row, int32_t kept_row, double ratio)
{
  result.postsolve.order.push_back(record_kind_t::parallel_row);
  result.postsolve.parallel_rows.push_back(parallel_row_record_t{removed_row, kept_row, ratio});
}

bool silent(const structure_report_t& s)
{
  return s.stack_error.empty() && s.lost_integers == 0 && s.bound_failures == 0 &&
         s.row_failures == 0 && s.kept_failures == 0 && s.added_failures == 0 &&
         s.postsolve_failures == 0;
}

// Columns x, y, s, t in this order.
constexpr int32_t x = 0, y = 1, s = 2, t = 3;

// Row 0: lower <= x + y + s <= 5, row 1: x + y = 3, with x, y in [0, 3], s in [-10, 10] and
// t in [0, 1], which no row holds.
model_t pair_model(double lower = -inf)
{
  return build({{{{x, 1.}, {y, 1.}, {s, 1.}}, lower, 5.}, {{{x, 1.}, {y, 1.}}, 3., 3.}},
               {0., 0., -10., 0.},
               {3., 3., 10., 1.});
}

const row_t equation{{{x, 1.}, {y, 1.}}, 3., 3.};

// The result of row 0 <- row 0 - row 1 on pair_model: s <= 2 and the equation.
result_t pair_result(const model_t& m, double lower, double upper)
{
  result_t result = reduced_to(m, {{{{s, 1.}}, lower, upper}, equation}, {0, 1});
  add(result, 0, 1, 1.);
  return result;
}

}  // namespace

GPRS_TEST(e2e_structure_reads_a_row_addition_record)
{
  model_t m            = pair_model();
  structure_report_t r = structure_check(m, pair_result(m, -inf, 2.));
  GPRS_CHECK(r.composed && silent(r));
  GPRS_CHECK_EQ(r.added_rows, 1);
  GPRS_CHECK_EQ(r.added_tested, 1);
  GPRS_CHECK_EQ(r.added_untested, 0);
  GPRS_CHECK_EQ(r.kept_tested, 1);
  GPRS_CHECK_EQ(r.rows_tested, 0);
  GPRS_CHECK_EQ(r.added_multiplier, 1.);

  auto error = [&](int32_t row, int32_t equation_row, double factor) {
    result_t result                   = pair_result(m, -inf, 2.);
    result.postsolve.row_additions[0] = row_addition_record_t{row, equation_row, factor};
    structure_report_t report         = structure_check(m, result);
    GPRS_CHECK(!report.composed);
    return report.stack_error;
  };
  GPRS_CHECK_EQ(error(0, 0, 1.), "row addition record 0");
  GPRS_CHECK_EQ(error(2, 1, 1.), "row addition record 0");
  GPRS_CHECK_EQ(error(0, -1, 1.), "row addition record 0");
  GPRS_CHECK_EQ(error(0, 1, 0.), "row addition record 0");
  GPRS_CHECK_EQ(error(0, 1, inf), "row addition record 0");
  GPRS_CHECK_EQ(error(0, 1, std::nan("")), "row addition record 0");

  // The order and the array of the records disagree.
  result_t result = pair_result(m, -inf, 2.);
  result.postsolve.row_additions.clear();
  GPRS_CHECK_EQ(structure_check(m, result).stack_error, "row addition records missing at 0");
  result = pair_result(m, -inf, 2.);
  result.postsolve.order.clear();
  GPRS_CHECK_EQ(structure_check(m, result).stack_error, "records outside the order 0");

  // A row that an earlier record took out of the model: the removed row of a parallel record,
  // the row of an alias.
  for (bool equation_gone : {false, true}) {
    result = reduced_to(m, {{{{s, 1.}}, -inf, 2.}, equation}, {0, 1});
    merge(result, equation_gone ? 1 : 0, equation_gone ? 0 : 1, 1.);
    add(result, 0, 1, 1.);
    GPRS_CHECK_EQ(structure_check(m, result).stack_error, "rows of row addition record 1");
  }
  result = reduced_to(m, {{{{s, 1.}}, -inf, 2.}, equation}, {0, 1});
  result.reduced =
    build({{{{2, 1.}}, -inf, 2.}, {{{x, 1.}, {y, 1.}}, 3., 3.}}, {0., 0., 0.}, {3., 3., 4.});
  result.postsolve.col_origin = {x, y, s};
  result.postsolve.order.push_back(record_kind_t::alias);
  result.postsolve.aliases.push_back(alias_record_t{t, s, 1, 1., 0.});
  add(result, 0, 1, 1.);
  GPRS_CHECK_EQ(structure_check(m, result).stack_error, "rows of row addition record 1");
  // The row may be merged afterwards.
  result = pair_result(m, -inf, 2.);
  merge(result, 0, 1, 1.);
  GPRS_CHECK(structure_check(m, result).stack_error.empty());

  // The messages of the other kinds are those of before.
  result = pair_result(m, -inf, 2.);
  merge(result, 0, 1, 0.);
  GPRS_CHECK_EQ(structure_check(m, result).stack_error, "parallel record 1");
  result = pair_result(m, -inf, 2.);
  result.postsolve.order.push_back(record_kind_t::parallel_row);
  GPRS_CHECK_EQ(structure_check(m, result).stack_error, "parallel records missing at 1");
}

GPRS_TEST(e2e_structure_proves_a_row_through_its_equation)
{
  // x + y + s <= 5 is s <= 2 plus the equation; s <= 2.5 admits x + y + s = 5.5.
  model_t m = pair_model();
  GPRS_CHECK(silent(structure_check(m, pair_result(m, -inf, 2.))));
  structure_report_t r = structure_check(m, pair_result(m, -inf, 2.5));
  GPRS_CHECK_EQ(r.added_failures, 1);
  GPRS_CHECK_EQ(r.added_tested, 1);
  GPRS_CHECK_EQ(r.added_worst_row, 0);
  GPRS_CHECK_EQ(r.added_worst, 0.5 / 3.);
  GPRS_CHECK_EQ(r.kept_failures, 0);
  // A tighter row is a restriction, which this check does not see.
  GPRS_CHECK(silent(structure_check(m, pair_result(m, -inf, 1.))));

  // Both sides: 1 <= x + y + s <= 5 is -2 <= s <= 2.
  m = pair_model(1.);
  GPRS_CHECK(silent(structure_check(m, pair_result(m, -2., 2.))));
  GPRS_CHECK_EQ(structure_check(m, pair_result(m, -2.5, 2.)).added_failures, 1);
  GPRS_CHECK_EQ(structure_check(m, pair_result(m, -inf, 2.)).added_failures, 1);

  // The factor of the record is not read: the multiplier comes from the rows.
  result_t result                          = pair_result(m, -2., 2.);
  result.postsolve.row_additions[0].factor = -7.;
  GPRS_CHECK(silent(structure_check(m, result)));

  // Without the record the row is tested against its own reduced row alone, and fails.
  result = pair_result(m, -2., 2.);
  result.postsolve.order.clear();
  result.postsolve.row_additions.clear();
  r = structure_check(m, result);
  GPRS_CHECK_EQ(r.kept_failures, 1);
  GPRS_CHECK_EQ(r.added_rows, 0);

  // Rows scaled after the subtraction: 4 * s <= 8 and (x + y) / 4 = 3 / 4. The multiplier of
  // the equation is 4 in the units of the reduced rows.
  m      = pair_model();
  result = reduced_to(m, {{{{s, 4.}}, -inf, 8.}, {{{x, 0.25}, {y, 0.25}}, 0.75, 0.75}}, {0, 1});
  add(result, 0, 1, 1.);
  r = structure_check(m, result);
  GPRS_CHECK(silent(r));
  GPRS_CHECK_EQ(r.added_multiplier, 4.);
  result.reduced.row_upper[0] = 8.5;
  GPRS_CHECK_EQ(structure_check(m, result).added_failures, 1);

  // A column of the equation without bounds leaves the proof as it is.
  m.col_lower[x] = -inf;
  m.col_upper[x] = inf;
  GPRS_CHECK(silent(structure_check(m, pair_result(m, -inf, 2.))));
  GPRS_CHECK_EQ(structure_check(m, pair_result(m, -inf, 2.5)).added_failures, 1);
}

GPRS_TEST(e2e_structure_proves_a_row_with_the_singleton_column_of_its_equation)
{
  // Row 0: 1 <= x + y + s <= 5, row 1: x + y + t = 3. Row 0 less row 1 is -2 <= s - t <= 2.
  model_t m = build({{{{x, 1.}, {y, 1.}, {s, 1.}}, 1., 5.}, {{{x, 1.}, {y, 1.}, {t, 1.}}, 3., 3.}},
                    {0., 0., -10., -1.},
                    {3., 3., 10., 1.});
  const row_t with_single{{{x, 1.}, {y, 1.}, {t, 1.}}, 3., 3.};
  auto check = [&](double entry, double lower, double upper, double scale) {
    result_t result = reduced_to(
      m, {{{{s, scale}, {t, scale * entry}}, scale * lower, scale * upper}, with_single}, {0, 1});
    add(result, 0, 1, 1.);
    structure_report_t r = structure_check(m, result);
    GPRS_CHECK(r.composed);
    GPRS_CHECK_EQ(r.added_tested, 1);
    GPRS_CHECK_EQ(r.added_untested, 0);
    GPRS_CHECK_EQ(r.kept_failures, 0);
    return r.added_failures;
  };
  for (double scale : {1., 3., 0.125}) {
    GPRS_CHECK_EQ(check(-1., -2., 2., scale), 0);
    // The entry with the wrong sign, half of it, none of it.
    GPRS_CHECK_EQ(check(1., -2., 2., scale), 1);
    GPRS_CHECK_EQ(check(-0.5, -2., 2., scale), 1);
    GPRS_CHECK_EQ(check(-1.5, -2., 2., scale), 1);
    // Sides that were not moved, or moved the wrong way.
    GPRS_CHECK_EQ(check(-1., 1., 5., scale), 1);
    GPRS_CHECK_EQ(check(-1., 4., 8., scale), 1);
    GPRS_CHECK_EQ(check(-1., -2., 2.5, scale), 1);
    GPRS_CHECK_EQ(check(-1., -2.5, 2., scale), 1);
  }
  // With t fixed by its bounds the entry does not matter.
  m.col_lower[t] = 1.;
  GPRS_CHECK_EQ(check(-1., -2., 2., 1.), 0);
  GPRS_CHECK_EQ(check(5., 4., 8., 1.), 0);
}

GPRS_TEST(e2e_structure_proves_a_row_that_left_the_model_after_a_subtraction)
{
  // s <= 2 holds over the bounds of s and was deleted: the equation alone proves the row.
  model_t m       = pair_model();
  m.col_upper[s]  = 2.;
  result_t result = reduced_to(m, {equation}, {1});
  add(result, 0, 1, 1.);
  structure_report_t r = structure_check(m, result);
  GPRS_CHECK(silent(r));
  GPRS_CHECK_EQ(r.added_tested, 1);
  GPRS_CHECK_EQ(r.rows_tested, 0);
  GPRS_CHECK_EQ(r.kept_tested, 1);
  // Over wider bounds it does not hold.
  result.reduced.col_upper[s] = 2.5;
  r                           = structure_check(m, result);
  GPRS_CHECK_EQ(r.added_failures, 1);
  GPRS_CHECK_EQ(r.row_failures, 0);
  GPRS_CHECK_EQ(r.bound_failures, 1);
  // Without the equation in the reduced model nothing proves it: the row is one that left the
  // model, and x + y ranges up to 6.
  result = reduced_to(m, {}, {});
  add(result, 0, 1, 1.);
  r = structure_check(m, result);
  GPRS_CHECK_EQ(r.added_rows, 1);
  GPRS_CHECK_EQ(r.added_tested, 0);
  GPRS_CHECK_EQ(r.rows_tested, 2);
  GPRS_CHECK_EQ(r.row_failures, 2);
}

GPRS_TEST(e2e_structure_tests_a_row_whose_equation_left_the_model_as_before)
{
  // The equation fixed x = 1 and y = 2 and was deleted; the row is 3 + s <= 5.
  model_t m = pair_model();
  result_t result;
  result.status  = status_t::reduced;
  result.reduced = build({{{{0, 1.}}, -inf, 2.}}, {0., 0.}, {4., 1.});
  result.postsolve.init(m);
  result.postsolve.row_origin = {0};
  result.postsolve.col_origin = {s, t};
  add(result, 0, 1, 1.);
  result.postsolve.push_fixed_col(x, 1.);
  result.postsolve.push_fixed_col(y, 2.);
  structure_report_t r = structure_check(m, result);
  GPRS_CHECK(silent(r));
  GPRS_CHECK_EQ(r.added_rows, 1);
  GPRS_CHECK_EQ(r.added_tested, 0);
  GPRS_CHECK_EQ(r.kept_tested, 1);
  GPRS_CHECK_EQ(r.rows_tested, 1);
  result.reduced.row_upper[0] = 2.5;
  GPRS_CHECK_EQ(structure_check(m, result).kept_failures, 1);
}

GPRS_TEST(e2e_structure_follows_the_merges_after_a_subtraction)
{
  // Row 0: 2 x + 2 y = 6, row 1: x + y = 3, row 2: x + y + s <= 5. Row 1 is subtracted from
  // row 2 and then merged into row 0, whose reduced row is the equation of row 2 from then on.
  const row_t twice{{{x, 2.}, {y, 2.}}, 6., 6.};
  model_t m = build(
    {twice, equation, {{{x, 1.}, {y, 1.}, {s, 1.}}, -inf, 5.}}, {0., 0., 0., 0.}, {3., 3., 4., 1.});
  result_t result = reduced_to(m, {twice, {{{s, 1.}}, -inf, 2.}}, {0, 2});
  add(result, 2, 1, 1.);
  merge(result, 1, 0, 0.5);
  structure_report_t r = structure_check(m, result);
  GPRS_CHECK(silent(r));
  GPRS_CHECK_EQ(r.added_tested, 1);
  GPRS_CHECK_EQ(r.kept_parallel, 1);
  GPRS_CHECK_EQ(r.added_multiplier, 0.5);
  result.reduced.row_upper[1] = 2.5;
  GPRS_CHECK_EQ(structure_check(m, result).added_failures, 1);
  // The merged row lost its lower side: the row that needs it fails, and so does row 1.
  result                      = reduced_to(m, {twice, {{{s, 1.}}, -inf, 2.}}, {0, 2});
  result.reduced.row_lower[0] = -inf;
  add(result, 2, 1, -1.);
  merge(result, 1, 0, 0.5);
  r = structure_check(m, result);
  GPRS_CHECK_EQ(r.added_failures, 0);
  GPRS_CHECK_EQ(r.kept_failures, 2);

  // Row 0: 2 s + 2 t <= 6, row 1: x + y + s + t <= 5, row 2: x + y = 3. Row 1 less row 2 is
  // s + t <= 2, which is merged into row 0: 2 s + 2 t <= 4.
  m = build(
    {{{{s, 2.}, {t, 2.}}, -inf, 6.}, {{{x, 1.}, {y, 1.}, {s, 1.}, {t, 1.}}, -inf, 5.}, equation},
    {0., 0., 0., 0.},
    {3., 3., 4., 1.});
  for (double side : {4., 6.}) {
    result = reduced_to(m, {{{{s, 2.}, {t, 2.}}, -inf, side}, equation}, {0, 2});
    add(result, 1, 2, 1.);
    merge(result, 1, 0, 0.5);
    r = structure_check(m, result);
    GPRS_CHECK(r.composed);
    GPRS_CHECK_EQ(r.added_tested, 1);
    GPRS_CHECK_EQ(r.kept_tested, 2);
    GPRS_CHECK_EQ(r.kept_failures, 0);
    GPRS_CHECK_EQ(r.added_failures, side == 4. ? 0 : 1);
  }
}

GPRS_TEST(e2e_structure_proves_a_row_through_several_equations)
{
  // Columns x, y, s, t, z, w. Row 0: x + y + z + w + s <= 9, row 1: x + y = 3, row 2:
  // z + w = 4. Row 0 less both is s <= 2.
  constexpr int32_t z = 4, w = 5;
  const row_t second{{{z, 1.}, {w, 1.}}, 4., 4.};
  model_t m = build({{{{x, 1.}, {y, 1.}, {z, 1.}, {w, 1.}, {s, 1.}}, -inf, 9.}, equation, second},
                    {0., 0., 0., 0., 0., 0.},
                    {3., 3., 4., 1., 4., 4.});
  for (double side : {2., 2.5}) {
    result_t result = reduced_to(m, {{{{s, 1.}}, -inf, side}, equation, second}, {0, 1, 2});
    add(result, 0, 1, 1.);
    add(result, 0, 2, 1.);
    structure_report_t r = structure_check(m, result);
    GPRS_CHECK(r.composed);
    GPRS_CHECK_EQ(r.added_rows, 1);
    GPRS_CHECK_EQ(r.added_tested, 1);
    GPRS_CHECK_EQ(r.added_failures, side == 2. ? 0 : 1);
  }
  // One of the two records is missing: the row cannot be proved.
  result_t result = reduced_to(m, {{{{s, 1.}}, -inf, 2.}, equation, second}, {0, 1, 2});
  add(result, 0, 2, 1.);
  GPRS_CHECK_EQ(structure_check(m, result).added_failures, 1);

  // Equations that share a column. Columns x, y, s, t, z. Row 0: x + y + z + s <= 9, row 1:
  // x + y = 3, row 2: y + z = 4. Row 0 less both is s - y <= 2.
  const row_t sharing{{{y, 1.}, {z, 1.}}, 4., 4.};
  m = build({{{{x, 1.}, {y, 1.}, {z, 1.}, {s, 1.}}, -inf, 9.}, equation, sharing},
            {0., 0., 0., 0., 0.},
            {3., 3., 4., 1., 4.});
  for (double side : {2., 2.5}) {
    result = reduced_to(m, {{{{s, 1.}, {y, -1.}}, -inf, side}, equation, sharing}, {0, 1, 2});
    add(result, 0, 1, 1.);
    add(result, 0, 2, 1.);
    structure_report_t r = structure_check(m, result);
    GPRS_CHECK(r.composed);
    GPRS_CHECK_EQ(r.added_rows, 1);
    GPRS_CHECK_EQ(r.added_tested, 1);
    GPRS_CHECK_EQ(r.added_failures, side == 2. ? 0 : 1);
  }

  // An equation that had an equation subtracted itself. Row 0: x + y + s <= 5, row 1:
  // x + y + t = 3, row 2: x + y = 1. Row 0 less row 1 is s - t <= 2; then row 1 less row 2 is
  // t = 2. Row 0 is made of its reduced row and of both equations, row 1 of its own and row 2.
  m = build({{{{x, 1.}, {y, 1.}, {s, 1.}}, -inf, 5.},
             {{{x, 1.}, {y, 1.}, {t, 1.}}, 3., 3.},
             {{{x, 1.}, {y, 1.}}, 1., 1.}},
            {0., 0., -10., 0.},
            {3., 3., 10., 2.});
  for (double side : {2., 2.5}) {
    result = reduced_to(
      m,
      {{{{s, 1.}, {t, -1.}}, -inf, side}, {{{t, 1.}}, 2., 2.}, {{{x, 1.}, {y, 1.}}, 1., 1.}},
      {0, 1, 2});
    add(result, 0, 1, 1.);
    add(result, 1, 2, 1.);
    structure_report_t r = structure_check(m, result);
    GPRS_CHECK(r.composed);
    GPRS_CHECK_EQ(r.added_rows, 2);
    GPRS_CHECK_EQ(r.added_tested, 2);
    GPRS_CHECK_EQ(r.added_untested, 0);
    GPRS_CHECK_EQ(r.added_failures, side == 2. ? 0 : 1);
  }
  // In the other order of the records row 0 is not made of row 2: row 1 was what it is now
  // when it was subtracted.
  result =
    reduced_to(m,
               {{{{s, 1.}, {t, -1.}}, -inf, 2.}, {{{t, 1.}}, 2., 2.}, {{{x, 1.}, {y, 1.}}, 1., 1.}},
               {0, 1, 2});
  add(result, 1, 2, 1.);
  add(result, 0, 1, 1.);
  GPRS_CHECK_EQ(structure_check(m, result).added_failures, 1);
}

GPRS_TEST(e2e_structure_counts_a_row_it_cannot_decide_as_not_tested)
{
  model_t m;
  // More equations than a row is tested with. Columns s, then x and y of each equation.
  for (int32_t count : {8, (int32_t)gprs_check::max_row_equations + 1}) {
    std::vector<row_t> rows(1), reduced(1);
    rows[0].lower = -inf;
    rows[0].upper = count + 2.;
    rows[0].entries.emplace_back(0, 1.);
    reduced[0] = row_t{{{0, 1.}}, -inf, 2.};
    for (int32_t j = 0; j < count; ++j) {
      row_t e{{{1 + 2 * j, 1.}, {2 + 2 * j, 1.}}, 1., 1.};
      rows[0].entries.insert(rows[0].entries.end(), e.entries.begin(), e.entries.end());
      rows.push_back(e);
      reduced.push_back(e);
    }
    model_t many =
      build(rows, std::vector<double>(1 + 2 * count, 0.), std::vector<double>(1 + 2 * count, 4.));
    std::vector<int32_t> origin(count + 1);
    std::iota(origin.begin(), origin.end(), 0);
    result_t result = reduced_to(many, reduced, origin);
    for (int32_t j = 0; j < count; ++j) {
      add(result, 0, 1 + j, 1.);
    }
    structure_report_t r = structure_check(many, result);
    GPRS_CHECK(silent(r));
    bool tested = count == 8;
    GPRS_CHECK_EQ(r.added_tested, tested ? 1 : 0);
    GPRS_CHECK_EQ(r.added_untested, tested ? 0 : 1);
  }

  // No point within the bounds meets the equation x + y = 7: a bound that proves the row does
  // so all the same, a search that finds none says nothing.
  m = pair_model();
  const row_t unmet{{{x, 1.}, {y, 1.}}, 7., 7.};
  for (double side : {-2., 2.}) {
    result_t result = reduced_to(m, {{{{s, 1.}}, -inf, side}, unmet}, {0, 1});
    add(result, 0, 1, 1.);
    structure_report_t r = structure_check(m, result);
    GPRS_CHECK_EQ(r.added_failures, 0);
    GPRS_CHECK_EQ(r.added_tested, side == -2. ? 1 : 0);
    GPRS_CHECK_EQ(r.added_untested, side == -2. ? 0 : 1);
  }

  // Row 0: x + y <= 0, row 1: x + y + t = 3, with x and y fixed at 1/2: row 0 is violated at
  // every postsolved point. Its reduced row is t >= 2.5 and that of row 1 is t = value, with t
  // in [2, 3]. Each is met by a point within the bounds; with value 2 no point meets both, and
  // the search has no smallest bound. With 2.75 one does, and the row fails.
  m = build({{{{x, 1.}, {y, 1.}}, -inf, 0.}, {{{x, 1.}, {y, 1.}, {t, 1.}}, 3., 3.}},
            {0., 0., 0., 2.},
            {3., 3., 4., 3.});
  for (double value : {2., 2.75}) {
    result_t result;
    result.status = status_t::reduced;
    result.reduced =
      build({{{{1, -1.}}, -inf, -2.5}, {{{1, 1.}}, value, value}}, {0., 2.}, {4., 3.});
    result.postsolve.init(m);
    result.postsolve.row_origin = {0, 1};
    result.postsolve.col_origin = {s, t};
    add(result, 0, 1, 1.);
    result.postsolve.push_fixed_col(x, 0.5);
    result.postsolve.push_fixed_col(y, 0.5);
    structure_report_t r = structure_check(m, result);
    GPRS_CHECK(r.composed);
    GPRS_CHECK_EQ(r.added_rows, 1);
    GPRS_CHECK_EQ(r.added_tested, value == 2. ? 0 : 1);
    GPRS_CHECK_EQ(r.added_untested, value == 2. ? 1 : 0);
    GPRS_CHECK_EQ(r.added_failures, value == 2. ? 0 : 1);
  }
}

namespace {

// One row of a small linear program: low <= a * r <= high.
struct lp_row_t {
  std::vector<double> a;
  double low;
  double high;
};

// The largest value of g * r over lower <= r <= upper and the rows, by enumeration of the basic
// solutions: every row is inactive or at one of its sides, as many columns as rows are active
// take the values that the active rows give, and the other columns sit on a bound. False when
// no point meets the rows. The bounds are finite and there are at most three rows.
bool brute_maximum(const std::vector<double>& g,
                   const std::vector<double>& lower,
                   const std::vector<double>& upper,
                   const std::vector<lp_row_t>& rows,
                   double& best)
{
  constexpr double eps = 1e-9;
  int32_t n = g.size(), m = rows.size();
  bool found = false;
  best       = -inf;
  std::vector<double> r(n);
  int32_t states = 1;
  for (int32_t i = 0; i < m; ++i) {
    states *= 3;
  }
  for (int32_t state = 0; state < states; ++state) {
    std::vector<int32_t> active;
    std::vector<double> side;
    bool usable = true;
    for (int32_t i = 0, rest = state; i < m; ++i, rest /= 3) {
      if (rest % 3 == 0) { continue; }
      double at = rest % 3 == 1 ? rows[i].low : rows[i].high;
      usable    = usable && std::isfinite(at);
      active.push_back(i);
      side.push_back(at);
    }
    if (!usable) { continue; }
    int32_t k = active.size();
    for (int32_t basic = 0; basic < (1 << n); ++basic) {
      std::vector<int32_t> cols;
      for (int32_t j = 0; j < n; ++j) {
        if ((basic >> j) & 1) { cols.push_back(j); }
      }
      if ((int32_t)cols.size() != k) { continue; }
      for (int32_t ends = 0; ends < (1 << n); ++ends) {
        if (ends & basic) { continue; }
        for (int32_t j = 0; j < n; ++j) {
          r[j] = (ends >> j) & 1 ? upper[j] : lower[j];
        }
        // Gaussian elimination with full pivoting on the active rows and the basic columns.
        std::vector<std::vector<double>> a(k, std::vector<double>(k + 1));
        for (int32_t p = 0; p < k; ++p) {
          double rhs = side[p];
          for (int32_t j = 0; j < n; ++j) {
            if (!((basic >> j) & 1)) { rhs -= rows[active[p]].a[j] * r[j]; }
          }
          for (int32_t q = 0; q < k; ++q) {
            a[p][q] = rows[active[p]].a[cols[q]];
          }
          a[p][k] = rhs;
        }
        std::vector<int32_t> order(k);
        std::iota(order.begin(), order.end(), 0);
        bool singular = false;
        for (int32_t p = 0; p < k && !singular; ++p) {
          int32_t pr = p, pc = p;
          for (int32_t u = p; u < k; ++u) {
            for (int32_t v = p; v < k; ++v) {
              if (std::fabs(a[u][v]) > std::fabs(a[pr][pc])) {
                pr = u;
                pc = v;
              }
            }
          }
          if (std::fabs(a[pr][pc]) < eps) {
            singular = true;
            break;
          }
          std::swap(a[p], a[pr]);
          for (int32_t u = 0; u < k; ++u) {
            std::swap(a[u][p], a[u][pc]);
          }
          std::swap(order[p], order[pc]);
          for (int32_t u = 0; u < k; ++u) {
            if (u == p) { continue; }
            double f = a[u][p] / a[p][p];
            for (int32_t v = p; v <= k; ++v) {
              a[u][v] -= f * a[p][v];
            }
          }
        }
        if (singular) { continue; }
        bool feasible = true;
        for (int32_t p = 0; p < k; ++p) {
          int32_t j = cols[order[p]];
          r[j]      = a[p][k] / a[p][p];
          feasible  = feasible && r[j] >= lower[j] - eps && r[j] <= upper[j] + eps;
        }
        double value = 0.;
        for (int32_t j = 0; j < n; ++j) {
          value += g[j] * r[j];
        }
        for (const lp_row_t& row : rows) {
          double activity = 0.;
          for (int32_t j = 0; j < n; ++j) {
            activity += row.a[j] * r[j];
          }
          feasible = feasible && activity >= row.low - eps && activity <= row.high + eps;
        }
        if (!feasible) { continue; }
        found = true;
        best  = std::max(best, value);
      }
    }
  }
  return found;
}

}  // namespace

GPRS_TEST(e2e_structure_search_finds_the_smallest_bound)
{
  // Random small rows with one or two equations. The smallest Lagrangian bound is the largest
  // value over the points that meet the rows (duality), which the enumeration gives; and the
  // search must say so when no point meets them.
  using gprs_check::combined_t;
  oracle::rng_t rng(20261127);
  int64_t compared = 0, without_point = 0, with_shared = 0;
  for (int32_t trial = 0; trial < 4000; ++trial) {
    int32_t n = rng.uniform(2, 6);
    std::vector<double> g(n), lower(n), upper(n), point(n);
    for (int32_t j = 0; j < n; ++j) {
      g[j]     = rng.uniform(-3, 3);
      lower[j] = rng.uniform(-2, 1);
      upper[j] = lower[j] + rng.uniform(0, 3);
      point[j] = rng.uniform((int32_t)lower[j], (int32_t)upper[j]);
    }
    // Sides around the activity of `point`, so that most cases have a point.
    auto sides = [&](lp_row_t& row, bool equation) {
      double activity = 0.;
      for (int32_t j = 0; j < n; ++j) {
        activity += row.a[j] * point[j];
      }
      if (rng.chance(0.15)) { activity += rng.uniform(-3, 3); }
      row.low  = equation || rng.chance(0.6) ? activity - (equation ? 0 : rng.uniform(0, 2)) : -inf;
      row.high = equation || rng.chance(0.6) ? activity + (equation ? 0 : rng.uniform(0, 2)) : inf;
    };
    std::vector<lp_row_t> rows;
    bool own = rng.chance(0.8);
    lp_row_t own_row{std::vector<double>(n, 0.), -inf, inf};
    if (own) {
      for (int32_t j = 0; j < n; ++j) {
        own_row.a[j] = rng.chance(0.6) ? rng.uniform(-2, 2) : 0;
      }
      sides(own_row, false);
      rows.push_back(own_row);
    }
    // One or two equations, of which one in four is a row with two sides, as a wrong engine
    // can leave it. They share columns with each other and with the row's own reduced row.
    int32_t count = rng.uniform(1, 2);
    for (int32_t e = 0; e < count; ++e) {
      lp_row_t equation{std::vector<double>(n, 0.), 0., 0.};
      for (int32_t j = 0; j < n; ++j) {
        if (rng.chance(0.5)) {
          equation.a[j] = rng.chance(0.5) ? rng.uniform(1, 3) : -rng.uniform(1, 3);
        }
      }
      sides(equation, !rng.chance(0.25));
      rows.push_back(equation);
    }
    combined_t p;
    p.own  = own;
    p.rows = rows.size();
    for (int32_t j = 0; j < n; ++j) {
      p.columns.push_back(combined_t::column_t{g[j], std::fabs(g[j]), lower[j], upper[j]});
      for (const lp_row_t& row : rows) {
        p.a.push_back(row.a[j]);
      }
    }
    for (const lp_row_t& row : rows) {
      p.low.push_back(row.low);
      p.high.push_back(row.high);
    }
    gprs_check::hyperplanes_t h(p);
    gprs_check::smallest_bound_t most, least;
    int64_t visited = 0;
    gprs_check::visit_vertices(p, h, 0, [&](const double* v) {
      ++visited;
      most.visit(p, 1., v);
      least.visit(p, -1., v);
    });
    GPRS_CHECK((double)visited <= h.count());
    double expected = 0.;
    bool has_point  = brute_maximum(g, lower, upper, rows, expected);
    GPRS_CHECK_EQ(gprs_check::no_common_point(p), !has_point);
    if (!has_point) {
      ++without_point;
      continue;
    }
    ++compared;
    with_shared += !h.general.empty();
    if (!(std::fabs(most.value - expected) <= 1e-9 * std::max(1., std::fabs(expected)))) {
      test::fail(__FILE__,
                 __LINE__,
                 "trial " + std::to_string(trial) + ": bound " + std::to_string(most.value) +
                   ", largest value " + std::to_string(expected));
    }
    // The smallest value, with the sign of the search turned.
    for (double& v : g) {
      v = -v;
    }
    GPRS_REQUIRE(brute_maximum(g, lower, upper, rows, expected));
    if (!(std::fabs(least.value - expected) <= 1e-9 * std::max(1., std::fabs(expected)))) {
      test::fail(__FILE__,
                 __LINE__,
                 "trial " + std::to_string(trial) + ", smallest value: bound " +
                   std::to_string(least.value) + ", largest value " + std::to_string(expected));
    }
  }
  GPRS_CHECK(compared > 2000 && without_point > 200 && with_shared > 1500);
}

GPRS_TEST(e2e_structure_rejects_a_row_that_is_no_equation_as_the_equation)
{
  // Row 1 is x + y <= 3 and was subtracted as if it were x + y = 3. The upper side of row 0
  // follows, with the multiplier on the finite side of row 1; its lower side does not.
  const row_t below{{{x, 1.}, {y, 1.}}, -inf, 3.};
  for (double lower : {-inf, 1.}) {
    model_t m =
      build({{{{x, 1.}, {y, 1.}, {s, 1.}}, lower, 5.}, below}, {0., 0., 0., 0.}, {3., 3., 4., 1.});
    result_t result = reduced_to(m, {{{{s, 1.}}, lower - 3., 2.}, below}, {0, 1});
    add(result, 0, 1, 1.);
    structure_report_t r = structure_check(m, result);
    GPRS_CHECK_EQ(r.added_tested, 1);
    GPRS_CHECK_EQ(r.added_failures, lower == 1. ? 1 : 0);
    // Subtracted the other way round, x + y >= 3 gives the lower side and not the upper one.
    result = reduced_to(m, {{{{s, 1.}}, lower - 3., 2.}, {{{x, 1.}, {y, 1.}}, 3., inf}}, {0, 1});
    add(result, 0, 1, 1.);
    r = structure_check(m, result);
    GPRS_CHECK_EQ(r.added_failures, 1);
    GPRS_CHECK_EQ(r.kept_failures, 1);
  }
}

GPRS_TEST(e2e_structure_passes_the_subtractions_of_the_engine)
{
  // x + y + z + s <= 5 less x + y + z = 3 is the bound s <= 2: the row leaves the model and
  // the equation stays.
  constexpr int32_t z = 3;
  model_t m =
    build({{{{x, 1.}, {y, 1.}, {z, 1.}, {s, 1.}}, -inf, 5.}, {{{x, 1.}, {y, 1.}, {z, 1.}}, 3., 3.}},
          {0., 0., 0., 0.},
          {3., 3., 4., 3.});
  m.col_type = {1, 1, 0, 1};
  m.cost     = {1., 2., -1., 3.};
  options_t options;
  options.primal_only = true;
  result_t result     = presolve(m, options);
  GPRS_REQUIRE(result.status == status_t::reduced);
  GPRS_CHECK_EQ(result.postsolve.row_additions.size(), 1u);
  structure_report_t r = structure_check(m, result);
  GPRS_CHECK(r.composed && silent(r));
  GPRS_CHECK_EQ(r.added_rows, 1);
  GPRS_CHECK_EQ(r.added_tested, 1);
  // The same result with the bound that the subtraction gave relaxed.
  GPRS_REQUIRE(result.reduced.ncols == 4 && result.reduced.col_upper[s] == 2.);
  result.reduced.col_upper[s] = 3.;
  r                           = structure_check(m, result);
  GPRS_CHECK_EQ(r.added_failures, 1);

  // Generated models with rows that are parallel up to singleton columns.
  oracle::generator_options_t g;
  g.p_parallel_row   = 0.5;
  g.p_equation       = 0.45;
  g.p_singleton_col  = 0.6;
  g.max_enum_cols    = 4;
  g.max_width        = 2;
  int64_t subtracted = 0, tested = 0, untested = 0, failed = 0;
  for (uint64_t seed = 9000000; seed < 9020000 && failed < 3; ++seed) {
    oracle::case_t c = oracle::generate(seed, g);
    for (bool primal_only : {false, true}) {
      options             = options_t();
      options.primal_only = primal_only;
      result              = presolve(c.model, options);
      if (!gprs_check::carries_model(result.status)) { continue; }
      r = structure_check(c.model, result);
      subtracted += !result.postsolve.row_additions.empty();
      tested += r.added_tested;
      untested += r.added_untested;
      if (silent(r)) { continue; }
      ++failed;
      test::fail(__FILE__,
                 __LINE__,
                 "seed " + std::to_string(seed) + ": " + r.stack_error + "\n" + oracle::to_code(c));
    }
  }
  if (std::getenv("GPRS_ORACLE_CENSUS")) {
    std::printf(
      "    results with a subtraction %lld, rows tested with an equation %lld, not "
      "tested %lld\n",
      (long long)subtracted,
      (long long)tested,
      (long long)untested);
  }
  GPRS_CHECK(subtracted > 300);
  GPRS_CHECK(tested > 250);
  GPRS_CHECK(untested * 20 < tested);
}
