/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The structure check and the crush check (check/) on results with col_from_rows records:
// results written by hand, where each obligation of a record must pass and where it must fail,
// and results of the engine on generated models.

#include "../check/structure_check.hpp"
#include "harness.hpp"
#include "oracle.hpp"

#include <cmath>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace gprs;
using gprs_check::crush_check;
using gprs_check::crush_report_t;
using gprs_check::point_set_t;
using gprs_check::structure_check;
using gprs_check::structure_report_t;

namespace {

using entries_t = std::vector<std::pair<int32_t, double>>;

struct row_t {
  entries_t entries;
  double lower;
  double upper;
};

// Continuous columns with zero cost; the entries of a row in increasing column order.
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
    for (const auto& entry : row.entries) {
      m.col_index.push_back(entry.first);
      m.value.push_back(entry.second);
    }
    m.row_start.push_back(m.value.size());
    m.row_lower.push_back(row.lower);
    m.row_upper.push_back(row.upper);
  }
  return m;
}

// The model that is left of `model` with the given columns and rows; the entries of a row name
// columns of the reduced model. The records are pushed by the test.
result_t reduced_to(const model_t& model,
                    const std::vector<int32_t>& col_origin,
                    const std::vector<row_t>& rows,
                    const std::vector<int32_t>& row_origin)
{
  std::vector<double> lower, upper;
  for (int32_t col : col_origin) {
    lower.push_back(model.col_lower[col]);
    upper.push_back(model.col_upper[col]);
  }
  result_t result;
  result.status  = status_t::reduced;
  result.reduced = build(rows, lower, upper);
  result.postsolve.init(model);
  result.postsolve.row_origin = row_origin;
  result.postsolve.col_origin = col_origin;
  return result;
}

saved_row_t save(postsolve_t& ps, int32_t row, const entries_t& entries, double lower, double upper)
{
  saved_row_t saved{row, lower, upper, (int64_t)ps.saved_entries.col.size(), 0};
  for (const auto& entry : entries) {
    ps.saved_entries.col.push_back(entry.first);
    ps.saved_entries.value.push_back(entry.second);
  }
  saved.end = ps.saved_entries.col.size();
  return saved;
}

record_col_t column(int32_t col, double lower, double upper)
{
  return record_col_t{col, lower, upper, lower > -inf ? lower : upper < inf ? upper : 0.};
}

bool silent(const structure_report_t& s)
{
  return s.composed && s.lost_integers == 0 && s.bound_failures == 0 && s.row_failures == 0 &&
         s.kept_failures == 0 && s.added_failures == 0 && s.box_failures == 0 &&
         s.integral_failures == 0 && s.snapshot_failures == 0 && s.objective_failures == 0 &&
         s.postsolve_failures == 0;
}

// The engine with the singleton-column rules alone.
options_t only_singleton_cols(bool primal_only = false)
{
  options_t options;
  options.enabled.fill(false);
  options.enabled[rule::singleton_cols] = true;
  options.primal_only                   = primal_only;
  return options;
}

}  // namespace

// Fixed model 1. E: x1 + x2 + x3 = 3 and T: x1 + x2 + x3 + s <= 5 with x in [0, 3]: T less E is
// s <= 2, and then x1 leaves E as a zero-cost singleton column, which widens E to
// 0 <= x2 + x3 <= 3. T is proved from its reduced row and from E as it was saved, with x1 in
// it. The same result with E widened to an upper side of 4 fails: the reduced point x2 = 3,
// x3 = 1 postsolves to x1 = 0 and T = 6.
GPRS_TEST(e2e_columns_row_addition_before_an_interval_record)
{
  constexpr int32_t x1 = 0, x2 = 1, x3 = 2, s = 3;
  model_t m = build(
    {{{{x1, 1.}, {x2, 1.}, {x3, 1.}}, 3., 3.}, {{{x1, 1.}, {x2, 1.}, {x3, 1.}, {s, 1.}}, -inf, 5.}},
    {0., 0., 0., 0.},
    {3., 3., 3., 10.});
  auto result = [&](double upper) {
    result_t r =
      reduced_to(m, {x2, x3, s}, {{{{0, 1.}, {1, 1.}}, 0., upper}, {{{2, 1.}}, -inf, 2.}}, {0, 1});
    r.postsolve.push_row_addition(1, 0, 1.);
    saved_row_t saved = save(r.postsolve, 0, {{x2, 1.}, {x3, 1.}}, 3., 3.);
    r.postsolve.push_interval_cols({column(x1, 0., 3.)}, {1.}, saved);
    return r;
  };
  structure_report_t r = structure_check(m, result(3.), false);
  GPRS_CHECK(silent(r));
  GPRS_CHECK_EQ(r.interval_records, 1);
  GPRS_CHECK_EQ(r.snapshot_tested, 1);
  GPRS_CHECK_EQ(r.added_rows, 1);
  GPRS_CHECK_EQ(r.added_tested, 1);
  GPRS_CHECK_EQ(r.kept_tested, 1);
  GPRS_CHECK_EQ(r.rows_tested, 0);

  r = structure_check(m, result(4.), false);
  GPRS_CHECK(r.composed);
  GPRS_CHECK_EQ(r.snapshot_failures, 1);
  GPRS_CHECK_EQ(r.snapshot_worst_row, 0);
  // x2 + x3 reaches 4 where 3 is asked for.
  GPRS_CHECK_EQ(r.snapshot_worst, 1. / 4.);
  // The reduced point that shows it.
  std::vector<double> x;
  result(4.).postsolve.postsolve({3., 1., 2.}, x);
  GPRS_CHECK(x == std::vector<double>({0., 3., 1., 2.}));

  // Such a record has no place in a result of primal-only mode.
  GPRS_CHECK_EQ(structure_check(m, result(3.), true).stack_error,
                "interval or forcing record in a primal-only result at 1");
}

// A row that loses singleton columns twice is saved twice. The snapshot of the first record is
// proved from the second saved row and that of the second from the reduced row.
GPRS_TEST(e2e_columns_two_interval_records_of_one_row)
{
  constexpr int32_t s1 = 0, s2 = 1, y = 2, z = 3;
  model_t m =
    build({{{{s1, 1.}, {s2, 1.}, {y, 1.}, {z, 1.}}, 2., 5.}, {{{y, 1.}, {z, -1.}}, -3., 3.}},
          {0., 0., 0., 0.},
          {1., 1., 4., 4.});
  auto result = [&](double second_lower, double reduced_upper) {
    result_t r = reduced_to(
      m, {y, z}, {{{{0, 1.}, {1, 1.}}, 0., reduced_upper}, {{{0, 1.}, {1, -1.}}, -3., 3.}}, {0, 1});
    r.postsolve.push_interval_cols(
      {column(s1, 0., 1.)}, {1.}, save(r.postsolve, 0, {{s2, 1.}, {y, 1.}, {z, 1.}}, 2., 5.));
    r.postsolve.push_interval_cols(
      {column(s2, 0., 1.)}, {1.}, save(r.postsolve, 0, {{y, 1.}, {z, 1.}}, second_lower, 5.));
    return r;
  };
  structure_report_t r = structure_check(m, result(1., 5.), false);
  GPRS_CHECK(silent(r));
  GPRS_CHECK_EQ(r.interval_records, 2);
  GPRS_CHECK_EQ(r.snapshot_tested, 2);
  // The row after the first record as it never was: the second snapshot then asks for
  // y + z >= 1, which the reduced row does not give.
  r = structure_check(m, result(2., 5.), false);
  GPRS_CHECK_EQ(r.snapshot_failures, 1);
  // The reduced row too wide for the second record.
  r = structure_check(m, result(1., 6.), false);
  GPRS_CHECK_EQ(r.snapshot_failures, 1);
  // A second saved row that is wider than the first record left the row fails the first
  // snapshot: s2 + y + z >= 1 does not follow from s2 + y + z >= 0.5.
  r = structure_check(m, result(0.5, 5.), false);
  GPRS_CHECK_EQ(r.snapshot_failures, 1);
}

// A row that an interval record widened and a parallel record removed afterwards: its snapshot
// is proved from the reduced row of the kept row.
GPRS_TEST(e2e_columns_interval_record_before_a_parallel_record)
{
  constexpr int32_t s = 0, y = 1, z = 2;
  model_t m   = build({{{{s, 1.}, {y, 1.}, {z, 1.}}, 2., 5.}, {{{y, 2.}, {z, 2.}}, 0., 8.}},
                      {0., 0., 0.},
                      {1., 4., 4.});
  auto result = [&](double kept_lower) {
    result_t r = reduced_to(m, {y, z}, {{{{0, 2.}, {1, 2.}}, kept_lower, 8.}}, {1});
    r.postsolve.push_interval_cols(
      {column(s, 0., 1.)}, {1.}, save(r.postsolve, 0, {{y, 1.}, {z, 1.}}, 2., 5.));
    r.postsolve.push_parallel_row(0, 1, 0.5);
    return r;
  };
  structure_report_t r = structure_check(m, result(2.), false);
  GPRS_CHECK(silent(r));
  GPRS_CHECK_EQ(r.snapshot_tested, 1);
  GPRS_CHECK_EQ(r.kept_tested, 2);
  r = structure_check(m, result(0.), false);
  GPRS_CHECK_EQ(r.snapshot_failures, 1);
}

// The objective through a defined record: the cost of the column leaves with it onto the saved
// columns and into the offset. The columns of interval and forcing records have none.
GPRS_TEST(e2e_columns_objective_of_the_records)
{
  constexpr int32_t k = 0, y = 1, z = 2;
  // k + 2 y + 3 z = 6 with a free k; two rows keep y and z.
  model_t m          = build({{{{k, 1.}, {y, 2.}, {z, 3.}}, 6., 6.},
                              {{{y, 1.}, {z, 1.}}, -inf, 2.},
                              {{{y, 1.}, {z, -1.}}, -1., inf}},
                             {-inf, 0., 0.},
                             {inf, 1., 1.});
  m.cost             = {2., 5., 7.};
  m.objective_offset = 1.;
  result_t result    = presolve(m, only_singleton_cols());
  GPRS_REQUIRE(result.postsolve.col_from_rows.size() == 1);
  GPRS_CHECK(silent(structure_check(m, result, false)));
  GPRS_CHECK(silent(structure_check(m, result, true)));
  GPRS_CHECK(result.reduced.cost == std::vector<double>({1., 1.}));
  GPRS_CHECK_EQ(result.reduced.objective_offset, 13.);
  // The cost of y not moved, and the offset not moved.
  result_t wrong        = result;
  wrong.reduced.cost[0] = 5.;
  structure_report_t r  = structure_check(m, wrong, false);
  GPRS_CHECK_EQ(r.objective_failures, 1);
  GPRS_CHECK_EQ(r.objective_worst_col, y);
  wrong                          = result;
  wrong.reduced.objective_offset = 1.;
  r                              = structure_check(m, wrong, false);
  GPRS_CHECK_EQ(r.objective_failures, 1);
  GPRS_CHECK_EQ(r.objective_constant, 12. / 14.);

  // An interval record and a forcing record whose columns have a cost.
  constexpr int32_t s = 0;
  model_t ranged = build({{{{s, 1.}, {y, 1.}, {z, 1.}}, 2., 5.}, {{{y, 1.}, {z, -1.}}, -3., 3.}},
                         {0., 0., 0.},
                         {1., 4., 4.});
  result         = presolve(ranged, only_singleton_cols());
  GPRS_REQUIRE(result.postsolve.col_from_rows.size() == 1);
  GPRS_CHECK(silent(structure_check(ranged, result, false)));
  ranged.cost[s] = 1.;
  r              = structure_check(ranged, result, false);
  GPRS_CHECK_EQ(r.objective_failures, 1);
  GPRS_CHECK_EQ(r.objective_worst_col, s);

  model_t forced = build({{{{k, 1.}, {y, 1.}}, -inf, 4.},
                          {{{k, -2.}, {z, 1.}}, -6., inf},
                          {{{y, 1.}, {z, -1.}}, -3., 3.}},
                         {-inf, 0., 0.},
                         {3., 4., 4.});
  result         = presolve(forced, only_singleton_cols());
  GPRS_REQUIRE(result.postsolve.col_from_rows.size() == 1);
  GPRS_CHECK(result.postsolve.col_from_rows[0].kind == col_from_rows_kind_t::forcing);
  GPRS_CHECK(silent(structure_check(forced, result, false)));
  forced.cost[k] = -1.;
  r              = structure_check(forced, result, false);
  GPRS_CHECK_EQ(r.objective_failures, 1);
  GPRS_CHECK_EQ(r.objective_worst_col, k);
}

// The contract of the records, part by part.
GPRS_TEST(e2e_columns_stack_contract)
{
  constexpr int32_t s1 = 0, s2 = 1, y = 2, z = 3;
  model_t m =
    build({{{{s1, 1.}, {s2, 2.}, {y, 1.}, {z, 1.}}, 2., 5.}, {{{y, 1.}, {z, -1.}}, -3., 3.}},
          {0., 0., 0., 0.},
          {1., 1., 4., 4.});
  result_t right = presolve(m, only_singleton_cols());
  GPRS_REQUIRE(right.postsolve.col_from_rows.size() == 1);
  GPRS_REQUIRE(right.postsolve.record_cols.size() == 2);
  GPRS_CHECK(silent(structure_check(m, right, false)));
  auto error = [&](void (*change)(model_t&, result_t&)) {
    model_t model   = m;
    result_t result = right;
    change(model, result);
    structure_report_t r = structure_check(model, result, false);
    GPRS_CHECK(!r.composed);
    return r.stack_error;
  };
  GPRS_CHECK_EQ(error([](model_t& model, result_t& r) {
                  model.col_type[0]       = 1;
                  r.postsolve.col_type[0] = 1;
                }),
                "integer column of interval record 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) { r.postsolve.col_type[0] = 1; }),
                "column types of the stack 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) { r.postsolve.saved_entries.col[0] = 1; }),
                "entry of col_from_rows record 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) { r.postsolve.saved_entries.value[0] = 0.; }),
                "entry of col_from_rows record 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) {
                  r.postsolve.col_from_rows[0].kind = col_from_rows_kind_t::defined;
                }),
                "kind or counts of col_from_rows record 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) { r.postsolve.col_from_rows[0].dual = true; }),
                "kind or counts of col_from_rows record 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) { r.postsolve.record_coefficients[1] = 0.; }),
                "coefficient of col_from_rows record 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) { r.postsolve.record_cols[0].lower = 2.; }),
                "bounds of col_from_rows record 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) {
                  std::swap(r.postsolve.record_cols[0], r.postsolve.record_cols[1]);
                }),
                "column of col_from_rows record 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) { r.postsolve.saved_rows[0].upper = inf; }),
                "sides of interval record 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) { r.postsolve.saved_rows[0].end = 9; }),
                "saved row of col_from_rows record 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) { r.postsolve.col_from_rows[0].rows_end = 2; }),
                "ranges of col_from_rows record 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) { r.postsolve.col_from_rows.clear(); }),
                "col_from_rows records missing at 0");
  GPRS_CHECK_EQ(error([](model_t&, result_t& r) {
                  r.postsolve.col_from_rows.push_back(r.postsolve.col_from_rows[0]);
                }),
                "records outside the order 0");

  // A defined record takes its row out of the model: a second record cannot save it again.
  constexpr int32_t k = 0;
  model_t pivot  = build({{{{k, 1.}, {1, 2.}, {2, 3.}}, 6., 6.}, {{{1, 1.}, {2, -1.}}, -1., 1.}},
                         {-inf, 0., 0.},
                         {inf, 1., 1.});
  result_t twice = reduced_to(pivot, {2}, {}, {});
  twice.postsolve.push_defined_col(
    k, 1., -inf, inf, save(twice.postsolve, 0, {{1, 2.}, {2, 3.}}, 6., 6.), false);
  twice.postsolve.push_defined_col(
    1, 2., 0., 1., save(twice.postsolve, 0, {{2, 3.}}, 6., 6.), false);
  GPRS_CHECK_EQ(structure_check(pivot, twice, false).stack_error,
                "saved row of col_from_rows record 1");
}

// Requirement 5 of the crush check (fixed model 14): xk + 1000 y1 - 1000 y2 = 0 with the point
// y1 = 100, y2 = 100.00000001, xk = 0, which misses the row by 1e-5. The postsolve gives
// xk = 1e-5, which is what the saved row asks for.
GPRS_TEST(e2e_columns_crush_allows_what_the_saved_row_asks_for)
{
  constexpr int32_t k = 0, y1 = 1, y2 = 2;
  model_t m       = build({{{{k, 1.}, {y1, 1000.}, {y2, -1000.}}, 0., 0.},
                           {{{y1, 1.}, {y2, 1.}}, -inf, 300.},
                           {{{y1, 1.}, {y2, -1.}}, -50., inf}},
                          {-inf, 0., 0.},
                          {inf, 200., 200.});
  result_t result = presolve(m, only_singleton_cols(true));
  GPRS_REQUIRE(result.postsolve.col_from_rows.size() == 1);
  GPRS_CHECK(silent(structure_check(m, result, true)));
  point_set_t set;
  set.add(m, {0., 100., 100.00000001});
  set.add(m, {0., 100., 100.});
  GPRS_REQUIRE(set.points.size() == 2);
  crush_report_t c = crush_check(m, result, true, set);
  GPRS_CHECK(c.exact);
  GPRS_CHECK_EQ(c.failed_reduced, 0);
  GPRS_CHECK_EQ(c.failed_postsolved, 0);
  GPRS_CHECK_EQ(c.failed_objective, 0);
  GPRS_CHECK_EQ(c.failed_round_trip, 0);
  GPRS_CHECK_EQ(c.moved_columns, 0);
  GPRS_CHECK_EQ(c.exempt_columns, 0);
  // The value moved by 1e-5, and all of it is allowed.
  std::vector<double> crushed, z;
  result.postsolve.crush(set.points[0], crushed);
  result.postsolve.postsolve(crushed, z);
  GPRS_CHECK(std::fabs(z[k] - 1e-5) < 1e-9);
  GPRS_CHECK(c.moved_continuous <= 1e-15);
}

// The columns of interval and forcing records, and the aliases of such columns, are not held
// to the round trip; an alias of a defined column is allowed what its rep is, times its scale.
GPRS_TEST(e2e_columns_crush_exempts_the_columns_with_a_range)
{
  constexpr int32_t a = 0, s = 1, y = 2, z = 3;
  // a = 2 s + 1, and 2 <= s + y + z <= 5 with s in [0, 1].
  model_t m       = build({{{{a, 1.}, {s, -2.}}, 1., 1.}, {{{s, 1.}, {y, 1.}, {z, 1.}}, 2., 5.}},
                          {1., 0., 0., 0.},
                          {3., 1., 4., 4.});
  result_t result = reduced_to(m, {y, z}, {{{{0, 1.}, {1, 1.}}, 1., 5.}}, {1});
  result.postsolve.push_alias(a, s, 0, 2., 1.);
  result.postsolve.push_interval_cols(
    {column(s, 0., 1.)}, {1.}, save(result.postsolve, 1, {{y, 1.}, {z, 1.}}, 2., 5.));
  GPRS_CHECK(silent(structure_check(m, result, false)));
  point_set_t set;
  set.add(m, {2.4, 0.7, 1., 1.5});
  GPRS_REQUIRE(set.points.size() == 1);
  crush_report_t c = crush_check(m, result, false, set);
  GPRS_CHECK(c.exact);
  GPRS_CHECK_EQ(c.exempt_columns, 2);
  GPRS_CHECK_EQ(c.failed_reduced + c.failed_postsolved + c.failed_objective, 0);
  GPRS_CHECK_EQ(c.failed_round_trip, 0);
  // The undo chose s = 0, so a = 1.
  std::vector<double> x;
  result.postsolve.postsolve({1., 1.5}, x);
  GPRS_CHECK(x == std::vector<double>({1., 0., 1., 1.5}));

  // b = 2 k + 1 with k defined by k + 1000 y1 - 1000 y2 = 0.
  constexpr int32_t b = 0, k = 1, y1 = 2, y2 = 3;
  model_t defined =
    build({{{{b, 1.}, {k, -2.}}, 1., 1.}, {{{k, 1.}, {y1, 1000.}, {y2, -1000.}}, 0., 0.}},
          {-inf, -inf, 0., 0.},
          {inf, inf, 200., 200.});
  result = reduced_to(defined, {y1, y2}, {}, {});
  result.postsolve.push_alias(b, k, 0, 2., 1.);
  result.postsolve.push_defined_col(
    k, 1., -inf, inf, save(result.postsolve, 1, {{y1, 1000.}, {y2, -1000.}}, 0., 0.), false);
  GPRS_CHECK(silent(structure_check(defined, result, true)));
  point_set_t points;
  points.add(defined, {1., 0., 100., 100.00000001});
  GPRS_REQUIRE(points.points.size() == 1);
  c = crush_check(defined, result, true, points);
  GPRS_CHECK_EQ(c.failed_round_trip, 0);
  GPRS_CHECK_EQ(c.exempt_columns, 0);
  // An alias with another offset is off by more than its rep is.
  result.postsolve.aliases[0].offset = 1.5;
  c                                  = crush_check(defined, result, true, points);
  GPRS_CHECK_EQ(c.failed_round_trip, 1);
}

// The engine on generated models with singleton columns and forcing columns, in both modes: the
// structure check is silent, and every kind of record occurs often enough to be tested.
GPRS_TEST(e2e_columns_structure_of_the_engine)
{
  oracle::generator_options_t g;
  g.p_singleton_col       = 0.5;
  g.p_singleton_line      = 0.7;
  g.p_singleton_zero_cost = 0.5;
  g.max_singleton_cols    = 4;
  g.max_row_singletons    = 2;
  g.p_forcing_col         = 0.5;
  g.p_equation            = 0.35;
  int64_t defined = 0, dual = 0, interval = 0, forcing = 0, boxes = 0, snapshots = 0, failed = 0;
  for (uint64_t seed = 9200000; seed < 9206000 && failed < 3; ++seed) {
    oracle::case_t c = oracle::generate(seed, g);
    for (bool primal_only : {false, true}) {
      options_t options;
      options.primal_only = primal_only;
      result_t result     = presolve(c.model, options);
      if (!gprs_check::carries_model(result.status)) { continue; }
      structure_report_t r = structure_check(c.model, result, primal_only);
      for (const col_from_rows_record_t& record : result.postsolve.col_from_rows) {
        defined += record.kind == col_from_rows_kind_t::defined && !record.dual;
        dual += record.dual;
        interval += record.kind == col_from_rows_kind_t::interval;
        forcing += record.kind == col_from_rows_kind_t::forcing;
        GPRS_CHECK(!primal_only || (record.kind == col_from_rows_kind_t::defined && !record.dual));
      }
      boxes += r.box_tested;
      snapshots += r.snapshot_tested;
      if (silent(r)) { continue; }
      ++failed;
      test::fail(__FILE__,
                 __LINE__,
                 "seed " + std::to_string(seed) + (primal_only ? " primal only: " : ": ") +
                   r.stack_error + "\n" + oracle::to_code(c));
    }
  }
  GPRS_CHECK_EQ(failed, 0);
  GPRS_CHECK(defined > 300 && dual > 100 && interval > 30 && forcing > 100);
  GPRS_CHECK(boxes > 100 && snapshots > 30);
}
