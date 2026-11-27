/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The col_from_rows records of the stack (postsolve.hpp): what a push stores, the undo of each
// kind and their place among the other records.

#include "harness.hpp"

#include <gprs/postsolve.hpp>
#include <gprs/presolve.hpp>

#include <cmath>
#include <utility>
#include <vector>

using namespace gprs;

namespace {

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

using entries_t = std::vector<std::pair<int32_t, double>>;

// A row saved into the pool of the stack, as save_row does it for a row of a state.
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

// The columns that no record removed are the reduced model, in index order.
void finish(postsolve_t& ps)
{
  ps.col_origin.clear();
  for (int32_t j = 0; j < ps.ncols; ++j) {
    if (!ps.col_removed[j]) { ps.col_origin.push_back(j); }
  }
}

std::vector<double> undo(const postsolve_t& ps, const std::vector<double>& x_reduced)
{
  std::vector<double> x;
  ps.postsolve(x_reduced, x);
  return x;
}

record_col_t column(int32_t col, double lower, double upper)
{
  return record_col_t{col, lower, upper, lower > -inf ? lower : upper < inf ? upper : 0.};
}

}  // namespace

GPRS_TEST(postsolve_defined_column)
{
  // x0 = (7 - 2 x1 + x2) / 2 from the saved row 2 x0 + 2 x1 - x2 = 7.
  postsolve_t ps;
  ps.init(bare_model({0, 0, 0, 1}, 2));
  GPRS_CHECK_EQ(ps.nnz, 0);
  saved_row_t row = save(ps, 1, {{1, 2.}, {2, -1.}}, 7., 7.);
  ps.push_defined_col(0, 2., -3., inf, row, true);
  GPRS_REQUIRE(ps.order.size() == 1 && ps.col_from_rows.size() == 1);
  GPRS_CHECK(ps.order[0] == record_kind_t::col_from_rows);
  const col_from_rows_record_t& r = ps.col_from_rows[0];
  GPRS_CHECK(r.kind == col_from_rows_kind_t::defined && r.dual);
  GPRS_CHECK_EQ(r.cols_begin, 0);
  GPRS_CHECK_EQ(r.cols_end, 1);
  GPRS_CHECK_EQ(r.rows_begin, 0);
  GPRS_CHECK_EQ(r.rows_end, 1);
  GPRS_CHECK_EQ(r.coefficients, 0);
  GPRS_CHECK(ps.record_coefficients == std::vector<double>({2.}));
  GPRS_CHECK_EQ(ps.record_cols[0].col, 0);
  GPRS_CHECK_EQ(ps.record_cols[0].lower, -3.);
  GPRS_CHECK_EQ(ps.record_cols[0].upper, inf);
  GPRS_CHECK_EQ(ps.record_cols[0].prefer, -3.);
  GPRS_CHECK_EQ(ps.saved_rows[0].row, 1);
  GPRS_CHECK_EQ(ps.saved_rows[0].begin, 0);
  GPRS_CHECK_EQ(ps.saved_rows[0].end, 2);
  GPRS_CHECK(ps.col_removed == std::vector<uint8_t>({1, 0, 0, 0}));
  finish(ps);
  GPRS_CHECK(undo(ps, {1., 3., 5.}) == std::vector<double>({4., 1., 3., 5.}));
  // The value is what the row gives, also outside the bounds of the record.
  GPRS_CHECK(undo(ps, {10., 1., 0.}) == std::vector<double>({-6., 10., 1., 0.}));
  std::vector<double> crushed;
  ps.crush({4., 1., 3., 5.}, crushed);
  GPRS_CHECK(crushed == std::vector<double>({1., 3., 5.}));

  // An integer column gets the nearest integer.
  postsolve_t integer;
  integer.init(bare_model({1, 1, 1}, 1));
  GPRS_CHECK(integer.integer_now(0) && integer.integer_at(1, 0));
  row = save(integer, 0, {{1, 1.0000000005}, {2, -2.}}, 3., 3.);
  integer.push_defined_col(0, 1., -inf, inf, row, false);
  GPRS_CHECK(!integer.col_from_rows[0].dual);
  finish(integer);
  GPRS_CHECK(undo(integer, {100., 7.}) == std::vector<double>({-83., 100., 7.}));
}

GPRS_TEST(postsolve_interval_columns)
{
  // 2 <= x0 + 2 x2 - x4 + x1 + x3 <= 5 without x0 in [0, 1], x2 in [0, 2] and x4 in [0, 1].
  postsolve_t ps;
  ps.init(bare_model({0, 0, 0, 0, 0}, 1));
  saved_row_t row = save(ps, 0, {{1, 1.}, {3, 1.}}, 2., 5.);
  ps.push_interval_cols(
    {column(0, 0., 1.), column(2, 0., 2.), column(4, 0., 1.)}, {1., 2., -1.}, row);
  GPRS_REQUIRE(ps.col_from_rows.size() == 1);
  const col_from_rows_record_t& r = ps.col_from_rows[0];
  GPRS_CHECK(r.kind == col_from_rows_kind_t::interval && !r.dual);
  GPRS_CHECK_EQ(r.cols_end - r.cols_begin, 3);
  GPRS_CHECK_EQ(r.rows_end - r.rows_begin, 1);
  GPRS_CHECK(ps.record_coefficients == std::vector<double>({1., 2., -1.}));
  GPRS_CHECK(ps.col_removed == std::vector<uint8_t>({1, 0, 1, 0, 1}));
  finish(ps);
  auto values = [&](double x1, double x3) {
    std::vector<double> x = undo(ps, {x1, x3});
    return std::vector<double>({x[0], x[2], x[4]});
  };
  // Inside the sides the columns stay at prefer.
  GPRS_CHECK(values(1., 2.) == std::vector<double>({0., 0., 0.}));
  GPRS_CHECK(values(2., 3.) == std::vector<double>({0., 0., 0.}));
  // Below: in the order of the record, each as far as needed or as it can go.
  GPRS_CHECK(values(1., 0.5) == std::vector<double>({0.5, 0., 0.}));
  GPRS_CHECK(values(0., 0.) == std::vector<double>({1., 0.5, 0.}));
  GPRS_CHECK(values(-3., 0.) == std::vector<double>({1., 2., 0.}));
  // Above: only the column with the negative coefficient can lower the row from prefer.
  GPRS_CHECK(values(3., 2.5) == std::vector<double>({0., 0., 0.5}));
  GPRS_CHECK(values(3., 3.) == std::vector<double>({0., 0., 1.}));
  // Outside the widened sides -3 and 6 the columns stop at their bounds.
  GPRS_CHECK(values(-9., 0.) == std::vector<double>({1., 2., 0.}));
  GPRS_CHECK(values(9., 0.) == std::vector<double>({0., 0., 1.}));

  // prefer: the lower bound, else the upper bound, else 0; a column without the bound it is
  // moved towards takes all that is missing.
  postsolve_t open;
  open.init(bare_model({0, 0, 0, 0}, 1));
  row = save(open, 0, {{3, 1.}}, -1., 1.);
  open.push_interval_cols(
    {column(0, -inf, 4.), column(1, -inf, inf), column(2, 2., inf)}, {1., -1., 1.}, row);
  GPRS_CHECK_EQ(open.record_cols[0].prefer, 4.);
  GPRS_CHECK_EQ(open.record_cols[1].prefer, 0.);
  GPRS_CHECK_EQ(open.record_cols[2].prefer, 2.);
  finish(open);
  // prefer gives 6 + x3. Above the upper side: x0 goes down without a limit.
  GPRS_CHECK(undo(open, {0.}) == std::vector<double>({-1., 0., 2., 0.}));
  // Below the lower side: x0 is at its upper bound, x1 goes down without a limit.
  GPRS_CHECK(undo(open, {-10.}) == std::vector<double>({4., -3., 2., -10.}));
}

// The column that ends between prefer and its bound gets the quotient of the row and not prefer
// plus a step. Found by the campaign (profile dense, seed 200045869): the step from a prefer of
// -1.28e9 lost eight digits of 597.33, which a parallel row of 2^18 times the scale read.
GPRS_TEST(postsolve_interval_column_far_from_prefer)
{
  postsolve_t ps;
  ps.init(bare_model({0, 1, 1}, 1));
  double a        = -3. / 128. / 256.;
  saved_row_t row = save(ps, 0, {{1, -3. / 256.}, {2, 2. / 256.}}, -12. / 256., -12. / 256.);
  ps.push_interval_cols({column(0, -1.28e9, 640.)}, {a}, row);
  finish(ps);
  GPRS_CHECK_EQ(undo(ps, {0., 1.})[0], (-12. / 256. - 2. / 256.) / a);
  GPRS_CHECK_EQ(undo(ps, {2., 0.})[0], (-12. / 256. + 6. / 256.) / a);
  // Beyond the bound the column stops there.
  GPRS_CHECK_EQ(undo(ps, {0., 2.})[0], 640.);

  // A column that goes all its way first: the next one gets the row with that term in it.
  postsolve_t two;
  two.init(bare_model({0, 0, 0}, 1));
  row = save(two, 0, {{1, 1.}}, 2., 3.);
  two.push_interval_cols({column(0, 0., 1.), column(2, -1e9, inf)}, {1., 1e-3}, row);
  finish(two);
  std::vector<double> x = undo(two, {0.5});
  GPRS_CHECK_EQ(x[0], 1.);
  GPRS_CHECK(std::fabs(x[2] - 500.) <= 1e-10);
  // And the mirror, from above with a negative coefficient.
  postsolve_t down;
  down.init(bare_model({0, 0}, 1));
  row = save(down, 0, {{1, 1.}}, -3., -2.);
  down.push_interval_cols({column(0, -inf, 1e9)}, {-1e-3}, row);
  finish(down);
  GPRS_CHECK(std::fabs(undo(down, {-4.})[0] - -1000.) <= 1e-10);
}

GPRS_TEST(postsolve_forcing_column)
{
  // x0 <= 3, x0 + x1 <= 4 and -2 x0 + x2 >= -6: the largest value that all three allow.
  for (uint8_t integer : {0, 1}) {
    postsolve_t ps;
    ps.init(bare_model({integer, 0, 0}, 2));
    std::vector<saved_row_t> rows = {save(ps, 0, {{1, 1.}}, -inf, 4.),
                                     save(ps, 1, {{2, 1.}}, -6., inf)};
    ps.push_forcing_col(column(0, -inf, 3.), {1., -2.}, rows);
    GPRS_REQUIRE(ps.col_from_rows.size() == 1);
    const col_from_rows_record_t& r = ps.col_from_rows[0];
    GPRS_CHECK(r.kind == col_from_rows_kind_t::forcing);
    GPRS_CHECK_EQ(r.cols_end - r.cols_begin, 1);
    GPRS_CHECK_EQ(r.rows_end - r.rows_begin, 2);
    GPRS_CHECK(ps.record_coefficients == std::vector<double>({1., -2.}));
    finish(ps);
    GPRS_CHECK_EQ(undo(ps, {0., 4.})[0], 3.);
    GPRS_CHECK_EQ(undo(ps, {2.5, 4.})[0], integer ? 1. : 1.5);
    GPRS_CHECK_EQ(undo(ps, {0., -3.})[0], integer ? 1. : 1.5);
    GPRS_CHECK_EQ(undo(ps, {7.5, 0.})[0], integer ? -4. : -3.5);
  }
  // The mirror, and a column without any bound: the rows alone decide, and the first row says
  // from which end.
  for (uint8_t integer : {0, 1}) {
    postsolve_t ps;
    ps.init(bare_model({integer, 0, 0}, 2));
    std::vector<saved_row_t> rows = {save(ps, 0, {{1, 1.}}, -inf, 4.),
                                     save(ps, 1, {{2, 1.}}, -6., inf)};
    ps.push_forcing_col(column(0, -inf, inf), {-1., 2.}, rows);
    finish(ps);
    // max(x1 - 4, -3 - x2 / 2).
    GPRS_CHECK_EQ(undo(ps, {2.5, 4.})[0], integer ? -1. : -1.5);
    GPRS_CHECK_EQ(undo(ps, {0., 1.})[0], integer ? -3. : -3.5);
    postsolve_t bounded;
    bounded.init(bare_model({integer, 0, 0}, 2));
    rows = {save(bounded, 0, {{1, 1.}}, -inf, 4.), save(bounded, 1, {{2, 1.}}, -6., inf)};
    bounded.push_forcing_col(column(0, -1., inf), {-1., 2.}, rows);
    finish(bounded);
    GPRS_CHECK_EQ(undo(bounded, {2.5, 4.})[0], -1.);
    GPRS_CHECK_EQ(undo(bounded, {6.5, 4.})[0], integer ? 3. : 2.5);
  }
}

// Records are undone in reverse order: a saved row names columns that later records remove,
// and gets their restored values.
GPRS_TEST(postsolve_from_rows_among_other_records)
{
  postsolve_t ps;
  ps.init(bare_model({0, 0, 0, 0, 0, 0}, 3));
  // x0 = 10 - x1 - x2 - x3; then x1 = 2 x4 + 1, x2 = 3, and x3 from x3 + x4 <= 5 with x3 <= 2.
  ps.push_defined_col(0, 1., -inf, inf, save(ps, 0, {{1, 1.}, {2, 1.}, {3, 1.}}, 10., 10.), false);
  ps.push_alias(1, 4, 1, 2., 1.);
  ps.push_fixed_col(2, 3.);
  ps.push_forcing_col(column(3, -inf, 2.), {1.}, {save(ps, 2, {{4, 1.}}, -inf, 5.)});
  GPRS_CHECK(ps.order == std::vector<record_kind_t>({record_kind_t::col_from_rows,
                                                     record_kind_t::alias,
                                                     record_kind_t::fixed_col,
                                                     record_kind_t::col_from_rows}));
  GPRS_CHECK_EQ(ps.col_from_rows[1].cols_begin, 1);
  GPRS_CHECK_EQ(ps.col_from_rows[1].rows_begin, 1);
  GPRS_CHECK_EQ(ps.col_from_rows[1].coefficients, 1);
  GPRS_CHECK_EQ(ps.saved_rows[1].begin, 3);
  finish(ps);
  GPRS_CHECK(ps.col_origin == std::vector<int32_t>({4, 5}));
  // x4 = 1: x3 = min(2, 4) = 2, x1 = 3, x0 = 10 - 3 - 3 - 2.
  GPRS_CHECK(undo(ps, {1., 9.}) == std::vector<double>({2., 3., 3., 2., 1., 9.}));
  // x4 = 4: x3 = 1, x1 = 9, x0 = -3.
  GPRS_CHECK(undo(ps, {4., 9.}) == std::vector<double>({-3., 9., 3., 1., 4., 9.}));
  std::vector<double> crushed;
  ps.crush({-3., 9., 3., 1., 4., 9.}, crushed);
  GPRS_CHECK(crushed == std::vector<double>({4., 9.}));
}

// init empties every array of the records.
GPRS_TEST(postsolve_init_empties_the_saved_rows)
{
  postsolve_t ps;
  ps.init(bare_model({0, 0}, 1));
  ps.push_defined_col(0, 1., 0., 1., save(ps, 0, {{1, 1.}}, 1., 1.), false);
  model_t m   = bare_model({0, 0, 0}, 2);
  m.row_start = {0, 2, 3};
  m.col_index = {0, 1, 2};
  m.value     = {1., 1., 1.};
  ps.init(m);
  GPRS_CHECK_EQ(ps.nnz, 3);
  GPRS_CHECK(ps.order.empty() && ps.col_from_rows.empty() && ps.record_cols.empty());
  GPRS_CHECK(ps.saved_rows.empty() && ps.record_coefficients.empty());
  GPRS_CHECK(ps.saved_entries.col.empty() && ps.saved_entries.value.empty());
  GPRS_CHECK(ps.col_removed == std::vector<uint8_t>({0, 0, 0}));
}

// postsolve_hash reads every field of the records.
GPRS_TEST(postsolve_hash_reads_the_saved_rows)
{
  auto stack = [](int32_t variant) {
    postsolve_t ps;
    ps.init(bare_model({0, 0, 0, 0}, 3));
    double a          = variant == 1 ? 3. : 2.;
    double lower      = variant == 2 ? -1. : -3.;
    double upper      = variant == 3 ? 9. : inf;
    double side       = variant == 4 ? 8. : 7.;
    double entry      = variant == 5 ? 2.5 : 2.;
    int32_t entry_col = variant == 6 ? 3 : 2;
    int32_t row       = variant == 7 ? 2 : 1;
    bool dual         = variant == 8;
    saved_row_t saved{row, side, side, 0, 2};
    ps.saved_entries.col   = {1, entry_col};
    ps.saved_entries.value = {entry, -1.};
    if (variant == 9) {
      ps.push_forcing_col(record_col_t{0, -inf, 5., 5.}, {a}, {saved_row_t{row, -inf, side, 0, 2}});
    } else if (variant == 10) {
      ps.push_interval_cols({record_col_t{0, lower, 5., lower}}, {a}, saved);
    } else {
      ps.push_defined_col(0, a, lower, upper, saved, dual);
    }
    return postsolve_hash(ps);
  };
  std::vector<uint64_t> seen;
  for (int32_t variant = 0; variant <= 10; ++variant) {
    uint64_t h = stack(variant);
    GPRS_CHECK_EQ(h, stack(variant));
    for (uint64_t other : seen) {
      GPRS_CHECK(h != other);
    }
    seen.push_back(h);
  }
}
