/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The knobs of the generator for the singleton-column rules (oracle.hpp): several singleton
// columns per row, their cost, and forcing columns. The cases stay in the oracle class, and
// the structure that the rules act on is drawn often.

#include "harness.hpp"
#include "oracle.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

generator_options_t columns_options()
{
  generator_options_t o;
  o.p_singleton_col       = 0.5;
  o.p_singleton_line      = 0.7;
  o.p_singleton_zero_cost = 0.5;
  o.max_singleton_cols    = 4;
  o.max_row_singletons    = 2;
  o.p_forcing_col         = 0.5;
  return o;
}

// Entries and locks of every column of a model.
struct column_view_t {
  std::vector<int32_t> entries;
  std::vector<int32_t> down_locks;
  std::vector<int32_t> up_locks;
  std::vector<int32_t> row;  // of the last entry

  explicit column_view_t(const model_t& m)
    : entries(m.ncols, 0), down_locks(m.ncols, 0), up_locks(m.ncols, 0), row(m.ncols, -1)
  {
    for (int32_t i = 0; i < m.nrows; ++i) {
      for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
        int32_t j     = m.col_index[e];
        bool positive = m.value[e] > 0;
        bool lower = m.row_lower[i] > -inf, upper = m.row_upper[i] < inf;
        ++entries[j];
        row[j] = i;
        up_locks[j] += positive ? upper : lower;
        down_locks[j] += positive ? lower : upper;
      }
    }
  }
};

}  // namespace

// The new knobs draw nothing where they are not set: a case with the knobs of before is the
// case with the new ones at their defaults spelled out.
GPRS_TEST(oracle_columns_knobs_are_off_by_default)
{
  generator_options_t before;
  before.p_singleton_col        = 0.6;
  generator_options_t spelled   = before;
  spelled.max_row_singletons    = 1;
  spelled.p_singleton_zero_cost = -3.;
  spelled.p_forcing_col         = 0.;
  spelled.max_forcing_cols      = 5;
  for (uint64_t seed = 8400000; seed < 8400500; ++seed) {
    GPRS_CHECK(
      same_model(oracle::generate(seed, before).base, oracle::generate(seed, spelled).base));
  }
  // With no singleton column drawn at all, a cost knob for them changes nothing either.
  generator_options_t plain;
  generator_options_t costed   = plain;
  costed.p_singleton_zero_cost = 0.9;
  costed.max_row_singletons    = 3;
  for (uint64_t seed = 8400000; seed < 8400500; ++seed) {
    GPRS_CHECK(same_model(oracle::generate(seed, plain).base, oracle::generate(seed, costed).base));
  }
}

GPRS_TEST(oracle_columns_cases_stay_in_the_class)
{
  generator_options_t o = columns_options();
  // forcing: columns that are forcing columns; not_forcing: the same count in the cases
  // without the knob; drawn: columns that the knob added.
  int64_t models = 0, two_in_a_row = 0, singletons = 0, zero_cost = 0, forcing = 0, not_forcing = 0;
  int64_t continuous_singletons = 0, optimal = 0, drawn = 0;
  for (uint64_t seed = 8410000; seed < 8414000; ++seed) {
    case_t c = oracle::generate(seed, o);
    GPRS_REQUIRE(well_formed(c.base) && well_formed(c.model));
    GPRS_CHECK(same_model(c.base, oracle::generate(seed, o).base));
    std::vector<uint8_t> line;
    GPRS_REQUIRE(oracle::classify(c.base, o.max_width, line));
    ++models;
    const model_t& m = c.base;
    column_view_t view(m);
    std::vector<int32_t> row_singletons(m.nrows, 0);
    for (int32_t j = 0; j < m.ncols; ++j) {
      bool lower = m.col_lower[j] > -inf, upper = m.col_upper[j] < inf;
      if (view.entries[j] == 1) {
        ++singletons;
        zero_cost += m.cost[j] == 0.;
        continuous_singletons += m.col_type[j] == 0;
        ++row_singletons[view.row[j]];
      }
      // A forcing column: no cost, free at one end without a lock there, held at the other.
      bool pattern = (!lower && view.down_locks[j] == 0 && view.up_locks[j] >= 1) ||
                     (!upper && view.up_locks[j] == 0 && view.down_locks[j] >= 1);
      forcing += pattern && m.cost[j] == 0.;
    }
    for (int32_t count : row_singletons) {
      two_in_a_row += count >= 2;
    }
    // The same case without the knob has the same columns up to the forcing ones.
    generator_options_t without = o;
    without.p_forcing_col       = 0.;
    const model_t plain         = oracle::generate(seed, without).base;
    column_view_t plain_view(plain);
    GPRS_CHECK(plain.ncols <= m.ncols && m.ncols <= plain.ncols + o.max_forcing_cols);
    for (int32_t j = 0; j < plain.ncols; ++j) {
      bool lower = plain.col_lower[j] > -inf, upper = plain.col_upper[j] < inf;
      bool pattern = (!lower && plain_view.down_locks[j] == 0 && plain_view.up_locks[j] >= 1) ||
                     (!upper && plain_view.up_locks[j] == 0 && plain_view.down_locks[j] >= 1);
      not_forcing += pattern && plain.cost[j] == 0.;
    }
    drawn += m.ncols - plain.ncols;
    if (oracle::assignments(m, line) <= 4096.) {
      optimal += oracle::solve(m, line).outcome == oracle::outcome_t::optimal;
    }
  }
  if (std::getenv("GPRS_ORACLE_CENSUS")) {
    std::printf(
      "    singleton columns %lld, without a cost %lld, continuous %lld, rows with two %lld, "
      "forcing columns %lld, without the knob %lld, columns drawn by it %lld, optimal %lld\n",
      (long long)singletons,
      (long long)zero_cost,
      (long long)continuous_singletons,
      (long long)two_in_a_row,
      (long long)forcing,
      (long long)not_forcing,
      (long long)drawn,
      (long long)optimal);
  }
  GPRS_CHECK_EQ(models, 4000);
  GPRS_CHECK(singletons > 3000 && two_in_a_row > 400);
  GPRS_CHECK(zero_cost * 3 > singletons && zero_cost * 3 < 2 * singletons);
  GPRS_CHECK(continuous_singletons > 500);
  // Most of the drawn columns are forcing columns, the rest have a cost or a lock too many.
  GPRS_CHECK(drawn > 400);
  GPRS_CHECK(forcing - not_forcing > drawn / 2 && forcing - not_forcing < drawn);
  GPRS_CHECK(optimal > 500);
}
