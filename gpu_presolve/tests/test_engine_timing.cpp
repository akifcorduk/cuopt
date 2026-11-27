/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "oracle.hpp"
#include "test_engine_support.hpp"

#include <gprs/presolve.hpp>

#include <cstring>
#include <vector>

using namespace gprs;
using namespace gprs::engine_test;

namespace {

bool same_doubles(const std::vector<double>& a, const std::vector<double>& b)
{
  return a.size() == b.size() &&
         (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

// Bit for bit.
bool same_model(const model_t& a, const model_t& b)
{
  return a.nrows == b.nrows && a.ncols == b.ncols && a.row_start == b.row_start &&
         a.col_index == b.col_index && same_doubles(a.value, b.value) &&
         same_doubles(a.row_lower, b.row_lower) && same_doubles(a.row_upper, b.row_upper) &&
         same_doubles(a.col_lower, b.col_lower) && same_doubles(a.col_upper, b.col_upper) &&
         same_doubles(a.cost, b.cost) && a.col_type == b.col_type &&
         same_doubles({a.objective_offset}, {b.objective_offset});
}

// The parts are intervals of the run that do not overlap, each read from the clock of
// result_t::seconds: none is negative and with the calls of the families they stay inside it.
void check_parts(const result_t& r)
{
  GPRS_CHECK(r.seconds_state >= 0.);
  GPRS_CHECK(r.seconds_start >= 0.);
  GPRS_CHECK(r.seconds_storage >= 0.);
  GPRS_CHECK(r.seconds_finish >= 0.);
  GPRS_CHECK(r.seconds_hash >= 0.);
  double inside =
    r.seconds_state + r.seconds_start + r.seconds_storage + r.seconds_finish + r.seconds_hash;
  for (const rule_stats_t& s : r.stats) {
    GPRS_CHECK(s.seconds >= 0.);
    inside += s.seconds;
  }
  GPRS_CHECK(inside <= r.seconds + 1e-9);
}

}  // namespace

GPRS_TEST(engine_seconds_parts)
{
  oracle::generator_options_t shape;
  shape.max_rows      = 12;
  shape.max_enum_cols = 8;
  for (int32_t mode = 0; mode < 3; ++mode) {
    options_t options;
    options.phase_hashes = mode == 1;
    if (mode == 2) {
      options.rebuild_dead_fraction  = 0.;
      options.rebuild_fixed_fraction = 0.;
    }
    for (uint64_t seed = 7000000; seed < 7000200; ++seed) {
      check_parts(presolve(oracle::generate(seed, shape).model, options));
    }
  }

  // A start step that ends the presolve: nothing is extracted.
  model_t crossing = make_model({{1., 1.}}, {0.}, {4.}, {0., 3.}, {1., 2.}, {1., 1.}, {0, 0});
  result_t r       = presolve(crossing, options_t());
  GPRS_CHECK_EQ(r.status, status_t::infeasible);
  check_parts(r);
  GPRS_CHECK_EQ(r.seconds_storage, 0.);
  GPRS_CHECK_EQ(r.seconds_finish, 0.);
}

// The last phase hash is over the rebuilt state, so with phase hashes the state is rebuilt at
// the end and without them it is not: the result is the same.
GPRS_TEST(engine_phase_hashes_do_not_change_the_result)
{
  oracle::generator_options_t shape;
  shape.max_rows         = 14;
  shape.max_enum_cols    = 8;
  shape.max_line_cols    = 4;
  shape.max_width        = 6;
  shape.p_singleton_row  = 0.05;
  shape.p_forcing_side   = 0.03;
  shape.p_outside_side   = 0.;
  shape.p_crossing       = 0.;
  shape.p_zero_cost      = 0.1;
  shape.p_infinite_bound = 0.1;
  int32_t reduced        = 0;
  for (uint64_t seed = 7100000; seed < 7101500; ++seed) {
    model_t m = oracle::generate(seed, shape).model;
    options_t options;
    if (seed % 2 == 1) { options.rebuild_dead_fraction = 0.; }
    result_t plain       = presolve(m, options);
    options.phase_hashes = true;
    result_t hashed      = presolve(m, options);
    GPRS_CHECK(plain.phase_hash.empty());
    GPRS_CHECK_EQ(plain.status, hashed.status);
    GPRS_CHECK_EQ(plain.hash, hashed.hash);
    GPRS_CHECK_EQ(plain.rounds, hashed.rounds);
    GPRS_CHECK_EQ(plain.rebuilds, hashed.rebuilds);
    GPRS_CHECK(same_model(plain.reduced, hashed.reduced));
    GPRS_CHECK(plain.postsolve.row_origin == hashed.postsolve.row_origin);
    GPRS_CHECK(plain.postsolve.col_origin == hashed.postsolve.col_origin);
    GPRS_CHECK_EQ(postsolve_hash(plain.postsolve), postsolve_hash(hashed.postsolve));
    for (int32_t rule = 0; rule < rule::count; ++rule) {
      const rule_stats_t& a = plain.stats[rule];
      const rule_stats_t& b = hashed.stats[rule];
      GPRS_CHECK_EQ(a.calls, b.calls);
      GPRS_CHECK_EQ(a.work, b.work);
      GPRS_CHECK_EQ(a.findings, b.findings);
      GPRS_CHECK_EQ(a.rows_removed, b.rows_removed);
      GPRS_CHECK_EQ(a.cols_removed, b.cols_removed);
      GPRS_CHECK_EQ(a.nnz_removed, b.nnz_removed);
    }
    reduced += plain.status == status_t::reduced;
  }
  GPRS_CHECK(reduced > 100);
}

// The last phase hash is over the rebuilt state in every build. No family runs here, so the
// only change is the start step, which leaves the entries of the columns it fixes in their
// rows: a presolve that rebuilds after its first phase and one that rebuilds only at the end
// reach the same state there, and differ in the state before it.
GPRS_TEST(engine_last_phase_hash_is_over_the_rebuilt_state)
{
  oracle::generator_options_t shape;
  shape.max_rows      = 10;
  shape.max_enum_cols = 8;
  shape.p_fixed_col   = 0.4;
  shape.p_crossing    = 0.;
  options_t early;
  early.enabled.fill(false);
  early.phase_hashes           = true;
  early.rebuild_dead_fraction  = 0.;
  early.rebuild_fixed_fraction = 0.;
  options_t late               = early;
  late.rebuild_dead_fraction   = 2.;
  late.rebuild_fixed_fraction  = 2.;
  int32_t carried              = 0;
  for (uint64_t seed = 7200000; seed < 7200300; ++seed) {
    model_t m  = oracle::generate(seed, shape).model;
    result_t a = presolve(m, early);
    result_t b = presolve(m, late);
    GPRS_CHECK_EQ(a.status, b.status);
    if (a.status == status_t::infeasible) { continue; }
    // After the start step, the three phases of the one round and the end.
    GPRS_REQUIRE(a.phase_hash.size() == 5 && b.phase_hash.size() == 5);
    GPRS_CHECK_EQ(a.phase_hash[4], b.phase_hash[4]);
    GPRS_CHECK_EQ(a.hash, b.hash);
    GPRS_CHECK_EQ(b.rebuilds, 1);
    if (a.rebuilds == 2) {
      GPRS_CHECK(a.phase_hash[3] != b.phase_hash[3]);
      ++carried;
    }
  }
  GPRS_CHECK(carried > 100);
}
