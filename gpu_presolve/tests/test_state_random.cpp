/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "test_state_support.hpp"

#include <gprs/state.hpp>

#include <array>
#include <cmath>
#include <string>
#include <vector>

using namespace gprs;
using namespace state_test;

namespace {

namespace op {
constexpr int32_t set_col_lower   = 0;
constexpr int32_t set_col_upper   = 1;
constexpr int32_t fix_col         = 2;
constexpr int32_t remove_col      = 3;
constexpr int32_t set_cost        = 4;
constexpr int32_t add_offset      = 5;
constexpr int32_t record_implied  = 6;
constexpr int32_t delete_row      = 7;
constexpr int32_t set_row_lower   = 8;
constexpr int32_t set_row_upper   = 9;
constexpr int32_t shift_row_sides = 10;
constexpr int32_t scale_row       = 11;
constexpr int32_t set_coefficient = 12;
constexpr int32_t delete_entry    = 13;
constexpr int32_t move_entry      = 14;
constexpr int32_t compact         = 15;
constexpr int32_t rebuild         = 16;
constexpr int32_t find_entry      = 17;
constexpr int32_t count           = 18;
}  // namespace op

const char* const op_name[op::count] = {"set_col_lower",
                                        "set_col_upper",
                                        "fix_col",
                                        "remove_col",
                                        "set_cost",
                                        "add_offset",
                                        "record_implied",
                                        "delete_row",
                                        "set_row_lower",
                                        "set_row_upper",
                                        "shift_row_sides",
                                        "scale_row",
                                        "set_coefficient",
                                        "delete_entry",
                                        "move_entry",
                                        "compact",
                                        "rebuild",
                                        "find_entry"};

// Upper end of each operation's share of 1000 draws.
const int32_t op_share[op::count] = {
  150, 300, 325, 335, 365, 385, 425, 432, 472, 512, 552, 592, 722, 762, 882, 912, 922, 1000};

struct entry_t {
  int32_t row  = -1;
  int32_t col  = -1;
  int64_t slot = -1;
};

struct run_t {
  state_t st;
  shadow_t sh;
  rng_t& rng;
  bool decimal;
  // Events the operation must raise, per row and column, on the rule lists.
  std::vector<uint8_t> row_events;
  std::vector<uint8_t> col_events;
  std::string failure;

  run_t(const model_t& m, const tolerances_t& tol, rng_t& generator, bool decimal_values)
    : st(m, tol),
      sh(m, tol),
      rng(generator),
      decimal(decimal_values),
      row_events(m.nrows, 0),
      col_events(m.ncols, 0)
  {
  }

  int32_t random_live_row()
  {
    if (st.nrows == 0) { return -1; }
    int32_t first = rng.below(st.nrows);
    for (int32_t k = 0; k < st.nrows; ++k) {
      int32_t i = (first + k) % st.nrows;
      if (sh.row_alive[i]) { return i; }
    }
    return -1;
  }

  int32_t random_live_col()
  {
    if (st.ncols == 0) { return -1; }
    int32_t first = rng.below(st.ncols);
    for (int32_t k = 0; k < st.ncols; ++k) {
      int32_t j = (first + k) % st.ncols;
      if (sh.live(j)) { return j; }
    }
    return -1;
  }

  // A live entry of a live column, found in the state's row.
  entry_t random_free_entry()
  {
    if (st.nrows == 0) { return {}; }
    int32_t first = rng.below(st.nrows);
    for (int32_t k = 0; k < st.nrows; ++k) {
      int32_t i    = (first + k) % st.nrows;
      int32_t free = 0;
      for (const auto& entry : sh.row[i]) {
        free += sh.live(entry.first);
      }
      if (free == 0) { continue; }
      int32_t pick = rng.below(free);
      for (int64_t e = st.csr.begin(i); e < st.csr.end(i); ++e) {
        if (!st.csr.live(e) || !sh.live(st.csr.index[e])) { continue; }
        if (pick-- == 0) { return {i, st.csr.index[e], e}; }
      }
    }
    return {};
  }

  void expect_rows_of_col(int32_t col, uint8_t events)
  {
    for (int32_t i = 0; i < sh.nrows; ++i) {
      if (sh.row[i].count(col)) { row_events[i] |= events; }
    }
  }

  void expect_live_cols_of_row(int32_t row)
  {
    for (const auto& entry : sh.row[row]) {
      if (sh.live(entry.first)) { col_events[entry.first] |= col_event::locks; }
    }
  }

  bool set_col_lower()
  {
    int32_t j = random_live_col();
    if (j < 0) { return false; }
    double v = rng.chance(0.12) ? -inf : random_bound_value(rng);
    if (v > sh.col_upper[j]) { v = sh.col_upper[j] - std::fabs(random_bound_value(rng)); }
    if (v == sh.col_lower[j]) { return false; }
    expect_rows_of_col(j, row_event::activity);
    st.set_col_lower(j, v);
    sh.col_lower[j] = v;
    return true;
  }

  bool set_col_upper()
  {
    int32_t j = random_live_col();
    if (j < 0) { return false; }
    double v = rng.chance(0.12) ? inf : random_bound_value(rng);
    if (v < sh.col_lower[j]) { v = sh.col_lower[j] + std::fabs(random_bound_value(rng)); }
    if (v == sh.col_upper[j]) { return false; }
    expect_rows_of_col(j, row_event::activity);
    st.set_col_upper(j, v);
    sh.col_upper[j] = v;
    return true;
  }

  bool fix_col()
  {
    int32_t j = random_live_col();
    if (j < 0) { return false; }
    double lower = sh.col_lower[j], upper = sh.col_upper[j];
    double v     = random_bound_value(rng);
    int32_t mode = rng.below(3);
    if (mode == 0 && lower > -inf) { v = lower; }
    if (mode == 1 && upper < inf) { v = upper; }
    if (mode == 2 && lower > -inf && upper < inf) { v = lower + (upper - lower) * rng.uniform(); }
    if (!(lower <= v && v <= upper)) { return false; }
    expect_rows_of_col(j, row_event::entries);
    st.fix_col(j, v);
    sh.fix_col(j, v);
    return true;
  }

  bool remove_col()
  {
    if (st.ncols == 0) { return false; }
    int32_t first = rng.below(st.ncols);
    for (int32_t k = 0; k < st.ncols; ++k) {
      int32_t j = (first + k) % st.ncols;
      if (!sh.live(j) || st.csc.size[j] != 0) { continue; }
      if (sh.cost[j] != 0.) {
        st.set_cost(j, 0.);
        sh.cost[j]    = 0.;
        col_events[j] = col_event::locks;
      }
      st.remove_col(j);
      sh.status[j] = col_status_t::removed;
      return true;
    }
    return false;
  }

  bool set_cost()
  {
    int32_t j = random_live_col();
    if (j < 0) { return false; }
    double v = rng.chance(0.2) ? 0. : random_coefficient(rng, decimal);
    st.set_cost(j, v);
    sh.cost[j]    = v;
    col_events[j] = col_event::locks;
    return true;
  }

  bool add_offset()
  {
    double a = random_coefficient(rng, decimal);
    double b = random_bound_value(rng);
    st.add_objective_offset(a, b);
    sh.offset += a * b;
    sh.offset_scale += std::fabs(a * b);
    return true;
  }

  bool record_implied()
  {
    int32_t j = random_live_col();
    int32_t i = random_live_row();
    if (j < 0 || i < 0 || sh.is_integer[j]) { return false; }
    double v = random_bound_value(rng);
    if (rng.chance(0.5)) {
      st.record_implied_lower(j, v, i);
      sh.implied_lower[j]         = v;
      sh.implied_lower_row[j]     = i;
      sh.implied_lower_version[j] = sh.version[i];
    } else {
      st.record_implied_upper(j, v, i);
      sh.implied_upper[j]         = v;
      sh.implied_upper_row[j]     = i;
      sh.implied_upper_version[j] = sh.version[i];
    }
    return true;
  }

  bool delete_row()
  {
    int32_t i = random_live_row();
    if (i < 0) { return false; }
    expect_live_cols_of_row(i);
    st.delete_row(i);
    sh.delete_row(i);
    return true;
  }

  bool set_row_lower()
  {
    int32_t i = random_live_row();
    if (i < 0) { return false; }
    double upper = st.row_upper[i];
    double v     = rng.chance(0.25) ? -inf : random_bound_value(rng);
    if (v > upper) { v = upper - std::fabs(random_bound_value(rng)); }
    if (v == st.row_lower[i]) { return false; }
    if ((v > -inf) != (st.row_lower[i] > -inf)) { expect_live_cols_of_row(i); }
    row_events[i] = row_event::side;
    st.set_row_lower(i, v);
    sh.row_lower[i] = v;
    ++sh.version[i];
    return true;
  }

  bool set_row_upper()
  {
    int32_t i = random_live_row();
    if (i < 0) { return false; }
    double lower = st.row_lower[i];
    double v     = rng.chance(0.25) ? inf : random_bound_value(rng);
    if (v < lower) { v = lower + std::fabs(random_bound_value(rng)); }
    if (v == st.row_upper[i]) { return false; }
    if ((v < inf) != (st.row_upper[i] < inf)) { expect_live_cols_of_row(i); }
    row_events[i] = row_event::side;
    st.set_row_upper(i, v);
    sh.row_upper[i] = v;
    ++sh.version[i];
    return true;
  }

  bool shift_row_sides()
  {
    int32_t i = random_live_row();
    if (i < 0) { return false; }
    double a = random_coefficient(rng, decimal);
    double b = random_bound_value(rng);
    if (std::fabs(a * b) > 1e25) { return false; }
    row_events[i] = row_event::side;
    st.shift_row_sides(i, a, b);
    sh.shift_row_sides(i, a, b);
    return true;
  }

  bool scale_row()
  {
    static const double factors[] = {0.5, 2., 10., 0.1, 3., 1. / 3., 100.};
    int32_t i                     = random_live_row();
    if (i < 0) { return false; }
    double factor = factors[rng.below(7)];
    bool integral = decimal && rng.chance(0.6);
    if (integral) { factor = rng.chance(0.5) ? 10. : 100.; }
    for (const auto& entry : sh.row[i]) {
      double v = std::fabs(entry.second * factor);
      if (v < 1e-8 || v > 1e8) { return false; }
      if (integral && sh.live(entry.first) && std::round(v) == 0.) { return false; }
    }
    row_events[i] = row_event::entries;
    st.scale_row(i, factor, integral);
    sh.scale_row(i, factor, integral);
    return true;
  }

  bool set_coefficient()
  {
    entry_t entry = random_free_entry();
    if (entry.slot < 0) { return false; }
    double v = random_coefficient(rng, decimal);
    st.set_coefficient(entry.slot, v);
    sh.row[entry.row][entry.col] = v;
    ++sh.version[entry.row];
    row_events[entry.row] = row_event::entries;
    col_events[entry.col] = col_event::locks;
    return true;
  }

  bool delete_entry()
  {
    entry_t entry = random_free_entry();
    if (entry.slot < 0) { return false; }
    st.delete_entry(entry.slot);
    sh.row[entry.row].erase(entry.col);
    ++sh.version[entry.row];
    row_events[entry.row] = row_event::entries;
    col_events[entry.col] = col_event::locks;
    return true;
  }

  bool move_entry()
  {
    entry_t entry = random_free_entry();
    if (entry.slot < 0) { return false; }
    int32_t target = -1;
    for (int32_t attempt = 0; attempt < 10 && target < 0; ++attempt) {
      int32_t j = random_live_col();
      if (!sh.row[entry.row].count(j)) { target = j; }
    }
    if (target < 0) { return false; }
    double v = random_coefficient(rng, decimal);
    st.move_entry(entry.slot, target, v);
    sh.row[entry.row].erase(entry.col);
    sh.row[entry.row][target] = v;
    ++sh.version[entry.row];
    row_events[entry.row] = row_event::entries;
    col_events[entry.col] = col_event::locks;
    col_events[target]    = col_event::locks;
    if (st.csr.index[entry.slot] != target) { failure = "move_entry changed the CSR slot"; }
    return true;
  }

  bool compact()
  {
    st.compact_sparse_lines();
    for (int32_t i = 0; i < st.nrows; ++i) {
      if (st.row_alive[i] && st.csr.used[i] > 2 * st.csr.size[i]) {
        failure = "a row is sparse after compaction";
      }
    }
    for (int32_t j = 0; j < st.ncols; ++j) {
      if (st.is_live(j) && st.csc.used[j] > 2 * st.csc.size[j]) {
        failure = "a column is sparse after compaction";
      }
    }
    if (!st.row_frontier.empty(row_list::compaction) ||
        !st.col_frontier.empty(col_list::compaction)) {
      failure = "a compaction list is not empty after compaction";
    }
    return true;
  }

  bool rebuild()
  {
    std::vector<int32_t> row_origin, col_origin, ref_row_origin, ref_col_origin;
    model_t before = reference_model(st, ref_row_origin, ref_col_origin);
    st.rebuild();
    sh.rebuild();
    model_t after = extract_model(st, row_origin, col_origin);
    if (!well_formed(after)) { failure = "extract_model: the model is not well formed"; }
    if (!same_model(before, after)) { failure = "rebuild changed the model"; }
    if (row_origin != ref_row_origin || col_origin != ref_col_origin) {
      failure = "extract_model: wrong origins";
    }
    if (st.csr.slots() != st.live_nnz || st.csc.slots() != st.live_nnz || st.fixed_nnz != 0 ||
        st.dead_fraction() != 0.) {
      failure = "rebuild left dead slots or fixed entries";
    }
    if (st.csr.used != st.csr.size || st.csr.capacity != st.csr.size ||
        st.csc.used != st.csc.size || st.csc.capacity != st.csc.size ||
        st.row_free_size != st.csr.size) {
      failure = "rebuild: used, size, capacity and row_free_size differ";
    }
    return true;
  }

  bool find_entry()
  {
    if (st.nrows == 0 || st.ncols == 0) { return false; }
    for (int32_t probe = 0; probe < 20; ++probe) {
      int32_t i    = rng.below(st.nrows);
      int32_t j    = rng.below(st.ncols);
      int64_t slot = st.find_entry(i, j);
      auto it      = sh.row[i].find(j);
      if (it == sh.row[i].end()) {
        if (slot != -1) { failure = "find_entry found an entry that does not exist"; }
        continue;
      }
      if (slot < st.csr.begin(i) || slot >= st.csr.end(i) || st.csr.index[slot] != j ||
          st.csr.value[slot] != it->second) {
        failure = "find_entry missed an entry or returned another slot";
      }
    }
    return true;
  }

  bool apply(int32_t kind)
  {
    switch (kind) {
      case op::set_col_lower: return set_col_lower();
      case op::set_col_upper: return set_col_upper();
      case op::fix_col: return fix_col();
      case op::remove_col: return remove_col();
      case op::set_cost: return set_cost();
      case op::add_offset: return add_offset();
      case op::record_implied: return record_implied();
      case op::delete_row: return delete_row();
      case op::set_row_lower: return set_row_lower();
      case op::set_row_upper: return set_row_upper();
      case op::shift_row_sides: return shift_row_sides();
      case op::scale_row: return scale_row();
      case op::set_coefficient: return set_coefficient();
      case op::delete_entry: return delete_entry();
      case op::move_entry: return move_entry();
      case op::compact: return compact();
      case op::rebuild: return rebuild();
      case op::find_entry: return find_entry();
      default: return false;
    }
  }

  // One random operation, then every check. False when the operation had no candidate.
  bool step(int32_t kind)
  {
    clear_rule_lists(st);
    row_events.assign(st.nrows, 0);
    col_events.assign(st.ncols, 0);
    if (!apply(kind)) { return false; }
    if (!failure.empty()) { return true; }
    for (int32_t i = 0; i < st.nrows; ++i) {
      if ((st.row_frontier.pending[i] & row_event::entries) != row_events[i]) {
        failure = "row " + std::to_string(i) + ": events " +
                  std::to_string(st.row_frontier.pending[i] & row_event::entries) + ", expected " +
                  std::to_string(row_events[i]);
        return true;
      }
    }
    for (int32_t j = 0; j < st.ncols; ++j) {
      if ((st.col_frontier.pending[j] & col_event::locks) != col_events[j]) {
        failure = "column " + std::to_string(j) + ": events " +
                  std::to_string(st.col_frontier.pending[j] & col_event::locks) + ", expected " +
                  std::to_string(col_events[j]);
        return true;
      }
    }
    failure = st.consistency_error();
    if (failure.empty()) { failure = compare(st, sh); }
    return true;
  }
};

model_options_t random_shape(rng_t& rng, int32_t& steps)
{
  model_options_t o;
  o.decimal    = rng.chance(0.4);
  int32_t kind = rng.below(10);
  if (kind < 5) {
    o.nrows   = 1 + rng.below(12);
    o.ncols   = 1 + rng.below(12);
    o.density = 0.5;
    steps     = 400;
  } else if (kind < 8) {
    o.nrows   = 20 + rng.below(21);
    o.ncols   = 20 + rng.below(21);
    o.density = 0.15;
    steps     = 800;
  } else {
    // Rows above max_light_row and above max_activity_lanes entries.
    o.nrows     = 3 + rng.below(4);
    o.ncols     = 80 + rng.below(320);
    o.density   = 0.03;
    o.long_rows = 2;
    steps       = 500;
  }
  return o;
}

// Runs random operations on models of random shape until `total_steps` operations were applied.
// Returns false at the first failure.
bool run_seed(uint64_t seed,
              const tolerances_t& tol,
              int32_t total_steps,
              std::array<int64_t, op::count>& applied,
              int64_t& summations)
{
  rng_t rng(seed);
  int32_t done = 0;
  for (int32_t episode = 0; done < total_steps; ++episode) {
    int32_t steps;
    model_options_t shape = random_shape(rng, steps);
    model_t m             = random_model(rng, shape);
    if (!well_formed(m)) {
      test::fail(__FILE__, __LINE__, "the generator made a model that is not well formed");
      return false;
    }
    run_t run(m, tol, rng, shape.decimal);
    std::string error = run.st.consistency_error();
    if (error.empty()) { error = compare(run.st, run.sh); }
    if (!error.empty()) {
      test::fail(__FILE__, __LINE__, "seed " + std::to_string(seed) + " at construction: " + error);
      return false;
    }
    int64_t visits_before = run.st.slot_visits;
    for (int32_t s = 0; s < steps; ++s) {
      int32_t draw = rng.below(1000);
      int32_t kind = 0;
      while (draw >= op_share[kind]) {
        ++kind;
      }
      if (!run.step(kind)) { continue; }
      ++applied[kind];
      ++done;
      if (!run.failure.empty()) {
        test::fail(__FILE__,
                   __LINE__,
                   "seed " + std::to_string(seed) + " episode " + std::to_string(episode) +
                     " step " + std::to_string(s) + " after " + op_name[kind] + ": " + run.failure);
        return false;
      }
    }
    summations += run.st.slot_visits - visits_before;
  }
  return true;
}

}  // namespace

// Thousands of random operations per seed on models with infinite and huge bounds, empty and
// long rows; after every operation the full consistency check, the events of the operation, and
// the comparison with the naive reference (entries, bounds, sides, locks, counters, versions,
// activities with exact counters and sums within the error bound).
GPRS_TEST(state_random_operations)
{
  std::array<int64_t, op::count> applied{};
  int64_t visits = 0;
  for (uint64_t seed = 1; seed <= 10; ++seed) {
    if (!run_seed(seed, tolerances_t(), 4000, applied, visits)) { return; }
  }
  for (int32_t kind = 0; kind < op::count; ++kind) {
    if (applied[kind] < 100) {
      test::fail(__FILE__, __LINE__, std::string(op_name[kind]) + " was applied too rarely");
    }
  }
}

// The same with an error share so small that every row is summed again as soon as it has twice
// its size in updates: the path through recompute_row_activity inside the operations.
GPRS_TEST(state_random_operations_with_resummation)
{
  tolerances_t tol;
  tol.activity_error_share = 1e-30;
  std::array<int64_t, op::count> applied{};
  int64_t visits = 0;
  for (uint64_t seed = 101; seed <= 106; ++seed) {
    if (!run_seed(seed, tol, 4000, applied, visits)) { return; }
  }
}

// A lower infinite-term limit puts ordinary terms on both sides of the rule.
GPRS_TEST(state_random_operations_low_infinite_limit)
{
  tolerances_t tol;
  tol.infinite_term = 50.;
  std::array<int64_t, op::count> applied{};
  int64_t visits = 0;
  for (uint64_t seed = 201; seed <= 204; ++seed) {
    if (!run_seed(seed, tol, 4000, applied, visits)) { return; }
  }
}
