/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The events of the column rules through the Stage 1 families and the storage steps between
// them: whatever a family, a compaction, a rebuild, make_explicit or absorb_fixed_terms changes
// of what the column rules read must be on their lists afterwards (rules.hpp, before (g)).
// tests/test_state_ops.cpp and test_state_random.cpp test the same per state operation.

#include "../src/saved_row.hpp"
#include "harness.hpp"
#include "oracle.hpp"

#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using namespace gprs;

namespace {

// What the column rules read of a live row: its sides less the terms of its fixed columns, its
// free entries, and the explicit bounds of their columns, which its activities are made of. The
// stored activities are left out: their last bits change with a fresh summation, which has no
// event.
struct row_view_t {
  bool alive   = false;
  double lower = 0.;
  double upper = 0.;
  struct entry_t {
    int32_t col;
    double value;
    double col_lower;
    double col_upper;
    bool operator==(const entry_t& o) const
    {
      return col == o.col && value == o.value && col_lower == o.col_lower &&
             col_upper == o.col_upper;
    }
  };
  std::vector<entry_t> entries;

  bool operator==(const row_view_t& o) const
  {
    return alive == o.alive && lower == o.lower && upper == o.upper && entries == o.entries;
  }
};

// What they read of a live column apart from its bounds, which have no column event: cost,
// locks and entries.
struct col_view_t {
  bool live    = false;
  double cost  = 0.;
  int32_t down = 0;
  int32_t up   = 0;
  std::vector<std::pair<int32_t, double>> entries;

  bool operator==(const col_view_t& o) const
  {
    return live == o.live && cost == o.cost && down == o.down && up == o.up && entries == o.entries;
  }
};

struct view_t {
  std::vector<row_view_t> rows;
  std::vector<col_view_t> cols;
};

view_t view_of(const state_t& st)
{
  view_t v;
  v.rows.resize(st.nrows);
  v.cols.resize(st.ncols);
  for (int32_t i = 0; i < st.nrows; ++i) {
    row_view_t& r = v.rows[i];
    r.alive       = st.row_alive[i];
    if (!r.alive) { continue; }
    dd_t fixed_terms = scan_row(st, i, [&](int32_t col, double value) {
      r.entries.push_back({col, value, st.col_lower[col], st.col_upper[col]});
    });
    std::sort(
      r.entries.begin(),
      r.entries.end(),
      [](const row_view_t::entry_t& a, const row_view_t::entry_t& b) { return a.col < b.col; });
    r.lower = free_side(st.row_lower[i], fixed_terms);
    r.upper = free_side(st.row_upper[i], fixed_terms);
  }
  for (int32_t j = 0; j < st.ncols; ++j) {
    col_view_t& c = v.cols[j];
    c.live        = st.is_live(j);
    if (!c.live) { continue; }
    c.cost = st.cost[j];
    c.down = st.down_locks[j];
    c.up   = st.up_locks[j];
    for (int64_t p = st.csc.begin(j); p < st.csc.end(j); ++p) {
      if (st.csc.live(p)) { c.entries.push_back({st.csc.index[p], st.csc.value[p]}); }
    }
    std::sort(c.entries.begin(), c.entries.end());
  }
  return v;
}

// The Stage 1 families on one model in the order of the engine, with the lists of the column
// rules emptied before every step and read after it.
struct driver_t {
  options_t opt;
  state_t st;
  postsolve_t ps;
  row_rules_t row_rules;
  bound_propagation_t propagation;
  row_tightening_t tightening;
  doubleton_t doubleton;
  parallel_rows_t parallel_rows;
  dual_fixing_t dual_fixing;
  std::string failure;
  bool ended          = false;
  int64_t row_changes = 0;
  int64_t col_changes = 0;
  int64_t rebuilds    = 0;

  explicit driver_t(const model_t& m) : st(m, opt.tol) { ps.init(m); }

  // The start step of presolve().
  bool start()
  {
    for (int32_t i = 0; i < st.nrows; ++i) {
      if (st.row_lower[i] <= st.row_upper[i]) { continue; }
      if (st.row_lower[i] - st.row_upper[i] > opt.tol.feasibility) { return false; }
      st.set_row_lower(i, st.row_upper[i]);
    }
    for (int32_t j = 0; j < st.ncols; ++j) {
      double old_lower = st.col_lower[j], old_upper = st.col_upper[j];
      double lower = old_lower, upper = old_upper;
      if (st.is_integer(j)) {
        lower = round_lower_integer(lower, opt.tol);
        upper = round_upper_integer(upper, opt.tol);
      }
      if (lower > upper) {
        if (lower - upper > opt.tol.feasibility) { return false; }
        lower = upper;
      }
      double first = lower < old_upper ? lower : old_upper;
      if (first != old_lower) { st.set_col_lower(j, first); }
      if (upper != old_upper) { st.set_col_upper(j, upper); }
      if (lower != first) { st.set_col_lower(j, lower); }
      if (lower == upper) { fix_column(st, ps, j, lower); }
    }
    return true;
  }

  void clear_column_lists()
  {
    std::vector<int32_t> taken;
    for (int32_t l = row_list::simple_probing; l <= row_list::parallel_cols; ++l) {
      st.row_frontier.take(l, taken);
    }
    for (int32_t l = col_list::singleton_cols; l <= col_list::parallel_cols; ++l) {
      st.col_frontier.take(l, taken);
    }
  }

  // One step. `change` returns the outcome of a family, ok for a storage step.
  template <typename change_t>
  void step(const char* name, change_t&& change)
  {
    if (ended || !failure.empty()) { return; }
    view_t before = view_of(st);
    clear_column_lists();
    if (change() != rule_outcome_t::ok) {
      ended = true;
      return;
    }
    view_t after = view_of(st);
    for (int32_t i = 0; i < st.nrows && failure.empty(); ++i) {
      if (!before.rows[i].alive || !after.rows[i].alive || before.rows[i] == after.rows[i]) {
        continue;
      }
      ++row_changes;
      if ((st.row_frontier.pending[i] & row_event::column_rules) != row_event::column_rules) {
        failure = std::string(name) + " changed row " + std::to_string(i) +
                  " and left it off a row list of the column rules, pending " +
                  std::to_string(st.row_frontier.pending[i]);
      }
    }
    for (int32_t j = 0; j < st.ncols && failure.empty(); ++j) {
      if (!before.cols[j].live || !after.cols[j].live || before.cols[j] == after.cols[j]) {
        continue;
      }
      ++col_changes;
      if ((st.col_frontier.pending[j] & col_event::column_rules) != col_event::column_rules) {
        failure = std::string(name) + " changed column " + std::to_string(j) +
                  " and left it off a column list of the column rules, pending " +
                  std::to_string(st.col_frontier.pending[j]);
      }
    }
    if (failure.empty()) {
      std::string error = st.consistency_error();
      if (!error.empty()) { failure = std::string(name) + ": " + error; }
    }
  }

  // As the engine: a family is called when its list holds a row or a column.
  template <typename family_t>
  void call(const char* name, int32_t rule, family_t& family)
  {
    bool pending = rule == rule::dual_fixing ? !st.col_frontier.empty(col_list::dual_fixing)
                                             : !st.row_frontier.empty(rule);
    if (!pending) { return; }
    step(name, [&] {
      work_budget_t budget = make_budget(opt.budget[rule], st);
      return family.run(st, budget, ps, opt).outcome;
    });
  }

  void storage(bool rebuild)
  {
    if (rebuild && !ended && st.fixed_terms_absorbable()) {
      ++rebuilds;
      step("rebuild", [&] {
        st.rebuild();
        return rule_outcome_t::ok;
      });
    } else {
      step("compact_sparse_lines", [&] {
        st.compact_sparse_lines();
        return rule_outcome_t::ok;
      });
    }
  }

  void round(bool rebuild)
  {
    for (int32_t pass = 0; pass < 3; ++pass) {
      call("row rules", rule::row_rules, row_rules);
      call("bound propagation", rule::propagation, propagation);
      call("row tightening", rule::tightening, tightening);
      call("dual fixing", rule::dual_fixing, dual_fixing);
      storage(false);
    }
    storage(rebuild);
    call("parallel rows", rule::parallel_rows, parallel_rows);
    storage(rebuild);
    call("doubleton equations", rule::doubleton, doubleton);
    storage(rebuild);
  }

  void run()
  {
    if (!start()) { return; }
    for (int32_t r = 0; r < 4; ++r) {
      round(r % 2 == 1);
    }
    step("make_explicit", [&] { return propagation.make_explicit(st, ps, opt).outcome; });
    for (int32_t r = 0; r < 3; ++r) {
      round(r % 2 == 0);
    }
    if (!ended && st.fixed_nnz > 0 && st.fixed_terms_absorbable()) {
      step("absorb_fixed_terms", [&] {
        st.absorb_fixed_terms();
        return rule_outcome_t::ok;
      });
      round(false);
    }
  }
};

}  // namespace

// No change of a row or a column that the column rules read is left off their lists, whichever
// Stage 1 family or storage step makes it.
GPRS_TEST(columns_events_follow_every_family)
{
  std::vector<oracle::generator_options_t> shapes(5);
  shapes[0].p_singleton_col      = 0.5;
  shapes[1].p_doubleton_equation = 0.45;
  shapes[1].p_equation           = 0.4;
  shapes[1].p_zero_cost          = 0.5;
  shapes[2].p_parallel_row       = 0.5;
  shapes[2].p_huge_bound         = 0.6;
  shapes[2].max_line_cols        = 3;
  shapes[2].p_singleton_col      = 0.3;
  shapes[3].p_common_factor      = 0.5;
  shapes[3].p_scaled             = 1.;
  shapes[3].max_coefficient      = 4;
  shapes[4].p_continuous         = 1.;
  shapes[4].max_line_cols        = 3;
  shapes[4].p_infinite_bound     = 0.25;
  shapes[4].p_forcing_side       = 0.25;
  int64_t models = 0, row_changes = 0, col_changes = 0, rebuilds = 0, aliases = 0, fixed = 0;
  int64_t additions = 0, failures = 0;
  for (size_t shape = 0; shape < shapes.size(); ++shape) {
    for (uint64_t seed = 7400000; seed < 7402500 && failures < 3; ++seed) {
      driver_t d(oracle::generate(seed, shapes[shape]).model);
      d.run();
      ++models;
      row_changes += d.row_changes;
      col_changes += d.col_changes;
      rebuilds += d.rebuilds;
      aliases += d.ps.aliases.size();
      fixed += d.ps.fixed_cols.size();
      additions += d.ps.row_additions.size();
      if (d.failure.empty()) { continue; }
      ++failures;
      test::fail(
        __FILE__,
        __LINE__,
        "shape " + std::to_string(shape) + " seed " + std::to_string(seed) + ": " + d.failure);
    }
  }
  GPRS_CHECK_EQ(failures, 0);
  GPRS_CHECK_EQ(models, 5 * 2500);
  GPRS_CHECK(row_changes > 5000 && col_changes > 5000);
  GPRS_CHECK(rebuilds > 5000 && aliases > 300 && fixed > 10000 && additions > 0);
}

// A list_mask_t has a bit for sixteen lists, and every one of them works as a list.
GPRS_TEST(columns_frontier_uses_every_bit)
{
  const int32_t lists = 8 * sizeof(list_mask_t);
  frontier_set_t f;
  f.init(lists, 4, list_bit(lists - 1) | list_bit(8) | list_bit(0));
  std::vector<int32_t> all = {0, 1, 2, 3}, taken;
  for (int32_t l = 0; l < lists; ++l) {
    bool listed = l == 0 || l == 8 || l == lists - 1;
    GPRS_CHECK(f.list[l] == (listed ? all : std::vector<int32_t>()));
  }
  f.take(lists - 1, taken);
  GPRS_CHECK(taken == all);
  GPRS_CHECK(f.empty(lists - 1));
  GPRS_CHECK_EQ(f.pending[2], list_bit(8) | list_bit(0));
  list_mask_t every = ~0;
  f.push(every, 2);
  for (int32_t l = 0; l < lists; ++l) {
    bool listed = l == 0 || l == 8;
    GPRS_CHECK(f.list[l] == (listed ? all : std::vector<int32_t>({2})));
  }
  GPRS_CHECK_EQ(f.pending[2], every);
  f.push(every, 2);
  GPRS_CHECK_EQ(f.list[lists - 1].size(), (size_t)1);
  for (int32_t l = 0; l < lists; ++l) {
    f.take(l, taken);
    GPRS_CHECK(f.empty(l));
  }
  for (int32_t i = 0; i < 4; ++i) {
    GPRS_CHECK_EQ(f.pending[i], 0);
  }
}
