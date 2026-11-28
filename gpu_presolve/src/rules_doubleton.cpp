/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "alias_apply.hpp"
#include "saved_row.hpp"

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <cmath>
#include <utility>
#include <vector>

namespace gprs {

struct doubleton_t::work_t {
  alias_batch_t batch;
  std::vector<int32_t> rows;
  // Collected rows that closed a cycle of the batch.
  std::vector<int32_t> cycle_rows;
  // By union-find root of the forest: a map of the batch lies in that tree. All 0 between calls.
  std::vector<uint8_t> tree_has_map;
};

namespace {

constexpr rule_outcome_t ok = rule_outcome_t::ok;

// The two free entries of the row, the smaller column first, and its side less the terms of its
// fixed columns, summed in storage order as rebuild() does.
alias_edge_t read_edge(const state_t& st, int32_t row)
{
  alias_edge_t edge{row, -1, -1, 0., 0., 0.};
  dd_t fixed_terms = scan_row(st, row, [&edge](int32_t col, double value) {
    if (edge.col_a < 0) {
      edge.col_a = col;
      edge.a     = value;
    } else {
      gprs_assert(edge.col_b < 0);
      edge.col_b = col;
      edge.b     = value;
    }
  });
  gprs_assert(edge.col_b >= 0);
  if (edge.col_a > edge.col_b) {
    std::swap(edge.col_a, edge.col_b);
    std::swap(edge.a, edge.b);
  }
  edge.rhs = dd_residual(st.row_upper[row], fixed_terms);
  return edge;
}

}  // namespace

doubleton_t::doubleton_t() : work(new work_t) {}

doubleton_t::~doubleton_t() = default;

rule_result_t doubleton_t::run(state_t& st,
                               work_budget_t& budget,
                               postsolve_t& ps,
                               const options_t& opt)
{
  work_t& w = *work;
  w.batch.init(st);
  if (w.tree_has_map.size() != (size_t)st.ncols) { w.tree_has_map.assign(st.ncols, 0); }
  const alias_forest_t& forest = w.batch.forest;
  const list_mask_t own_list   = list_bit(row_list::doubleton);
  rule_result_t result;

  st.row_frontier.take(row_list::doubleton, w.rows);
  w.cycle_rows.clear();
  size_t next = 0;
  for (; next < w.rows.size(); ++next) {
    if (budget.stop()) {
      result.stopped = true;
      break;
    }
    int32_t row = w.rows[next];
    if (!st.row_alive[row]) { continue; }
    budget.spend(line_visit_work);
    double rhs = st.row_upper[row];
    if (st.row_free_size[row] != 2 || st.row_lower[row] != rhs || !std::isfinite(rhs)) { continue; }
    alias_edge_t edge = read_edge(st, row);
    budget.spend((int64_t)st.csr.used[row] + st.csc.size[edge.col_a] + st.csc.size[edge.col_b]);
    if (!w.batch.add(edge)) { w.cycle_rows.push_back(row); }
  }
  for (; next < w.rows.size(); ++next) {
    if (st.row_alive[w.rows[next]]) { st.row_frontier.push(own_list, w.rows[next]); }
  }

  // No row of a map is left without its substitution from here on: the budget is not tested.
  result.outcome =
    apply_aliases(st, ps, opt.tol, w.batch, alias_origin_t::doubleton, budget, result);

  if (result.outcome == ok) {
    // A row that closed a cycle can be a tree edge of the next batch. So can a cut edge of a
    // tree that lost a column: the batch tried it in one direction only. A cut edge of a tree
    // without a map is inadmissible in both directions and comes back with its next event.
    for (int32_t row : w.cycle_rows) {
      st.row_frontier.push(own_list, row);
    }
    for (const alias_map_t& map : w.batch.maps) {
      w.tree_has_map[tree_of(forest, map.col)] = 1;
    }
    for (const alias_edge_t& edge : forest.edges) {
      if (st.row_alive[edge.row] && w.tree_has_map[tree_of(forest, edge.col_a)]) {
        st.row_frontier.push(own_list, edge.row);
      }
    }
    for (const alias_map_t& map : w.batch.maps) {
      w.tree_has_map[tree_of(forest, map.col)] = 0;
    }
  }
  w.batch.clear();
  return result;
}

}  // namespace gprs
