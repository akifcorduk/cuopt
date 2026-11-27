/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "saved_row.hpp"

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <cmath>
#include <utility>
#include <vector>

namespace gprs {

struct doubleton_t::work_t {
  alias_forest_t forest;
  std::vector<alias_map_t> maps;
  std::vector<int32_t> rows;
  // Collected rows that closed a cycle of the batch.
  std::vector<int32_t> cycle_rows;
  // CSR slot of the entry of the root being substituted in each row, -1 without one. All -1
  // outside the maps of one root.
  std::vector<int64_t> root_slot;
  // By union-find root of the forest: a map of the batch lies in that tree. All 0 between calls.
  std::vector<uint8_t> tree_has_map;
};

namespace {

constexpr rule_outcome_t ok         = rule_outcome_t::ok;
constexpr rule_outcome_t infeasible = rule_outcome_t::infeasible;

// A transferred bound is a difference divided by the scale, two rounded operations. Twice
// their error is allowed for when the bound of an integer root is rounded.
constexpr double transfer_error_units = 4.;

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

// The bounds of map.col as explicit bounds of the root, which is live or was fixed by an earlier
// map of the batch.
rule_outcome_t transfer_bounds(state_t& st,
                               postsolve_t& ps,
                               const tolerances_t& tol,
                               const alias_map_t& map)
{
  int32_t root = map.root;
  gprs_assert(st.col_status[root] != col_status_t::removed);
  double lower = (st.col_lower[map.col] - map.offset) / map.scale;
  double upper = (st.col_upper[map.col] - map.offset) / map.scale;
  if (map.scale < 0) { std::swap(lower, upper); }
  if (st.is_integer(root)) {
    // An integer root is rounded here with the tolerance in the units of the column: a value of
    // the root that the rounding of tighten_lower / tighten_upper would let through moves the
    // column by |scale| times the distance to its bound.
    double slack = tol.feasibility / std::fmax(1., std::fabs(map.scale));
    double error = transfer_error_units * rounding_unit;
    if (std::isfinite(lower)) { lower = std::ceil(lower - slack - error * std::fabs(lower)); }
    if (std::isfinite(upper)) { upper = std::floor(upper + slack + error * std::fabs(upper)); }
  }
  if (lower > -inf) {
    if (st.is_live(root)) {
      if (tighten_lower(st, ps, root, lower).outcome != ok) { return infeasible; }
    } else if (lower - st.col_lower[root] > tol.feasibility) {
      return infeasible;
    }
  }
  if (upper < inf) {
    if (st.is_live(root)) {
      if (tighten_upper(st, ps, root, upper).outcome != ok) { return infeasible; }
    } else if (st.col_upper[root] - upper > tol.feasibility) {
      return infeasible;
    }
  }
  return ok;
}

void set_root_slots(const state_t& st, int32_t root, std::vector<int64_t>& root_slot, bool set)
{
  const line_store_t& csc = st.csc;
  for (int64_t p = csc.begin(root); p < csc.end(root); ++p) {
    if (csc.live(p)) { root_slot[csc.index[p]] = set ? csc.twin[p] : -1; }
  }
}

// x[map.col] = map.scale * x[map.root] + map.offset in every row of the column and in the cost.
void substitute(state_t& st,
                postsolve_t& ps,
                const tolerances_t& tol,
                const alias_map_t& map,
                std::vector<int64_t>& root_slot)
{
  int32_t col = map.col, root = map.root;
  // move_entry can reallocate the arrays of the CSC: every access goes through st.csc.
  for (int64_t p = st.csc.begin(col); p < st.csc.end(col); ++p) {
    if (!st.csc.live(p)) { continue; }
    int32_t row  = st.csc.index[p];
    int64_t slot = st.csc.twin[p];
    double a     = st.csc.value[p];
    double moved = a * map.scale;
    if (map.offset != 0.) { st.shift_row_sides(row, a, map.offset); }
    int64_t at = root_slot[row];
    if (at < 0) {
      st.move_entry(slot, root, moved);
      root_slot[row] = slot;
      continue;
    }
    double old = st.csr.value[at];
    double sum = old + moved;
    if (std::fabs(sum) <= tol.cancellation * (std::fabs(old) + std::fabs(moved))) {
      st.delete_entry(at);
      root_slot[row] = -1;
    } else {
      st.set_coefficient(at, sum);
    }
    st.delete_entry(slot);
  }
  double cost = st.cost[col];
  if (cost != 0.) {
    double moved = cost * map.scale;
    double old   = st.cost[root];
    double sum   = old + moved;
    if (std::fabs(sum) <= tol.cancellation * (std::fabs(old) + std::fabs(moved))) { sum = 0.; }
    st.set_cost(root, sum);
    st.add_objective_offset(cost, map.offset);
    st.set_cost(col, 0.);
  }
  st.remove_col(col);
  ps.push_alias(col, root, map.row, map.scale, map.offset);
}

int32_t tree_of(const alias_forest_t& forest, int32_t col)
{
  while (forest.link[col] != col) {
    col = forest.link[col];
  }
  return col;
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
  if (w.forest.link.size() != (size_t)st.ncols) {
    w.forest.init(st.ncols);
    w.tree_has_map.assign(st.ncols, 0);
  }
  if (w.root_slot.size() != (size_t)st.nrows) { w.root_slot.assign(st.nrows, -1); }
  const tolerances_t& tol    = opt.tol;
  const list_mask_t own_list = list_bit(row_list::doubleton);
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
    if (!w.forest.add_edge(edge)) { w.cycle_rows.push_back(row); }
  }
  for (; next < w.rows.size(); ++next) {
    if (st.row_alive[w.rows[next]]) { st.row_frontier.push(own_list, w.rows[next]); }
  }

  w.forest.freeze(st, w.maps);
  // No row of a map is left without its substitution from here on: the budget is not tested.
  for (const alias_map_t& map : w.maps) {
    st.delete_row(map.row);
  }
  size_t k = 0;
  while (k < w.maps.size() && result.outcome == ok) {
    int32_t root = w.maps[k].root;
    bool marked  = false;
    for (; k < w.maps.size() && w.maps[k].root == root; ++k) {
      const alias_map_t& map = w.maps[k];
      gprs_assert(st.is_live(map.col));
      result.outcome = transfer_bounds(st, ps, tol, map);
      if (result.outcome != ok) { break; }
      if (st.is_live(root)) {
        if (!marked) {
          set_root_slots(st, root, w.root_slot, true);
          marked = true;
        }
        substitute(st, ps, tol, map, w.root_slot);
      } else {
        double lower = st.col_lower[map.col], upper = st.col_upper[map.col];
        double value = map.scale * st.col_lower[root] + map.offset;
        if (value < lower) { value = lower; }
        if (value > upper) { value = upper; }
        fix_column(st, ps, map.col, value);
      }
      ++result.findings;
      budget.found();
    }
    if (marked) { set_root_slots(st, root, w.root_slot, false); }
  }

  if (result.outcome == ok) {
    // A row that closed a cycle can be a tree edge of the next batch. So can a cut edge of a
    // tree that lost a column: the batch tried it in one direction only. A cut edge of a tree
    // without a map is inadmissible in both directions and comes back with its next event.
    for (int32_t row : w.cycle_rows) {
      st.row_frontier.push(own_list, row);
    }
    for (const alias_map_t& map : w.maps) {
      w.tree_has_map[tree_of(w.forest, map.col)] = 1;
    }
    for (const alias_edge_t& edge : w.forest.edges) {
      if (st.row_alive[edge.row] && w.tree_has_map[tree_of(w.forest, edge.col_a)]) {
        st.row_frontier.push(own_list, edge.row);
      }
    }
    for (const alias_map_t& map : w.maps) {
      w.tree_has_map[tree_of(w.forest, map.col)] = 0;
    }
  }
  w.forest.clear();
  return result;
}

}  // namespace gprs
