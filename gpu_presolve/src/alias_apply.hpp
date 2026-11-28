/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// The aliases of one call of a family as one batch through the alias forest: the equations
// a * x[col_a] + b * x[col_b] = rhs that the family collects are the edges, freeze() picks a
// root for every tree and composes the map of every other column to it, and each column is
// substituted once, into its root. Shared by the doubleton equations (rules.hpp (d)), whose
// rows are such equations, and by simple probing (rules.hpp (g)), whose rows give one such
// equation per column other than the binary.

#include <gprs/assert.hpp>
#include <gprs/postsolve.hpp>
#include <gprs/rules.hpp>
#include <gprs/state.hpp>

#include <cmath>
#include <utility>
#include <vector>

namespace gprs {

// A transferred bound is a difference divided by the scale, two rounded operations. Twice
// their error is allowed for when the bound of an integer root is rounded.
constexpr double transfer_error_units = 4.;

// The batch of one call. The arrays by row live for the whole presolve.
struct alias_batch_t {
  alias_forest_t forest;
  std::vector<alias_map_t> maps;
  // By row: the edges that the row gave the forest and that have not become a map. All 0
  // between calls.
  std::vector<int32_t> open_edges;
  // CSR slot of the entry of the root being substituted in each row, -1 without one. All -1
  // outside the maps of one root.
  std::vector<int64_t> root_slot;

  // Before the first edge of a call. O(rows + columns) once, else O(1).
  void init(const state_t& st)
  {
    if (forest.link.size() != (size_t)st.ncols) { forest.init(st.ncols); }
    if (root_slot.size() != (size_t)st.nrows) {
      root_slot.assign(st.nrows, -1);
      open_edges.assign(st.nrows, 0);
    }
  }

  // alias_forest_t::add_edge: false, with nothing added, when the two columns are connected.
  bool add(const alias_edge_t& edge)
  {
    if (!forest.add_edge(edge)) { return false; }
    ++open_edges[edge.row];
    return true;
  }

  // Ends the batch. O(columns and edges of the batch).
  void clear()
  {
    for (const alias_edge_t& edge : forest.edges) {
      open_edges[edge.row] = 0;
    }
    forest.clear();
  }
};

// The union-find root of a column of the batch: two columns are connected by edges of the
// batch when they have the same one. O(log(columns of the tree)).
inline int32_t tree_of(const alias_forest_t& forest, int32_t col)
{
  while (forest.link[col] != col) {
    col = forest.link[col];
  }
  return col;
}

// The bounds of map.col as explicit bounds of the root, which is live or was fixed by an earlier
// map of the batch.
inline rule_outcome_t transfer_bounds(state_t& st,
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
      if (tighten_lower(st, ps, root, lower).outcome != rule_outcome_t::ok) {
        return rule_outcome_t::infeasible;
      }
    } else if (lower - st.col_lower[root] > tol.feasibility) {
      return rule_outcome_t::infeasible;
    }
  }
  if (upper < inf) {
    if (st.is_live(root)) {
      if (tighten_upper(st, ps, root, upper).outcome != rule_outcome_t::ok) {
        return rule_outcome_t::infeasible;
      }
    } else if (st.col_upper[root] - upper > tol.feasibility) {
      return rule_outcome_t::infeasible;
    }
  }
  return rule_outcome_t::ok;
}

inline void set_root_slots(const state_t& st,
                           int32_t root,
                           std::vector<int64_t>& root_slot,
                           bool set)
{
  const line_store_t& csc = st.csc;
  for (int64_t p = csc.begin(root); p < csc.end(root); ++p) {
    if (csc.live(p)) { root_slot[csc.index[p]] = set ? csc.twin[p] : -1; }
  }
}

// x[map.col] = map.scale * x[map.root] + map.offset in every row of the column and in the cost.
// The record names the row of the map when the batch has deleted it, else no row.
inline void substitute(state_t& st,
                       postsolve_t& ps,
                       const tolerances_t& tol,
                       const alias_map_t& map,
                       alias_origin_t origin,
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
  ps.push_alias(col, root, st.row_alive[map.row] ? -1 : map.row, map.scale, map.offset, origin);
}

// Applies the collected batch in full, since a deleted row without its substitutions is a wrong
// model: freeze() returns the maps grouped by root, and a row is deleted when every edge that
// it gave the forest has become a map (a row with a cut edge stays). Then, per map and in the
// order of the maps: the bounds of the column are transferred to the root as explicit bounds;
// if the root is fixed afterwards, the column is fixed at its image, moved into its bounds when
// rounding puts it outside; otherwise the column is replaced by the root in its rows and in the
// cost and the alias is recorded with the origin. Every map is one finding. Returns infeasible
// when transferred bounds cross; the state is then not used again.
// The work, the entries of the substituted columns and of their roots, is not spent here: the
// caller spends it when it collects an edge, where it tests its budget.
inline rule_outcome_t apply_aliases(state_t& st,
                                    postsolve_t& ps,
                                    const tolerances_t& tol,
                                    alias_batch_t& batch,
                                    alias_origin_t origin,
                                    work_budget_t& budget,
                                    rule_result_t& result)
{
  batch.forest.freeze(st, batch.maps);
  for (const alias_map_t& map : batch.maps) {
    gprs_assert(batch.open_edges[map.row] > 0);
    if (--batch.open_edges[map.row] == 0) { st.delete_row(map.row); }
  }
  rule_outcome_t outcome = rule_outcome_t::ok;
  size_t k               = 0;
  while (k < batch.maps.size() && outcome == rule_outcome_t::ok) {
    int32_t root = batch.maps[k].root;
    bool marked  = false;
    for (; k < batch.maps.size() && batch.maps[k].root == root; ++k) {
      const alias_map_t& map = batch.maps[k];
      gprs_assert(st.is_live(map.col));
      outcome = transfer_bounds(st, ps, tol, map);
      if (outcome != rule_outcome_t::ok) { break; }
      if (st.is_live(root)) {
        if (!marked) {
          set_root_slots(st, root, batch.root_slot, true);
          marked = true;
        }
        substitute(st, ps, tol, map, origin, batch.root_slot);
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
    if (marked) { set_root_slots(st, root, batch.root_slot, false); }
  }
  return outcome;
}

}  // namespace gprs
