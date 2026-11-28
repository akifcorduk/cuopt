/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "alias_apply.hpp"
#include "forcing.hpp"
#include "implied.hpp"

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace gprs {

namespace {

// A column of the row that can be its binary: integer with bounds one apart, so that its term
// takes two values, `step` apart.
struct binary_t {
  int32_t col;
  double a;
  double step;
};

}  // namespace

struct simple_probing_t::work_t {
  std::vector<int32_t> rows;
  alias_batch_t batch;
  // Rows that the batch of the call holds.
  int64_t taken = 0;
  // Of the row that is being scanned: the columns that can be its binary in increasing index,
  // and, for one of them, the alias of every other free column as an edge.
  std::vector<binary_t> binaries;
  std::vector<alias_edge_t> edges;
  // By union-find root of the forest: the row scan that last met a column of that tree.
  std::vector<uint32_t> met;
  uint32_t scan = 0;
  // By column: the call that last scanned it, and what reach() found.
  std::vector<uint32_t> scanned;
  std::vector<double> reach;
  uint32_t call = 0;
};

namespace {

constexpr rule_outcome_t ok = rule_outcome_t::ok;

struct context_t {
  state_t& st;
  work_budget_t& budget;
  const options_t& opt;
  const tolerances_t& tol;
  simple_probing_t::work_t& w;
  rule_result_t& result;
};

// The two rows that an equation becomes with the term of its binary at its larger and at its
// smaller value: what is left between the side and the minimum activity of the other entries
// (low), and between their maximum activity and the side (high), each with the error bound of
// the activity it is taken from. The step enters the compensated sum, so that a gap is right to
// its own rounding whatever the size of the activities.
struct gaps_t {
  double low;
  double high;
  double low_error;
  double high_error;
};

gaps_t gaps(const state_t& st, int32_t row, double side, double step)
{
  const activity_t& min = st.min_act;
  const activity_t& max = st.max_act;
  dd_t low{min.hi[row], min.lo[row]};
  dd_t high{max.hi[row], max.lo[row]};
  dd_add_product(low, step, 1.);
  dd_sub_product(high, step, 1.);
  gaps_t g;
  g.low        = dd_residual(side, low);
  g.high       = -dd_residual(side, high);
  g.low_error  = activity_error(min.updates[row] + 1, std::fmax(min.max_term[row], step));
  g.high_error = activity_error(max.updates[row] + 1, std::fmax(max.max_term[row], step));
  return g;
}

// An integer column with the bounds 0 and 1.
bool is_unit(const state_t& st, int32_t col)
{
  return st.is_integer(col) && st.col_lower[col] == 0. && st.col_upper[col] == 1.;
}

// What the offset of an alias of the column meets: it enters the side of every row of the
// column, times the coefficient there. The largest magnitude of the live entries of the column,
// or infinity when one of its rows has a finite side beyond the size that a rule reads. One
// scan of the column per call: the state does not change while a call collects.
double reach(context_t& c, int32_t col)
{
  const state_t& st           = c.st;
  simple_probing_t::work_t& w = c.w;
  if (w.scanned.size() != (size_t)st.ncols) {
    w.scanned.assign(st.ncols, 0);
    w.reach.assign(st.ncols, 0.);
  }
  if (w.scanned[col] == w.call) { return w.reach[col]; }
  const line_store_t& csc = st.csc;
  double largest          = 0.;
  for (int64_t p = csc.begin(col); p < csc.end(col); ++p) {
    if (!csc.live(p)) { continue; }
    int32_t row = csc.index[p];
    if (std::fabs(csc.value[p]) > largest) { largest = std::fabs(csc.value[p]); }
    for (double side : {st.row_lower[row], st.row_upper[row]}) {
      if (!std::isinf(side) && !side_readable(c.tol, side)) { largest = inf; }
    }
  }
  c.budget.spend(csc.used[col]);
  w.scanned[col] = w.call;
  w.reach[col]   = largest;
  return largest;
}

// An offset of the given size, or below it, may enter the rows of the column: in every side
// its rounding stays below the tolerance (tolerances_t::side_error, as for a side that a rule
// reads).
bool offset_fits(context_t& c, int32_t col, double size)
{
  return size * reach(c, col) * c.tol.side_error <= c.tol.feasibility;
}

enum class probe_t : uint8_t { no, unit_refused, scale_refused, narrow_refused, size_refused, yes };

// The equation with p as its binary: yes when the other free entries are forcing at both values
// of the term of p and the aliases pass their limits; they are then in w.edges.
probe_t probe(context_t& c, int32_t row, double side, const binary_t& p)
{
  state_t& st             = c.st;
  const line_store_t& csr = st.csr;
  gaps_t g                = gaps(st, row, side, p.step);
  // The other entries at the bounds of the minimum activity and at those of the maximum one.
  forcing_entries_t at_min, at_max;
  bool units = is_unit(st, p.col), in_range = true;
  // Of the continuous columns among them: each range is wide against the rounding of its two
  // bounds, and the smallest range of a term.
  bool wide        = true;
  double narrowest = inf;
  c.w.edges.clear();
  c.budget.spend(csr.used[row]);
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    if (col == p.col || !st.is_live(col)) { continue; }
    double a     = csr.value[e];
    double lower = st.col_lower[col], upper = st.col_upper[col];
    gprs_assert(std::isfinite(lower) && std::isfinite(upper));
    bool continuous = !st.is_integer(col);
    at_min.add(a, min_activity_bound(a, lower, upper), continuous);
    at_max.add(a, max_activity_bound(a, lower, upper), continuous);
    // The term of the binary is at its larger value where the other terms are at their
    // smallest: a column whose coefficient has the sign of that of the binary falls from its
    // upper bound to its lower one while the binary rises from 0 to 1.
    bool falls   = (p.a > 0) == (a > 0);
    double range = upper - lower;
    double scale = falls ? -range : range;
    double start = falls ? upper : lower;
    if (!(range <= c.opt.alias_scale_limit)) { in_range = false; }
    if (!continuous && range == 1. && !is_unit(st, col)) { units = false; }
    if (continuous) {
      if (!(bound_rounding(lower) + bound_rounding(upper) <= c.tol.zero * range)) { wide = false; }
      if (std::fabs(a) * range < narrowest) { narrowest = std::fabs(a) * range; }
    }
    // x[col] = start + scale * x[p].
    c.w.edges.push_back(col < p.col ? alias_edge_t{row, col, p.col, 1., -scale, start}
                                    : alias_edge_t{row, p.col, col, -scale, 1., start});
  }
  if (!forcing(c.tol, at_min, g.low, g.low_error, true) ||
      !forcing(c.tol, at_max, -g.high, g.high_error, false)) {
    return probe_t::no;
  }
  if (!units) { return probe_t::unit_refused; }
  if (!in_range) { return probe_t::scale_refused; }
  // The forcing test lets a continuous column sit off its bound by the rounding of the row, and
  // a bound of such a column is itself rounded. Fixed at the bound, that is rounding. As an
  // alias it is the scale that carries it: where the range of the column, or of its term, is
  // not far above that rounding, the two bounds are one value as far as the row can tell, and
  // an alias between them puts coefficients of rounding size into the other rows of the column.
  double slack =
    std::fmax(std::fmax(g.low, 0.) + g.low_error, std::fmax(g.high, 0.) + g.high_error);
  if (!wide || !(slack <= c.tol.zero * narrowest)) { return probe_t::narrow_refused; }
  // Whichever column the batch makes the root, the alias of a column to it has one of the two
  // bounds of the column as its offset, and 0 or 1 for the binary and the other columns with
  // those bounds.
  if (!offset_fits(c, p.col, 1.)) { return probe_t::size_refused; }
  for (const alias_edge_t& edge : c.w.edges) {
    int32_t col = edge.col_a == p.col ? edge.col_b : edge.col_a;
    double size = std::fmax(std::fabs(st.col_lower[col]), std::fabs(st.col_upper[col]));
    if (!offset_fits(c, col, size)) { return probe_t::size_refused; }
  }
  return probe_t::yes;
}

// The aliases of the row enter the batch, unless edges of the batch connect two of its columns
// already: the row then waits for the next call.
void take(context_t& c, int32_t row, const binary_t& p)
{
  state_t& st                 = c.st;
  simple_probing_t::work_t& w = c.w;
  w.batch.init(st);
  if (w.met.size() != (size_t)st.ncols) { w.met.assign(st.ncols, 0); }
  const alias_forest_t& forest = w.batch.forest;
  ++w.scan;
  auto meets = [&w, &forest](int32_t col) {
    if (forest.link[col] == -1) { return false; }
    int32_t tree = tree_of(forest, col);
    if (w.met[tree] == w.scan) { return true; }
    w.met[tree] = w.scan;
    return false;
  };
  bool connected = meets(p.col);
  for (const alias_edge_t& edge : w.edges) {
    connected = connected || meets(edge.col_a == p.col ? edge.col_b : edge.col_a);
  }
  if (connected) {
    ++c.result.columns.probing_connected;
    st.row_frontier.push(list_bit(row_list::simple_probing), row);
    return;
  }
  int64_t entries = st.csc.size[p.col];
  for (size_t k = 0; k < w.edges.size(); ++k) {
    const alias_edge_t& edge = w.edges[k];
    bool added               = w.batch.add(edge);
    gprs_assert(added);
    entries += st.csc.size[edge.col_a == p.col ? edge.col_b : edge.col_a];
  }
  c.budget.spend(entries);
  ++w.taken;
}

// A listed live row: an equation with three or more free entries that a binary splits into two
// forcing rows gives its aliases to the batch.
void collect(context_t& c, int32_t row)
{
  state_t& st = c.st;
  double side = st.row_upper[row];
  if (st.row_free_size[row] < 3 || st.row_lower[row] != side || !side_readable(c.tol, side)) {
    return;
  }
  const activity_t& min = st.min_act;
  const activity_t& max = st.max_act;
  if (min.inf_count[row] != 0 || max.inf_count[row] != 0) { return; }
  if (refresh_activity(st, row)) { c.budget.spend(st.csr.used[row]); }
  // (maximum activity - side) - (side - minimum activity): zero when a binary carries half of
  // the range of the row.
  dd_t ends      = dd_add(dd_t{min.hi[row], min.lo[row]}, dd_t{max.hi[row], max.lo[row]});
  double balance = dd_residual(2. * side, ends);
  double error   = st.min_activity_error(row) + st.max_activity_error(row);
  if (!(std::fabs(balance) - error <= c.opt.probing_balance * c.tol.feasibility)) { return; }

  const line_store_t& csr         = st.csr;
  std::vector<binary_t>& binaries = c.w.binaries;
  binaries.clear();
  c.budget.spend(csr.used[row]);
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    if (!st.is_live(col)) { continue; }
    double a     = csr.value[e];
    double range = st.col_upper[col] - st.col_lower[col];
    if (!st.is_integer(col) || range != 1.) { continue; }
    double step = std::fabs(a);
    gaps_t g    = gaps(st, row, side, step);
    if (forcing_possible(c.tol, g.low, g.low_error) &&
        forcing_possible(c.tol, g.high, g.high_error)) {
      binaries.push_back(binary_t{col, a, step});
    }
  }
  if (binaries.size() > 1) {
    std::sort(binaries.begin(), binaries.end(), [](const binary_t& a, const binary_t& b) {
      return a.col < b.col;
    });
  }
  // What the row was left for, when a binary splits it into two forcing rows and none is taken:
  // the first of these that a binary met.
  bool unit_refused = false, scale_refused = false, narrow_refused = false;
  bool size_refused = false;
  for (const binary_t& p : binaries) {
    probe_t found = probe(c, row, side, p);
    if (found == probe_t::yes) {
      take(c, row, p);
      return;
    }
    unit_refused   = unit_refused || found == probe_t::unit_refused;
    scale_refused  = scale_refused || found == probe_t::scale_refused;
    narrow_refused = narrow_refused || found == probe_t::narrow_refused;
    size_refused   = size_refused || found == probe_t::size_refused;
  }
  column_stats_t& stats = c.result.columns;
  if (unit_refused) {
    ++stats.probing_unit_refused;
  } else if (scale_refused) {
    ++stats.probing_scale_refused;
  } else if (narrow_refused) {
    ++stats.probing_narrow_refused;
  } else if (size_refused) {
    ++stats.probing_size_refused;
    // The size test reads the other rows of the columns, which change without an event of this
    // row: no list brings the row back when they do.
    c.result.limited = true;
  }
}

}  // namespace

simple_probing_t::simple_probing_t() : work(new work_t) {}

simple_probing_t::~simple_probing_t() = default;

rule_result_t simple_probing_t::run(state_t& st,
                                    work_budget_t& budget,
                                    postsolve_t& ps,
                                    const options_t& opt)
{
  work_t& w = *work;
  rule_result_t result;
  context_t c{st, budget, opt, opt.tol, w, result};
  column_stats_t& stats = result.columns;

  st.row_frontier.take(row_list::simple_probing, w.rows);
  ++w.call;
  w.taken     = 0;
  size_t next = 0;
  for (; next < w.rows.size(); ++next) {
    if (budget.stop()) {
      result.stopped = true;
      break;
    }
    int32_t row = w.rows[next];
    if (!st.row_alive[row]) { continue; }
    budget.spend(line_visit_work);
    collect(c, row);
  }
  for (; next < w.rows.size(); ++next) {
    if (st.row_alive[w.rows[next]]) {
      st.row_frontier.push(list_bit(row_list::simple_probing), w.rows[next]);
    }
  }
  if (w.taken == 0) { return result; }

  int32_t rows_before    = st.live_rows;
  int64_t changes_before = st.coefficient_changes;
  // No row of the batch is left without its substitutions from here on: the budget is not
  // tested.
  result.outcome = apply_aliases(st, ps, opt.tol, w.batch, alias_origin_t::probing, budget, result);
  if (result.outcome == ok) {
    const alias_forest_t& forest = w.batch.forest;
    for (const alias_map_t& map : w.batch.maps) {
      int32_t depth = forest.depth[forest.position[map.col]];
      if (depth > stats.alias_depth) { stats.alias_depth = depth; }
      if (std::fabs(map.scale) > stats.probing_max_scale) {
        stats.probing_max_scale = std::fabs(map.scale);
      }
    }
    stats.probing_rows += w.taken;
    stats.probing_rows_deleted += rows_before - st.live_rows;
    stats.probing_rows_cut += w.taken - (rows_before - st.live_rows);
    stats.probing_deep_batches += stats.alias_depth > 1;
    stats.probing_entry_changes += st.coefficient_changes - changes_before;
  }
  w.batch.clear();
  return result;
}

}  // namespace gprs
