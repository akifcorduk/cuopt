/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "implied.hpp"
#include "integral.hpp"
#include "saved_row.hpp"

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace gprs {

namespace {

// A singleton column of the row that is being visited.
struct single_t {
  int32_t col;
  int64_t slot;
  double a;
};

// Sizes that two counters compare a cost with (column_count::); they decide no reduction.
constexpr double near_cancellation = 1e-9;
constexpr double small_cost        = 1e-7;

}  // namespace

struct singleton_cols_t::work_t {
  std::vector<int32_t> cols;
  std::vector<int32_t> rows;
  // By column: what the free singleton columns substituted so far can multiply a deviation of
  // the reduced point by before it reaches the column. It holds no fact about a type.
  std::vector<double> growth;
  // By row: the pass over the lists that last visited it.
  std::vector<uint32_t> stamp;
  uint32_t pass = 0;
  // By row: a forcing column was left because a side of the row is not readable. A side changes
  // without an event of the columns of its row, so the row lists them when it is listed.
  std::vector<uint8_t> unreadable;
  // Entries saved of rows that stayed in the model, over the whole presolve.
  int64_t kept_saved = 0;
  // The bounds of continuous columns are explicit (bound_propagation_t::make_explicit).
  bool explicit_bounds = false;
  // Of one visit.
  std::vector<single_t> singles;
  std::vector<record_col_t> record_cols;
  std::vector<double> coefficients;
  std::vector<saved_row_t> saved;
  std::vector<int64_t> slots;
  // Columns of the row that the growth factor of earlier substitutions alone refused.
  std::vector<int32_t> refused;
  std::vector<int32_t> col_rows;
  // By column: taken by the zero-cost rule in the visit that is running. All 0 between visits.
  std::vector<uint8_t> taken;
  // By column: counted in growth_refused_row (bit 0) or in growth_refused (bit 1), so that the
  // two statistics count columns and not visits.
  std::vector<uint8_t> counted;
  // For column_count::rows_rescanned. epoch moves on with every call and every reduction; by
  // row, its value at the last scan of the row.
  uint32_t epoch = 0;
  std::vector<uint32_t> scanned_at;
  // The singleton columns of a visit by decreasing coefficient, as indices of `singles`.
  std::vector<int32_t> pivot_order;
};

namespace {

struct context_t {
  state_t& st;
  work_budget_t& budget;
  postsolve_t& ps;
  const options_t& opt;
  const tolerances_t& tol;
  singleton_cols_t::work_t& w;
  rule_result_t& result;
  // Reductions of this call.
  int64_t reductions = 0;
  // The counter of the test that pivot() makes next: of the refusal, when the test fails.
  int32_t cause = 0;
  // The end of the range that implied_free tested last is the one at the upper bound.
  bool upper_end = false;

  void finding(int32_t sub, int64_t count, bool dual)
  {
    result.findings += count;
    result.sub_findings[sub] += count;
    if (dual) { result.dual_findings += count; }
    ++reductions;
    ++w.epoch;
    budget.found();
  }

  void count(int32_t counter, int64_t events = 1)
  {
    if (events != 0) { result.columns.counts[counter] += events; }
  }
};

// What a visit reads of its row in one scan.
struct row_scan_t {
  dd_t fixed_terms;
  int32_t free_entries = 0;
  // The two largest magnitudes among the free entries, and where the largest one is.
  double largest       = 0.;
  double second        = 0.;
  int64_t largest_slot = -1;
  // The smallest magnitude among the free entries.
  double smallest = inf;
};

row_scan_t scan(context_t& c, int32_t row)
{
  const state_t& st       = c.st;
  const line_store_t& csr = st.csr;
  row_scan_t s;
  c.w.singles.clear();
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    double a    = csr.value[e];
    if (!st.is_live(col)) {
      dd_add_product(s.fixed_terms, a, st.col_lower[col]);
      continue;
    }
    ++s.free_entries;
    double size = std::fabs(a);
    if (size > s.largest) {
      s.second       = s.largest;
      s.largest      = size;
      s.largest_slot = e;
    } else if (size > s.second) {
      s.second = size;
    }
    if (size < s.smallest) { s.smallest = size; }
    if (st.csc.size[col] == 1) { c.w.singles.push_back(single_t{col, e, a}); }
  }
  if (c.w.singles.size() > 1) {
    std::sort(c.w.singles.begin(), c.w.singles.end(), [](const single_t& a, const single_t& b) {
      return a.col < b.col;
    });
  }
  return s;
}

// A side of the free part of a row that a reduction may read: of a size whose rounding stays
// below the tolerance, as side_readable asks of the stored side.
bool free_side_readable(const tolerances_t& tol, double side)
{
  return std::fabs(side) * tol.side_error <= tol.feasibility;
}

// One end of the range of (side - rest) / a over the explicit bounds of the other columns of
// the row: with the minimum of the rest, or with its maximum. False without one: the rest holds
// an infinite term, or the error guard of bound propagation drops it after a fresh summation.
bool range_end(context_t& c, int32_t row, int64_t e, double side, bool minimum, candidate_t& k)
{
  state_t& st           = c.st;
  const activity_t& act = minimum ? st.min_act : st.max_act;
  for (int32_t attempt = 0; attempt < 2; ++attempt) {
    row_rest_t rest;
    c.cause = column_count::refused_infinite_rest;
    if (!row_rest(st, row, e, minimum, rest)) { return false; }
    k.a     = st.csr.value[e];
    k.side  = side;
    k.rest  = rest.sum;
    k.error = rest.error;
    c.cause = column_count::refused_bound;
    if (!(k.error > c.tol.activity_error_share * c.tol.feasibility * std::fabs(k.a))) {
      k.bound = dd_residual(side, k.rest) / k.a;
      return std::isfinite(k.bound);
    }
    c.cause = column_count::refused_guard;
    if (activity_fresh(st, act, row)) { return false; }
    c.budget.spend(st.csr.used[row]);
    st.recompute_row_activity(row);
  }
  return false;
}

// The band around a bound of a column within which its row implies it.
double implied_band(const context_t& c, double bound)
{
  return c.opt.implied_band_scale * implied_tolerance(c.tol, bound);
}

// The test of implied_free on the end at the lower bound alone, read from the state as it is
// and without a fresh sum: an end whose error bound would ask for one counts as not implied.
// For the counters refused_ahead and refused_behind only.
bool lower_end_implied(const context_t& c, int32_t row, const single_t& s, double side)
{
  const state_t& st = c.st;
  double lower      = st.col_lower[s.col];
  if (lower == -inf) { return true; }
  row_rest_t rest;
  if (!row_rest(st, row, s.slot, s.a < 0, rest)) { return false; }
  if (rest.error > c.tol.activity_error_share * c.tol.feasibility * std::fabs(s.a)) {
    return false;
  }
  candidate_t k{dd_residual(side, rest.sum) / s.a, side, rest.sum, s.a, rest.error};
  return std::isfinite(k.bound) && k.bound - candidate_slack(k) >= lower - implied_band(c, lower);
}

// The row with `side` as its right-hand side gives the column of the singleton a value inside
// its explicit bounds whatever the other columns are inside theirs: each end of the range of
// (side - rest) / a, moved outward by its rounding, is within implied_band of the bound it must
// not pass; an infinite bound needs nothing. When it fails, c.cause and c.upper_end say why and
// at which end.
bool implied_free(context_t& c, int32_t row, const single_t& s, double side)
{
  const state_t& st = c.st;
  double lower = st.col_lower[s.col], upper = st.col_upper[s.col];
  candidate_t k;
  c.upper_end = true;
  if (upper < inf) {
    if (!range_end(c, row, s.slot, side, s.a > 0, k)) { return false; }
    if (!(k.bound + candidate_slack(k) <= upper + implied_band(c, upper))) { return false; }
  }
  c.upper_end = false;
  if (lower > -inf) {
    if (!range_end(c, row, s.slot, side, s.a < 0, k)) { return false; }
    if (!(k.bound - candidate_slack(k) >= lower - implied_band(c, lower))) { return false; }
  }
  return true;
}

// The largest absolute coefficient among the other free entries of the row in continuous
// columns (growth_ratio_t::continuous_columns); 0 without one.
double largest_continuous(context_t& c, int32_t row, int64_t slot)
{
  const state_t& st       = c.st;
  const line_store_t& csr = st.csr;
  double largest          = 0.;
  c.budget.spend(csr.used[row]);
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (e == slot || !csr.live(e)) { continue; }
    int32_t col = csr.index[e];
    if (!st.is_live(col) || st.is_integer(col)) { continue; }
    largest = std::fmax(largest, std::fabs(csr.value[e]));
  }
  return largest;
}

// The free singleton column leaves with its row: the row is saved without it and deleted, its
// cost moves onto the other columns of the row and into the offset. False, with nothing
// changed, when a cost would not be finite.
bool substitute(context_t& c,
                int32_t row,
                const single_t& s,
                double free_side,
                bool dual,
                double growth,
                double largest_other)
{
  state_t& st       = c.st;
  row_pool_t& pool  = c.ps.saved_entries;
  int32_t col       = s.col;
  double q          = st.cost[col] / s.a;
  size_t pool_begin = pool.col.size();
  c.budget.spend(st.csr.used[row]);
  saved_row_t saved = save_row(st, row, pool, [col](int32_t other) { return other != col; });
  saved.lower       = free_side;
  saved.upper       = free_side;
  if (q != 0.) {
    for (int64_t e = saved.begin; e < saved.end; ++e) {
      if (!std::isfinite(st.cost[pool.col[e]] - q * pool.value[e])) {
        pool.col.resize(pool_begin);
        pool.value.resize(pool_begin);
        return false;
      }
    }
  }
  double lower = st.col_lower[col], upper = st.col_upper[col];
  c.budget.spend(st.csr.used[row] + (saved.end - saved.begin));
  st.delete_row(row);
  if (q != 0.) {
    for (int64_t e = saved.begin; e < saved.end; ++e) {
      int32_t other = pool.col[e];
      double old    = st.cost[other];
      double moved  = -q * pool.value[e];
      double sum    = old + moved;
      double size   = std::fabs(old) + std::fabs(moved);
      if (std::fabs(sum) <= c.tol.cancellation * size) {
        c.count(column_count::costs_cancelled, sum != 0.);
        sum = 0.;
      } else {
        c.count(column_count::costs_near_cancelled, std::fabs(sum) <= near_cancellation * size);
      }
      if (sum == old) { continue; }
      st.set_cost(other, sum);
      c.result.columns.cost_fill += old == 0.;
    }
    st.add_objective_offset(q, free_side);
    st.set_cost(col, 0.);
  }
  st.remove_col(col);
  c.ps.push_defined_col(col, s.a, lower, upper, saved, dual);
  double size = std::fabs(s.a);
  if (growth > 0.) {
    for (int64_t e = saved.begin; e < saved.end; ++e) {
      double reached = growth * (std::fabs(pool.value[e]) / size);
      if (reached > c.w.growth[pool.col[e]]) { c.w.growth[pool.col[e]] = reached; }
    }
  }
  double factor = growth * (largest_other / size);
  if (factor > c.result.columns.max_growth) { c.result.columns.max_growth = factor; }
  c.result.columns.saved_entries += saved.end - saved.begin;
  c.finding(dual ? sub_rule::free_singleton_dual : sub_rule::free_singleton, 1, dual);
  return true;
}

// The free singleton column, with the equation or the tight side of the inequality as pivot.
// True when the column and the row left the model.
bool pivot(context_t& c, int32_t row, const row_scan_t& scanned, const single_t& s, bool equation)
{
  state_t& st = c.st;
  double cost = st.cost[s.col];
  double side = st.row_upper[row];
  if (!equation) {
    c.cause = column_count::refused_zero_cost;
    if (cost == 0.) { return false; }
    c.cause = column_count::refused_integer;
    if (!c.opt.dual_integer_pivots && st.is_integer(s.col)) { return false; }
    // The column moves to lower cost until the row stops it: at its upper side when the
    // activity rises with that move, else at its lower side.
    bool rises = (cost < 0) == (s.a > 0);
    side       = rises ? st.row_upper[row] : st.row_lower[row];
  }
  c.cause = std::isinf(side) ? column_count::refused_side_infinite : column_count::refused_side;
  if (!side_readable(c.tol, side)) { return false; }
  double free_side = dd_residual(side, scanned.fixed_terms);
  c.cause          = column_count::refused_free_side;
  if (!free_side_readable(c.tol, free_side)) { return false; }
  double q = cost / s.a;
  c.cause  = column_count::refused_cost;
  // A quotient that underflows to zero would leave the cost where it is.
  if (!std::isfinite(q) || (q == 0.) != (cost == 0.) || !std::isfinite(q * free_side)) {
    return false;
  }
  if (!implied_free(c, row, s, side)) {
    c.result.columns.free_singleton_rejected_explicit += c.w.explicit_bounds;
    if (!equation) {
      // The column moves down when its cost is positive: the lower bound is the one ahead.
      bool ahead = c.upper_end ? cost < 0 || !lower_end_implied(c, row, s, side) : cost > 0;
      c.count(ahead ? column_count::refused_ahead : column_count::refused_behind);
    }
    return false;
  }
  if (st.is_integer(s.col)) {
    c.budget.spend(st.csr.used[row]);
    integral_form_t form = integral_for(st, c.tol, row, s.slot, free_side);
    c.cause = form.integral ? column_count::refused_drift : column_count::refused_integral;
    if (!form.integral ||
        !drift_within(c.tol, c.tol.integral_drift_share, s.a, form, drift_units_t::row)) {
      return false;
    }
  }
  // A column with a bound can pass it by the deviation of the reduced point times the ratio.
  double largest_other = s.slot == scanned.largest_slot ? scanned.second : scanned.largest;
  if (c.opt.growth_ratio == growth_ratio_t::continuous_columns) {
    largest_other = largest_continuous(c, row, s.slot);
  } else if (c.opt.growth_ratio == growth_ratio_t::integer_pivot_exempt && st.is_integer(s.col)) {
    largest_other = 0.;
  }
  double ratio = largest_other / std::fabs(s.a);
  bool bounded = st.col_lower[s.col] > -inf || st.col_upper[s.col] < inf;
  c.cause      = column_count::refused_growth_row;
  if (bounded && ratio > c.opt.pivot_growth_limit) {
    c.result.columns.growth_refused_row += !(c.w.counted[s.col] & 1);
    c.w.counted[s.col] |= 1;
    return false;
  }
  double growth = std::fmax(c.w.growth[s.col], bounded ? 1. : 0.);
  c.cause       = column_count::refused_growth_history;
  if (growth * ratio > c.opt.pivot_growth_limit) {
    c.result.columns.growth_refused += !(c.w.counted[s.col] & 2);
    c.w.counted[s.col] |= 2;
    c.w.refused.push_back(s.col);
    return false;
  }
  c.cause = column_count::refused_new_cost;
  return substitute(c, row, s, free_side, !equation, growth, largest_other);
}

// The value at which the undo of an interval record starts a column with these bounds.
double start_value(interval_start_t start, double lower, double upper)
{
  switch (start) {
    case interval_start_t::lower: return lower > -inf ? lower : upper < inf ? upper : 0.;
    case interval_start_t::upper: return upper < inf ? upper : lower > -inf ? lower : 0.;
    case interval_start_t::nearest_zero: return lower > 0. ? lower : upper < 0. ? upper : 0.;
    case interval_start_t::middle:
      return lower == -inf  ? (upper < inf ? upper : 0.)
             : upper == inf ? lower
                            : 0.5 * lower + 0.5 * upper;
  }
  gprs_assert(false && "start_value: not a start");
  return 0.;
}

// Of the zero-cost integer singleton columns of the row, those for which the row is integral by
// its ratios: every other free column is integer and has a coefficient that is a multiple of
// theirs. Only a coefficient of the smallest absolute value of the row can divide every other
// one, so one more scan, and only of a row that has such a column, decides for all of them. For
// a counter; no work is spent.
int64_t integral_singletons(const context_t& c, int32_t row, const row_scan_t& scanned)
{
  const state_t& st = c.st;
  int64_t found     = 0;
  for (const single_t& s : c.w.singles) {
    found += st.cost[s.col] == 0. && st.is_integer(s.col) && std::fabs(s.a) == scanned.smallest;
  }
  if (found == 0) { return 0; }
  const line_store_t& csr = st.csr;
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (!csr.live(e) || !st.is_live(csr.index[e])) { continue; }
    double ratio = std::fabs(csr.value[e]) / scanned.smallest;
    if (!st.is_integer(csr.index[e]) || ratio != std::round(ratio)) { return 0; }
  }
  return found;
}

// The zero-cost continuous singleton columns of a row with two finite sides leave it together;
// the row keeps its other entries with the sides that their values must meet for values of the
// removed columns to exist. A side that is infinite (opt.zero_cost_one_sided) stays so. True
// when the columns left.
bool zero_cost(context_t& c, int32_t row, const row_scan_t& scanned)
{
  state_t& st  = c.st;
  double lower = st.row_lower[row], upper = st.row_upper[row];
  // The largest and the smallest sum of the terms of the taken columns.
  dd_t most, least;
  bool most_infinite = lower == -inf, least_infinite = upper == inf;
  std::vector<record_col_t>& cols = c.w.record_cols;
  std::vector<double>& values     = c.w.coefficients;
  cols.clear();
  values.clear();
  c.w.slots.clear();
  int64_t integers = 0;
  for (const single_t& s : c.w.singles) {
    if (st.cost[s.col] != 0.) {
      c.count(column_count::zero_cost_small_cost,
              std::fabs(st.cost[s.col]) <= small_cost && !st.is_integer(s.col));
      continue;
    }
    if (st.is_integer(s.col)) {
      ++integers;
      continue;
    }
    double low = st.col_lower[s.col], high = st.col_upper[s.col];
    // A bound whose term is beyond the size that a side holds within the tolerance would put
    // that term into a new side.
    if ((low > -inf && !free_side_readable(c.tol, s.a * low)) ||
        (high < inf && !free_side_readable(c.tol, s.a * high))) {
      c.count(column_count::zero_cost_term);
      continue;
    }
    double at_most = s.a > 0 ? high : low, at_least = s.a > 0 ? low : high;
    if (std::isinf(at_most)) {
      most_infinite = true;
    } else {
      dd_add_product(most, s.a, at_most);
    }
    if (std::isinf(at_least)) {
      least_infinite = true;
    } else {
      dd_add_product(least, s.a, at_least);
    }
    cols.push_back(record_col_t{s.col, low, high, start_value(c.opt.interval_start, low, high)});
    values.push_back(s.a);
    c.w.slots.push_back(s.slot);
  }
  if (integers > 0) {
    c.count(column_count::zero_cost_integer, integers);
    c.count(column_count::zero_cost_integer_integral, integral_singletons(c, row, scanned));
  }
  int32_t taken = cols.size();
  if (taken == 0) { return false; }
  if (scanned.free_entries == taken && !c.opt.zero_cost_whole_row) {
    c.count(column_count::zero_cost_whole_row);
    c.count(column_count::zero_cost_whole_row_columns, taken);
    return false;
  }
  double new_lower  = most_infinite ? -inf : dd_residual(lower, most);
  double new_upper  = least_infinite ? inf : dd_residual(upper, least);
  double free_lower = lower == -inf ? -inf : dd_residual(lower, scanned.fixed_terms);
  double free_upper = upper == inf ? inf : dd_residual(upper, scanned.fixed_terms);
  if ((new_lower > -inf && !free_side_readable(c.tol, new_lower)) ||
      (new_upper < inf && !free_side_readable(c.tol, new_upper)) ||
      (free_lower > -inf && !free_side_readable(c.tol, free_lower)) ||
      (free_upper < inf && !free_side_readable(c.tol, free_upper))) {
    c.count(column_count::zero_cost_side);
    return false;
  }
  // Two differences that are equal to the last bit can end the other way round.
  if (new_lower > new_upper) {
    c.count(column_count::zero_cost_crossing);
    return false;
  }
  bool row_stays  = new_lower > -inf || new_upper < inf;
  int64_t entries = scanned.free_entries - taken;
  if (row_stays && c.w.kept_saved + entries > c.opt.saved_row_factor * c.ps.nnz) {
    ++c.result.columns.saved_row_refused;
    c.result.limited = true;
    return false;
  }

  for (const record_col_t& col : cols) {
    c.w.taken[col.col] = 1;
  }
  const std::vector<uint8_t>& is_taken = c.w.taken;
  c.budget.spend(st.csr.used[row]);
  saved_row_t saved =
    save_row(st, row, c.ps.saved_entries, [&is_taken](int32_t other) { return !is_taken[other]; });
  saved.lower = free_lower;
  saved.upper = free_upper;
  for (const record_col_t& col : cols) {
    c.w.taken[col.col] = 0;
  }
  gprs_assert(saved.end - saved.begin == entries);

  for (int64_t slot : c.w.slots) {
    st.delete_entry(slot);
  }
  if (!row_stays) {
    c.budget.spend(st.csr.used[row]);
    st.delete_row(row);
  } else {
    // The side that moves away from the other one first, so that they never cross.
    bool upper_first = new_lower > upper;
    if (upper_first && new_upper != upper) { st.set_row_upper(row, new_upper); }
    if (new_lower != lower) { st.set_row_lower(row, new_lower); }
    if (!upper_first && new_upper != upper) { st.set_row_upper(row, new_upper); }
    c.w.kept_saved += entries;
  }
  for (const record_col_t& col : cols) {
    st.remove_col(col.col);
  }
  c.ps.push_interval_cols(cols, values, saved);
  c.result.columns.saved_entries += entries;
  c.count(column_count::zero_cost_records);
  c.finding(sub_rule::zero_cost_singleton, taken, false);
  return true;
}

// The singleton columns of the visit by decreasing absolute coefficient, as indices of singles
// (singleton_order_t::largest_pivot). The index breaks a tie: singles is in increasing index
// and the sort is stable.
void order_pivots(context_t& c)
{
  const std::vector<single_t>& singles = c.w.singles;
  std::vector<int32_t>& order          = c.w.pivot_order;
  order.resize(singles.size());
  for (size_t k = 0; k < order.size(); ++k) {
    order[k] = k;
  }
  std::stable_sort(order.begin(), order.end(), [&singles](int32_t a, int32_t b) {
    return std::fabs(singles[a].a) > std::fabs(singles[b].a);
  });
}

// The row with its singleton columns: the free singleton column with the smallest index that
// passes, else the zero-cost singleton columns together (opt.singleton_order has the other
// orders). At most once per pass over the lists.
void visit(context_t& c, int32_t row)
{
  state_t& st = c.st;
  if (c.w.stamp[row] == c.w.pass) { return; }
  c.w.stamp[row] = c.w.pass;
  if (st.row_singletons[row] == 0) {
    c.count(column_count::rows_without_singleton);
    return;
  }
  c.budget.spend(st.csr.used[row]);
  c.count(column_count::rows_scanned);
  c.count(column_count::scanned_slots, st.csr.used[row]);
  if (c.w.scanned_at[row] == c.w.epoch) {
    c.count(column_count::rows_rescanned);
    c.count(column_count::rescanned_slots, st.csr.used[row]);
  }
  c.w.scanned_at[row] = c.w.epoch;
  row_scan_t scanned  = scan(c, row);
  gprs_assert((int32_t)c.w.singles.size() == st.row_singletons[row]);
  double lower = st.row_lower[row], upper = st.row_upper[row];
  bool equation = lower == upper;
  int32_t sub   = equation ? sub_rule::free_singleton : sub_rule::free_singleton_dual;
  c.w.refused.clear();
  c.count(column_count::single_entry_rows, scanned.free_entries < 2);
  bool projects = sub_rule_active(c.opt, sub_rule::zero_cost_singleton);
  // The sides that the zero-cost singleton columns need of their row.
  bool sided =
    c.opt.zero_cost_one_sided ? lower > -inf || upper < inf : lower > -inf && upper < inf;
  if (projects && !sided) {
    for (const single_t& s : c.w.singles) {
      c.count(column_count::zero_cost_one_sided, st.cost[s.col] == 0. && !st.is_integer(s.col));
    }
  }
  if (c.opt.singleton_order == singleton_order_t::zero_cost_first && projects && sided &&
      zero_cost(c, row, scanned)) {
    return;
  }
  int64_t growth_refusals = c.result.columns.counts[column_count::refused_growth_row];
  if (scanned.free_entries >= 2 && sub_rule_active(c.opt, sub)) {
    if (refresh_activity(st, row)) { c.budget.spend(st.csr.used[row]); }
    bool largest_first = c.opt.singleton_order == singleton_order_t::largest_pivot;
    if (largest_first) { order_pivots(c); }
    for (size_t n = 0; n < c.w.singles.size(); ++n) {
      const single_t& s = c.w.singles[largest_first ? c.w.pivot_order[n] : n];
      if (pivot(c, row, scanned, s, equation)) {
        if (st.is_integer(s.col)) {
          c.count(equation ? column_count::integer_pivots : column_count::integer_pivots_dual);
        }
        c.count(column_count::pivots_among_several, c.w.singles.size() > 1);
        return;
      }
      c.count(c.cause);
    }
  }
  if (c.opt.singleton_order != singleton_order_t::zero_cost_first && projects && sided &&
      zero_cost(c, row, scanned)) {
    c.count(column_count::growth_then_zero_cost,
            c.result.columns.counts[column_count::refused_growth_row] > growth_refusals);
  }
  // A column that only its growth factor kept in the row is a reduction left undone, which a
  // presolve of the reduced model would make: it starts without the factor.
  for (int32_t col : c.w.refused) {
    if (st.is_live(col)) { c.result.limited = true; }
  }
}

// The row of the one live entry of a singleton column.
int32_t row_of_singleton(context_t& c, int32_t col)
{
  const line_store_t& csc = c.st.csc;
  int64_t p               = csc.begin(col);
  while (!csc.live(p)) {
    ++p;
    gprs_assert(p < csc.end(col));
  }
  c.budget.spend(p - csc.begin(col) + 1);
  return csc.index[p];
}

// The live columns of a row that a forcing column was left for go back on the column list: the
// row has changed, and the column that was left has had no event of its own.
void list_columns(context_t& c, int32_t row)
{
  state_t& st             = c.st;
  const line_store_t& csr = st.csr;
  c.w.unreadable[row]     = 0;
  c.budget.spend(csr.used[row]);
  for (int64_t e = csr.begin(row); e < csr.end(row); ++e) {
    if (csr.live(e) && st.is_live(csr.index[e])) {
      st.col_frontier.push(list_bit(col_list::singleton_cols), csr.index[e]);
    }
  }
}

// The forcing column: no cost, no bound and no lock in one direction, and locks in the other.
// Every value of its other columns leaves it a value, so its rows restrict nothing: they are
// saved without it and deleted. False when the rows do not fit the rest of the budget: the
// column then waits for the next call. The first reduction of a call is never left for that.
bool forcing_column(context_t& c, int32_t col)
{
  state_t& st = c.st;
  if (st.cost[col] != 0. && std::fabs(st.cost[col]) <= small_cost) {
    c.count(column_count::forcing_small_cost,
            (st.down_locks[col] == 0 && st.col_lower[col] == -inf && st.up_locks[col] >= 1) ||
              (st.up_locks[col] == 0 && st.col_upper[col] == inf && st.down_locks[col] >= 1));
  }
  if (st.cost[col] != 0.) { return true; }
  double lower = st.col_lower[col], upper = st.col_upper[col];
  bool down = st.down_locks[col] == 0 && lower == -inf && st.up_locks[col] >= 1;
  bool up   = st.up_locks[col] == 0 && upper == inf && st.down_locks[col] >= 1;
  if (!down && !up) { return true; }
  c.count(column_count::forcing_tested);
  std::vector<int32_t>& rows  = c.w.col_rows;
  std::vector<double>& values = c.w.coefficients;
  rows.clear();
  values.clear();
  int64_t slots = st.csc.used[col];
  for (int64_t p = st.csc.begin(col); p < st.csc.end(col); ++p) {
    if (!st.csc.live(p)) { continue; }
    rows.push_back(st.csc.index[p]);
    values.push_back(st.csc.value[p]);
    slots += 2 * (int64_t)st.csr.used[st.csc.index[p]];
  }
  if (c.reductions > 0 && c.budget.used + slots > c.budget.limit) {
    c.count(column_count::forcing_budget);
    return false;
  }
  c.budget.spend(slots);

  row_pool_t& pool                = c.ps.saved_entries;
  size_t pool_begin               = pool.col.size();
  std::vector<saved_row_t>& saved = c.w.saved;
  saved.clear();
  size_t kept = 0;
  bool usable = true;
  for (size_t k = 0; k < rows.size() && usable; ++k) {
    int32_t row = rows[k];
    // The side that stops the column: every row has at most that one, by the lock counts.
    bool upper_side = down == (values[k] > 0);
    double side     = upper_side ? st.row_upper[row] : st.row_lower[row];
    gprs_assert(std::isinf(upper_side ? st.row_lower[row] : st.row_upper[row]));
    if (std::isinf(side)) { continue; }
    saved_row_t copy = save_row(st, row, pool, [col](int32_t other) { return other != col; });
    usable =
      side_readable(c.tol, side) && free_side_readable(c.tol, upper_side ? copy.upper : copy.lower);
    if (!usable) { c.w.unreadable[row] = 1; }
    saved.push_back(copy);
    values[kept++] = values[k];
  }
  if (!usable) {
    pool.col.resize(pool_begin);
    pool.value.resize(pool_begin);
    c.count(column_count::forcing_side);
    return true;
  }
  values.resize(kept);
  for (int32_t row : rows) {
    st.delete_row(row);
  }
  st.remove_col(col);
  record_col_t record{col, lower, upper, lower > -inf ? lower : upper < inf ? upper : 0.};
  c.ps.push_forcing_col(record, values, saved);
  c.result.columns.saved_entries += pool.col.size() - pool_begin;
  c.finding(sub_rule::forcing_column, 1, false);
  return true;
}

}  // namespace

singleton_cols_t::singleton_cols_t() : work(new work_t) {}

singleton_cols_t::~singleton_cols_t() = default;

void singleton_cols_t::bounds_became_explicit() { work->explicit_bounds = true; }

rule_result_t singleton_cols_t::run(state_t& st,
                                    work_budget_t& budget,
                                    postsolve_t& ps,
                                    const options_t& opt)
{
  work_t& w = *work;
  if (w.growth.size() != (size_t)st.ncols) {
    w.growth.assign(st.ncols, 0.);
    w.taken.assign(st.ncols, 0);
    w.counted.assign(st.ncols, 0);
  }
  if (w.stamp.size() != (size_t)st.nrows) {
    w.stamp.assign(st.nrows, 0);
    w.unreadable.assign(st.nrows, 0);
    w.scanned_at.assign(st.nrows, 0);
  }
  ++w.epoch;
  gprs_assert(choices_in_range(opt));
  // The sides that the family reads are those of opt.side_guard_scale times the size that the
  // tolerances allow.
  tolerances_t tol = opt.tol;
  tol.side_error   = opt.tol.side_error / opt.side_guard_scale;
  rule_result_t result;
  context_t c{st, budget, ps, opt, tol, w, result};
  bool visits = sub_rule_active(opt, sub_rule::free_singleton) ||
                sub_rule_active(opt, sub_rule::free_singleton_dual) ||
                sub_rule_active(opt, sub_rule::zero_cost_singleton);
  bool forcing = sub_rule_active(opt, sub_rule::forcing_column);
  if (!visits && !forcing) {
    st.col_frontier.take(col_list::singleton_cols, w.cols);
    st.row_frontier.take(row_list::singleton_cols, w.rows);
    return result;
  }
  // A reduction lists the columns and rows it touches: the lists are served until they stay
  // empty, so that a column which a deleted row leaves as a singleton is visited in this call.
  while (!result.stopped) {
    st.col_frontier.take(col_list::singleton_cols, w.cols);
    st.row_frontier.take(row_list::singleton_cols, w.rows);
    if (w.cols.empty() && w.rows.empty()) { break; }
    ++w.pass;
    c.count(column_count::passes);
    size_t k = 0, r = 0;
    for (; k < w.cols.size() && !budget.stop(); ++k) {
      int32_t col = w.cols[k];
      if (!st.is_live(col)) { continue; }
      budget.spend(line_visit_work);
      if (visits && st.csc.size[col] == 1) { visit(c, row_of_singleton(c, col)); }
      if (forcing && st.is_live(col) && !forcing_column(c, col)) { break; }
    }
    if (k == w.cols.size()) {
      for (; r < w.rows.size() && !budget.stop(); ++r) {
        int32_t row = w.rows[r];
        if (!st.row_alive[row]) { continue; }
        budget.spend(line_visit_work);
        if (w.unreadable[row]) { list_columns(c, row); }
        if (visits) { visit(c, row); }
      }
    }
    result.stopped = k < w.cols.size() || r < w.rows.size();
    for (; k < w.cols.size(); ++k) {
      if (st.is_live(w.cols[k])) {
        st.col_frontier.push(list_bit(col_list::singleton_cols), w.cols[k]);
      }
    }
    for (; r < w.rows.size(); ++r) {
      if (st.row_alive[w.rows[r]]) {
        st.row_frontier.push(list_bit(row_list::singleton_cols), w.rows[r]);
      }
    }
    // What the reductions of this pass listed stays on the lists for the call of the next
    // round. A pass without a reduction can list columns as well (list_columns): the next pass
    // serves them, so that no call ends with a list and without a finding.
    if (opt.singleton_order == singleton_order_t::single_pass && c.reductions > 0) { break; }
  }
  c.count(column_count::calls_stopped, result.stopped);
  c.count(column_count::calls_limited, result.limited);
  // Two sums of the other counters: a visit scans its row or finds no singleton column in it,
  // and a column that is tried leaves or is counted by the test that refuses it.
  std::array<int64_t, column_count::count>& counts = result.columns.counts;
  counts[column_count::rows_visited] =
    counts[column_count::rows_scanned] + counts[column_count::rows_without_singleton];
  int64_t tried = result.sub_findings[sub_rule::free_singleton] +
                  result.sub_findings[sub_rule::free_singleton_dual];
  for (int32_t k = column_count::refused_zero_cost; k <= column_count::refused_new_cost; ++k) {
    if (k != column_count::refused_ahead && k != column_count::refused_behind) {
      tried += counts[k];
    }
  }
  counts[column_count::pivots_tried] = tried;
  return result;
}

}  // namespace gprs
