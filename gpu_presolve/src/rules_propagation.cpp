/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <cmath>
#include <vector>

namespace gprs {

struct bound_propagation_t::work_t {
  std::vector<int32_t> rows;
};

bound_propagation_t::bound_propagation_t() : work(new work_t) {}

bound_propagation_t::~bound_propagation_t() = default;

namespace {

// A candidate is rounded twice, in the residual and in the division, by up to this many rounding
// units of its size in all. The integer rounding allows for the feasibility tolerance only, which
// is less than that from about 1e10 on.
constexpr double candidate_rounding_units = 4.;

struct context_t {
  state_t& st;
  work_budget_t& budget;
  postsolve_t& ps;
  const tolerances_t& tol;
  rule_result_t& result;

  void finding()
  {
    ++result.findings;
    budget.found();
  }
};

// The bound that one side of the row implies for the column of entry e: the upper side with
// the minimum activity (upper_side) or the lower side with the maximum activity. That activity
// has no infinite term, or its only one at e. It is the same for every entry of the row: a
// candidate changes the bound of its column that this activity does not hold, and a column
// fixed by its candidate is fixed at the bound that the activity holds.
bool candidate(const context_t& c, int32_t row, int64_t e, bool upper_side, double& bound)
{
  const state_t& st     = c.st;
  const activity_t& act = upper_side ? st.min_act : st.max_act;
  int32_t infinite      = act.inf_count[row];
  gprs_assert(infinite == 0 || (infinite == 1 && act.inf_slot[row] == e));
  int32_t col = st.csr.index[e];
  double a    = st.csr.value[e];
  dd_t rest{act.hi[row], act.lo[row]};
  uint32_t updates = act.updates[row];
  if (infinite == 0) {
    double own = upper_side ? min_activity_bound(a, st.col_lower[col], st.col_upper[col])
                            : max_activity_bound(a, st.col_lower[col], st.col_upper[col]);
    gprs_assert(!is_infinite_term(a, own, c.tol));
    dd_sub_product(rest, a, own);
    ++updates;
  }
  double error = activity_error(updates, act.max_term[row]);
  if (error > c.tol.activity_error_share * c.tol.feasibility * std::fabs(a)) { return false; }
  bound = dd_residual(upper_side ? st.row_upper[row] : st.row_lower[row], rest) / a;
  // An overflow of the division says nothing about the column.
  return std::isfinite(bound);
}

rule_outcome_t propagate_entry(context_t& c, int32_t row, int64_t e, bool upper_side)
{
  state_t& st = c.st;
  double bound;
  if (!candidate(c, row, e, upper_side, bound)) { return rule_outcome_t::ok; }
  int32_t col   = st.csr.index[e];
  bool is_upper = (st.csr.value[e] > 0) == upper_side;
  if (st.is_integer(col)) {
    // Moved outward by its own rounding error first, or an integer point that meets the row
    // exactly would be cut off.
    double rounding       = candidate_rounding_units * rounding_unit * std::fabs(bound);
    bound_change_t change = is_upper ? tighten_upper(st, c.ps, col, bound + rounding)
                                     : tighten_lower(st, c.ps, col, bound - rounding);
    if (change.changed) {
      c.budget.spend(st.csc.used[col]);
      c.finding();
    }
    return change.outcome;
  }
  double step = c.tol.implied_bound_step * c.tol.feasibility;
  if (is_upper) {
    double record = st.implied_upper_valid(col) ? st.implied_upper[col] : inf;
    if (bound < record - step) {
      st.record_implied_upper(col, bound, row);
      c.finding();
    }
  } else {
    double record = st.implied_lower_valid(col) ? st.implied_lower[col] : -inf;
    if (bound > record + step) {
      st.record_implied_lower(col, bound, row);
      c.finding();
    }
  }
  return rule_outcome_t::ok;
}

rule_outcome_t propagate_side(context_t& c, int32_t row, bool upper_side)
{
  state_t& st = c.st;
  double side = upper_side ? st.row_upper[row] : st.row_lower[row];
  if (std::isinf(side) || std::fabs(side) * c.tol.side_error > c.tol.feasibility) {
    return rule_outcome_t::ok;
  }
  const activity_t& act = upper_side ? st.min_act : st.max_act;
  if (act.inf_count[row] > 1) { return rule_outcome_t::ok; }
  if (act.inf_count[row] == 1) {
    int64_t e = act.inf_slot[row];
    gprs_assert(st.csr.begin(row) <= e && e < st.csr.end(row) && st.csr.live(e));
    c.budget.spend(1);
    if (!st.is_live(st.csr.index[e])) { return rule_outcome_t::ok; }
    return propagate_entry(c, row, e, upper_side);
  }
  c.budget.spend(st.csr.used[row]);
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (!st.csr.live(e) || !st.is_live(st.csr.index[e])) { continue; }
    rule_outcome_t outcome = propagate_entry(c, row, e, upper_side);
    if (outcome != rule_outcome_t::ok) { return outcome; }
  }
  return rule_outcome_t::ok;
}

}  // namespace

rule_result_t bound_propagation_t::run(state_t& st,
                                       work_budget_t& budget,
                                       postsolve_t& ps,
                                       const options_t& opt)
{
  rule_result_t result;
  context_t c{st, budget, ps, opt.tol, result};
  std::vector<int32_t>& rows = work->rows;
  // A tightened bound puts the rows of its column on the list again: the call runs until the
  // list stays empty or the budget stops it.
  while (!st.row_frontier.empty(row_list::propagation)) {
    st.row_frontier.take(row_list::propagation, rows);
    for (size_t k = 0; k < rows.size(); ++k) {
      if (budget.stop()) {
        for (; k < rows.size(); ++k) {
          if (st.row_alive[rows[k]]) {
            st.row_frontier.push(list_bit(row_list::propagation), rows[k]);
          }
        }
        result.stopped = true;
        return result;
      }
      int32_t row = rows[k];
      if (!st.row_alive[row]) { continue; }
      budget.spend(line_visit_work);
      for (bool upper_side : {true, false}) {
        result.outcome = propagate_side(c, row, upper_side);
        if (result.outcome != rule_outcome_t::ok) { return result; }
      }
    }
  }
  return result;
}

rule_result_t bound_propagation_t::make_explicit(state_t& st, postsolve_t& ps, const options_t&)
{
  rule_result_t result;
  for (int32_t col = 0; col < st.ncols; ++col) {
    if (!st.is_live(col) || st.is_integer(col)) { continue; }
    if (st.implied_lower_valid(col) && st.implied_lower[col] > st.col_lower[col]) {
      bound_change_t change = tighten_lower(st, ps, col, st.implied_lower[col]);
      result.outcome        = change.outcome;
      if (result.outcome != rule_outcome_t::ok) { return result; }
      result.findings += change.changed;
    }
    if (st.is_live(col) && st.implied_upper_valid(col) &&
        st.implied_upper[col] < st.col_upper[col]) {
      bound_change_t change = tighten_upper(st, ps, col, st.implied_upper[col]);
      result.outcome        = change.outcome;
      if (result.outcome != rule_outcome_t::ok) { return result; }
      result.findings += change.changed;
    }
  }
  return result;
}

}  // namespace gprs
