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
  // Marks of the continuous columns, cleared by make_explicit; empty until the first mark.
  // lower_pending / upper_pending: a row gave a candidate that is better than the explicit bound
  // by more than the step. lower_stale / upper_stale: a visit found the record of the column
  // void, so a candidate that was not recorded because the lost record was as good can be better
  // than the record of now.
  std::vector<uint8_t> pending;
  bool explicit_bounds = false;
  // After make_explicit: how often run() tightened the lower (2 * col) and the upper
  // (2 * col + 1) bound of a continuous column.
  std::vector<uint8_t> steps;
};

bound_propagation_t::bound_propagation_t() : work(new work_t) {}

bound_propagation_t::~bound_propagation_t() = default;

namespace {

constexpr uint8_t lower_pending = 1;
constexpr uint8_t upper_pending = 2;
constexpr uint8_t lower_stale   = 4;
constexpr uint8_t upper_stale   = 8;

// Tightenings of one explicit bound of a continuous column by run(). Bounds that depend on each
// other through coefficients below 1 (a decay, an average of neighbours) improve in ever
// smaller steps, thousands of them before a step falls below the tolerance; the local testbed
// needs 3. A bound that has had its steps stays where it is, and the call reports limited.
constexpr uint8_t max_explicit_steps = 16;

// A candidate is rounded twice, in the residual and in the division, by up to this many rounding
// units of its size in all.
constexpr double candidate_rounding_units = 4.;

inline double bound_rounding(double bound)
{
  return candidate_rounding_units * rounding_unit * std::fabs(bound);
}

// What a bound of a continuous column must improve on to count: implied_bound_step feasibility
// tolerances, or as many times what a bound of its size loses when it is stored, and never less
// than two candidates of one bound can differ by: each is moved outward by its rounding and by
// the error bound of its activity, which the guard keeps below a share of the tolerance and
// which is another one in every presolve.
inline double improvement_step(const tolerances_t& tol, double bound)
{
  double step =
    tol.implied_bound_step * std::fmax(tol.feasibility, tol.side_error * std::fabs(bound));
  return std::fmax(step, 2. * (bound_rounding(bound) + tol.activity_error_share * tol.feasibility));
}

struct context_t {
  state_t& st;
  work_budget_t& budget;
  postsolve_t& ps;
  const tolerances_t& tol;
  bound_propagation_t::work_t& work;
  rule_result_t& result;
  // The error guard dropped a candidate of the side that is being visited.
  bool guarded = false;

  void finding()
  {
    ++result.findings;
    budget.found();
  }

  void record()
  {
    ++result.records;
    budget.found();
  }
};

// The bound that one side of the row implies for the column of entry e: the upper side with
// the minimum activity (upper_side) or the lower side with the maximum activity. That activity
// has no infinite term, or its only one at e. It is the same for every entry of the row: a
// candidate changes the bound of its column that this activity does not hold, and a column
// fixed by its candidate is fixed at the bound that the activity holds.
struct candidate_t {
  double bound;
  // What it was computed from: bound = dd_residual(side, rest) / a, with the error bound of rest.
  double side;
  dd_t rest;
  double a;
  double error;
};

bool candidate(context_t& c, int32_t row, int64_t e, bool upper_side, candidate_t& k)
{
  const state_t& st     = c.st;
  const activity_t& act = upper_side ? st.min_act : st.max_act;
  int32_t infinite      = act.inf_count[row];
  gprs_assert(infinite == 0 || (infinite == 1 && act.inf_slot[row] == e));
  int32_t col      = st.csr.index[e];
  k.a              = st.csr.value[e];
  k.side           = upper_side ? st.row_upper[row] : st.row_lower[row];
  k.rest           = dd_t{act.hi[row], act.lo[row]};
  uint32_t updates = act.updates[row];
  if (infinite == 0) {
    double own = upper_side ? min_activity_bound(k.a, st.col_lower[col], st.col_upper[col])
                            : max_activity_bound(k.a, st.col_lower[col], st.col_upper[col]);
    gprs_assert(!is_infinite_term(k.a, own, c.tol));
    dd_sub_product(k.rest, k.a, own);
    ++updates;
  }
  k.error = activity_error(updates, act.max_term[row]);
  if (k.error > c.tol.activity_error_share * c.tol.feasibility * std::fabs(k.a)) {
    c.guarded = true;
    return false;
  }
  k.bound = dd_residual(k.side, k.rest) / k.a;
  // An overflow of the division says nothing about the column.
  return std::isfinite(k.bound);
}

// What the true quotient can lie beyond the candidate by: nothing when the residual and the
// division are exact for the stored sum (a candidate of 8 is 8, whatever the history of the
// sum), else their rounding and the error bound of the activity. A bound of a continuous column
// is the candidate moved outward by it, so that the bound holds for every point of the row: a
// bound that is a rounding step too tight cuts a point off, and a row with a large coefficient
// of the column carries that on, multiplied (131072 twice turns one step into 4e-6).
double candidate_slack(const candidate_t& k)
{
  // dd_residual(k.side, k.rest), with what its two additions round away.
  double carry, tail_rounding, residual_rounding;
  double difference = two_sum(k.side, -k.rest.hi, carry);
  double tail       = two_sum(carry, -k.rest.lo, tail_rounding);
  double residual   = two_sum(difference, tail, residual_rounding);
  if (tail_rounding == 0. && residual_rounding == 0. && std::fma(k.bound, k.a, -residual) == 0.) {
    return 0.;
  }
  return bound_rounding(k.bound) +
         (k.error + std::fabs(tail_rounding) + std::fabs(residual_rounding)) / std::fabs(k.a);
}

rule_outcome_t propagate_entry(context_t& c, int32_t row, int64_t e, bool upper_side)
{
  state_t& st = c.st;
  candidate_t k;
  if (!candidate(c, row, e, upper_side, k)) { return rule_outcome_t::ok; }
  int32_t col   = st.csr.index[e];
  bool is_upper = (st.csr.value[e] > 0) == upper_side;
  bool pending  = false;
  double bound  = k.bound;
  if (st.is_integer(col)) {
    pending = true;
  } else {
    double slack = candidate_slack(k);
    bound        = is_upper ? bound + slack : bound - slack;
    double step  = improvement_step(c.tol, bound);
    bool lost    = false;
    if (is_upper) {
      bool valid    = st.implied_upper_valid(col);
      double record = valid ? st.implied_upper[col] : inf;
      lost          = !valid && st.implied_upper_row[col] >= 0;
      if (bound < record - step) {
        st.record_implied_upper(col, bound, row);
        c.record();
      }
      pending = bound < st.col_upper[col] - step;
    } else {
      bool valid    = st.implied_lower_valid(col);
      double record = valid ? st.implied_lower[col] : -inf;
      lost          = !valid && st.implied_lower_row[col] >= 0;
      if (bound > record + step) {
        st.record_implied_lower(col, bound, row);
        c.record();
      }
      pending = bound > st.col_lower[col] + step;
    }
    if (!c.work.explicit_bounds) {
      if (pending || lost) {
        if (c.work.pending.empty()) { c.work.pending.assign(st.ncols, 0); }
        uint8_t marks = 0;
        if (pending) { marks |= is_upper ? upper_pending : lower_pending; }
        if (lost) { marks |= is_upper ? upper_stale : lower_stale; }
        c.work.pending[col] |= marks;
      }
      pending = false;
    } else if (pending) {
      int64_t at     = col;
      uint8_t& steps = c.work.steps[2 * at + is_upper];
      if (steps == max_explicit_steps) {
        c.result.limited = true;
        pending          = false;
      } else {
        ++steps;
      }
    }
  }
  if (!pending) { return rule_outcome_t::ok; }
  // The integer rounding allows for the feasibility tolerance only, which is less than the
  // rounding of the candidate from about 1e10 on.
  double rounding       = st.is_integer(col) ? bound_rounding(bound) : 0.;
  bound_change_t change = is_upper ? tighten_upper(st, c.ps, col, bound + rounding)
                                   : tighten_lower(st, c.ps, col, bound - rounding);
  if (change.changed) {
    c.budget.spend(st.csc.used[col]);
    c.finding();
  }
  return change.outcome;
}

rule_outcome_t propagate_side(context_t& c, int32_t row, bool upper_side)
{
  state_t& st = c.st;
  double side = upper_side ? st.row_upper[row] : st.row_lower[row];
  if (std::isinf(side) || std::fabs(side) * c.tol.side_error > c.tol.feasibility) {
    return rule_outcome_t::ok;
  }
  const activity_t& act = upper_side ? st.min_act : st.max_act;
  // A second time when the error guard dropped a candidate and the activity was not fresh: its
  // error bound then counted a term that the sum no longer holds.
  for (int32_t attempt = 0; attempt < 2; ++attempt) {
    c.guarded = false;
    if (act.inf_count[row] > 1) { return rule_outcome_t::ok; }
    if (act.inf_count[row] == 1) {
      int64_t e = act.inf_slot[row];
      gprs_assert(st.csr.begin(row) <= e && e < st.csr.end(row) && st.csr.live(e));
      c.budget.spend(1);
      if (!st.is_live(st.csr.index[e])) { return rule_outcome_t::ok; }
      rule_outcome_t outcome = propagate_entry(c, row, e, upper_side);
      if (outcome != rule_outcome_t::ok) { return outcome; }
    } else {
      c.budget.spend(st.csr.used[row]);
      for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
        if (!st.csr.live(e) || !st.is_live(st.csr.index[e])) { continue; }
        rule_outcome_t outcome = propagate_entry(c, row, e, upper_side);
        if (outcome != rule_outcome_t::ok) { return outcome; }
      }
    }
    if (!c.guarded || activity_fresh(st, act, row)) { break; }
    c.budget.spend(st.csr.used[row]);
    st.recompute_row_activity(row);
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
  context_t c{st, budget, ps, opt.tol, *work, result};
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

rule_result_t bound_propagation_t::make_explicit(state_t& st, postsolve_t& ps, const options_t& opt)
{
  rule_result_t result;
  work->explicit_bounds = true;
  int64_t bounds        = st.ncols;
  work->steps.assign(2 * bounds, 0);
  std::vector<uint8_t>& pending = work->pending;
  if (pending.empty()) { return result; }
  const tolerances_t& tol = opt.tol;
  const line_store_t& csc = st.csc;
  for (int32_t col = 0; col < st.ncols; ++col) {
    uint8_t marks = pending[col];
    pending[col]  = 0;
    if (!(marks & (lower_pending | upper_pending)) || !st.is_live(col)) { continue; }
    // The record is the best candidate that a row gave. When no record of the column was lost,
    // every candidate that was not recorded is within the step of it, so no row has more to
    // give for this column and only the rows that the new bound reaches are visited again
    // (state_t::set_col_upper). Otherwise all rows of the column are.
    bool covered = true;
    if (marks & lower_pending) {
      bool taken = false;
      if (st.implied_lower_valid(col)) {
        double bound = st.implied_lower[col];
        if (bound > st.col_lower[col] + improvement_step(tol, bound)) {
          bound_change_t change = tighten_lower(st, ps, col, bound);
          result.outcome        = change.outcome;
          if (result.outcome != rule_outcome_t::ok) { return result; }
          result.findings += change.changed;
          taken = true;
        }
      }
      covered = covered && taken && !(marks & lower_stale);
    }
    if ((marks & upper_pending) && st.is_live(col)) {
      bool taken = false;
      if (st.implied_upper_valid(col)) {
        double bound = st.implied_upper[col];
        if (bound < st.col_upper[col] - improvement_step(tol, bound)) {
          bound_change_t change = tighten_upper(st, ps, col, bound);
          result.outcome        = change.outcome;
          if (result.outcome != rule_outcome_t::ok) { return result; }
          result.findings += change.changed;
          taken = true;
        }
      }
      covered = covered && taken && !(marks & upper_stale);
    }
    if (covered || !st.is_live(col)) { continue; }
    for (int64_t p = csc.begin(col); p < csc.end(col); ++p) {
      if (csc.live(p)) { st.row_frontier.push(list_bit(row_list::propagation), csc.index[p]); }
    }
  }
  return result;
}

}  // namespace gprs
