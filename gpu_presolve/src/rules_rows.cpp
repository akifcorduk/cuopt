/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "forcing.hpp"

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>
#include <gprs/rules.hpp>

#include <cmath>
#include <vector>

namespace gprs {

struct row_rules_t::work_t {
  std::vector<int32_t> rows;
  // Free columns of a forcing row and the value each is fixed at.
  std::vector<int32_t> cols;
  std::vector<double> values;
};

row_rules_t::row_rules_t() : work(new work_t) {}

row_rules_t::~row_rules_t() = default;

namespace {

// A bound from a singleton row is rounded twice, in the side less the fixed terms and in the
// division, by up to this many rounding units of its size in all. The integer rounding allows
// for the feasibility tolerance only, which is less than that from about 1e10 on.
constexpr double bound_rounding_units = 4.;

struct context_t {
  state_t& st;
  work_budget_t& budget;
  postsolve_t& ps;
  const tolerances_t& tol;
  row_rules_t::work_t& work;
  rule_result_t& result;

  void finding()
  {
    ++result.findings;
    budget.found();
  }
};

// A row without a free entry is the constant sum of its fixed terms.
rule_outcome_t empty_row(context_t& c, int32_t row)
{
  state_t& st = c.st;
  dd_t constant;
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (!st.csr.live(e)) { continue; }
    dd_add_product(constant, st.csr.value[e], st.col_lower[st.csr.index[e]]);
  }
  c.budget.spend(st.csr.used[row]);
  double value = dd_value(constant);
  if (value > st.row_upper[row] + c.tol.feasibility ||
      value < st.row_lower[row] - c.tol.feasibility) {
    return rule_outcome_t::infeasible;
  }
  st.delete_row(row);
  c.finding();
  return rule_outcome_t::ok;
}

rule_outcome_t singleton_row(context_t& c, int32_t row)
{
  state_t& st = c.st;
  dd_t fixed_terms;
  int64_t free_slot = -1;
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (!st.csr.live(e)) { continue; }
    int32_t j = st.csr.index[e];
    if (st.is_live(j)) {
      free_slot = e;
    } else {
      dd_add_product(fixed_terms, st.csr.value[e], st.col_lower[j]);
    }
  }
  c.budget.spend(st.csr.used[row]);
  gprs_assert(free_slot >= 0);
  int32_t col = st.csr.index[free_slot];
  double a    = st.csr.value[free_slot];

  double lower = -inf, upper = inf;
  if (st.row_lower[row] > -inf) {
    (a > 0 ? lower : upper) = dd_residual(st.row_lower[row], fixed_terms) / a;
  }
  if (st.row_upper[row] < inf) {
    (a > 0 ? upper : lower) = dd_residual(st.row_upper[row], fixed_terms) / a;
  }
  // An integer bound is moved outward by its own rounding error, or an integer point that meets
  // the row exactly would be cut off.
  if (st.is_integer(col)) {
    if (std::isfinite(lower)) { lower -= bound_rounding_units * rounding_unit * std::fabs(lower); }
    if (std::isfinite(upper)) { upper += bound_rounding_units * rounding_unit * std::fabs(upper); }
  }
  // The row passed the infeasibility test in its own units. With |a| < 1 the same violation is
  // larger in the units of the column, where tighten_lower / tighten_upper would reject it.
  double magnitude = std::fabs(a);
  if (lower > st.col_upper[col] && magnitude * (lower - st.col_upper[col]) <= c.tol.feasibility) {
    lower = st.col_upper[col];
  }
  if (upper < st.col_lower[col] && magnitude * (st.col_lower[col] - upper) <= c.tol.feasibility) {
    upper = st.col_lower[col];
  }
  // tighten_lower can fix the column at its upper bound, after which the upper bound from the
  // row cannot be applied: a row that leaves no integer between its bounds is caught here.
  if (st.is_integer(col) && round_lower_integer(lower, c.tol) > round_upper_integer(upper, c.tol)) {
    return rule_outcome_t::infeasible;
  }

  st.delete_row(row);
  c.finding();
  bound_change_t change = tighten_lower(st, c.ps, col, lower);
  if (change.outcome != rule_outcome_t::ok) { return change.outcome; }
  if (change.changed) {
    c.budget.spend(st.csc.used[col]);
    c.finding();
  }
  if (!st.is_live(col)) { return rule_outcome_t::ok; }
  change = tighten_upper(st, c.ps, col, upper);
  if (change.outcome != rule_outcome_t::ok) { return change.outcome; }
  if (change.changed) {
    c.budget.spend(st.csc.used[col]);
    c.finding();
  }
  return rule_outcome_t::ok;
}

// The forcing row: the free entries of the row are forcing (src/forcing.hpp) at its minimum
// activity (at_minimum) or at its maximum one, gap being side - activity and error the error
// bound of the activity. Every free column is fixed at the bound that gives the activity and
// the row is deleted.
bool forcing_row(context_t& c, int32_t row, double gap, double error, bool at_minimum)
{
  state_t& st = c.st;
  if (!forcing_possible(c.tol, gap, error)) { return false; }
  forcing_entries_t entries;
  c.work.cols.clear();
  c.work.values.clear();
  for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
    if (!st.csr.live(e)) { continue; }
    int32_t col = st.csr.index[e];
    if (!st.is_live(col)) { continue; }
    double a = st.csr.value[e];
    c.work.cols.push_back(col);
    c.work.values.push_back(at_minimum
                              ? min_activity_bound(a, st.col_lower[col], st.col_upper[col])
                              : max_activity_bound(a, st.col_lower[col], st.col_upper[col]));
    entries.add(a, c.work.values.back(), !st.is_integer(col));
  }
  c.budget.spend(st.csr.used[row]);
  if (!forcing(c.tol, entries, gap, error, at_minimum)) { return false; }

  st.delete_row(row);
  c.finding();
  for (size_t k = 0; k < c.work.cols.size(); ++k) {
    int32_t col = c.work.cols[k];
    c.budget.spend(st.csc.used[col]);
    fix_column(st, c.ps, col, c.work.values[k]);
    c.finding();
  }
  return true;
}

rule_outcome_t visit_row(context_t& c, int32_t row)
{
  state_t& st = c.st;
  c.budget.spend(line_visit_work);
  if (st.row_free_size[row] == 0) { return empty_row(c, row); }
  if (refresh_activity(st, row)) { c.budget.spend(st.csr.used[row]); }

  double lower    = st.row_lower[row];
  double upper    = st.row_upper[row];
  double feas     = c.tol.feasibility;
  bool min_finite = st.min_act.inf_count[row] == 0;
  bool max_finite = st.max_act.inf_count[row] == 0;
  dd_t min_sum{st.min_act.hi[row], st.min_act.lo[row]};
  dd_t max_sum{st.max_act.hi[row], st.max_act.lo[row]};
  double min_error = st.min_activity_error(row);
  double max_error = st.max_activity_error(row);
  // upper - minimum activity and lower - maximum activity, where both are finite.
  double upper_gap = upper < inf && min_finite ? dd_residual(upper, min_sum) : inf;
  double lower_gap = lower > -inf && max_finite ? dd_residual(lower, max_sum) : -inf;

  if (upper_gap < -(feas + min_error) || lower_gap > feas + max_error) {
    return rule_outcome_t::infeasible;
  }

  bool upper_redundant =
    upper == inf || (max_finite && dd_residual(upper, max_sum) >= max_error - feas);
  bool lower_redundant =
    lower == -inf || (min_finite && dd_residual(lower, min_sum) <= feas - min_error);
  if (upper_redundant && lower_redundant) {
    st.delete_row(row);
    c.finding();
    return rule_outcome_t::ok;
  }

  if (st.row_free_size[row] == 1) { return singleton_row(c, row); }

  if (upper_gap < inf && forcing_row(c, row, upper_gap, min_error, true)) {
    return rule_outcome_t::ok;
  }
  if (lower_gap > -inf) { forcing_row(c, row, lower_gap, max_error, false); }
  return rule_outcome_t::ok;
}

}  // namespace

rule_result_t row_rules_t::run(state_t& st,
                               work_budget_t& budget,
                               postsolve_t& ps,
                               const options_t& opt)
{
  rule_result_t result;
  context_t c{st, budget, ps, opt.tol, *work, result};
  std::vector<int32_t>& rows = work->rows;
  // A reduction puts the rows it changes on the list again: the call runs until the list stays
  // empty or the budget stops it.
  while (!st.row_frontier.empty(row_list::row_rules)) {
    st.row_frontier.take(row_list::row_rules, rows);
    for (size_t k = 0; k < rows.size(); ++k) {
      if (budget.stop()) {
        for (; k < rows.size(); ++k) {
          if (st.row_alive[rows[k]]) {
            st.row_frontier.push(list_bit(row_list::row_rules), rows[k]);
          }
        }
        result.stopped = true;
        return result;
      }
      if (!st.row_alive[rows[k]]) { continue; }
      result.outcome = visit_row(c, rows[k]);
      if (result.outcome != rule_outcome_t::ok) { return result; }
    }
  }
  return result;
}

rule_result_t row_rules_t::remove_empty_rows(state_t& st, postsolve_t& ps, const options_t& opt)
{
  rule_result_t result;
  work_budget_t budget;
  context_t c{st, budget, ps, opt.tol, *work, result};
  for (int32_t row = 0; row < st.nrows; ++row) {
    if (!st.row_alive[row] || st.row_free_size[row] != 0) { continue; }
    result.outcome = empty_row(c, row);
    if (result.outcome != rule_outcome_t::ok) { return result; }
  }
  return result;
}

}  // namespace gprs
