/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <gprs/postsolve.hpp>
#include <gprs/state.hpp>

#include <cstdint>
#include <memory>

namespace gprs {

struct options_t;

// Work is counted in the units of the Stage 0 cost model: one unit per matrix entry read or
// written, line_visit_work units per row or column visited. It never depends on time.
constexpr int64_t line_visit_work = 8;

// Budget of one call of a rule family. The call stops when stop() holds, which it tests at
// least once per row or column it visits.
struct work_budget_t {
  int64_t limit           = 0;  // work the call may spend
  int64_t stall_limit     = 0;  // work that may pass without a finding
  int64_t used            = 0;
  int64_t used_at_finding = 0;

  void spend(int64_t units) { used += units; }
  void found() { used_at_finding = used; }
  bool stop() const { return used >= limit || used - used_at_finding >= stall_limit; }
};

enum class rule_outcome_t : uint8_t { ok, infeasible, unbounded_or_infeasible };

struct rule_result_t {
  rule_outcome_t outcome = rule_outcome_t::ok;
  // Changes of the model: rows deleted, columns fixed or substituted, bounds, sides or
  // coefficients changed. Removed rows, columns and nonzeros are read from the counters of the
  // state.
  int64_t findings = 0;
  // Implied bounds recorded (state_t::record_implied_lower / record_implied_upper). A record
  // changes no row, column or explicit bound and is not a finding.
  int64_t records = 0;
  // The call stopped on its budget and returned the unprocessed part of its list to the state.
  bool stopped = false;
  // The call left undone a reduction that no list holds: bound propagation, when a bound of a
  // continuous column has had its steps (see (b)). The state is then no fixed point of the
  // rules, although every list can be empty.
  bool limited = false;
};

// Contract of every family below:
// - run() takes its list from the state (state_t::row_frontier / col_frontier), skips rows and
//   columns that are no longer live, and changes the state only through state_t's operations and
//   the bound functions at the end of this file.
// - The free entries of a row are its live entries in live columns, in no particular order
//   (line_store_t); the terms of its fixed columns are constants of the row that both
//   activities contain. A column of the state is live with lower < upper, or fixed, or removed.
// - Its work is proportional to the entries of the rows and columns it visits; it spends that
//   work on the budget and calls budget.found() at every finding and at every record. It tests
//   budget.stop() between two rows or columns of its list, never inside the reduction of one
//   (doubleton_t: between two edges of the batch it collects), and pushes the unvisited part of
//   its list back when it stops.
// - A test on an activity allows for the error bound of that activity
//   (state_t::min_activity_error / max_activity_error): a reduction is applied only if its
//   condition holds for every value within the bound. The state does not keep the bound below a
//   fixed share of the tolerance (state_t::recompute_row_activity); a family that tests an
//   activity brings the bound down first where it can decide the test (refresh_activity), so
//   that the test does not depend on what the sum held earlier.
// - With outcome ok the state is consistent and every reduction has its postsolve record. With
//   another outcome the state is not used again.
// - The result depends on the state, the options and the budget only: never on the iteration
//   order of a hash container, on an address or on a clock.
// - work_t is the family's data that outlives a call (hashes, marks, the alias forest); it is
//   defined in the family's source file. It holds rows and columns by index and no slot: lines
//   are compacted and the state is rebuilt between calls.
// - A family reads opt.tol and the fields of options_t its description names. Any other
//   constant it needs is a named constant in its own source file.

// (a) Rows of row_list::row_rules, the tests in this order. Empty row (no free entry): deleted,
// or infeasible. Infeasible row: minimum activity above the upper side or maximum activity
// below the lower side by more than the feasibility tolerance. Redundant row: within the
// feasibility tolerance the maximum activity is not above the upper side and the minimum
// activity not below the lower side (an infinite side needs nothing); deleted. Singleton row
// (one free entry): its sides, less the terms of the fixed columns and divided by the
// coefficient, become explicit bounds of the column (tighten_lower / tighten_upper) and the row
// is deleted. Forcing row: an activity meets its side within feasibility / dynamism of the row;
// every free column is fixed at the bound that gives that activity and the row is deleted.
// With a free continuous column the row is forcing only when the activity can move towards the
// side by no more than its rounding (its error bound plus a few rounding units of its terms):
// the points of the row lie up to that distance, divided by the coefficient, off the bound of
// such a column, fixing it moves them onto the bound, and another row that holds the column
// with a larger coefficient would be violated by more. An activity beyond the side, within
// the tolerance, has no such point and is forcing as before.
// Every test reads the activities over the explicit bounds. A recorded implied bound enters no
// test: it becomes an explicit bound first (bound_propagation_t::make_explicit).
struct row_rules_t {
  struct work_t;
  row_rules_t();
  ~row_rules_t();
  rule_result_t run(state_t& st, work_budget_t& budget, postsolve_t& ps, const options_t& opt);
  // The empty-row test of run() on every live row without a free entry, whatever the list holds
  // and without a budget: the row is deleted, or the result is infeasible. For a presolve that a
  // limit stopped before the list was served; a reduced model then still has no row without
  // entries. O(rows + entries of the rows without a free entry).
  rule_result_t remove_empty_rows(state_t& st, postsolve_t& ps, const options_t& opt);
  std::unique_ptr<work_t> work;
};

// (b) Rows of row_list::propagation. A finite upper side with the minimum activity, a finite
// lower side with the maximum activity: no infinite term, every free column gets a candidate;
// exactly one, only the column at inf_slot, if it is live, in O(1); two or more, none. The
// candidate is dd_residual(side, activity without the column's own term) / a. No candidate when
// the error bound of the activity exceeds tol.activity_error_share * tol.feasibility * |a|
// (with the activity summed from scratch first if it was not fresh), or when
// |side| * tol.side_error > tol.feasibility. Integer column: the explicit bound is tightened
// (tighten_lower / tighten_upper, which round).
// Continuous column: a candidate whose residual and division are exact for the stored sum stays
// as it is. Any other is moved outward by what the true quotient can lie beyond it by: the
// rounding of the residual and of the division and the error bound of the activity over |a|.
// The rounding of one bound is then never a restriction that other rows carry on and multiply
// by their coefficients. With step = tol.implied_bound_step *
// max(tol.feasibility, tol.side_error * |candidate|), the second term being what a bound of that
// size loses when it is stored, and at least what two candidates of one bound differ by (twice
// their rounding plus twice tol.activity_error_share * tol.feasibility):
// - The candidate is recorded as implied bound (a record, not a finding) when it improves the
//   valid record by more than the step, whatever the explicit bound is.
// - A candidate that improves the explicit bound by more than the step is pending. Before
//   make_explicit the explicit bound is not touched and the column is marked. After it the
//   explicit bound is tightened to the candidate (tighten_lower / tighten_upper; a finding), as
//   for an integer column, but at most max_explicit_steps times per bound (a constant in the
//   source file): a candidate for a bound that has had them is dropped and the call reports
//   limited.
// An explicit bound is the candidate of a row that is live when the bound is written, over
// explicit bounds that are at least as tight as those the candidate was computed on: in run()
// the candidate of the row it visits, in make_explicit a record whose source row is alive and
// unchanged since (state_t::implied_lower_valid / implied_upper_valid; explicit bounds only
// tighten). It therefore holds for every point that satisfies that row and the explicit bounds,
// and from then on it is a bound of the model like any other, whatever happens to the row. A
// record whose source row was deleted or changed is never used; if another row still implies
// the bound, that row gives the candidate.
struct bound_propagation_t {
  struct work_t;
  bound_propagation_t();
  ~bound_propagation_t();
  rule_result_t run(state_t& st, work_budget_t& budget, postsolve_t& ps, const options_t& opt);
  // From this call on run() tightens the explicit bounds of continuous columns. For every column
  // with a pending bound: a valid record that improves the explicit bound by more than the step
  // becomes that bound (a finding; infeasible when the bounds cross by more than the
  // tolerance). The live rows of the column go on row_list::propagation, so
  // that the next run() derives what is pending from the state as it is, unless the record was
  // taken and no visit of the column ever found its record void: every candidate that was not
  // recorded is then within the step of the record, no row has more to give for the column, and
  // the rows that the new bound reaches are listed by state_t::set_col_lower / set_col_upper.
  // The marks are cleared. Called once, when the phases that need the implied bounds apart from
  // the explicit ones have nothing left to do: in Stage 1 after the rounds, before probing from
  // Stage 2 on. O(columns + entries of the marked columns).
  rule_result_t make_explicit(state_t& st, postsolve_t& ps, const options_t& opt);
  std::unique_ptr<work_t> work;
};

// (c) Rows of row_list::tightening whose free columns are all integer, one composed edit per row,
// each step on the result of the previous: scaling to integral coefficients and division by the
// gcd (scale_row with integral set), rounding of the sides less the terms of the fixed columns
// (Chvatal-Gomory rounding included), then coefficient strengthening. Infeasible when an
// equation gets a fractional right-hand side or the rounded sides cross.
// Coefficient strengthening applies to any row with exactly one finite side b and finite
// maximum activity M (for <=; mirrored for >=): an integer column with |a| > M - b + feasibility,
// where M - b > feasibility, gets its coefficient and the side reduced (set_coefficient, then
// one set_row_upper or set_row_lower for the row). Redundancy is left to row_rules, which sees
// the row again through its list.
struct row_tightening_t {
  struct work_t;
  row_tightening_t();
  ~row_tightening_t();
  rule_result_t run(state_t& st, work_budget_t& budget, postsolve_t& ps, const options_t& opt);
  std::unique_ptr<work_t> work;
};

// (d) Rows of row_list::doubleton that are equations with two free entries. The call collects
// them as the edges of one batch (alias_forest_t::add_edge) and tests the budget between two
// edges, spending for an edge the live entries of its two columns; rows it does not collect go
// back on the list. A collected batch is applied in full, since a deleted edge row without its
// substitution is a wrong model: freeze() picks the tree edges and returns the maps grouped by
// root, and the rows of the maps (alias_map_t::row) are deleted. Then, per map and in the order
// of the maps: the bounds of the substituted column are transferred to the root as explicit
// bounds (tighten_lower / tighten_upper; infeasible when they cross); if the root is fixed
// afterwards, the column is fixed at its image, moved into its bounds when rounding puts it
// outside; otherwise the column is replaced by the root in its rows, the sides of those rows
// (shift_row_sides), the cost and the offset follow, the column is removed (remove_col) and the
// alias is recorded. In a row that has the root, the root's coefficient is changed
// (set_coefficient) or, when the combined coefficient is zero within tol.cancellation, deleted
// (delete_entry), and the column's entry is deleted; in a row without it the column's entry
// becomes the root's (move_entry). Whether a row has the root is read from a mark per row that
// is set once per root and kept up to date, not from find_entry, so the work is at most the
// entries of the batch rows, of the substituted columns and of their roots.
struct doubleton_t {
  struct work_t;
  doubleton_t();
  ~doubleton_t();
  rule_result_t run(state_t& st, work_budget_t& budget, postsolve_t& ps, const options_t& opt);
  std::unique_ptr<work_t> work;
};

// (e) Rows whose free entries are proportional: the same free columns and one ratio r with
// |b - r * a| <= tol.zero * |b| for every pair of coefficients. work_t keeps, for every live row
// with at least two free entries, a hash of its free entries that does not depend on their
// storage order (coefficients divided by the one of the smallest column index and rounded, so
// that parallel rows get one hash except at a rounding boundary). The rows of
// row_list::parallel_rows get a new hash. A bucket is the set of those rows with equal hash and
// equal free length, in index order. Each listed row, in index order, is compared exactly with
// its neighbours in its bucket, listed or not: at most opt.parallel_bucket_limit below it and as
// many above it. The work is thus bounded by the entries of the listed rows times twice that
// limit, and a row that changed is still found parallel to a row that did not.
// Of two parallel rows the one with the smaller index is kept: the other row's sides, less its
// fixed terms, divided by the ratio, plus the fixed terms of the kept row, are intersected with
// the kept row's sides (set_row_lower / set_row_upper; infeasible when they cross by more than
// the feasibility tolerance, a smaller crossing sets the new side to the opposite one), and the
// other row is deleted and recorded.
// A pair whose kept row has fixed terms f with |f| * tol.side_error > tol.feasibility is left
// alone and counts as not parallel towards the limit: a side of that row is stored with f in it,
// as one double, and would lose more than the tolerance (a column fixed at a huge bound).
struct parallel_rows_t {
  struct work_t;
  parallel_rows_t();
  ~parallel_rows_t();
  rule_result_t run(state_t& st, work_budget_t& budget, postsolve_t& ps, const options_t& opt);
  std::unique_ptr<work_t> work;
};

// (f) Columns of col_list::dual_fixing. A column with cost >= 0 and no down lock can move to its
// lower bound, one with cost <= 0 and no up lock to its upper bound; an empty column is the case
// without locks. A finite target bound: the column is fixed there (zero cost and no lock at all:
// the lower bound if finite, else the upper bound, else 0). An infinite target bound: cost != 0
// gives unbounded_or_infeasible; cost == 0 with a lock in the other direction leaves the column
// (Stage 3 removes it with its rows). Not called in primal-only mode.
struct dual_fixing_t {
  struct work_t;
  dual_fixing_t();
  ~dual_fixing_t();
  rule_result_t run(state_t& st, work_budget_t& budget, postsolve_t& ps, const options_t& opt);
  std::unique_ptr<work_t> work;
};

// Explicit bound changes, one policy for every family.
struct bound_change_t {
  rule_outcome_t outcome = rule_outcome_t::ok;
  bool changed           = false;
};

// Tightens an explicit bound of a live column; the bound is not NaN. An integer column gets the
// rounded bound (round_lower_integer / round_upper_integer). A bound that is not tighter than
// the current one changes nothing. A bound that crosses the opposite bound by more than the
// feasibility tolerance gives infeasible, a lower bound of +inf and an upper bound of -inf
// included; one that crosses it by less is set to the opposite bound. When lower == upper
// afterwards the column is fixed and recorded (fix_column), so col_status tells the caller.
// O(csc.used[col]) when the state changes, O(1) otherwise.
bound_change_t tighten_lower(state_t& st, postsolve_t& ps, int32_t col, double lower);
bound_change_t tighten_upper(state_t& st, postsolve_t& ps, int32_t col, double upper);

// Fixes a live column at a finite value inside its bounds, integral for an integer column, and
// records it (state_t::fix_col, postsolve_t::push_fixed_col). O(csc.used[col]).
void fix_column(state_t& st, postsolve_t& ps, int32_t col, double value);

// An activity of the row is fresh when it has had no update since it was summed from scratch:
// its error bound is then the smallest it can have, and the one that a presolve of the same
// rows starts with. The error bound of an activity that is not fresh still counts every term
// the sum ever held, a huge one that has left it included.
bool activity_fresh(const state_t& st, const activity_t& act, int32_t row);

// Sums the activities of a live row from scratch (state_t::recompute_row_activity) when one of
// them is not fresh and its error bound is above tol.zero * tol.feasibility, below which it
// decides no test against the feasibility tolerance. Returns whether it did. The row rules and
// row tightening call it before they test the activities of a row; bound propagation, whose
// guard scales with the coefficient, sums again when the guard drops a candidate and the
// activity is not fresh. O(1) unless it sums.
bool refresh_activity(state_t& st, int32_t row);

}  // namespace gprs
