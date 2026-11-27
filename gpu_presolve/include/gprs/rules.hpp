/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <gprs/postsolve.hpp>
#include <gprs/state.hpp>

#include <array>
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

// The reductions of the column rules, (g) to (j) below, which hold more than one: each has its
// switch (options_t::sub_enabled, sub_rule_active) and its finding counter
// (result_t::sub_findings). Simple probing is a family with one reduction and has neither.
namespace sub_rule {
// (h) A singleton column implied free by its row: an equation as pivot row.
constexpr int32_t free_singleton = 0;
// (h) The zero-cost continuous singleton columns of a row with two finite sides.
constexpr int32_t zero_cost_singleton = 1;
// (h) A zero-cost column without a lock in a direction in which its bound is infinite.
constexpr int32_t forcing_column = 2;
// (h) free_singleton with an inequality as pivot row, tight by the cost of the column.
constexpr int32_t free_singleton_dual = 3;
// (j) Of two parallel columns the one that the other dominates is fixed.
constexpr int32_t parallel_dominance = 4;
// (i) A bound of a column from the rows that lock it against its cost.
constexpr int32_t dual_bound = 5;
// (i) A column with a single lock against its cost, aliased to a binary of its lock row.
constexpr int32_t lock_alias = 6;
// (j) Two parallel columns with equal cost become one.
constexpr int32_t parallel_merge = 7;
constexpr int32_t count          = 8;
}  // namespace sub_rule

const char* sub_rule_name(int32_t sub);

// What the column rules count besides their findings, for one call (rule_result_t) and summed
// or, for the two largest values, taken over the calls of a presolve (result_t).
struct column_stats_t {
  // Columns whose cost became nonzero when a free singleton column took its pivot row out of
  // the matrix and into the objective.
  int64_t cost_fill = 0;
  // Entries of the rows that records saved.
  int64_t saved_entries = 0;
  // Largest growth factor of a substituted free singleton column: what a restored value can
  // multiply a deviation of the reduced point by.
  double max_growth = 0.;
  // Largest depth of a tree in the alias forest of one call of simple probing: the aliases
  // composed between a substituted column and its root.
  int32_t alias_depth = 0;
  // Singleton columns that the implied-free test refused after the switch to explicit bounds
  // (bound_propagation_t::make_explicit).
  int64_t free_singleton_rejected_explicit = 0;
};

struct rule_result_t {
  rule_outcome_t outcome = rule_outcome_t::ok;
  // Changes of the model: rows deleted, columns fixed or substituted, bounds, sides or
  // coefficients changed. Removed rows, columns and nonzeros are read from the counters of the
  // state.
  int64_t findings = 0;
  // Of the findings, those that keep an optimal point and not every feasible one: a reduction by
  // a lock or cost argument. None in primal-only mode.
  int64_t dual_findings = 0;
  // Of the findings of a column rule, those of each of its reductions (sub_rule::).
  std::array<int64_t, sub_rule::count> sub_findings = {};
  column_stats_t columns;
  // Implied bounds recorded (state_t::record_implied_lower / record_implied_upper). A record
  // changes no row, column or explicit bound and is not a finding.
  int64_t records = 0;
  // The call stopped on its budget and returned the unprocessed part of its list to the state.
  bool stopped = false;
  // The call left undone a reduction that no list holds, or cannot tell: bound propagation,
  // when a bound of a continuous column has had its steps (see (b)); the parallel rows, when
  // they leave a pair alone only because the presolve has changed one of its rows, or end the
  // comparisons of a row on opt.parallel_bucket_limit with rows of its bucket left (see (e)).
  // The state is then no fixed point of the rules, although every list can be empty.
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

// (e) Rows whose cores are proportional. A singleton column is a live column with one live
// entry; the core of a row is its free entries outside singleton columns. Two rows are parallel
// when their cores have the same columns and one ratio r with |b - r * a| <= tol.zero * |b| for
// every pair of core coefficients. With opt.near_parallel_rows off the core of a row is all its
// free entries, and what is said of singleton columns below does not occur.
// work_t keeps, for every live row with at least two core entries, a hash of its core that does
// not depend on the storage order (coefficients divided by the one of the smallest column index
// and rounded, so that parallel rows get one hash except at a rounding boundary). The rows of
// row_list::parallel_rows are those with a change of their free entries or of a side
// (row_event::side: the sides decide what is done with a pair, below). Those whose row_version
// or number of free entries changed get a new hash, and so do the rows of
// row_list::singletons, whose core changed with their free entries as they were. A bucket is the
// set of those rows with equal hash and equal core length, in index order. Each listed row, in
// index order, is compared exactly with its neighbours in its bucket, listed or not: at most
// opt.parallel_bucket_limit above it, then as many below it. The work is thus bounded by the
// entries of the listed rows times twice that limit, and a row that changed is still found
// parallel to a row that did not.
// A row whose comparisons end on that limit while its bucket has a further live row in that
// direction has not been compared with it, and no list will bring the two together when the
// rows between them leave the bucket (deleted, merged, subtracted from, or with other
// entries): only a presolve of the reduced model compares them. The call reports limited
// (rule_result_t::limited). While no call has, every two rows of a bucket have been compared
// since the last change of either.
// Two parallel rows without a singleton column: the one with the smaller index is kept: the
// other row's sides, less its fixed terms, divided by the ratio, plus the fixed terms of the
// kept row, are intersected with the kept row's sides (set_row_lower / set_row_upper;
// infeasible when they cross by more than the feasibility tolerance, a smaller crossing sets
// the new side to the opposite one), and the other row is deleted and recorded.
// A pair whose kept row has fixed terms f with |f| * tol.side_error > tol.feasibility is left
// alone and counts as not parallel towards the limit: a side of that row is stored with f in it,
// as one double, and would lose more than the tolerance (a column fixed at a huge bound).
// Two parallel rows with a singleton column in one of them at least: an equation of the pair is
// subtracted from the other row, the target. The equation qualifies when it has no singleton
// column, or one while the target has at most one; of two equations that qualify, the one with
// fewer singleton columns is subtracted, then the one with the smaller index. With f = core of
// the target / core of the equation (the coefficients of the smallest core column), in this
// order: the first core entry of the target, in storage order, becomes the entry of the
// equation's singleton column with -f times its coefficient there (move_entry), if it has one;
// the other core entries of the target are deleted in storage order (delete_entry); the finite
// sides of the target are shifted by f times the right-hand side of the equation less its fixed
// terms (shift_row_sides), unless that is zero; the step is recorded (push_row_addition). No row
// is deleted: the target is left with its own singleton columns and that of the equation, with
// one or two entries or with a part of the entries it had, to the row rules and the doubleton
// equations. The two rows have the same solutions before and after, integer columns or not.
// Such a step takes at least one entry from the model, since a core has two entries or more and
// the equation brings at most one, and no family adds an entry: their number is finite.
// After the step the target row of the model is the new target plus f times the equation, so
// what a later step does to the equation within its tolerance reaches that row times |f|; and
// once the singleton column of the equation is taken out through the target, the equation is
// the old target divided by f, so what is done to the target reaches it divided by |f|. The
// step is made only where neither grows: with |f| <= 1, and with |f| = 1 when the equation has
// a singleton column.
// The step also carries the errors with which the constants of the two rows are stored into
// the singleton columns. That of the equation reaches a singleton column of the target with
// the coefficient c times |f| / |c|: it grows by g_E = max(1, |f| / min |c|). That of the
// target reaches the singleton column of the equation, with the coefficient a there, divided
// by |a|: it grows by g_T = max(1, 1 / |a|), and by 1 without such a column.
// The pair is left alone, and counts as not parallel towards the limit, when neither row
// qualifies; when f or the new coefficient is zero or not finite; when |f| is not as above; when
// g_E or g_T is above 1000; when
// |v| * tol.side_error > tol.feasibility for v the shift, g_E times the right-hand side of the
// equation, g_E times its fixed terms, g_T times a finite side of the target or, with g_T > 1,
// g_T times the fixed terms of the target (with g_T = 1 these stay in its entries and are not
// read); when g_E > 1 and the presolve has changed the equation, or g_T > 1 and it has changed
// the target (row_version != 0): such a row holds the rounding of those changes, which keeps
// the tolerance in that row only (a side that was scaled next to a fixed term of 1e7 is off by
// 1e-9), and no limit on a size bounds it; or when max |a| / min |a| over the free entries of
// the target would end above 1e6 and above its value before. That last test and the step read
// the target once more and spend its used slots.
// Every test but the one on row_version reads the two rows as they are, so the pair is tested
// again when it can end otherwise: a change of an entry, of a side or of the core lists the
// row, and with g_E and g_T at most 1000 the terms of fixed columns that leave a row without
// listing it (state_t::absorb_fixed_terms: sides and terms up to 1e5) are below every limit
// on a size, in the row and in the side. The test on row_version reads what the presolve has
// done: a pair that fails it alone would be subtracted by a presolve of the reduced model,
// which finds the row as a model gave it. The call then reports limited
// (rule_result_t::limited), and the presolve does not end at a fixed point.
// A reduction can change the core of a row that stays: of the kept row or the equation, where a
// column that only the two rows held becomes a singleton column, and of the equation, whose
// singleton column gets a second entry. Such a row and the target have no key until the next
// call, which lists them, and when the listed row is one of them its turn ends.
// Integer columns: the subtraction is a row operation, so it changes no solution. It changes
// what a single row tells. A target whose columns were all integer can get a continuous
// singleton column of the equation and is then no row of (c); a bound that the target gave a
// core column is now read from the equation and the bounds of its singleton column, which are
// explicit only for an integer column or after the target has become a singleton row.
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
// (Stage 3 removes it with its rows). Not called in primal-only mode. Every finding is a dual
// finding (rule_result_t::dual_findings).
struct dual_fixing_t {
  struct work_t;
  dual_fixing_t();
  ~dual_fixing_t();
  rule_result_t run(state_t& st, work_budget_t& budget, postsolve_t& ps, const options_t& opt);
  std::unique_ptr<work_t> work;
};

// The column rules, (g) to (j). Each is called once per round, in this order, after the
// doubleton equations, on the state that the row rules of the round leave; dual fixing stays in
// the cascade and so runs before them on every state. They are called in primal-only mode as
// well, where each of their reductions decides for itself (sub_rule_active).
// What every test of them reads, and nothing else: explicit bounds, sides, entries, lock counts,
// costs, types and the stored activities of the state as it is. A bound counts as implied by a
// row when the candidate of that row, recomputed at the test (src/implied.hpp), is within
// implied_tolerance of it; a recorded implied bound is never read, since records depend on the
// order of the visits. Each premise has an event that lists what reads it: cost, locks and
// entries of a column its column lists (col_event::locks, col_event::entries); sides, entries
// and both activities of a row, and with the activities the bounds of every column of the row,
// its row lists (row_event::column_rules). A family that reads columns through a row list
// expands the listed rows to their live columns.
// A family takes its lists when it is called and, when it stops on its budget, returns the part
// it has not served. A reduction that it leaves undone for a reason that is no function of the
// state reports limited.
// A family whose reductions are not built takes its lists and reports nothing: none of the four
// holds a reduction so far.

// Largest range upper - lower of a column that simple probing or the lock alias replaces by a
// binary: it is the scale of the alias, which multiplies every coefficient of the column into
// the rows of the binary.
constexpr double alias_scale_limit = 1e6;

// (g) Rows of row_list::simple_probing, which holds every row at construction.
struct simple_probing_t {
  struct work_t;
  simple_probing_t();
  ~simple_probing_t();
  rule_result_t run(state_t& st, work_budget_t& budget, postsolve_t& ps, const options_t& opt);
  std::unique_ptr<work_t> work;
};

// (h) Columns of col_list::singleton_cols and rows of row_list::singleton_cols. The reductions
// sub_rule::free_singleton, zero_cost_singleton, forcing_column and free_singleton_dual belong
// here. state_t::row_singletons tells for a listed row, in O(1), whether it holds a singleton
// column at all.
struct singleton_cols_t {
  struct work_t;
  singleton_cols_t();
  ~singleton_cols_t();
  rule_result_t run(state_t& st, work_budget_t& budget, postsolve_t& ps, const options_t& opt);
  std::unique_ptr<work_t> work;
};

// (i) Columns of col_list::dual_cols and rows of row_list::dual_cols. The reductions
// sub_rule::dual_bound and lock_alias belong here.
struct dual_cols_t {
  struct work_t;
  dual_cols_t();
  ~dual_cols_t();
  rule_result_t run(state_t& st, work_budget_t& budget, postsolve_t& ps, const options_t& opt);
  std::unique_ptr<work_t> work;
};

// (j) Columns of col_list::parallel_cols and rows of row_list::parallel_cols. The reductions
// sub_rule::parallel_dominance and parallel_merge belong here.
struct parallel_cols_t {
  struct work_t;
  parallel_cols_t();
  ~parallel_cols_t();
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
