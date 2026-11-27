/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Ownership of the Stage 1 host engine. A part edits its own files only. Every src/*.cpp and
// tests/*.cpp is picked up by CMakeLists.txt and build_host_local.sh without an edit, so a part
// adds source and test files under its own names. A threshold that a rule needs and options_t
// does not have is a named constant in the rule's source file; part E moves it into options_t
// when it has to become a setting.
//
// Shared, interfaces fixed: include/gprs/assert.hpp, numerics.hpp, rules.hpp; src/assert.cpp;
//    tests/harness.hpp, main.cpp, test_numerics.cpp; CMakeLists.txt; build_host_local.sh.
// A  State and matrix operations: include/gprs/state.hpp; src/state_build.cpp, state_ops.cpp,
//    state_storage.cpp; tests/test_state_*.cpp.
// B  Postsolve and alias forest: include/gprs/postsolve.hpp; src/postsolve.cpp,
//    alias_forest.cpp; tests/test_postsolve_*.cpp, test_alias_*.cpp.
// C  Row rules, bound propagation, tightening: src/rules_bounds.cpp, rules_rows.cpp,
//    rules_propagation.cpp, rules_tightening.cpp; tests/test_bounds_*.cpp, test_rows_*.cpp,
//    test_propagation_*.cpp, test_tightening_*.cpp.
// D  Doubleton equations and parallel rows: src/rules_doubleton.cpp, rules_parallel_rows.cpp;
//    tests/test_doubleton_*.cpp, test_parallel_*.cpp.
// E  Dual fixing, engine driver, hash, IO, tool: include/gprs/presolve.hpp, io.hpp;
//    src/rules_dual_fixing.cpp, presolve.cpp, hash.cpp, io.cpp; tools/gprs_presolve.cpp;
//    tests/test_dual_*.cpp, test_engine_*.cpp, test_hash_*.cpp, test_io.cpp.
// F  Oracle, generators, end-to-end tests: tests/oracle.hpp, oracle.cpp, test_oracle_*.cpp,
//    test_e2e_*.cpp.

#include <gprs/numerics.hpp>
#include <gprs/postsolve.hpp>
#include <gprs/rules.hpp>
#include <gprs/state.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace gprs {

// Rule families, in the order of the statistics.
namespace rule {
constexpr int32_t row_rules     = 0;
constexpr int32_t propagation   = 1;
constexpr int32_t tightening    = 2;
constexpr int32_t doubleton     = 3;
constexpr int32_t parallel_rows = 4;
constexpr int32_t dual_fixing   = 5;
constexpr int32_t count         = 6;
}  // namespace rule

const char* rule_name(int32_t rule);

// Budget of one call of a family, in work units, from the work of one visit of every row and
// column at the call, pass = free nonzeros + line_visit_work * (live rows + live columns):
// limit = floor + passes * pass, stall limit = floor + stall_passes * pass. With the nonzeros
// alone as the measure, a first scan of a model with short rows (8 units per row against 2.4
// entries) would stall before it has seen every row once.
struct budget_t {
  double passes;
  double stall_passes;
  int64_t floor;
};

// The budget of a call on the state as it is.
inline work_budget_t make_budget(const budget_t& b, const state_t& st)
{
  double pass = st.free_nnz() + line_visit_work * ((int64_t)st.live_rows + st.live_cols);
  work_budget_t budget;
  budget.limit       = b.floor + b.passes * pass;
  budget.stall_limit = b.floor + b.stall_passes * pass;
  return budget;
}

// Every threshold and budget of the engine. The defaults are the same on every machine; nothing
// is read from the machine.
struct options_t {
  tolerances_t tol;

  // Lock arguments off: every feasible point of the original model crushes to a feasible point
  // of the reduced model with the same objective value.
  bool primal_only = false;

  std::array<bool, rule::count> enabled = {true, true, true, true, true, true};

  std::array<budget_t, rule::count> budget = {{{16., 2., 100000},
                                               {16., 2., 100000},
                                               {16., 2., 100000},
                                               {16., 4., 100000},
                                               {8., 8., 100000},
                                               {4., 2., 100000}}};

  // Rounds of cascade, parallel rows and doubleton equations, counted over the whole presolve.
  int32_t max_rounds = 20;
  // Passes over the four cascade families within one round.
  int32_t max_cascade_passes = 100;

  // Seconds. Only an abort that returns status limit with a valid partial result; it never
  // changes which reductions are tried.
  double time_limit = inf;

  // The engine rebuilds the state between phases when the dead share of the slots, or the share
  // of the live entries that belong to fixed columns, exceeds these.
  double rebuild_dead_fraction  = 0.25;
  double rebuild_fixed_fraction = 0.25;

  // Kept rows a row is compared with inside one hash bucket of the parallel-row search.
  int32_t parallel_bucket_limit = 4;

  // Record a hash of state and postsolve stack after every phase in result_t::phase_hash.
  bool phase_hashes = false;
};

// reduced, unchanged, reduced_to_empty and limit carry a valid reduced model and postsolve stack.
// failed is an assertion in this project and is never returned.
enum class status_t : uint8_t {
  reduced,
  unchanged,
  reduced_to_empty,
  infeasible,
  unbounded_or_infeasible,
  limit,
  failed
};

const char* status_name(status_t status);

struct rule_stats_t {
  int64_t calls    = 0;
  int64_t work     = 0;
  int64_t findings = 0;
  // Implied bounds recorded (rule_result_t::records); only bound propagation has them.
  int64_t records      = 0;
  double seconds       = 0.;
  int64_t rows_removed = 0;
  int64_t cols_removed = 0;
  int64_t nnz_removed  = 0;
};

struct result_t {
  status_t status = status_t::failed;
  model_t reduced;
  // Carries row_origin and col_origin of the reduced model.
  postsolve_t postsolve;
  std::array<rule_stats_t, rule::count> stats;
  // The start step: one call; findings are the sides and bounds it changed plus the columns it
  // fixed, seconds is seconds_start, and the removals are those of the fixed columns.
  rule_stats_t start;
  int32_t rounds = 0;
  // Of `rounds`: those after the switch to explicit bounds (bound_propagation_t::make_explicit).
  int32_t explicit_rounds = 0;
  int32_t rebuilds        = 0;
  // The presolve ended because nothing was left to do: the last round changed nothing, no call
  // of it stopped on its budget, no list of the cascade is left, no family reported limited
  // (rule_result_t::limited), no row holds a term of a fixed column that a family could read,
  // and no limit (max_rounds, max_cascade_passes, time_limit) cut it short.
  // presolve(reduced, options) then returns unchanged; see presolve().
  bool fixed_point = false;
  // From the input model to the finished reduced model.
  double seconds = 0.;
  // model_hash of the reduced model.
  uint64_t hash = 0;
  std::vector<uint64_t> phase_hash;
  // The parts of `seconds` outside the calls of the families (rule_stats_t::seconds):
  // the postsolve stack and the state from the model; the start step; compaction, rebuilds and
  // phase hashes between passes and phases; the reduced model from the state and the status;
  // model_hash of the reduced model, a chain of one word after the other that costs the same
  // per word whatever the engine does.
  double seconds_state   = 0.;
  double seconds_start   = 0.;
  double seconds_storage = 0.;
  double seconds_finish  = 0.;
  double seconds_hash    = 0.;
  // Not one of the parts above: the seconds from the switch to explicit bounds to the end of the
  // last round, the calls of the families in them included.
  double seconds_explicit = 0.;
};

// Runs the Stage 1 host engine on a well_formed model.
// Start, before any rule and in index order, which sets up what the rules assume of the state
// (lower <= upper on every row and column, integral bounds on integer columns, no live column
// with lower == upper). Rows: sides that cross by more than the feasibility tolerance give
// infeasible; a smaller crossing sets the lower side to the upper one. Columns: the bounds of
// an integer column are rounded first (round_lower_integer / round_upper_integer); bounds that
// then cross by more than the feasibility tolerance give infeasible; a smaller crossing sets
// the lower bound to the upper one; a column with lower == upper is fixed (fix_column).
// Round: (1) cascade: row rules, bound propagation, row tightening, then dual fixing on the
// locks they leave, repeated until their lists are empty, a budget stops them or
// max_cascade_passes is reached; (2) parallel rows on the state the cascade leaves; (3) doubleton
// equations. Rounds repeat until one changes nothing or max_rounds is reached. Between phases
// the state compacts sparse lines and rebuilds by the two fractions above; it rebuilds only
// when every row can take the terms of its fixed columns (state_t::fixed_terms_absorbable), so
// that no rule reads a side that the move has rounded by more than the tolerance.
// A family is called with make_budget(options.budget[rule], state).
// The order of the whole presolve:
// 1. Rounds with the implied bounds of continuous columns apart from the explicit ones: bound
//    propagation records them and marks the columns whose explicit bound a row would improve.
//    Later stages put the phases that need the two apart here (dual and structural rules).
// 2. bound_propagation_t::make_explicit, when bound propagation is switched on: the recorded
//    bounds of the marked columns become explicit, and from here on a continuous column gets
//    its explicit bounds as an integer column does. When that changed a bound or listed a row,
//    the rounds go on.
// 3. While the rounds ended with a round that changed nothing and a row still holds a term of
//    a fixed column: state_t::absorb_fixed_terms, and the rounds go on with the rows that it
//    lists. A family can treat a row with such a term otherwise than the same row without it
//    (the side of the free part is a difference of two doubles; a term above the infinite-term
//    limit makes an activity infinite; the limits on the size of a side), so only this state
//    is the one a second presolve would see. A rebuild between the phases lists such rows for
//    the same reason (state_t::rebuild). Skipped when only dual fixing and doubleton equations
//    are switched on, which read no side and no activity, and when a row cannot take its terms
//    (state_t::fixed_terms_absorbable): the rules would read a side that the move has rounded
//    by more than the tolerance. The terms then stay in the rows until the reduced model is
//    read, as before the switch, and the presolve does not end at a fixed point.
// 4. The reduced model is read from the state.
// Rounds that ended on a budget or on max_cascade_passes are not taken up again in steps 2 and
// 3: only a new reason starts them.
// Why the bounds become explicit before the last rounds and not in one step at the end: a bound
// that becomes explicit makes rows redundant, singleton or forcing, makes coefficients
// strengthen and gives further bounds, of integer columns too. The rounds after the switch
// apply all of that, so the reduced model is a fixed point of the rules. No reduction is made
// on a recorded bound: a row is tested on explicit bounds only, each of which is the candidate
// of a row that was in the model when the bound was written (rules.hpp, (b)), so that the bound
// holds for every feasible point and stays a bound of the model whatever happens to that row.
// Fixed point: with result_t::fixed_point set, presolve(result.reduced, options) returns
// unchanged, and the reduced model has, for the families that are switched on, no empty row,
// no singleton row, no row that is redundant or forcing on its own bounds (forcing as in
// rules.hpp, (a), which leaves a row with a continuous column off its bound), no coefficient that
// row tightening would change and no bound that propagation would tighten. What it rests on:
// every row was visited by every family after its last change; no row holds the term of a fixed
// column, and a row that lost one other than plainly was visited again; an activity whose error
// bound could decide a test was summed from scratch before the test (refresh_activity). What
// it does not cover: a stored activity differs from the sum that the second presolve starts
// with by its rounding, so a test that the last bits of an activity decide can go the other
// way there.
// Added with the implementation:
// - A family is called when it is switched on and its list is not empty; dual fixing never in
//   primal-only mode. A call that stops on its budget ends the cascade after the pass it is in,
//   and the family goes on with a new budget in the cascade of the next round. A round changes
//   nothing when its calls report no finding; an implied bound that is only recorded is not a
//   finding (rule_result_t::records) and does not start another round.
// - Sparse lines are compacted after every cascade pass and after every phase. The state is
//   rebuilt only after a phase (cascade, parallel rows, doubleton equations) and at the end: a
//   rebuild costs O(rows + columns) of the input model, whatever is left of it.
// - Statistics: rows, columns and nonzeros removed by a call are the differences of live_rows,
//   live_cols and free_nnz(). The start step has its own entry, result_t::start. make_explicit
//   is booked under rule::propagation with its findings, seconds and removals, not as a call.
// - Empty rows: when the presolve does not end at a fixed point and the row rules are switched
//   on, row_rules_t::remove_empty_rows runs before the reduced model is read (booked under
//   rule::row_rules, not as a call), so that a reduced model has no row without entries
//   whatever stopped the rounds. With the row rules switched off an empty row of the input
//   stays.
// - Status: infeasible and unbounded_or_infeasible as the start step or a family reports them,
//   with an empty result_t::reduced. limit when the time limit, tested before every call, ended
//   the rounds; make_explicit is still called and the reduced model read from the state.
//   Otherwise unchanged when the reduced model equals the input in every array,
//   reduced_to_empty when it has no row and no column, else reduced. Budgets and the limits on
//   rounds and passes never give limit.
// - phase_hash: one entry after the start step, three per round (after the cascade, parallel
//   rows and doubleton equations, each after its compaction or rebuild, also when the family
//   was not called) and one after the extraction; steps 2 and 3 add none of their own. A
//   presolve that ends inside a round has the entries up to there and, with status limit, the
//   last one.
// - result_t::hash is model_hash(result_t::reduced) for every status.
// - The end: the reduced model is read from the state as the rounds leave it (reduced_model),
//   which is the model of the rebuilt state. The rebuild itself is carried out only where
//   something reads the state afterwards: with phase_hashes, whose last entry is over the
//   rebuilt state, and in a build with assertions, which compares reduced_model with
//   extract_model of the rebuilt state. result_t::rebuilds counts it in every case.
result_t presolve(const model_t& model, const options_t& options);

// The model and the two origin maps that st.rebuild() followed by extract_model(st, ...) give,
// bit for bit, read from the state as it is: the state is not changed and need not be rebuilt.
// The terms of the fixed columns leave the finite sides of their rows as in rebuild(); the
// entries of a row are in increasing column order. O(used slots of the live rows + rows +
// columns), plus the sort of the rows that move_entry left out of order.
model_t reduced_model(const state_t& st,
                      std::vector<int32_t>& row_origin,
                      std::vector<int32_t>& col_origin);

// Hash over sizes and the bit patterns of every array of the model.
// The three hashes chain 64-bit words (an integer as its value, a double as its bit pattern, an
// array as its length and its elements), so they are the same on every platform and depend on
// no address.
uint64_t model_hash(const model_t& model);

// Hash over the bit patterns of bounds, sides, activities with their counters, live entries in
// storage order, flags, locks, cost and offset of the state.
// Added with the implementation: and over everything else a later step reads, which is the
// layout of both stores (start, used, capacity and size of every line, tombstones by position),
// row versions, recorded implied bounds, the counters, the tolerances and the lists in their
// order. Not hashed: slot_visits, lane_scratch, value and twin of a tombstone.
uint64_t state_hash(const state_t& st);

// Hash over the records of the stack in order.
// Added with the implementation: and over its sizes, column types, row_origin and col_origin.
uint64_t postsolve_hash(const postsolve_t& ps);

}  // namespace gprs
