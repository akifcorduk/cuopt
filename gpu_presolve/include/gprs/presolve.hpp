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

  // Rounds of cascade, parallel rows and doubleton equations.
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
  int64_t calls        = 0;
  int64_t work         = 0;
  int64_t findings     = 0;
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
  int32_t rounds   = 0;
  int32_t rebuilds = 0;
  // From the input model to the finished reduced model.
  double seconds = 0.;
  // model_hash of the reduced model.
  uint64_t hash = 0;
  std::vector<uint64_t> phase_hash;
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
// the state compacts sparse lines and rebuilds by the two fractions above. At the end recorded
// implied bounds are made explicit, the state is rebuilt and the reduced model extracted.
// A family is called with make_budget(options.budget[rule], state).
result_t presolve(const model_t& model, const options_t& options);

// Hash over sizes and the bit patterns of every array of the model.
uint64_t model_hash(const model_t& model);

// Hash over the bit patterns of bounds, sides, activities with their counters, live entries in
// storage order, flags, locks, cost and offset of the state.
uint64_t state_hash(const state_t& st);

// Hash over the records of the stack in order.
uint64_t postsolve_hash(const postsolve_t& ps);

}  // namespace gprs
