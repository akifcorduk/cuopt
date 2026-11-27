/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "test_engine_support.hpp"

#include <gprs/presolve.hpp>

#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace gprs;
using namespace gprs::engine_test;

namespace {

//   r0:  x0 + 2 x1 - x2      <= 6
//   r1:  x0 - x1 + x2 + 3 x3 in [-4, 8]
//   r2:        x1      - x3  >= -1
model_t hash_model()
{
  model_t m          = make_model({{1., 2., -1., 0.}, {1., -1., 1., 3.}, {0., 1., 0., -1.}},
                                  {-inf, -4., -1.},
                                  {6., 8., inf},
                                  {1., 0., 0., -2.},
                                  {4., 3., 2., 7.},
                                  {2., -1., -3., 0.},
                                  {1, 0, 0, 1});
  m.objective_offset = 10.;
  return m;
}

double next_up(double v) { return std::nextafter(v, inf); }

template <typename object_t>
struct change_t {
  const char* name;
  std::function<void(object_t&)> apply;
};

// Every change gives a hash that differs from the base and from every other change.
template <typename object_t, typename hash_t>
void check_changes(const object_t& base,
                   hash_t hash,
                   const std::vector<change_t<object_t>>& changes)
{
  uint64_t base_hash = hash(base);
  std::vector<uint64_t> seen;
  for (const change_t<object_t>& change : changes) {
    object_t changed = base;
    change.apply(changed);
    uint64_t h = hash(changed);
    if (h == base_hash) {
      gprs::test::fail(__FILE__, __LINE__, std::string(change.name) + ": hash does not change");
    }
    for (size_t k = 0; k < seen.size(); ++k) {
      if (seen[k] != h) { continue; }
      gprs::test::fail(
        __FILE__, __LINE__, std::string(change.name) + ": same hash as " + changes[k].name);
    }
    seen.push_back(h);
  }
}

uint64_t hash_of_state(const state_t& st) { return state_hash(st); }
uint64_t hash_of_model(const model_t& m) { return model_hash(m); }
uint64_t hash_of_stack(const postsolve_t& ps) { return postsolve_hash(ps); }

postsolve_t empty_stack()
{
  postsolve_t ps;
  ps.init(hash_model());
  return ps;
}

}  // namespace

GPRS_TEST(hash_model_is_a_function_of_the_values)
{
  model_t a = hash_model();
  model_t b = hash_model();
  b.value.reserve(1000);
  b.col_index.shrink_to_fit();
  std::vector<double> moved = std::move(b.cost);
  b.cost.assign(moved.begin(), moved.end());
  GPRS_CHECK_EQ(model_hash(a), model_hash(b));
  GPRS_CHECK_EQ(model_hash(a), model_hash(model_t(a)));
  GPRS_CHECK(model_hash(a) != 0);
  // The value on every platform: words are hashed, not bytes. It was computed a second time
  // outside this project from the description of the chain in hash.cpp.
  GPRS_CHECK_EQ(model_hash(a), 0x9df5ad20c1657f10ull);
}

GPRS_TEST(hash_model_depends_on_every_field)
{
  using change = change_t<model_t>;
  check_changes<model_t>(
    hash_model(),
    hash_of_model,
    {change{"nrows", [](model_t& m) { ++m.nrows; }},
     change{"ncols", [](model_t& m) { ++m.ncols; }},
     change{"row_start", [](model_t& m) { ++m.row_start[1]; }},
     change{"col_index", [](model_t& m) { m.col_index[4] = 3; }},
     change{"value, last bit", [](model_t& m) { m.value[0] = next_up(m.value[0]); }},
     change{"value, sign", [](model_t& m) { m.value[3] = -m.value[3]; }},
     change{"value, order", [](model_t& m) { std::swap(m.value[0], m.value[1]); }},
     change{"row_lower", [](model_t& m) { m.row_lower[1] = next_up(m.row_lower[1]); }},
     change{"row_lower infinite", [](model_t& m) { m.row_lower[1] = -inf; }},
     change{"row_upper", [](model_t& m) { m.row_upper[0] = next_up(m.row_upper[0]); }},
     change{"col_lower", [](model_t& m) { m.col_lower[2] = next_up(m.col_lower[2]); }},
     change{"col_upper", [](model_t& m) { m.col_upper[3] = next_up(m.col_upper[3]); }},
     change{"cost", [](model_t& m) { m.cost[0] = next_up(m.cost[0]); }},
     change{"cost, sign of zero", [](model_t& m) { m.cost[3] = -0.; }},
     change{"col_type", [](model_t& m) { m.col_type[1] = 1; }},
     change{"objective_offset", [](model_t& m) { m.objective_offset = next_up(10.); }},
     change{"sides swapped", [](model_t& m) { std::swap(m.row_lower, m.row_upper); }},
     change{"bounds of two columns swapped",
            [](model_t& m) {
              std::swap(m.col_lower[0], m.col_lower[1]);
              std::swap(m.col_upper[0], m.col_upper[1]);
            }},
     // The same words in the same order, in other arrays.
     change{"value moved to the next array",
            [](model_t& m) {
              m.col_upper.insert(m.col_upper.begin(), m.col_lower.back());
              m.col_lower.pop_back();
            }},
     change{"zero appended", [](model_t& m) { m.cost.push_back(0.); }},
     change{"entry appended", [](model_t& m) {
              m.col_index.push_back(0);
              m.value.push_back(0.);
            }}});
}

GPRS_TEST(hash_state_is_a_function_of_the_state)
{
  model_t m = hash_model();
  state_t a(m, tolerances_t());
  state_t b(m, tolerances_t());
  GPRS_CHECK_EQ(state_hash(a), state_hash(b));
  state_t c(a);
  c.csr.value.reserve(4096);
  c.col_lower.shrink_to_fit();
  GPRS_CHECK_EQ(state_hash(a), state_hash(c));

  // The same operations give the same hash.
  for (state_t* st : {&a, &b}) {
    st->set_col_upper(1, 2.);
    st->delete_row(2);
    st->fix_col(0, 3.);
  }
  GPRS_CHECK_EQ(state_hash(a), state_hash(b));
  GPRS_CHECK(state_hash(a) != state_hash(c));

  // What is no part of the state: the visit counter and the scratch lanes of a summation.
  uint64_t before = state_hash(a);
  GPRS_CHECK(a.find_entry(0, 1) >= 0);
  GPRS_CHECK(a.slot_visits != b.slot_visits);
  side_sum_t min_side, max_side;
  a.fold_row_activity(1, min_side, max_side);
  GPRS_CHECK_EQ(state_hash(a), before);
  GPRS_CHECK_EQ(state_hash(a), state_hash(b));
}

GPRS_TEST(hash_state_depends_on_every_operation)
{
  using change = change_t<state_t>;
  state_t base(hash_model(), tolerances_t());
  std::vector<int32_t> taken;
  check_changes<state_t>(
    base,
    hash_of_state,
    {change{"set_col_lower", [](state_t& st) { st.set_col_lower(1, 1.); }},
     change{"set_col_lower, last bit", [](state_t& st) { st.set_col_lower(1, next_up(1.)); }},
     change{"set_col_upper", [](state_t& st) { st.set_col_upper(1, 1.); }},
     change{"set_col_upper infinite", [](state_t& st) { st.set_col_upper(2, inf); }},
     change{"set_row_lower", [](state_t& st) { st.set_row_lower(1, -3.); }},
     change{"set_row_lower finite", [](state_t& st) { st.set_row_lower(0, -3.); }},
     change{"set_row_upper", [](state_t& st) { st.set_row_upper(1, 7.); }},
     change{"set_cost", [](state_t& st) { st.set_cost(3, 1.); }},
     change{"add_objective_offset", [](state_t& st) { st.add_objective_offset(1., 0x1p-60); }},
     change{"fix_col", [](state_t& st) { st.fix_col(0, 2.); }},
     change{"fix_col at another value", [](state_t& st) { st.fix_col(0, 3.); }},
     change{"remove_col",
            [](state_t& st) {
              st.delete_row(1);
              st.delete_row(2);
              st.remove_col(3);
            }},
     change{"two rows deleted",
            [](state_t& st) {
              st.delete_row(1);
              st.delete_row(2);
            }},
     change{"delete_row", [](state_t& st) { st.delete_row(2); }},
     change{"shift_row_sides", [](state_t& st) { st.shift_row_sides(1, 1., 1.); }},
     change{"scale_row", [](state_t& st) { st.scale_row(1, 2., false); }},
     change{"set_coefficient", [](state_t& st) { st.set_coefficient(st.find_entry(1, 3), 4.); }},
     change{"delete_entry", [](state_t& st) { st.delete_entry(st.find_entry(1, 3)); }},
     change{"move_entry", [](state_t& st) { st.move_entry(st.find_entry(2, 1), 0, 1.); }},
     change{"record_implied_lower", [](state_t& st) { st.record_implied_lower(1, 0.5, 2); }},
     change{"record_implied_upper", [](state_t& st) { st.record_implied_upper(1, 2.5, 0); }},
     change{"recorded from another row", [](state_t& st) { st.record_implied_upper(1, 2.5, 1); }},
     change{"row list taken",
            [&taken](state_t& st) { st.row_frontier.take(row_list::propagation, taken); }},
     change{"another row list taken",
            [&taken](state_t& st) { st.row_frontier.take(row_list::tightening, taken); }},
     change{"column list taken",
            [&taken](state_t& st) { st.col_frontier.take(col_list::dual_fixing, taken); }},
     change{"row list of a column rule taken",
            [&taken](state_t& st) { st.row_frontier.take(row_list::simple_probing, taken); }},
     change{"column list of a column rule taken",
            [&taken](state_t& st) { st.col_frontier.take(col_list::parallel_cols, taken); }},
     change{"row on the list of a column rule",
            [](state_t& st) { st.row_frontier.push(list_bit(row_list::dual_cols), 1); }},
     change{"column list in another order",
            [&taken](state_t& st) {
              st.col_frontier.take(col_list::dual_fixing, taken);
              for (int32_t col : {3, 2, 1, 0}) {
                st.col_frontier.push(col_event::locks, col);
              }
            }},
     change{"rebuild after a fixing",
            [](state_t& st) {
              st.fix_col(0, 2.);
              st.rebuild();
            }},
     change{"tolerance", [](state_t& st) { st.tol.feasibility = 1e-7; }}});
}

// An operation changes several fields at once (a bound, two activities, a counter), which
// hides a field that the hash leaves out. Here one element of one array changes at a time; the
// result need not be a consistent state.
GPRS_TEST(hash_state_depends_on_every_array)
{
  using change = change_t<state_t>;
  state_t base(hash_model(), tolerances_t());
  std::vector<change> changes = {
    change{"nrows", [](state_t& st) { ++st.nrows; }},
    change{"ncols", [](state_t& st) { ++st.ncols; }},
    change{"row_lower", [](state_t& st) { st.row_lower[1] = -3.; }},
    change{"row_upper", [](state_t& st) { st.row_upper[1] = -3.; }},
    change{"row_alive", [](state_t& st) { st.row_alive[1] = 0; }},
    change{"row_free_size", [](state_t& st) { ++st.row_free_size[1]; }},
    change{"row_version", [](state_t& st) { ++st.row_version[1]; }},
    change{"row_singletons", [](state_t& st) { ++st.row_singletons[1]; }},
    change{"col_lower", [](state_t& st) { st.col_lower[1] = 0.5; }},
    change{"col_upper", [](state_t& st) { st.col_upper[1] = 0.5; }},
    change{"cost", [](state_t& st) { st.cost[1] = 0.5; }},
    change{"col_type", [](state_t& st) { st.col_type[1] = col_type_t::integer; }},
    change{"col_status", [](state_t& st) { st.col_status[1] = col_status_t::fixed; }},
    change{"col_status removed", [](state_t& st) { st.col_status[1] = col_status_t::removed; }},
    change{"down_locks", [](state_t& st) { ++st.down_locks[1]; }},
    change{"up_locks", [](state_t& st) { ++st.up_locks[1]; }},
    change{"implied_lower", [](state_t& st) { st.implied_lower[1] = 0.5; }},
    change{"implied_upper", [](state_t& st) { st.implied_upper[1] = 0.5; }},
    change{"implied_lower_row", [](state_t& st) { st.implied_lower_row[1] = 2; }},
    change{"implied_upper_row", [](state_t& st) { st.implied_upper_row[1] = 2; }},
    change{"implied_lower_version", [](state_t& st) { ++st.implied_lower_version[1]; }},
    change{"implied_upper_version", [](state_t& st) { ++st.implied_upper_version[1]; }},
    change{"objective_offset.hi", [](state_t& st) { st.objective_offset.hi = 0.5; }},
    change{"objective_offset.lo", [](state_t& st) { st.objective_offset.lo = 0.5; }},
    change{"live_rows", [](state_t& st) { ++st.live_rows; }},
    change{"live_cols", [](state_t& st) { ++st.live_cols; }},
    change{"live_nnz", [](state_t& st) { ++st.live_nnz; }},
    change{"fixed_nnz", [](state_t& st) { ++st.fixed_nnz; }},
    change{"bound_changes", [](state_t& st) { ++st.bound_changes; }},
    change{"side_changes", [](state_t& st) { ++st.side_changes; }},
    change{"coefficient_changes", [](state_t& st) { ++st.coefficient_changes; }},
    change{"tol.feasibility", [](state_t& st) { st.tol.feasibility = 0.5; }},
    change{"tol.dual", [](state_t& st) { st.tol.dual = 0.5; }},
    change{"tol.zero", [](state_t& st) { st.tol.zero = 0.5; }},
    change{"tol.integrality", [](state_t& st) { st.tol.integrality = 0.5; }},
    change{"tol.infinite_term", [](state_t& st) { st.tol.infinite_term = 0.5; }},
    change{"tol.activity_error_share", [](state_t& st) { st.tol.activity_error_share = 0.5; }},
    change{"tol.implied_bound_step", [](state_t& st) { st.tol.implied_bound_step = 0.5; }},
    change{"tol.side_error", [](state_t& st) { st.tol.side_error = 0.5; }},
    change{"tol.cancellation", [](state_t& st) { st.tol.cancellation = 0.5; }},
    change{"row lists, one more", [](state_t& st) { st.row_frontier.list.emplace_back(); }},
    change{"column lists, one more", [](state_t& st) { st.col_frontier.list.emplace_back(); }}};

  for (line_store_t state_t::* p : {&state_t::csr, &state_t::csc}) {
    changes.push_back({"start", [p](state_t& st) { ++(st.*p).start[1]; }});
    changes.push_back({"used", [p](state_t& st) { --(st.*p).used[1]; }});
    changes.push_back({"capacity", [p](state_t& st) { ++(st.*p).capacity[1]; }});
    changes.push_back({"size", [p](state_t& st) { --(st.*p).size[1]; }});
    changes.push_back({"index", [p](state_t& st) { (st.*p).index[1] = 3; }});
    changes.push_back({"value", [p](state_t& st) { (st.*p).value[1] = 0.5; }});
    changes.push_back({"twin", [p](state_t& st) { (st.*p).twin[1] = 9; }});
    changes.push_back({"slots", [p](state_t& st) {
                         (st.*p).index.push_back(dead_index);
                         (st.*p).value.push_back(0.);
                         (st.*p).twin.push_back(0);
                       }});
  }
  for (activity_t state_t::* p : {&state_t::min_act, &state_t::max_act}) {
    changes.push_back({"hi", [p](state_t& st) { (st.*p).hi[1] = 0.5; }});
    changes.push_back({"lo", [p](state_t& st) { (st.*p).lo[1] = 0.5; }});
    changes.push_back({"inf_count", [p](state_t& st) { ++(st.*p).inf_count[1]; }});
    changes.push_back({"inf_slot", [p](state_t& st) { ++(st.*p).inf_slot[1]; }});
    changes.push_back({"max_term", [p](state_t& st) { (st.*p).max_term[1] = 0.5; }});
    changes.push_back({"updates", [p](state_t& st) { ++(st.*p).updates[1]; }});
  }
  for (int32_t l = 0; l < row_list::count; ++l) {
    changes.push_back({"row list", [l](state_t& st) { st.row_frontier.list[l].push_back(1); }});
  }
  for (int32_t l = 0; l < col_list::count; ++l) {
    changes.push_back({"column list", [l](state_t& st) { st.col_frontier.list[l].push_back(1); }});
  }
  GPRS_CHECK_EQ(changes.size(), (size_t)(42 + 2 * 8 + 2 * 6 + row_list::count + col_list::count));
  check_changes<state_t>(base, hash_of_state, changes);
}

// Two states with the same entries stored in another order are two states: rules walk a line
// in storage order.
GPRS_TEST(hash_state_depends_on_the_storage_order)
{
  state_t a(hash_model(), tolerances_t());
  state_t b(a);
  int64_t e0 = b.csr.begin(1), e1 = e0 + 1;
  std::swap(b.csr.index[e0], b.csr.index[e1]);
  std::swap(b.csr.value[e0], b.csr.value[e1]);
  std::swap(b.csr.twin[e0], b.csr.twin[e1]);
  b.csc.twin[b.csr.twin[e0]] = e0;
  b.csc.twin[b.csr.twin[e1]] = e1;
  GPRS_CHECK_EQ(b.consistency_error(), std::string());
  GPRS_CHECK(state_hash(a) != state_hash(b));

  // The same within a column.
  state_t c(a);
  int64_t p0 = c.csc.begin(1), p1 = p0 + 1;
  std::swap(c.csc.index[p0], c.csc.index[p1]);
  std::swap(c.csc.value[p0], c.csc.value[p1]);
  std::swap(c.csc.twin[p0], c.csc.twin[p1]);
  c.csr.twin[c.csc.twin[p0]] = p0;
  c.csr.twin[c.csc.twin[p1]] = p1;
  GPRS_CHECK_EQ(c.consistency_error(), std::string());
  GPRS_CHECK(state_hash(a) != state_hash(c));
  GPRS_CHECK(state_hash(b) != state_hash(c));
}

// Value and twin of a tombstone are leftovers of the deleted entry.
GPRS_TEST(hash_state_ignores_the_leftovers_of_tombstones)
{
  state_t a(hash_model(), tolerances_t());
  int64_t slot = a.find_entry(1, 3);
  int64_t twin = a.csr.twin[slot];
  a.delete_entry(slot);
  state_t b(a);
  b.csr.value[slot] = 99.;
  b.csr.twin[slot]  = 5;
  b.csc.value[twin] = -99.;
  b.csc.twin[twin]  = 7;
  GPRS_CHECK_EQ(state_hash(a), state_hash(b));
  // A live entry next to it is hashed.
  b.csr.value[slot - 1] = 99.;
  GPRS_CHECK(state_hash(a) != state_hash(b));
}

GPRS_TEST(hash_postsolve_depends_on_the_records_and_their_order)
{
  using change     = change_t<postsolve_t>;
  postsolve_t base = empty_stack();
  GPRS_CHECK_EQ(postsolve_hash(base), postsolve_hash(empty_stack()));
  GPRS_CHECK_EQ(postsolve_hash(base), postsolve_hash(postsolve_t(base)));
  check_changes<postsolve_t>(
    base,
    hash_of_stack,
    {change{"nrows", [](postsolve_t& ps) { ++ps.nrows; }},
     change{"ncols", [](postsolve_t& ps) { ++ps.ncols; }},
     change{"col_type", [](postsolve_t& ps) { ps.col_type[2] = 1; }},
     change{"row_origin", [](postsolve_t& ps) { ps.row_origin = {0, 2}; }},
     change{"col_origin", [](postsolve_t& ps) { ps.col_origin = {0, 2}; }},
     change{"col_origin, other", [](postsolve_t& ps) { ps.col_origin = {0, 3}; }},
     change{"fixed", [](postsolve_t& ps) { ps.push_fixed_col(0, 2.); }},
     change{"fixed, value", [](postsolve_t& ps) { ps.push_fixed_col(0, 3.); }},
     change{"fixed, value bit", [](postsolve_t& ps) { ps.push_fixed_col(1, next_up(2.)); }},
     change{"fixed, column", [](postsolve_t& ps) { ps.push_fixed_col(1, 2.); }},
     change{"two fixed",
            [](postsolve_t& ps) {
              ps.push_fixed_col(0, 2.);
              ps.push_fixed_col(1, 2.);
            }},
     change{"two fixed, other order",
            [](postsolve_t& ps) {
              ps.push_fixed_col(1, 2.);
              ps.push_fixed_col(0, 2.);
            }},
     change{"alias", [](postsolve_t& ps) { ps.push_alias(1, 2, 0, 2., 1.); }},
     change{"alias, columns swapped", [](postsolve_t& ps) { ps.push_alias(2, 1, 0, 2., 1.); }},
     change{"alias, rep", [](postsolve_t& ps) { ps.push_alias(1, 3, 0, 2., 1.); }},
     change{"alias, column", [](postsolve_t& ps) { ps.push_alias(2, 3, 0, 2., 1.); }},
     change{"alias, row", [](postsolve_t& ps) { ps.push_alias(1, 2, 1, 2., 1.); }},
     change{"alias, scale", [](postsolve_t& ps) { ps.push_alias(1, 2, 0, -2., 1.); }},
     change{"alias, offset", [](postsolve_t& ps) { ps.push_alias(1, 2, 0, 2., 0.); }},
     change{"alias, scale and offset swapped",
            [](postsolve_t& ps) { ps.push_alias(1, 2, 0, 1., 2.); }},
     change{"alias, origin",
            [](postsolve_t& ps) { ps.push_alias(1, 2, 0, 2., 1., alias_origin_t::probing); }},
     change{"alias, origin without a row",
            [](postsolve_t& ps) { ps.push_alias(1, 2, -1, 2., 1., alias_origin_t::probing); }},
     change{"alias, lock",
            [](postsolve_t& ps) { ps.push_alias(1, 2, -1, 2., 1., alias_origin_t::lock); }},
     change{"parallel", [](postsolve_t& ps) { ps.push_parallel_row(2, 0, 0.5); }},
     change{"parallel, rows swapped", [](postsolve_t& ps) { ps.push_parallel_row(0, 2, 0.5); }},
     change{"parallel, kept row", [](postsolve_t& ps) { ps.push_parallel_row(2, 1, 0.5); }},
     change{"parallel, removed row", [](postsolve_t& ps) { ps.push_parallel_row(1, 0, 0.5); }},
     change{"parallel, ratio", [](postsolve_t& ps) { ps.push_parallel_row(2, 0, -0.5); }},
     change{"addition", [](postsolve_t& ps) { ps.push_row_addition(2, 0, 0.5); }},
     change{"addition, rows swapped", [](postsolve_t& ps) { ps.push_row_addition(0, 2, 0.5); }},
     change{"addition, equation", [](postsolve_t& ps) { ps.push_row_addition(2, 1, 0.5); }},
     change{"addition, row", [](postsolve_t& ps) { ps.push_row_addition(1, 0, 0.5); }},
     change{"addition, factor", [](postsolve_t& ps) { ps.push_row_addition(2, 0, -0.5); }},
     change{"parallel, addition",
            [](postsolve_t& ps) {
              ps.push_parallel_row(2, 0, 0.5);
              ps.push_row_addition(2, 0, 0.5);
            }},
     change{"addition, parallel",
            [](postsolve_t& ps) {
              ps.push_row_addition(2, 0, 0.5);
              ps.push_parallel_row(2, 0, 0.5);
            }},
     // The same records per kind, interleaved differently.
     change{"fixed, parallel, fixed",
            [](postsolve_t& ps) {
              ps.push_fixed_col(0, 2.);
              ps.push_parallel_row(2, 0, 0.5);
              ps.push_fixed_col(1, 2.);
            }},
     change{"fixed, fixed, parallel",
            [](postsolve_t& ps) {
              ps.push_fixed_col(0, 2.);
              ps.push_fixed_col(1, 2.);
              ps.push_parallel_row(2, 0, 0.5);
            }},
     change{"alias, fixed",
            [](postsolve_t& ps) {
              ps.push_alias(1, 2, 0, 2., 1.);
              ps.push_fixed_col(2, 2.);
            }},
     change{"fixed, alias", [](postsolve_t& ps) {
              ps.push_fixed_col(0, 2.);
              ps.push_alias(1, 2, 0, 2., 1.);
            }}});
}
