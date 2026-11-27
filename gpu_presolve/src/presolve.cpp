/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/presolve.hpp>

#include <chrono>

namespace gprs {

const char* rule_name(int32_t rule)
{
  static const char* const names[rule::count] = {
    "row_rules", "propagation", "tightening", "doubleton", "parallel_rows", "dual_fixing"};
  gprs_assert(rule >= 0 && rule < rule::count);
  return names[rule];
}

const char* status_name(status_t status)
{
  switch (status) {
    case status_t::reduced: return "reduced";
    case status_t::unchanged: return "unchanged";
    case status_t::reduced_to_empty: return "reduced_to_empty";
    case status_t::infeasible: return "infeasible";
    case status_t::unbounded_or_infeasible: return "unbounded_or_infeasible";
    case status_t::limit: return "limit";
    case status_t::failed: return "failed";
  }
  gprs_assert(false && "status_name: not a status");
  return "";
}

namespace {

// The list of a row family has the family's index.
static_assert(rule::row_rules == row_list::row_rules &&
                rule::propagation == row_list::propagation &&
                rule::tightening == row_list::tightening &&
                rule::doubleton == row_list::doubleton &&
                rule::parallel_rows == row_list::parallel_rows,
              "rule:: and row_list:: number the five row families alike");

using engine_clock_t = std::chrono::steady_clock;

inline double seconds_since(engine_clock_t::time_point start)
{
  return std::chrono::duration<double>(engine_clock_t::now() - start).count();
}

// The start step of presolve(). False when a row or a column admits no value.
bool start_step(state_t& st, postsolve_t& ps)
{
  double feasibility = st.tol.feasibility;
  for (int32_t i = 0; i < st.nrows; ++i) {
    double lower = st.row_lower[i];
    double upper = st.row_upper[i];
    if (lower <= upper) { continue; }
    if (lower - upper > feasibility) { return false; }
    st.set_row_lower(i, upper);
  }
  for (int32_t j = 0; j < st.ncols; ++j) {
    double old_lower = st.col_lower[j];
    double old_upper = st.col_upper[j];
    double lower     = old_lower;
    double upper     = old_upper;
    if (st.is_integer(j)) {
      lower = round_lower_integer(lower, st.tol);
      upper = round_upper_integer(upper, st.tol);
    }
    if (lower > upper) {
      if (lower - upper > feasibility) { return false; }
      lower = upper;
    }
    // The state takes a bound only if lower <= upper holds afterwards, and the new lower bound
    // can lie above the old upper one (old bounds that cross, or [1.2, 1.9999999] of an integer
    // column, which becomes [2, 2]): the lower bound goes below the old upper bound first.
    double first = lower < old_upper ? lower : old_upper;
    if (first != old_lower) { st.set_col_lower(j, first); }
    if (upper != old_upper) { st.set_col_upper(j, upper); }
    if (lower != first) { st.set_col_lower(j, lower); }
    if (lower == upper) { fix_column(st, ps, j, lower); }
  }
  return true;
}

bool same_model(const model_t& a, const model_t& b)
{
  return a.nrows == b.nrows && a.ncols == b.ncols && a.row_start == b.row_start &&
         a.col_index == b.col_index && a.value == b.value && a.row_lower == b.row_lower &&
         a.row_upper == b.row_upper && a.col_lower == b.col_lower && a.col_upper == b.col_upper &&
         a.cost == b.cost && a.col_type == b.col_type && a.objective_offset == b.objective_offset;
}

inline status_t ended_status(rule_outcome_t outcome)
{
  gprs_assert(outcome != rule_outcome_t::ok);
  return outcome == rule_outcome_t::infeasible ? status_t::infeasible
                                               : status_t::unbounded_or_infeasible;
}

struct sizes_t {
  int32_t rows;
  int32_t cols;
  int64_t nnz;
};

struct engine_t {
  const options_t& opt;
  state_t& st;
  result_t& result;
  engine_clock_t::time_point start;

  row_rules_t row_rules;
  bound_propagation_t propagation;
  row_tightening_t tightening;
  doubleton_t doubleton;
  parallel_rows_t parallel_rows;
  dual_fixing_t dual_fixing;

  rule_outcome_t outcome = rule_outcome_t::ok;
  bool out_of_time       = false;
  int64_t round_findings = 0;
  bool pass_stopped      = false;

  engine_t(const options_t& options,
           state_t& state,
           result_t& res,
           engine_clock_t::time_point start_time)
    : opt(options), st(state), result(res), start(start_time)
  {
  }

  sizes_t sizes() const { return sizes_t{st.live_rows, st.live_cols, st.free_nnz()}; }

  bool active(int32_t rule) const
  {
    return opt.enabled[rule] && !(rule == rule::dual_fixing && opt.primal_only);
  }

  bool pending(int32_t rule) const
  {
    if (rule == rule::dual_fixing) { return !st.col_frontier.empty(col_list::dual_fixing); }
    return !st.row_frontier.empty(rule);
  }

  // Books what a family did since `before`. False when its outcome ends the presolve.
  bool book(int32_t rule,
            const rule_result_t& r,
            const sizes_t& before,
            engine_clock_t::time_point begin)
  {
    rule_stats_t& stats = result.stats[rule];
    stats.findings += r.findings;
    stats.seconds += seconds_since(begin);
    if (r.outcome != rule_outcome_t::ok) {
      outcome = r.outcome;
      return false;
    }
    stats.rows_removed += before.rows - st.live_rows;
    stats.cols_removed += before.cols - st.live_cols;
    stats.nnz_removed += before.nnz - st.free_nnz();
    return true;
  }

  // One call of a family that is switched on and has a list. False when the presolve ends here.
  template <typename family_t>
  bool call(int32_t rule, family_t& family)
  {
    if (!active(rule) || !pending(rule)) { return true; }
    if (seconds_since(start) >= opt.time_limit) {
      out_of_time = true;
      return false;
    }
    work_budget_t budget             = make_budget(opt.budget[rule], st);
    sizes_t before                   = sizes();
    engine_clock_t::time_point begin = engine_clock_t::now();
    rule_result_t r                  = family.run(st, budget, result.postsolve, opt);
    ++result.stats[rule].calls;
    result.stats[rule].work += budget.used;
    if (!book(rule, r, before, begin)) { return false; }
    round_findings += r.findings;
    pass_stopped = pass_stopped || r.stopped;
    return true;
  }

  bool cascade_pending() const
  {
    for (int32_t rule : {rule::row_rules, rule::propagation, rule::tightening, rule::dual_fixing}) {
      if (active(rule) && pending(rule)) { return true; }
    }
    return false;
  }

  // False when the presolve ends inside the cascade. Lines are compacted between passes; the
  // state is rebuilt between phases only, a rebuild being O(rows + columns) of the input.
  bool cascade()
  {
    for (int32_t pass = 0; pass < opt.max_cascade_passes; ++pass) {
      pass_stopped = false;
      if (!call(rule::row_rules, row_rules) || !call(rule::propagation, propagation) ||
          !call(rule::tightening, tightening) || !call(rule::dual_fixing, dual_fixing)) {
        return false;
      }
      st.compact_sparse_lines();
      if (pass_stopped || !cascade_pending()) { break; }
    }
    return true;
  }

  void record_phase()
  {
    gprs_assert(st.consistency_error().empty());
    if (!opt.phase_hashes) { return; }
    uint64_t a = state_hash(st);
    uint64_t b = postsolve_hash(result.postsolve);
    result.phase_hash.push_back(a ^ (b + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2)));
  }

  void end_phase()
  {
    if (st.dead_fraction() > opt.rebuild_dead_fraction ||
        st.fixed_nnz > opt.rebuild_fixed_fraction * st.live_nnz) {
      st.rebuild();
      ++result.rebuilds;
    } else {
      st.compact_sparse_lines();
    }
    record_phase();
  }

  void rounds()
  {
    for (int32_t round = 0; round < opt.max_rounds; ++round) {
      ++result.rounds;
      round_findings = 0;
      if (!cascade()) { return; }
      end_phase();
      if (!call(rule::parallel_rows, parallel_rows)) { return; }
      end_phase();
      if (!call(rule::doubleton, doubleton)) { return; }
      end_phase();
      if (round_findings == 0) { return; }
    }
  }

  status_t run(const model_t& model)
  {
    if (!start_step(st, result.postsolve)) { return status_t::infeasible; }
    record_phase();
    rounds();
    if (outcome != rule_outcome_t::ok) { return ended_status(outcome); }

    // Without bound propagation no implied bound is recorded.
    if (opt.enabled[rule::propagation]) {
      sizes_t before                   = sizes();
      engine_clock_t::time_point begin = engine_clock_t::now();
      rule_result_t r                  = propagation.make_explicit(st, result.postsolve, opt);
      if (!book(rule::propagation, r, before, begin)) { return ended_status(outcome); }
    }
    st.rebuild();
    ++result.rebuilds;
    result.reduced = extract_model(st, result.postsolve.row_origin, result.postsolve.col_origin);
    record_phase();

    if (out_of_time) { return status_t::limit; }
    if (same_model(result.reduced, model)) { return status_t::unchanged; }
    if (result.reduced.nrows == 0 && result.reduced.ncols == 0) {
      return status_t::reduced_to_empty;
    }
    return status_t::reduced;
  }
};

}  // namespace

result_t presolve(const model_t& model, const options_t& options)
{
  engine_clock_t::time_point start = engine_clock_t::now();
  result_t result;
  result.postsolve.init(model);
  state_t st(model, options.tol);
  engine_t engine(options, st, result, start);
  result.status  = engine.run(model);
  result.hash    = model_hash(result.reduced);
  result.seconds = seconds_since(start);
  return result;
}

}  // namespace gprs
