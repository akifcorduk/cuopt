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
  static const char* const names[rule::count] = {"row_rules",
                                                 "propagation",
                                                 "tightening",
                                                 "doubleton",
                                                 "parallel_rows",
                                                 "dual_fixing",
                                                 "simple_probing",
                                                 "singleton_cols",
                                                 "dual_cols",
                                                 "parallel_cols"};
  gprs_assert(rule >= 0 && rule < rule::count);
  return names[rule];
}

const char* sub_rule_name(int32_t sub)
{
  static const char* const names[sub_rule::count] = {"free_singleton",
                                                     "zero_cost_singleton",
                                                     "forcing_column",
                                                     "free_singleton_dual",
                                                     "parallel_dominance",
                                                     "dual_bound",
                                                     "lock_alias",
                                                     "parallel_merge"};
  gprs_assert(sub >= 0 && sub < sub_rule::count);
  return names[sub];
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

#if defined(GPRS_ASSERT) && GPRS_ASSERT
constexpr bool assertions_on = true;
#else
constexpr bool assertions_on = false;
#endif

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
  simple_probing_t simple_probing;
  singleton_cols_t singleton_cols;
  dual_cols_t dual_cols;
  parallel_cols_t parallel_cols;

  rule_outcome_t outcome = rule_outcome_t::ok;
  bool out_of_time       = false;
  int64_t round_findings = 0;
  bool pass_stopped      = false;
  bool round_stopped     = false;
  // The last round changed nothing, none of its calls stopped on its budget and its cascade
  // left no list.
  bool settled         = false;
  bool explicit_bounds = false;
  // A call left a reduction undone that no list holds (rule_result_t::limited).
  bool limited = false;

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

  // A list of the family holds a row or a column.
  bool pending(int32_t rule) const
  {
    const frontier_set_t& rows = st.row_frontier;
    const frontier_set_t& cols = st.col_frontier;
    switch (rule) {
      case rule::dual_fixing: return !cols.empty(col_list::dual_fixing);
      case rule::simple_probing: return !rows.empty(row_list::simple_probing);
      case rule::singleton_cols:
        return !cols.empty(col_list::singleton_cols) || !rows.empty(row_list::singleton_cols);
      case rule::dual_cols:
        return !cols.empty(col_list::dual_cols) || !rows.empty(row_list::dual_cols);
      case rule::parallel_cols:
        return !cols.empty(col_list::parallel_cols) || !rows.empty(row_list::parallel_cols);
      default: return !rows.empty(rule);
    }
  }

  // Books what a family did since `before`. False when its outcome ends the presolve.
  bool book(int32_t rule,
            const rule_result_t& r,
            const sizes_t& before,
            engine_clock_t::time_point begin)
  {
    rule_stats_t& stats = result.stats[rule];
    stats.findings += r.findings;
    stats.records += r.records;
    stats.seconds += seconds_since(begin);
    gprs_assert(r.dual_findings <= r.findings && (r.dual_findings == 0 || !opt.primal_only));
    result.dual_findings += r.dual_findings;
    for (int32_t sub = 0; sub < sub_rule::count; ++sub) {
      result.sub_findings[sub] += r.sub_findings[sub];
    }
    column_stats_t& columns = result.columns;
    columns.cost_fill += r.columns.cost_fill;
    columns.saved_entries += r.columns.saved_entries;
    if (r.columns.max_growth > columns.max_growth) { columns.max_growth = r.columns.max_growth; }
    if (r.columns.alias_depth > columns.alias_depth) {
      columns.alias_depth = r.columns.alias_depth;
    }
    columns.free_singleton_rejected_explicit += r.columns.free_singleton_rejected_explicit;
    columns.growth_refused += r.columns.growth_refused;
    columns.growth_refused_row += r.columns.growth_refused_row;
    columns.saved_row_refused += r.columns.saved_row_refused;
    columns.probing_rows += r.columns.probing_rows;
    columns.probing_rows_deleted += r.columns.probing_rows_deleted;
    columns.probing_rows_cut += r.columns.probing_rows_cut;
    columns.probing_connected += r.columns.probing_connected;
    columns.probing_unit_refused += r.columns.probing_unit_refused;
    columns.probing_scale_refused += r.columns.probing_scale_refused;
    columns.probing_size_refused += r.columns.probing_size_refused;
    columns.probing_deep_batches += r.columns.probing_deep_batches;
    columns.probing_entry_changes += r.columns.probing_entry_changes;
    if (r.columns.probing_max_scale > columns.probing_max_scale) {
      columns.probing_max_scale = r.columns.probing_max_scale;
    }
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
    if (result.stats[rule].calls == 0) { result.stats[rule].first_work = budget.used; }
    ++result.stats[rule].calls;
    result.stats[rule].work += budget.used;
    if (!book(rule, r, before, begin)) { return false; }
    round_findings += r.findings;
    pass_stopped  = pass_stopped || r.stopped;
    round_stopped = round_stopped || r.stopped;
    limited       = limited || r.limited;
    return true;
  }

  // The family is switched on and its list holds a row or a column.
  bool waiting(int32_t rule) const { return active(rule) && pending(rule); }

  // A column family, which are the families behind dual fixing, is waiting.
  bool column_rules_waiting() const
  {
    for (int32_t rule = rule::simple_probing; rule < rule::count; ++rule) {
      if (waiting(rule)) { return true; }
    }
    return false;
  }

  // One column family as a phase of the round. False when the presolve ends here.
  template <typename family_t>
  bool column_phase(int32_t rule, family_t& family)
  {
    if (!call(rule, family)) { return false; }
    end_phase();
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
      engine_clock_t::time_point begin = engine_clock_t::now();
      st.compact_sparse_lines();
      result.seconds_storage += seconds_since(begin);
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
    engine_clock_t::time_point begin = engine_clock_t::now();
    // A rebuild moves the terms of the fixed columns into the sides, which the rules of the
    // next phase read.
    if ((st.dead_fraction() > opt.rebuild_dead_fraction ||
         st.fixed_nnz > opt.rebuild_fixed_fraction * st.live_nnz) &&
        st.fixed_terms_absorbable()) {
      st.rebuild();
      ++result.rebuilds;
    } else {
      st.compact_sparse_lines();
    }
    record_phase();
    result.seconds_storage += seconds_since(begin);
  }

  // Rounds until one changes nothing or max_rounds, which counts the rounds of the whole
  // presolve, is reached. Returns early, with outcome or out_of_time set, when the presolve ends
  // inside a round.
  void rounds()
  {
    settled = false;
    while (result.rounds < opt.max_rounds) {
      ++result.rounds;
      result.explicit_rounds += explicit_bounds;
      round_findings = 0;
      round_stopped  = false;
      if (!cascade()) { return; }
      bool served = !cascade_pending();
      end_phase();
      if (!call(rule::parallel_rows, parallel_rows)) { return; }
      end_phase();
      if (!call(rule::doubleton, doubleton)) { return; }
      end_phase();
      if (!column_phase(rule::simple_probing, simple_probing) ||
          !column_phase(rule::singleton_cols, singleton_cols) ||
          !column_phase(rule::dual_cols, dual_cols) ||
          !column_phase(rule::parallel_cols, parallel_cols)) {
        return;
      }
      if (round_findings == 0) {
        settled = served && !round_stopped;
        // A row that closed a cycle of doubleton equations is back on its list: the next batch
        // tries it as a tree edge, which this one did not.
        if (!settled || !waiting(rule::doubleton)) { return; }
        settled = false;
      }
    }
  }

  // A row holds the term of a fixed column and a family that reads sides or activities is
  // switched on: such a family can treat the row otherwise than a second presolve, which finds
  // the term in the sides.
  bool fixed_terms_matter() const
  {
    if (st.fixed_nnz == 0) { return false; }
    for (int32_t rule = 0; rule < rule::count; ++rule) {
      if (rule != rule::doubleton && rule != rule::dual_fixing && opt.enabled[rule]) {
        return true;
      }
    }
    return false;
  }

  void absorb_fixed_terms()
  {
    engine_clock_t::time_point begin = engine_clock_t::now();
    st.absorb_fixed_terms();
    result.seconds_storage += seconds_since(begin);
  }

  bool run_start_step()
  {
    engine_clock_t::time_point begin = engine_clock_t::now();
    sizes_t before                   = sizes();
    int64_t changes                  = st.side_changes + st.bound_changes;
    bool feasible                    = start_step(st, result.postsolve);
    if (feasible) { record_phase(); }
    result.seconds_start = seconds_since(begin);
    rule_stats_t& stats  = result.start;
    stats.calls          = 1;
    stats.seconds        = result.seconds_start;
    if (!feasible) { return false; }
    stats.rows_removed = before.rows - st.live_rows;
    stats.cols_removed = before.cols - st.live_cols;
    stats.nnz_removed  = before.nnz - st.free_nnz();
    stats.findings     = st.side_changes + st.bound_changes - changes + stats.cols_removed;
    return true;
  }

  status_t run(const model_t& model)
  {
    if (!run_start_step()) { return status_t::infeasible; }
    rounds();
    if (outcome != rule_outcome_t::ok) { return ended_status(outcome); }

    engine_clock_t::time_point explicit_begin = engine_clock_t::now();
    // The rounds go on only for a new reason: a bound that became explicit, a row that
    // make_explicit listed, the rows that lost the terms of fixed columns. Rounds that ended on
    // a budget or on max_cascade_passes are not taken up again.
    bool renewed = false;
    // Without bound propagation no column is marked and no bound is pending.
    if (opt.enabled[rule::propagation]) {
      sizes_t before  = sizes();
      size_t listed   = st.row_frontier.list[row_list::propagation].size();
      rule_result_t r = propagation.make_explicit(st, result.postsolve, opt);
      singleton_cols.bounds_became_explicit();
      explicit_bounds = true;
      if (!book(rule::propagation, r, before, explicit_begin)) { return ended_status(outcome); }
      renewed = r.findings > 0 || st.row_frontier.list[row_list::propagation].size() > listed;
    }
    while (!out_of_time) {
      if (renewed) {
        rounds();
        if (outcome != rule_outcome_t::ok) { return ended_status(outcome); }
      }
      if (!settled || out_of_time || !fixed_terms_matter() || !st.fixed_terms_absorbable()) {
        break;
      }
      absorb_fixed_terms();
      renewed = cascade_pending() || waiting(rule::parallel_rows) || column_rules_waiting();
    }
    result.seconds_explicit = seconds_since(explicit_begin);
    result.fixed_point      = settled && !limited && !out_of_time && !fixed_terms_matter() &&
                         !cascade_pending() && !waiting(rule::doubleton) && !column_rules_waiting();

    if (opt.enabled[rule::row_rules]) {
      if (result.fixed_point) {
        gprs_assert(no_empty_row());
      } else {
        sizes_t before                   = sizes();
        engine_clock_t::time_point begin = engine_clock_t::now();
        rule_result_t r                  = row_rules.remove_empty_rows(st, result.postsolve, opt);
        if (!book(rule::row_rules, r, before, begin)) { return ended_status(outcome); }
      }
    }

    engine_clock_t::time_point finish_begin = engine_clock_t::now();
    result.reduced = reduced_model(st, result.postsolve.row_origin, result.postsolve.col_origin);
    ++result.rebuilds;
    // The state has no reader from here on but the last phase hash and the assertions.
    if (opt.phase_hashes || assertions_on) {
      st.rebuild();
      gprs_assert(reduced_is_model_of_rebuilt_state());
    }
    record_phase();
    status_t status       = final_status(model);
    result.seconds_finish = seconds_since(finish_begin);
    return status;
  }

  bool no_empty_row() const
  {
    for (int32_t row = 0; row < st.nrows; ++row) {
      if (st.row_alive[row] && st.row_free_size[row] == 0) { return false; }
    }
    return true;
  }

  bool reduced_is_model_of_rebuilt_state() const
  {
    std::vector<int32_t> row_origin, col_origin;
    model_t model = extract_model(st, row_origin, col_origin);
    return model_hash(model) == model_hash(result.reduced) &&
           row_origin == result.postsolve.row_origin && col_origin == result.postsolve.col_origin;
  }

  status_t final_status(const model_t& model) const
  {
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
  result.seconds_state = seconds_since(start);
  engine_t engine(options, st, result, start);
  result.status                         = engine.run(model);
  engine_clock_t::time_point hash_begin = engine_clock_t::now();
  result.hash                           = model_hash(result.reduced);
  result.seconds_hash                   = seconds_since(hash_begin);
  result.seconds                        = seconds_since(start);
  return result;
}

}  // namespace gprs
