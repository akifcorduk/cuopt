"""instrument.py <source of rules_singleton_cols.cpp> <output>: the scratch instrumentation of
the singleton-column family: counters of every refusal by cause, trace lines, and three
measurement overrides read from the environment (diag.hpp). Every replacement must match once."""
import sys

src = open(sys.argv[1]).read()


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old)
    src = src.replace(old, new)


rep('#include "implied.hpp"\n', '#include "diag.hpp"\n#include "implied.hpp"\n')

# The side guard of the family, with the measurement scale.
rep("""  return std::fabs(side) * tol.side_error <= tol.feasibility;
}""", """  return std::fabs(side) * tol.side_error <= tol.feasibility * gprs_diag::data().side_scale;
}

bool family_side_readable(const tolerances_t& tol, double side)
{
  return !std::isinf(side) && free_side_readable(tol, side);
}""")

# range_end: why it fails.
rep("""    row_rest_t rest;
    if (!row_rest(st, row, e, minimum, rest)) { return false; }""",
    """    row_rest_t rest;
    if (!row_rest(st, row, e, minimum, rest)) {
      c.why = 1;
      return false;
    }""")
rep("""    if (activity_fresh(st, act, row)) { return false; }""",
    """    if (activity_fresh(st, act, row)) {
      c.why = 2;
      return false;
    }""")
rep("""  // Reductions of this call.
  int64_t reductions = 0;
""", """  // Reductions of this call.
  int64_t reductions = 0;
  // Why the implied-bound test failed: 1 an infinite term, 2 the error guard, 3 the bound.
  int32_t why = 0;
""")
rep("""    if (!(k.bound + candidate_slack(k) <= upper + implied_tolerance(c.tol, upper))) {
      return false;
    }""", """    if (!(k.bound + candidate_slack(k) <=
          upper + gprs_diag::data().implied_tol_scale * implied_tolerance(c.tol, upper))) {
      c.why  = 3;
      c.miss = k.bound - upper;
      return false;
    }""")
rep("""    if (!(k.bound - candidate_slack(k) >= lower - implied_tolerance(c.tol, lower))) {
      return false;
    }""", """    if (!(k.bound - candidate_slack(k) >=
          lower - gprs_diag::data().implied_tol_scale * implied_tolerance(c.tol, lower))) {
      c.why  = 3;
      c.miss = lower - k.bound;
      return false;
    }""")
rep("""  int32_t why = 0;
""", """  int32_t why = 0;
  double miss = 0.;
""")

# pivot: every refusal by cause.
rep("""  if (!equation) {
    if (cost == 0.) { return false; }""", """  using namespace gprs_diag;
  count(piv_tried);
  auto trace = [&](const char* cause, double x, double y) {
    if (!data().trace) { return; }
    std::fprintf(stderr,
                 "TRACE stage=%d cause=%s row=%d col=%d int=%d a=%.9g cost=%.9g lower=%.9g "
                 "upper=%.9g eq=%d free=%d rowL=%.9g rowU=%.9g largest=%.9g second=%.9g x=%.9g "
                 "y=%.9g\\n",
                 data().stage, cause, row, s.col, (int)st.is_integer(s.col), s.a, cost,
                 st.col_lower[s.col], st.col_upper[s.col], (int)equation, scanned.free_entries,
                 st.row_lower[row], st.row_upper[row], scanned.largest, scanned.second, x, y);
  };
  if (!equation) {
    if (cost == 0.) {
      count(piv_ineq_cost0);
      return false;
    }""")
rep("""  if (!side_readable(c.tol, side)) { return false; }
  double free_side = dd_residual(side, scanned.fixed_terms);
  if (!free_side_readable(c.tol, free_side)) { return false; }
  double q = cost / s.a;
  if (!std::isfinite(q) || !std::isfinite(q * free_side)) { return false; }
  if (!implied_free(c, row, s, side)) {
    c.result.columns.free_singleton_rejected_explicit += c.w.explicit_bounds;
    return false;
  }""", """  if (!family_side_readable(c.tol, side)) {
    count(std::isinf(side) ? piv_side_infinite : piv_side_unreadable);
    trace(std::isinf(side) ? "side_infinite" : "side_unreadable", side, 0.);
    return false;
  }
  double free_side = dd_residual(side, scanned.fixed_terms);
  if (!free_side_readable(c.tol, free_side)) {
    count(piv_free_side_unreadable);
    trace("free_side_unreadable", free_side, 0.);
    return false;
  }
  double q = cost / s.a;
  if (!std::isfinite(q) || !std::isfinite(q * free_side)) {
    count(piv_q_nonfinite);
    return false;
  }
  c.why = 0;
  if (!implied_free(c, row, s, side)) {
    c.result.columns.free_singleton_rejected_explicit += c.w.explicit_bounds;
    count(piv_not_implied);
    if (!equation) {
      // Which end fails: the bound ahead of the move to lower cost, or only the one behind it.
      bool lower_ahead = cost > 0;
      candidate_t k;
      double lower = st.col_lower[s.col], upper = st.col_upper[s.col];
      bool upper_ok = !(upper < inf) ||
                      (range_end(c, row, s.slot, side, s.a > 0, k) &&
                       k.bound + candidate_slack(k) <= upper + implied_tolerance(c.tol, upper));
      bool lower_ok = !(lower > -inf) ||
                      (range_end(c, row, s.slot, side, s.a < 0, k) &&
                       k.bound - candidate_slack(k) >= lower - implied_tolerance(c.tol, lower));
      bool ahead_ok = lower_ahead ? lower_ok : upper_ok;
      count(ahead_ok ? piv_r5_fail_behind_only : piv_r5_fail_ahead);
    }
    count(c.why == 1 ? piv_not_implied_infinite_rest
                     : c.why == 2 ? piv_not_implied_guard : piv_not_implied_bound);
    trace(c.why == 1 ? "not_implied_infinite_rest"
                     : c.why == 2 ? "not_implied_guard" : "not_implied_bound",
          c.miss, c.w.explicit_bounds);
    return false;
  }""")
rep("""    if (!form.integral ||
        !drift_within(c.tol, c.tol.integral_drift_share, s.a, form, drift_units_t::row)) {
      return false;
    }""", """    if (!form.integral) {
      count(piv_not_integral);
      trace("not_integral", free_side, 0.);
      return false;
    }
    if (!drift_within(c.tol, c.tol.integral_drift_share, s.a, form, drift_units_t::row)) {
      count(piv_drift);
      trace("drift", free_side, 0.);
      return false;
    }""")
rep("""  double largest_other = s.slot == scanned.largest_slot ? scanned.second : scanned.largest;
  double ratio         = largest_other / std::fabs(s.a);""",
    """  double largest_other = s.slot == scanned.largest_slot ? scanned.second : scanned.largest;
  if (data().growth_mode & 1) {
    // Measurement only: the largest coefficient of the other continuous columns.
    largest_other = 0.;
    for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
      if (!st.csr.live(e) || e == s.slot) { continue; }
      int32_t other = st.csr.index[e];
      if (!st.is_live(other) || st.is_integer(other)) { continue; }
      largest_other = std::fmax(largest_other, std::fabs(st.csr.value[e]));
    }
  }
  double ratio         = largest_other / std::fabs(s.a);
  // Measurement only: an integer column, which the undo rounds, has no ratio.
  if ((data().growth_mode & 2) && st.is_integer(s.col)) { ratio = 0.; }""")
rep("""    c.result.columns.growth_refused_row += !(c.w.counted[s.col] & 1);
    c.w.counted[s.col] |= 1;
    return false;""", """    c.result.columns.growth_refused_row += !(c.w.counted[s.col] & 1);
    c.w.counted[s.col] |= 1;
    count(piv_growth_row);
    if (data().trace) {
      // The column with the largest other coefficient: its type and range.
      int32_t big = -1;
      double size = 0.;
      int32_t integers = 0, continuous = 0;
      for (int64_t e = st.csr.begin(row); e < st.csr.end(row); ++e) {
        if (!st.csr.live(e) || e == s.slot || !st.is_live(st.csr.index[e])) { continue; }
        (st.is_integer(st.csr.index[e]) ? integers : continuous) += 1;
        if (std::fabs(st.csr.value[e]) > size) {
          size = std::fabs(st.csr.value[e]);
          big  = st.csr.index[e];
        }
      }
      std::fprintf(stderr, "TRACEG big_int=%d big_lower=%.9g big_upper=%.9g others_int=%d "
                   "others_cont=%d\\n", (int)st.is_integer(big), st.col_lower[big],
                   st.col_upper[big], integers, continuous);
    }
    trace("growth_row", ratio, largest_other);
    return false;""")
rep("""    c.w.counted[s.col] |= 2;
    c.w.refused.push_back(s.col);
    return false;""", """    c.w.counted[s.col] |= 2;
    c.w.refused.push_back(s.col);
    count(piv_growth_history);
    trace("growth_history", growth, ratio);
    return false;""")
rep("""  return substitute(c, row, s, free_side, !equation, growth, largest_other);""",
    """  if (!substitute(c, row, s, free_side, !equation, growth, largest_other)) {
    count(piv_cost_nonfinite);
    return false;
  }
  count(equation ? piv_ok_equation : piv_ok_inequality);
  if (c.w.singles.size() > 1) { count(piv_ok_multi); }
  return true;""")

# The cancellation rule of a moved cost, and costs that are small and not zero.
rep("""      if (std::fabs(sum) <= c.tol.cancellation * (std::fabs(old) + std::fabs(moved))) { sum = 0.; }""",
    """      if (std::fabs(sum) <= c.tol.cancellation * (std::fabs(old) + std::fabs(moved))) {
        if (sum != 0.) { gprs_diag::count(gprs_diag::cost_cancel_inexact); }
        sum = 0.;
      } else if (std::fabs(sum) <= 1e-9 * (std::fabs(old) + std::fabs(moved))) {
        gprs_diag::count(gprs_diag::cost_near_cancel);
      }""")
rep("""  if (st.cost[col] != 0.) { return true; }
  double lower = st.col_lower[col], upper = st.col_upper[col];""",
    """  if (st.cost[col] != 0.) {
    if (std::fabs(st.cost[col]) <= 1e-7 &&
        ((st.down_locks[col] == 0 && st.col_lower[col] == -inf && st.up_locks[col] >= 1) ||
         (st.up_locks[col] == 0 && st.col_upper[col] == inf && st.down_locks[col] >= 1))) {
      gprs_diag::count(gprs_diag::fc_small_cost);
    }
    return true;
  }
  double lower = st.col_lower[col], upper = st.col_upper[col];""")

# zero_cost: every refusal by cause.
rep("""    if (st.is_integer(s.col) || st.cost[s.col] != 0.) { continue; }""",
    """    if (st.cost[s.col] != 0.) { continue; }
    if (st.is_integer(s.col)) {
      gprs_diag::count(gprs_diag::zc_col_integer);
      continue;
    }""")
rep("""        (high < inf && !free_side_readable(c.tol, s.a * high))) {
      continue;
    }""", """        (high < inf && !free_side_readable(c.tol, s.a * high))) {
      gprs_diag::count(gprs_diag::zc_col_term_unreadable);
      continue;
    }""")
rep("""  if (taken == 0 || scanned.free_entries == taken) { return; }""",
    """  if (taken == 0) { return; }
  if (scanned.free_entries == taken) {
    gprs_diag::count(gprs_diag::zc_all_taken);
    gprs_diag::count(gprs_diag::zc_all_taken_cols, taken);
    if (gprs_diag::data().trace) {
      std::fprintf(stderr, "TRACE stage=%d cause=zc_all_taken row=%d taken=%d rowL=%.9g rowU=%.9g\\n",
                   gprs_diag::data().stage, row, taken, lower, upper);
    }
    return;
  }""")
rep("""      !free_side_readable(c.tol, free_lower) || !free_side_readable(c.tol, free_upper)) {
    return;
  }""", """      !free_side_readable(c.tol, free_lower) || !free_side_readable(c.tol, free_upper)) {
    gprs_diag::count(gprs_diag::zc_side_unreadable);
    return;
  }""")
rep("""  if (new_lower > new_upper) { return; }""", """  if (new_lower > new_upper) {
    gprs_diag::count(gprs_diag::zc_cross);
    return;
  }""")
rep("""    ++c.result.columns.saved_row_refused;
    c.result.limited = true;
    return;""", """    ++c.result.columns.saved_row_refused;
    c.result.limited = true;
    gprs_diag::count(gprs_diag::zc_saved_limit);
    return;""")
rep("""  c.finding(sub_rule::zero_cost_singleton, taken, false);""",
    """  c.finding(sub_rule::zero_cost_singleton, taken, false);
  gprs_diag::count(gprs_diag::zc_ok);
  gprs_diag::count(gprs_diag::zc_ok_cols, taken);""")

rep("""    if (st.cost[s.col] != 0.) { continue; }
    if (st.is_integer(s.col)) {""", """    if (st.cost[s.col] != 0.) {
      if (std::fabs(st.cost[s.col]) <= 1e-7 && !st.is_integer(s.col)) {
        gprs_diag::count(gprs_diag::zc_col_small_cost);
      }
      continue;
    }
    if (st.is_integer(s.col)) {""")

# visit.
rep("""  c.w.stamp[row] = c.w.pass;
  if (st.row_singletons[row] == 0) { return; }""", """  c.w.stamp[row] = c.w.pass;
  gprs_diag::count(gprs_diag::row_listed);
  if (st.row_singletons[row] == 0) {
    gprs_diag::count(gprs_diag::row_listed_no_singleton);
    return;
  }
  gprs_diag::count(gprs_diag::row_scanned);""")
rep("""  if (scanned.free_entries >= 2 && sub_rule_active(c.opt, sub)) {""",
    """  if (scanned.free_entries < 2) { gprs_diag::count(gprs_diag::row_one_free_entry); }
  int64_t refused_before = gprs_diag::data().c[gprs_diag::data().stage][gprs_diag::piv_growth_row];
  if (!(lower > -inf && upper < inf)) {
    for (const single_t& s : c.w.singles) {
      if (st.cost[s.col] == 0. && !st.is_integer(s.col)) {
        gprs_diag::count(gprs_diag::zc_col_one_sided_row);
      }
    }
  }
  if (scanned.free_entries >= 2 && sub_rule_active(c.opt, sub)) {""")
rep("""    zero_cost(c, row, scanned);
  }""", """    int64_t ok_before = gprs_diag::data().c[gprs_diag::data().stage][gprs_diag::zc_ok];
    zero_cost(c, row, scanned);
    if (gprs_diag::data().c[gprs_diag::data().stage][gprs_diag::zc_ok] > ok_before &&
        gprs_diag::data().c[gprs_diag::data().stage][gprs_diag::piv_growth_row] > refused_before) {
      gprs_diag::count(gprs_diag::piv_refused_then_zero_cost);
    }
  }""")

# Scans of a row that no reduction of this call has followed since its last visit in this call.
rep("""  // By row: the pass over the lists that last visited it.
  std::vector<uint32_t> stamp;""", """  // By row: the pass over the lists that last visited it.
  std::vector<uint32_t> stamp;
  // Measurement: the call that last scanned the row and the reductions of that call then.
  std::vector<uint32_t> scan_call;
  std::vector<int64_t> scan_reductions;
  uint32_t call = 0;""")
rep("""  gprs_diag::count(gprs_diag::row_scanned);""", """  gprs_diag::count(gprs_diag::row_scanned);
  gprs_diag::count(gprs_diag::scan_slots, st.csr.used[row]);
  if (c.w.scan_call.size() != (size_t)st.nrows) {
    c.w.scan_call.assign(st.nrows, 0);
    c.w.scan_reductions.assign(st.nrows, 0);
  }
  if (c.w.scan_call[row] == c.w.call && c.w.scan_reductions[row] == c.reductions) {
    gprs_diag::count(gprs_diag::scan_redundant);
    gprs_diag::count(gprs_diag::scan_redundant_slots, st.csr.used[row]);
  }
  c.w.scan_call[row]       = c.w.call;
  c.w.scan_reductions[row] = c.reductions;""")
rep("""  rule_result_t result;
  context_t c{st, budget, ps, opt, opt.tol, w, result};""", """  rule_result_t result;
  ++w.call;
  context_t c{st, budget, ps, opt, opt.tol, w, result};""")

# forcing column.
rep("""  if (c.reductions > 0 && c.budget.used + slots > c.budget.limit) { return false; }""",
    """  gprs_diag::count(gprs_diag::fc_tested);
  if (c.reductions > 0 && c.budget.used + slots > c.budget.limit) {
    gprs_diag::count(gprs_diag::fc_budget);
    return false;
  }""")
rep("""    usable =
      side_readable(c.tol, side) && free_side_readable(c.tol, upper_side ? copy.upper : copy.lower);""",
    """    usable = family_side_readable(c.tol, side) &&
             free_side_readable(c.tol, upper_side ? copy.upper : copy.lower);""")
rep("""    pool.value.resize(pool_begin);
    return true;
  }
  values.resize(kept);""", """    pool.value.resize(pool_begin);
    gprs_diag::count(gprs_diag::fc_unusable);
    return true;
  }
  gprs_diag::count(gprs_diag::fc_ok);
  values.resize(kept);""")

# run.
rep("""    ++w.pass;
    size_t k = 0, r = 0;""", """    ++w.pass;
    gprs_diag::count(gprs_diag::passes);
    size_t k = 0, r = 0;""")
rep("""  return result;
}

}  // namespace gprs""", """  gprs_diag::count(gprs_diag::calls);
  gprs_diag::count(gprs_diag::calls_stopped, result.stopped);
  gprs_diag::count(gprs_diag::calls_limited, result.limited);
  if (gprs_diag::data().stage == 1) { gprs_diag::data().stage = 2; }
  return result;
}

}  // namespace gprs""")

open(sys.argv[2], "w").write(src)
