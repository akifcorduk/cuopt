/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// gprs_fuzz <first seed> <seeds per profile> [--singletons] [--columns] [--profile <name>]
// Long random runs of the engine, for every rule change (tools/run_campaign.sh splits one run
// over processes). For each seed and each generator profile: one generated model, the presolve
// with options chosen by the seed, and three verdicts.
//   oracle     tests/oracle.hpp, exact by enumeration: status, optimal value, postsolved points,
//              and crushed points of the base model, which must be feasible for the reduced
//              model and cost the same there, or no more when the result has a dual finding.
//              The line "crush" counts the results where that test ran and those among them
//              with a dual finding.
//   second     On one seed in four: when the result is a fixed point (result_t::fixed_point),
//              the presolve of its reduced model returns unchanged.
//   structure  check/structure_check.hpp on a result that carries a reduced model. It is the
//              check that runs on real instances without a solver, and it sees relaxations
//              and, with its objective part, a cost or an offset that a reduction lost.
//              Its summary line ends with the results that have a row addition record
//              (subtracted), the rows that are made of equations as well (added), those tested
//              against the reduced row of such an equation and those not tested, and the
//              largest multiplier of an equation in those tests.
//   The line "columns" counts what the singleton-column rules reached: the results with a
//   finding of each of their reductions and the findings, the results with a col_from_rows
//   record of each kind (defined, defined with the dual flag, interval, forcing), the box ends
//   and snapshots that the structure check tested, and the results that the family left as
//   limited.
// With the engine as it is, all three must be silent. A structure failure that the oracle does
// not share is a false alarm of the structure check (or a relaxation without effect on the
// optimum); an oracle failure that the structure check does not share is, on a result with a
// reduced model, a restriction, which the structure check cannot see by design, or a
// relaxation or an objective error that it missed. The two are told apart on the feasible
// points of the reduced model, all that the solver of the oracle lists and not the sample that
// the oracle takes: a result is relaxed when one of them postsolves to a point that the model
// rejects.
// --singletons adds columns with one entry to the generated model (rows parallel up to singleton
// columns). --columns makes the generator draw what the singleton-column rules act on: up to
// two singleton columns per row, most of them line columns and half of them without a cost,
// and forcing columns (oracle.hpp). GPRS_FUZZ_SHOW=<n> prints the first n models of each kind
// of disagreement. GPRS_FUZZ_OFF=<rule family> switches that family off for every seed, on top
// of the options of the seed: with a family off the run can be compared, line by line, with
// that of an engine that does not have it.
// The profiles are those of tests/test_e2e_presolve.cpp and four more.
// Exit code 1 when the oracle, the second presolve or the structure check failed on any model.

#include "../check/structure_check.hpp"
#include "../tests/oracle.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace gprs;
using oracle::case_t;
using oracle::generator_options_t;
using oracle::verdict_t;

namespace {

struct profile_t {
  const char* name;
  generator_options_t options;
};

std::vector<profile_t> profiles()
{
  std::vector<profile_t> list;
  generator_options_t o;
  list.push_back({"default", o});

  o               = generator_options_t();
  o.max_line_cols = 0;
  o.max_enum_cols = 6;
  o.max_rows      = 6;
  list.push_back({"integer", o});

  o                      = generator_options_t();
  o.p_doubleton_equation = 0.45;
  o.p_equation           = 0.4;
  o.p_singleton_row      = 0.05;
  o.p_empty_row          = 0.02;
  o.p_zero_cost          = 0.5;
  list.push_back({"equations", o});

  o                = generator_options_t();
  o.p_parallel_row = 0.5;
  o.max_rows       = 7;
  list.push_back({"parallel", o});

  o          = generator_options_t();
  o.p_scaled = 1.;
  list.push_back({"scaled", o});

  o                  = generator_options_t();
  o.max_line_cols    = 3;
  o.p_infinite_bound = 0.7;
  o.p_huge_bound     = 0.6;
  o.p_zero_cost      = 0.5;
  list.push_back({"open_ends", o});

  o                 = generator_options_t();
  o.density         = 0.9;
  o.max_rows        = 8;
  o.max_enum_cols   = 4;
  o.max_width       = 2;
  o.p_singleton_row = 0.05;
  o.p_empty_row     = 0.02;
  o.p_forcing_side  = 0.3;
  list.push_back({"dense", o});

  o                 = generator_options_t();
  o.max_width       = 1;
  o.max_side        = 6;
  o.max_coefficient = 4;
  o.max_line_cols   = 1;
  o.p_common_factor = 0.4;
  o.p_fixed_col     = 0.05;
  list.push_back({"binary", o});

  // Not in the test file: parallel rows among equations, and continuous columns with finite
  // bounds in rows of binaries (implied bounds, coefficient strengthening on them).
  o                 = generator_options_t();
  o.p_parallel_row  = 0.6;
  o.p_equation      = 0.5;
  o.max_rows        = 6;
  o.max_enum_cols   = 4;
  o.max_line_cols   = 1;
  o.max_width       = 1;
  o.p_singleton_row = 0.03;
  o.p_empty_row     = 0.01;
  list.push_back({"parallel_equations", o});

  o                  = generator_options_t();
  o.max_width        = 1;
  o.max_line_cols    = 3;
  o.p_continuous     = 1.;
  o.p_infinite_bound = 0.25;
  o.p_huge_bound     = 0.1;
  o.max_rows         = 7;
  o.p_singleton_row  = 0.05;
  o.p_empty_row      = 0.01;
  o.p_forcing_side   = 0.25;
  list.push_back({"bounded_continuous", o});

  // Huge finite bounds on most line columns, with parallel rows and equations: columns fixed at
  // a huge bound stay in their rows as huge constants.
  o                  = generator_options_t();
  o.max_line_cols    = 3;
  o.max_enum_cols    = 4;
  o.p_infinite_bound = 0.2;
  o.p_huge_bound     = 0.9;
  o.p_parallel_row   = 0.45;
  o.p_zero_cost      = 0.6;
  o.max_rows         = 6;
  list.push_back({"huge_parallel", o});

  o                      = generator_options_t();
  o.max_line_cols        = 3;
  o.max_enum_cols        = 4;
  o.p_infinite_bound     = 0.2;
  o.p_huge_bound         = 0.9;
  o.p_equation           = 0.45;
  o.p_doubleton_equation = 0.3;
  o.p_parallel_row       = 0.2;
  o.p_zero_cost          = 0.6;
  o.max_rows             = 6;
  list.push_back({"huge_equations", o});
  return list;
}

// Up to three new columns with one entry each, at the end of rows that have entries. A row that
// has a line column gets a narrow integer column; any other row a narrow integer column or a
// line column.
case_t add_singletons(const case_t& c, uint64_t seed, const generator_options_t& g)
{
  oracle::rng_t rng(seed * 0x9e3779b97f4a7c15ull + 12345);
  const model_t& b = c.base;
  std::vector<uint8_t> line;
  if (!oracle::classify(b, g.max_width, line)) { return c; }
  model_t m;
  m.nrows            = b.nrows;
  m.ncols            = b.ncols;
  m.row_lower        = b.row_lower;
  m.row_upper        = b.row_upper;
  m.col_lower        = b.col_lower;
  m.col_upper        = b.col_upper;
  m.cost             = b.cost;
  m.col_type         = b.col_type;
  m.objective_offset = b.objective_offset;
  m.row_start.push_back(0);
  std::vector<double> col_scale = c.col_scale;
  int32_t added                 = 0;
  for (int32_t i = 0; i < b.nrows; ++i) {
    bool has_line = false;
    for (int64_t e = b.row_start[i]; e < b.row_start[i + 1]; ++e) {
      m.col_index.push_back(b.col_index[e]);
      m.value.push_back(b.value[e]);
      has_line = has_line || line[b.col_index[e]];
    }
    if (b.row_start[i + 1] > b.row_start[i] && added < 3 && rng.chance(0.6)) {
      ++added;
      int32_t a = rng.uniform(1, g.max_coefficient);
      if (rng.chance(0.5)) { a = -a; }
      m.col_index.push_back(m.ncols++);
      m.value.push_back(a);
      double cost = rng.chance(0.4) ? 0. : rng.uniform(-3, 3);
      m.cost.push_back(cost);
      if (!has_line && rng.chance(0.4)) {
        bool integer = rng.chance(0.3);
        double lower = rng.chance(0.3) ? -INFINITY : rng.uniform(-3, 3);
        double upper = rng.chance(0.4)
                         ? INFINITY
                         : (std::isinf(lower) ? 0. : lower) + g.max_width + 1 + rng.uniform(0, 6);
        m.col_type.push_back(integer ? 1 : 0);
        m.col_lower.push_back(lower);
        m.col_upper.push_back(upper);
      } else {
        double lower = rng.uniform(-2, 2);
        m.col_type.push_back(1);
        m.col_lower.push_back(lower);
        m.col_upper.push_back(lower + rng.uniform(1, g.max_width < 2 ? 1 : 2));
      }
      col_scale.push_back(1.);
    }
    m.row_start.push_back(m.col_index.size());
  }
  if (added == 0 || !well_formed(m)) { return c; }
  return oracle::make_case(m, c.row_scale, col_scale);
}

// --columns: the knobs of the generator that draw singleton columns and forcing columns.
generator_options_t with_columns(generator_options_t o)
{
  o.p_singleton_col       = std::fmax(o.p_singleton_col, 0.5);
  o.p_singleton_line      = 0.7;
  o.p_singleton_zero_cost = 0.5;
  o.max_singleton_cols    = 4;
  o.max_row_singletons    = 2;
  o.p_forcing_col         = 0.5;
  return o;
}

// The options of a seed: three seeds in ten keep the defaults; one each has primal-only mode;
// one family or one reduction of the column rules switched off; a random subset of the families
// and of those reductions (each on with probability 3/4), a random mode and a random setting of
// merge_integers; a rebuild after every phase; no rebuild; tiny budgets; few rounds and passes,
// half of them with the two limits of the singleton-column rules at their tightest.
options_t variant(uint64_t seed)
{
  options_t o;
  uint64_t v = seed % 10, w = seed / 10;
  switch (v) {
    case 3: o.primal_only = true; break;
    case 4: {
      uint64_t pick = w % (rule::count + sub_rule::count);
      if (pick < rule::count) {
        o.enabled[pick] = false;
      } else {
        o.sub_enabled[pick - rule::count] = false;
      }
      break;
    }
    case 5: {
      oracle::rng_t rng(w);
      for (int32_t r = 0; r < rule::count; ++r) {
        o.enabled[r] = rng.chance(0.75);
      }
      for (int32_t s = 0; s < sub_rule::count; ++s) {
        o.sub_enabled[s] = rng.chance(0.75);
      }
      o.primal_only    = rng.chance(0.5);
      o.merge_integers = (merge_integers_t)rng.uniform(0, 2);
      break;
    }
    case 6:
      o.rebuild_dead_fraction  = 0.;
      o.rebuild_fixed_fraction = 0.;
      break;
    case 7:
      o.rebuild_dead_fraction  = 2.;
      o.rebuild_fixed_fraction = 2.;
      break;
    case 8:
      for (budget_t& b : o.budget) {
        b = (w & 1) ? budget_t{0.25, 0.125, 0} : budget_t{0., 0., 20};
      }
      break;
    case 9:
      o.max_rounds            = 1 + (w & 1);
      o.max_cascade_passes    = 1 + ((w >> 1) & 1);
      o.parallel_bucket_limit = 1;
      if ((w >> 2) & 1) {
        o.pivot_growth_limit = 1.;
        o.saved_row_factor   = 0.;
      }
      break;
    default: break;
  }
  return o;
}

// One seed in four, whatever its variant is: the reduced model of its result is presolved again.
bool sampled(uint64_t seed) { return ((seed * 0x9e3779b97f4a7c15ull) >> 62) == 0; }

// A feasible point of the reduced model, or its far point, postsolves to a point that the model
// rejects. The stack keeps its contract. False for a reduced model outside the oracle class.
bool relaxed(const case_t& c, const options_t& options, const result_t& result)
{
  std::vector<uint8_t> line;
  model_t in_class;
  if (!well_formed(result.reduced) ||
      !oracle::into_class(result.reduced, oracle::check_max_assignments, in_class, line)) {
    return false;
  }
  oracle::solution_t red = oracle::solve(in_class, line, (int32_t)oracle::check_max_assignments);
  std::vector<double> x;
  auto rejected = [&](const std::vector<double>& point) {
    result.postsolve.postsolve(point, x);
    return !point_feasible(check_point(c.model, x), options.tol);
  };
  if (red.outcome == oracle::outcome_t::optimal && rejected(red.x)) { return true; }
  for (size_t k = 0; k < red.feasible.size(); ++k) {
    if (rejected(red.feasible[k]) || rejected(red.far[k])) { return true; }
  }
  return false;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: gprs_fuzz <first seed> <seeds per profile> [--singletons] [--columns] "
                 "[--profile <name>]\n");
    return 2;
  }
  uint64_t first   = std::strtoull(argv[1], nullptr, 10);
  uint64_t count   = std::strtoull(argv[2], nullptr, 10);
  bool singletons  = false;
  bool columns     = false;
  const char* only = nullptr;
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--singletons")) { singletons = true; }
    if (!std::strcmp(argv[i], "--columns")) { columns = true; }
    if (!std::strcmp(argv[i], "--profile") && i + 1 < argc) { only = argv[++i]; }
  }
  std::map<std::string, int64_t> tally;
  int64_t models = 0, outside = 0, base_outside = 0, failures = 0, reduced = 0, mutated = 0;
  int64_t findings[rule::count] = {};
  int64_t dual_findings         = 0;
  int64_t crush_results = 0, crush_dual = 0, crush_points = 0, crush_skipped = 0;
  int64_t second_runs = 0, second_changed = 0;
  int64_t o_only_carried = 0, carried = 0, s_fail = 0, s_only = 0, s_outside = 0, both = 0,
          o_only = 0, missed_relaxed = 0;
  int64_t rows_tested = 0, kept_tested = 0, kept_parallel = 0, kept_untested = 0;
  int64_t subtracted = 0, added_rows = 0, added_tested = 0, added_untested = 0;
  double worst = 0., multiplier = 0.;
  // By reduction of the singleton-column rules: results with a finding, and findings.
  const int32_t reductions[4] = {sub_rule::free_singleton,
                                 sub_rule::free_singleton_dual,
                                 sub_rule::zero_cost_singleton,
                                 sub_rule::forcing_column};
  int64_t sub_results[4] = {}, sub_found[4] = {};
  // Results with a record of the kind: defined, defined with the dual flag, interval, forcing.
  int64_t kind_results[4] = {};
  int64_t box_tested = 0, snapshot_tested = 0, snapshot_untested = 0, limited_results = 0;
  std::map<std::string, int64_t> parts;
  int64_t show_missed =
    std::getenv("GPRS_FUZZ_SHOW") ? std::atoll(std::getenv("GPRS_FUZZ_SHOW")) : 0;
  int32_t off = -1;
  if (const char* name = std::getenv("GPRS_FUZZ_OFF")) {
    for (int32_t r = 0; r < rule::count; ++r) {
      if (!std::strcmp(name, rule_name(r))) { off = r; }
    }
    if (off < 0) {
      std::fprintf(stderr, "GPRS_FUZZ_OFF: %s is no rule family\n", name);
      return 2;
    }
  }
  for (const profile_t& profile : profiles()) {
    if (only && std::strcmp(only, profile.name)) { continue; }
    generator_options_t generator = columns ? with_columns(profile.options) : profile.options;
    for (uint64_t seed = first; seed < first + count; ++seed) {
      case_t c = oracle::generate(seed, generator);
      if (singletons) {
        case_t d = add_singletons(c, seed, profile.options);
        mutated += d.model.ncols != c.model.ncols;
        c = d;
      }
      options_t options = variant(seed);
      if (off >= 0) { options.enabled[off] = false; }
      result_t result = presolve(c.model, options);
      verdict_t v     = oracle::check_result(c, options, result);
      ++models;
      dual_findings += result.dual_findings;
      for (int32_t k = 0; k < 4; ++k) {
        sub_results[k] += result.sub_findings[reductions[k]] > 0;
        sub_found[k] += result.sub_findings[reductions[k]];
      }
      bool has_kind[4] = {};
      for (const col_from_rows_record_t& r : result.postsolve.col_from_rows) {
        has_kind[r.kind == col_from_rows_kind_t::defined    ? (r.dual ? 1 : 0)
                 : r.kind == col_from_rows_kind_t::interval ? 2
                                                            : 3] = true;
      }
      for (int32_t k = 0; k < 4; ++k) {
        kind_results[k] += has_kind[k];
      }
      limited_results += result.columns.growth_refused + result.columns.saved_row_refused > 0;
      if (v.ok && v.crushed_points > 0) {
        ++crush_results;
        crush_dual += result.dual_findings > 0;
        crush_points += v.crushed_points;
      }
      crush_skipped += v.skipped_points;
      if (result.fixed_point && gprs_check::carries_model(result.status) && sampled(seed)) {
        ++second_runs;
        if (presolve(result.reduced, options).status != status_t::unchanged) {
          if (++second_changed <= show_missed + 3) {
            std::printf(
              "SECOND profile %s seed %llu variant %llu: the presolve of the reduced "
              "model of a fixed point changes it\n%s\n",
              profile.name,
              (unsigned long long)seed,
              (unsigned long long)(seed % 10),
              second_changed <= show_missed ? oracle::to_code(c).c_str() : "");
          }
        }
      }
      bool s_bad = false;
      std::string part;
      if (gprs_check::carries_model(result.status)) {
        ++carried;
        gprs_check::structure_report_t s =
          gprs_check::structure_check(c.model, result, options.primal_only);
        rows_tested += s.rows_tested;
        kept_tested += s.kept_tested;
        kept_parallel += s.kept_parallel;
        kept_untested += s.kept_untested;
        subtracted += !result.postsolve.row_additions.empty();
        added_rows += s.added_rows;
        added_tested += s.added_tested;
        added_untested += s.added_untested;
        if (s.added_multiplier > multiplier) { multiplier = s.added_multiplier; }
        box_tested += s.box_tested;
        snapshot_tested += s.snapshot_tested;
        snapshot_untested += s.snapshot_untested;
        if (!s.stack_error.empty()) { part += " stack"; }
        if (s.lost_integers) { part += " types"; }
        if (s.bound_failures) { part += " bounds"; }
        if (s.row_failures) { part += " rows"; }
        if (s.kept_failures) { part += " kept_rows"; }
        if (s.added_failures) { part += " added_rows"; }
        if (s.box_failures || s.integral_failures || s.snapshot_failures) { part += " columns"; }
        if (s.objective_failures) { part += " objective"; }
        if (s.postsolve_failures) { part += " postsolve"; }
        s_bad = !part.empty();
        double w =
          std::fmax(std::fmax(s.bound_worst, s.row_worst), std::fmax(s.kept_worst, s.added_worst));
        w = std::fmax(w, std::fmax(s.box_worst, s.snapshot_worst));
        if (!s_bad && w > worst) { worst = w; }
        if (s_bad) {
          ++s_fail;
          ++parts[part];
          bool o_bad = !v.ok && v.message.find("outside the oracle class") == std::string::npos;
          bool o_out = !v.ok && !o_bad;
          both += o_bad;
          s_outside += o_out;
          if (v.ok) {
            if (++s_only <= show_missed) {
              std::printf(
                "STRUCTURE-ONLY profile %s seed %llu variant %llu:%s | %s | bound %g col %d "
                "row %g row %d kept %g row %d added %g row %d box %g col %d integral %lld col %d "
                "snapshot %g row %d objective %g col %d constant %g "
                "post %g col %d\n%s\n",
                profile.name,
                (unsigned long long)seed,
                (unsigned long long)(seed % 10),
                part.c_str(),
                s.stack_error.c_str(),
                s.bound_worst,
                s.bound_worst_col,
                s.row_worst,
                s.row_worst_row,
                s.kept_worst,
                s.kept_worst_row,
                s.added_worst,
                s.added_worst_row,
                s.box_worst,
                s.box_worst_col,
                (long long)s.integral_failures,
                s.first_integral_col,
                s.snapshot_worst,
                s.snapshot_worst_row,
                s.objective_worst,
                s.objective_worst_col,
                s.objective_constant,
                s.postsolve_worst,
                s.postsolve_worst_col,
                oracle::to_code(c).c_str());
            }
          }
        }
      }
      if (!s_bad && !v.ok && v.message.find("outside the oracle class") == std::string::npos) {
        ++o_only;
        if (gprs_check::carries_model(result.status)) {
          ++o_only_carried;
          bool is_relaxed = relaxed(c, options, result);
          missed_relaxed += is_relaxed;
          if (is_relaxed ? missed_relaxed <= show_missed : o_only_carried <= show_missed) {
            std::printf("%s profile %s seed %llu variant %llu status %s: %s\n%s\n",
                        is_relaxed ? "MISSED-RELAXED" : "MISSED",
                        profile.name,
                        (unsigned long long)seed,
                        (unsigned long long)(seed % 10),
                        status_name(result.status),
                        v.message.c_str(),
                        oracle::to_code(c).c_str());
          }
        }
      }
      reduced += result.status == status_t::reduced || result.status == status_t::reduced_to_empty;
      for (int32_t r = 0; r < rule::count; ++r) {
        findings[r] += result.stats[r].findings;
      }
      if (v.ok) { continue; }
      if (v.message == "the reduced model is outside the oracle class") {
        ++outside;
        continue;
      }
      if (v.message == "the base model is outside the oracle class") {
        ++base_outside;
        continue;
      }
      ++failures;
      ++tally[v.message.substr(0, 48)];
      if (failures <= 6) {
        std::printf("FAIL profile %s seed %llu variant %llu: %s\n%s\n",
                    profile.name,
                    (unsigned long long)seed,
                    (unsigned long long)(seed % 10),
                    v.message.c_str(),
                    failures <= 2 ? oracle::to_code(c).c_str() : "");
      }
    }
  }
  std::printf(
    "models %lld mutated %lld reduced %lld failures %lld reduced-outside-class %lld "
    "base-outside-class %lld | findings",
    (long long)models,
    (long long)mutated,
    (long long)reduced,
    (long long)failures,
    (long long)outside,
    (long long)base_outside);
  for (int32_t r = 0; r < rule::count; ++r) {
    std::printf(" %s %lld", rule_name(r), (long long)findings[r]);
  }
  std::printf("\n");
  // carried: results with a reduced model; alone: failed while the oracle passed; missed: the
  // oracle failed and the structure check passed, with and without a reduced model; worst: the
  // largest violation that passed; undecided: failed on a model the oracle cannot decide;
  // missed_relaxed: of the missed with a reduced model, the relaxed ones.
  std::printf(
    "structure carried %lld failed %lld alone %lld missed %lld missed_no_model %lld "
    "worst %g both %lld undecided %lld rows %lld kept %lld parallel %lld untested %lld "
    "subtracted %lld added %lld added_tested %lld added_untested %lld multiplier %g "
    "missed_relaxed %lld\n",
    (long long)carried,
    (long long)s_fail,
    (long long)s_only,
    (long long)o_only_carried,
    (long long)(o_only - o_only_carried),
    worst,
    (long long)both,
    (long long)s_outside,
    (long long)rows_tested,
    (long long)kept_tested,
    (long long)kept_parallel,
    (long long)kept_untested,
    (long long)subtracted,
    (long long)added_rows,
    (long long)added_tested,
    (long long)added_untested,
    multiplier,
    (long long)missed_relaxed);
  // results: those whose crush test ran and passed; with_dual: of them, the results with a dual
  // finding, where a crushed point need only cost no more; points: base points crushed;
  // skipped: base points that the scaled model rejects. second: presolves of the reduced model
  // of a sampled fixed point; changed: those that did not return unchanged.
  std::printf(
    "crush results %lld with_dual %lld points %lld skipped %lld dual_findings %lld "
    "second %lld changed %lld\n",
    (long long)crush_results,
    (long long)crush_dual,
    (long long)crush_points,
    (long long)crush_skipped,
    (long long)dual_findings,
    (long long)second_runs,
    (long long)second_changed);
  // Per reduction of the singleton-column rules the results with a finding and the findings;
  // the results with a record of each kind; what the structure check tested of them; the
  // results in which a limit of the family refused a reduction.
  std::printf(
    "columns free_singleton %lld %lld free_singleton_dual %lld %lld zero_cost_singleton %lld "
    "%lld forcing_column %lld %lld | records defined %lld defined_dual %lld interval %lld "
    "forcing %lld | box_tested %lld snapshot_tested %lld snapshot_untested %lld limited %lld\n",
    (long long)sub_results[0],
    (long long)sub_found[0],
    (long long)sub_results[1],
    (long long)sub_found[1],
    (long long)sub_results[2],
    (long long)sub_found[2],
    (long long)sub_results[3],
    (long long)sub_found[3],
    (long long)kind_results[0],
    (long long)kind_results[1],
    (long long)kind_results[2],
    (long long)kind_results[3],
    (long long)box_tested,
    (long long)snapshot_tested,
    (long long)snapshot_untested,
    (long long)limited_results);
  for (const auto& t : parts) {
    std::printf("  part %lld x%s\n", (long long)t.second, t.first.c_str());
  }
  for (const auto& t : tally) {
    std::printf("  %lld x %s\n", (long long)t.second, t.first.c_str());
  }
  return failures != 0 || s_fail != 0 || second_changed != 0;
}
