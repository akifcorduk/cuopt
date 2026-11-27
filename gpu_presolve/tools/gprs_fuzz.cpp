/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// gprs_fuzz <first seed> <seeds per profile> [--singletons] [--profile <name>]
// Long random runs of the engine, for every rule change (tools/run_campaign.sh splits one run
// over processes). For each seed and each generator profile: one generated model, the presolve
// with options chosen by the seed, and two verdicts.
//   oracle     tests/oracle.hpp, exact by enumeration: status, optimal value, postsolved points.
//   structure  check/structure_check.hpp on a result that carries a reduced model. It is the
//              check that runs on real instances without a solver, and it sees relaxations only.
//              Its summary line ends with the results that have a row addition record
//              (subtracted), the rows that are made of equations as well (added), those tested
//              against the reduced row of such an equation and those not tested, and the
//              largest multiplier of an equation in those tests.
// With the engine as it is, both must be silent. A structure failure that the oracle does not
// share is a false alarm of the structure check (or a relaxation without effect on the optimum);
// an oracle failure that the structure check does not share is, on a result with a reduced model,
// a restriction or an objective error, which the structure check cannot see by design, or a
// relaxation that it missed. The two are told apart on the feasible points of the reduced
// model, all that the solver of the oracle lists and not the sample that the oracle takes: a
// result is relaxed when one of them postsolves to a point that the model rejects.
// --singletons adds columns with one entry to the generated model (rows parallel up to singleton
// columns). GPRS_FUZZ_SHOW=<n> prints the first n models of each kind of disagreement.
// The profiles are those of tests/test_e2e_presolve.cpp and four more.
// Exit code 1 when the oracle or the structure check failed on any model.

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

options_t variant(uint64_t seed)
{
  options_t o;
  uint64_t v = seed % 10, w = seed / 10;
  switch (v) {
    case 3: o.primal_only = true; break;
    case 4: o.enabled[w % rule::count] = false; break;
    case 5:
      for (int32_t r = 0; r < rule::count; ++r) {
        o.enabled[r] = ((w >> r) & 1) || ((w >> (r + 6)) & 1);
      }
      o.primal_only = (w >> 12) & 1;
      break;
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
      break;
    default: break;
  }
  return o;
}

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
                 "usage: gprs_fuzz <first seed> <seeds per profile> [--singletons] "
                 "[--profile <name>]\n");
    return 2;
  }
  uint64_t first   = std::strtoull(argv[1], nullptr, 10);
  uint64_t count   = std::strtoull(argv[2], nullptr, 10);
  bool singletons  = false;
  const char* only = nullptr;
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--singletons")) { singletons = true; }
    if (!std::strcmp(argv[i], "--profile") && i + 1 < argc) { only = argv[++i]; }
  }
  std::map<std::string, int64_t> tally;
  int64_t models = 0, outside = 0, base_outside = 0, failures = 0, reduced = 0, mutated = 0;
  int64_t findings[rule::count] = {0, 0, 0, 0, 0, 0};
  int64_t o_only_carried = 0, carried = 0, s_fail = 0, s_only = 0, s_outside = 0, both = 0,
          o_only = 0, missed_relaxed = 0;
  int64_t rows_tested = 0, kept_tested = 0, kept_parallel = 0, kept_untested = 0;
  int64_t subtracted = 0, added_rows = 0, added_tested = 0, added_untested = 0;
  double worst = 0., multiplier = 0.;
  std::map<std::string, int64_t> parts;
  int64_t show_missed =
    std::getenv("GPRS_FUZZ_SHOW") ? std::atoll(std::getenv("GPRS_FUZZ_SHOW")) : 0;
  for (const profile_t& profile : profiles()) {
    if (only && std::strcmp(only, profile.name)) { continue; }
    for (uint64_t seed = first; seed < first + count; ++seed) {
      case_t c = oracle::generate(seed, profile.options);
      if (singletons) {
        case_t d = add_singletons(c, seed, profile.options);
        mutated += d.model.ncols != c.model.ncols;
        c = d;
      }
      options_t options = variant(seed);
      result_t result   = presolve(c.model, options);
      verdict_t v       = oracle::check_result(c, options, result);
      ++models;
      bool s_bad = false;
      std::string part;
      if (gprs_check::carries_model(result.status)) {
        ++carried;
        gprs_check::structure_report_t s = gprs_check::structure_check(c.model, result);
        rows_tested += s.rows_tested;
        kept_tested += s.kept_tested;
        kept_parallel += s.kept_parallel;
        kept_untested += s.kept_untested;
        subtracted += !result.postsolve.row_additions.empty();
        added_rows += s.added_rows;
        added_tested += s.added_tested;
        added_untested += s.added_untested;
        if (s.added_multiplier > multiplier) { multiplier = s.added_multiplier; }
        if (!s.stack_error.empty()) { part += " stack"; }
        if (s.lost_integers) { part += " types"; }
        if (s.bound_failures) { part += " bounds"; }
        if (s.row_failures) { part += " rows"; }
        if (s.kept_failures) { part += " kept_rows"; }
        if (s.added_failures) { part += " added_rows"; }
        if (s.postsolve_failures) { part += " postsolve"; }
        s_bad = !part.empty();
        double w =
          std::fmax(std::fmax(s.bound_worst, s.row_worst), std::fmax(s.kept_worst, s.added_worst));
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
                "row %g row %d kept %g row %d added %g row %d post %g col %d\n%s\n",
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
  for (const auto& t : parts) {
    std::printf("  part %lld x%s\n", (long long)t.second, t.first.c_str());
  }
  for (const auto& t : tally) {
    std::printf("  %lld x %s\n", (long long)t.second, t.first.c_str());
  }
  return failures != 0 || s_fail != 0;
}
