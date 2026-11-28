// Attack on the two new tests with the right engine: generated models whose costs and
// objective offset are scaled far from the generator's small integers. The oracle (with its
// crush test on every result) and the structure check (with its objective part) must stay
// silent.
//
//   costfuzz <first seed> <seeds per profile> [--singletons]

#include "ident/profiles.hpp"
#include "structure_check.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

using namespace gprs;
using oracle::case_t;

namespace {

case_t rescaled(const case_t& c, uint64_t seed)
{
  oracle::rng_t rng(seed * 0x2545f4914f6cdd1dull + 99);
  model_t base   = c.base;
  int32_t kind   = seed % 6;
  double uniform = 1.;
  double offset  = 0.;
  if (kind == 0) {
    uniform = std::ldexp(1., 30);
    offset  = 0.5;
  } else if (kind == 1) {
    uniform = std::ldexp(1., -20);
    offset  = -1048576.;
  } else if (kind == 2) {
    offset = 1073741824.;
  } else if (kind == 3) {
    offset = 1099511627776.;
  } else if (kind == 4) {
    uniform = std::ldexp(1., 50);
    offset  = -3.;
  }
  for (int32_t j = 0; j < base.ncols; ++j) {
    double factor = uniform;
    if (kind == 2) { factor = std::ldexp(1., 10 * rng.uniform(0, 2)); }
    if (kind == 5) { factor = std::ldexp(1., 10 * rng.uniform(-1, 1)); }
    base.cost[j] *= factor;
  }
  base.objective_offset += offset;
  return oracle::make_case(base, c.row_scale, c.col_scale);
}

options_t variant(uint64_t seed)
{
  options_t o;
  uint64_t v = (seed / 6) % 5;
  if (v == 1) { o.primal_only = true; }
  if (v == 2) {
    o.rebuild_dead_fraction  = 0.;
    o.rebuild_fixed_fraction = 0.;
  }
  if (v == 3) {
    for (budget_t& b : o.budget) {
      b = budget_t{0.25, 0.125, 0};
    }
  }
  if (v == 4) { o.enabled[rule::dual_fixing] = false; }
  return o;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 3) { return 2; }
  uint64_t first  = std::strtoull(argv[1], nullptr, 10);
  uint64_t count  = std::strtoull(argv[2], nullptr, 10);
  bool singletons = argc > 3 && !std::strcmp(argv[3], "--singletons");
  bool columns    = argc > 3 && !std::strcmp(argv[3], "--columns");
  int64_t models = 0, oracle_failures = 0, structure_failures = 0, objective_failures = 0;
  int64_t crush_results = 0, crush_dual = 0, carried = 0, shown = 0;
  double worst_coefficient = 0., worst_constant = 0.;
  std::map<std::string, int64_t> tally;
  for (const ident::profile_t& profile : ident::profiles()) {
    for (uint64_t seed = first; seed < first + count; ++seed) {
      oracle::generator_options_t g = profile.options;
      if (columns) {
        g.p_singleton_col       = std::fmax(g.p_singleton_col, 0.5);
        g.p_singleton_line      = 0.7;
        g.p_singleton_zero_cost = 0.5;
        g.max_singleton_cols    = 4;
        g.max_row_singletons    = 2;
        g.p_forcing_col         = 0.5;
      }
      case_t c = oracle::generate(seed, g);
      if (singletons) { c = ident::add_singletons(c, seed, profile.options); }
      c = rescaled(c, seed);
      options_t options = variant(seed);
      result_t result   = presolve(c.model, options);
      oracle::verdict_t v = oracle::check_result(c, options, result);
      ++models;
#ifndef COSTFUZZ_BASE
      if (v.ok && v.crushed_points > 0) {
        ++crush_results;
        crush_dual += result.dual_findings > 0;
      }
#endif
      if (!v.ok) {
        ++oracle_failures;
        ++tally[v.message.substr(0, 40)];
        std::printf("ALARM %s %llu kind %llu dual %lld fs %lld fsd %lld zs %lld fc %lld: %s\n",
                    profile.name,
                    (unsigned long long)seed,
                    (unsigned long long)(seed % 6),
                    (long long)result.dual_findings,
#ifndef COSTFUZZ_BASE
                    (long long)result.sub_findings[sub_rule::free_singleton],
                    (long long)result.sub_findings[sub_rule::free_singleton_dual],
                    (long long)result.sub_findings[sub_rule::zero_cost_singleton],
                    (long long)result.sub_findings[sub_rule::forcing_column],
#else
                    0ll, 0ll, 0ll, 0ll,
#endif
                    v.message.substr(0, 160).c_str());
        if (++shown <= 5) {
          std::printf("ORACLE profile %s seed %llu kind %llu: %s\n%s\n",
                      profile.name,
                      (unsigned long long)seed,
                      (unsigned long long)(seed % 6),
                      v.message.substr(0, 300).c_str(),
                      oracle::to_code(c).c_str());
        }
      }
      if (result.status == status_t::infeasible ||
          result.status == status_t::unbounded_or_infeasible) {
        continue;
      }
      ++carried;
#ifndef COSTFUZZ_BASE
      gprs_check::structure_report_t s =
        gprs_check::structure_check(c.model, result, options.primal_only);
      bool bad = !s.composed || s.lost_integers || s.bound_failures || s.row_failures ||
                 s.kept_failures || s.added_failures || s.objective_failures ||
                 s.postsolve_failures || s.box_failures || s.integral_failures ||
                 s.snapshot_failures;
      structure_failures += bad;
      objective_failures += s.objective_failures != 0;
      if (s.objective_worst > worst_coefficient) { worst_coefficient = s.objective_worst; }
      if (s.objective_constant > worst_constant) { worst_constant = s.objective_constant; }
      if (bad && ++shown <= 5) {
        std::printf("STRUCTURE profile %s seed %llu kind %llu: stack '%s' objective %lld "
                    "worst %g col %d constant %g\n%s\n",
                    profile.name,
                    (unsigned long long)seed,
                    (unsigned long long)(seed % 6),
                    s.stack_error.c_str(),
                    (long long)s.objective_failures,
                    s.objective_worst,
                    s.objective_worst_col,
                    s.objective_constant,
                    oracle::to_code(c).c_str());
      }
#endif
    }
  }
  std::printf("models %lld carried %lld oracle_failures %lld structure_failures %lld "
              "objective_failures %lld crush_results %lld with_dual %lld worst_coefficient %g "
              "worst_constant %g\n",
              (long long)models,
              (long long)carried,
              (long long)oracle_failures,
              (long long)structure_failures,
              (long long)objective_failures,
              (long long)crush_results,
              (long long)crush_dual,
              worst_coefficient,
              worst_constant);
  for (const auto& t : tally) {
    std::printf("  %lld x %s\n", (long long)t.second, t.first.c_str());
  }
  return oracle_failures != 0 || structure_failures != 0;
}
