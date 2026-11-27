/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// End-to-end runs on random models that the oracle (oracle.hpp) cannot solve: rows with several
// continuous columns, chains of such rows, big-M rows, coefficients of sizes 1e-3 to 1e3. The
// models are built around planted points, which are feasible by construction, so no solver is
// needed:
// - the presolve does not report infeasible;
// - a planted point, crushed, is feasible for the reduced model, and taken back by postsolve it
//   is feasible for the model and costs no more: a dual reduction moves a feasible point to a
//   feasible one that is not worse, and every other reduction keeps every feasible point. It
//   costs the same when the result has no dual finding (result_t::dual_findings), which
//   primal-only mode guarantees;
// - with result_t::fixed_point a second presolve returns unchanged.
// Bound propagation writes explicit bounds of continuous columns along such chains, which the
// classes of the oracle (one continuous column per row) never reach.

#include "harness.hpp"

#include <gprs/postsolve.hpp>
#include <gprs/presolve.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace gprs;

namespace {

struct rng_t {
  uint64_t state;
  explicit rng_t(uint64_t seed) : state(seed * 0x9e3779b97f4a7c15ull + 0x1234567) {}
  uint64_t next()
  {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  int32_t uniform(int32_t lo, int32_t hi)
  {
    return lo + (int32_t)(next() % (uint64_t)(hi - lo + 1));
  }
  bool chance(double p) { return (next() >> 11) * 0x1p-53 < p; }
  double pick(const std::vector<double>& list) { return list[next() % list.size()]; }
};

struct profile_t {
  const char* name;
  int32_t max_cols        = 7;
  int32_t max_rows        = 9;
  double p_continuous     = 0.6;
  double p_infinite_bound = 0.4;
  double p_tight_bound    = 0.3;
  double p_scaled         = 0.1;   // per coefficient: times a power of ten from 1e-3 to 1e3
  double p_difference_row = 0.3;   // a x - b y between two columns
  double p_big_m_row      = 0.15;  // a continuous column against a binary with a large coefficient
  double p_decay_row      = 0.;    // x - f y with 0 < f <= 1 between continuous columns
  double p_parallel_row   = 0.08;  // a multiple of an earlier row with its own sides
  double p_tight_side     = 0.5;
  double p_two_sided      = 0.2;
};

std::vector<profile_t> profiles()
{
  std::vector<profile_t> list;
  profile_t p;
  p.name = "mixed";
  list.push_back(p);

  p                  = profile_t();
  p.name             = "chain";
  p.p_continuous     = 0.9;
  p.p_infinite_bound = 0.75;
  p.p_difference_row = 0.75;
  p.p_big_m_row      = 0.05;
  p.max_cols         = 9;
  p.max_rows         = 14;
  p.p_tight_side     = 0.6;
  list.push_back(p);

  p                  = profile_t();
  p.name             = "decay";
  p.p_continuous     = 1.;
  p.p_infinite_bound = 0.6;
  p.p_difference_row = 0.2;
  p.p_decay_row      = 0.6;
  p.p_big_m_row      = 0.;
  p.max_cols         = 6;
  p.max_rows         = 10;
  list.push_back(p);

  p                  = profile_t();
  p.name             = "big_m";
  p.p_continuous     = 0.5;
  p.p_infinite_bound = 0.5;
  p.p_big_m_row      = 0.5;
  p.p_difference_row = 0.2;
  p.max_rows         = 12;
  list.push_back(p);

  p                = profile_t();
  p.name           = "parallel";
  p.p_parallel_row = 0.4;
  p.p_continuous   = 0.8;
  list.push_back(p);
  return list;
}

struct planted_t {
  model_t model;
  std::vector<std::vector<double>> points;
};

using entries_t = std::vector<std::pair<int32_t, double>>;

// One to three points in quarters (integers for the integer columns), bounds around them, and
// rows whose sides lie at or beyond the activities of the points, rounded outward.
planted_t generate(uint64_t seed, const profile_t& p)
{
  rng_t rng(seed);
  planted_t out;
  model_t& m      = out.model;
  int32_t n       = rng.uniform(2, p.max_cols);
  int32_t rows    = rng.uniform(1, p.max_rows);
  int32_t npoints = rng.uniform(1, 3);
  out.points.assign(npoints, std::vector<double>(n, 0.));
  m.ncols = n;
  std::vector<int32_t> continuous, integers;
  for (int32_t j = 0; j < n; ++j) {
    bool integer = !rng.chance(p.p_continuous);
    bool binary  = integer && rng.chance(0.6);
    double low = inf, high = -inf;
    for (int32_t q = 0; q < npoints; ++q) {
      double value     = binary    ? rng.uniform(0, 1)
                         : integer ? rng.uniform(-3, 3)
                                   : 0.25 * rng.uniform(-16, 16);
      out.points[q][j] = value;
      low              = std::fmin(low, value);
      high             = std::fmax(high, value);
    }
    double lower = binary ? 0. : -inf, upper = binary ? 1. : inf;
    for (int32_t side = 0; side < 2 && !binary; ++side) {
      if (rng.chance(p.p_infinite_bound)) { continue; }
      double slack = rng.chance(p.p_tight_bound) ? 0. : rng.pick({0.25, 1., 1., 3., 10., 1000.});
      if (integer) { slack = std::ceil(slack); }
      (side ? upper : lower) = side ? high + slack : low - slack;
    }
    m.col_type.push_back(integer);
    m.col_lower.push_back(lower);
    m.col_upper.push_back(upper);
    m.cost.push_back(rng.chance(0.3) ? 0. : rng.uniform(-3, 3));
    (integer ? integers : continuous).push_back(j);
  }

  auto coefficient = [&]() {
    double a = rng.uniform(1, 3) * (rng.chance(0.5) ? 1. : -1.);
    if (rng.chance(p.p_scaled)) { a *= rng.pick({10., 100., 1000., 0.1, 0.01, 0.001}); }
    return a;
  };
  std::vector<entries_t> all;
  m.row_start.push_back(0);
  for (int32_t i = 0; i < rows; ++i) {
    entries_t row;
    auto add = [&](int32_t col, double a) {
      for (const auto& entry : row) {
        if (entry.first == col) { return; }
      }
      row.push_back({col, a});
    };
    if (!all.empty() && rng.chance(p.p_parallel_row)) {
      double factor = rng.pick({1., -1., 2., -3., 0.5, 10.});
      for (const auto& entry : all[rng.next() % all.size()]) {
        add(entry.first, entry.second * factor);
      }
    } else if (continuous.size() >= 2 && rng.chance(p.p_decay_row)) {
      add(continuous[rng.next() % continuous.size()], 1.);
      add(continuous[rng.next() % continuous.size()], -rng.pick({0.5, 0.75, 0.9, 0.99, 0.25, 1.}));
    } else if (rng.chance(p.p_difference_row)) {
      add(rng.uniform(0, n - 1), rng.chance(0.7) ? 1. : coefficient());
      add(rng.uniform(0, n - 1), rng.chance(0.7) ? -1. : coefficient());
    } else if (!continuous.empty() && !integers.empty() && rng.chance(p.p_big_m_row)) {
      add(continuous[rng.next() % continuous.size()], 1.);
      double big = rng.pick({10., 100., 1000., 1e4, 12.5, 50.});
      add(integers[rng.next() % integers.size()], rng.chance(0.5) ? -big : big);
      if (rng.chance(0.3)) { add(rng.uniform(0, n - 1), coefficient()); }
    } else {
      int32_t length = rng.uniform(2, 5);
      for (int32_t k = 0; k < length; ++k) {
        add(rng.uniform(0, n - 1), coefficient());
      }
    }
    for (size_t a = 1; a < row.size(); ++a) {
      for (size_t b = a; b > 0 && row[b - 1].first > row[b].first; --b) {
        std::swap(row[b - 1], row[b]);
      }
    }
    double low = inf, high = -inf;
    for (int32_t q = 0; q < npoints; ++q) {
      dd_t sum;
      for (const auto& entry : row) {
        dd_add_product(sum, entry.second, out.points[q][entry.first]);
      }
      double rest;
      double value = two_sum(sum.hi, sum.lo, rest);
      low          = std::fmin(low, rest == 0. ? value : std::nextafter(value, -inf));
      high         = std::fmax(high, rest == 0. ? value : std::nextafter(value, inf));
    }
    bool two     = rng.chance(p.p_two_sided);
    bool has_up  = two || rng.chance(0.5);
    bool has_low = two || !has_up;
    auto slack   = [&]() {
      return rng.chance(p.p_tight_side) ? 0. : rng.pick({0.25, 1., 2., 5., 100.});
    };
    double upper = has_up ? high + slack() : inf;
    double lower = has_low ? low - slack() : -inf;
    for (const auto& entry : row) {
      m.col_index.push_back(entry.first);
      m.value.push_back(entry.second);
    }
    m.row_start.push_back(m.col_index.size());
    m.row_lower.push_back(lower);
    m.row_upper.push_back(upper);
    all.push_back(row);
  }
  m.nrows = rows;
  return out;
}

// Rows and bounds within two tolerances plus 1e-9 of the size of the row: a row that the
// presolve scaled, or that a parallel row took over, holds its tolerance in other units.
bool holds(const model_t& m, const std::vector<double>& x, double feasibility)
{
  for (int32_t j = 0; j < m.ncols; ++j) {
    double allowed = 2. * feasibility + 1e-9 * std::fabs(x[j]);
    if (x[j] < m.col_lower[j] - allowed || x[j] > m.col_upper[j] + allowed) { return false; }
  }
  for (int32_t i = 0; i < m.nrows; ++i) {
    dd_t sum;
    double size = 0.;
    for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
      dd_add_product(sum, m.value[e], x[m.col_index[e]]);
      size += std::fabs(m.value[e] * x[m.col_index[e]]);
    }
    double allowed = 2. * feasibility + 1e-9 * size;
    if (m.row_upper[i] < inf && -dd_residual(m.row_upper[i], sum) > allowed) { return false; }
    if (m.row_lower[i] > -inf && dd_residual(m.row_lower[i], sum) > allowed) { return false; }
  }
  return true;
}

double objective(const model_t& m, const std::vector<double>& x)
{
  dd_t sum{m.objective_offset, 0.};
  for (int32_t j = 0; j < m.ncols; ++j) {
    dd_add_product(sum, m.cost[j], x[j]);
  }
  return dd_value(sum);
}

// Empty when the presolve of the case is right, else what is wrong.
std::string planted_problem(const planted_t& c, const options_t& options)
{
  result_t result = presolve(c.model, options);
  if (result.status == status_t::infeasible) { return "infeasible, with a feasible point"; }
  if (result.status == status_t::unbounded_or_infeasible) { return std::string(); }
  if (options.primal_only && result.dual_findings != 0) {
    return "a dual finding in primal-only mode";
  }
  for (const std::vector<double>& x : c.points) {
    std::vector<double> crushed, back;
    result.postsolve.crush(x, crushed);
    if (!holds(result.reduced, crushed, options.tol.feasibility)) {
      return "a crushed feasible point violates the reduced model";
    }
    result.postsolve.postsolve(crushed, back);
    if (!holds(c.model, back, options.tol.feasibility)) {
      return "a feasible point violates the model after crush and postsolve";
    }
    double before = objective(c.model, x), after = objective(c.model, back);
    if (after > before + 1e-5) { return "a feasible point costs more after crush and postsolve"; }
    if (result.dual_findings == 0 && after < before - 1e-5) {
      return "a feasible point costs less after crush and postsolve, without a dual finding";
    }
  }
  if (result.fixed_point && presolve(result.reduced, options).status != status_t::unchanged) {
    return "the second presolve changes the reduced model";
  }
  return std::string();
}

struct known_t {
  const char* profile;
  uint64_t seed;
  bool primal_only;
};

}  // namespace

// Models on which the presolve reported infeasible before a forcing row left continuous columns
// alone that are off their bound (rules.hpp, (a)), found by the property below on other seeds.
GPRS_TEST(e2e_planted_known_cases)
{
  const std::vector<known_t> known = {{"mixed", 21569, false},
                                      {"chain", 74086, true},
                                      {"chain", 122798, false},
                                      {"decay", 18573, false},
                                      {"decay", 68640, true},
                                      {"big_m", 106338, false},
                                      {"big_m", 162230, true},
                                      {"parallel", 298364, false}};
  GPRS_CHECK(known.size() >= 4);
  for (const known_t& k : known) {
    for (const profile_t& profile : profiles()) {
      if (std::string(profile.name) != k.profile) { continue; }
      options_t options;
      options.primal_only = k.primal_only;
      planted_t c         = generate(k.seed, profile);
      GPRS_REQUIRE(well_formed(c.model));
      std::string problem = planted_problem(c, options);
      if (!problem.empty()) {
        std::fprintf(stderr,
                     "    profile %s seed %llu%s: %s\n",
                     k.profile,
                     (unsigned long long)k.seed,
                     k.primal_only ? " primal only" : "",
                     problem.c_str());
      }
      GPRS_CHECK(problem.empty());
    }
  }
}

GPRS_TEST(e2e_planted_points_survive)
{
  int64_t models = 0, failures = 0;
  for (const profile_t& profile : profiles()) {
    for (uint64_t seed = 7100000; seed < 7103000; ++seed) {
      planted_t c = generate(seed, profile);
      GPRS_REQUIRE(well_formed(c.model));
      for (const std::vector<double>& x : c.points) {
        point_check_t check = check_point(c.model, x);
        GPRS_REQUIRE(check.row_violation == 0. && check.bound_violation == 0. &&
                     check.integrality_violation == 0.);
      }
      for (bool primal_only : {false, true}) {
        options_t options;
        options.primal_only = primal_only;
        std::string problem = planted_problem(c, options);
        ++models;
        if (problem.empty()) { continue; }
        ++failures;
        if (failures <= 5) {
          std::fprintf(stderr,
                       "    profile %s seed %llu%s: %s\n",
                       profile.name,
                       (unsigned long long)seed,
                       primal_only ? " primal only" : "",
                       problem.c_str());
        }
      }
    }
  }
  GPRS_CHECK_EQ(failures, 0);
  GPRS_CHECK(models == 30000);
}
