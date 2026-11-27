/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"
#include "oracle.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>

using namespace gprs;
using oracle::case_t;
using oracle::generator_options_t;
using oracle::outcome_t;
using oracle::solution_t;

namespace {

bool same_model(const model_t& a, const model_t& b)
{
  return a.nrows == b.nrows && a.ncols == b.ncols && a.row_start == b.row_start &&
         a.col_index == b.col_index && a.value == b.value && a.row_lower == b.row_lower &&
         a.row_upper == b.row_upper && a.col_lower == b.col_lower && a.col_upper == b.col_upper &&
         a.cost == b.cost && a.col_type == b.col_type && a.objective_offset == b.objective_offset;
}

// FNV-1a over the bit patterns of a case.
struct fnv_t {
  uint64_t hash = 0xcbf29ce484222325ull;

  void bytes(const void* data, size_t size)
  {
    const unsigned char* p = (const unsigned char*)data;
    for (size_t k = 0; k < size; ++k) {
      hash = (hash ^ p[k]) * 0x100000001b3ull;
    }
  }
  template <typename T>
  void add(const std::vector<T>& v)
  {
    uint64_t size = v.size();
    bytes(&size, sizeof(size));
    if (!v.empty()) { bytes(v.data(), v.size() * sizeof(T)); }
  }
  void add(const model_t& m)
  {
    bytes(&m.nrows, sizeof(m.nrows));
    bytes(&m.ncols, sizeof(m.ncols));
    add(m.row_start);
    add(m.col_index);
    add(m.value);
    add(m.row_lower);
    add(m.row_upper);
    add(m.col_lower);
    add(m.col_upper);
    add(m.cost);
    add(m.col_type);
    bytes(&m.objective_offset, sizeof(m.objective_offset));
  }
};

bool is_power_of_two(double v)
{
  int exponent;
  return v > 0 && std::frexp(v, &exponent) == 0.5;
}

bool small(double v, int32_t limit) { return v == std::floor(v) && std::fabs(v) <= limit; }

bool half_step(double v, int32_t limit)
{
  return 2. * v == std::floor(2. * v) && std::fabs(v) <= limit + 0.5;
}

// What the generator promises of a case, tested from scratch. Counts what it finds.
struct census_t {
  std::map<std::string, int64_t> count;
  int64_t cases = 0;

  void add(const char* name, bool hit = true) { count[name] += hit; }
  int64_t operator[](const char* name) const
  {
    auto it = count.find(name);
    return it == count.end() ? 0 : it->second;
  }
};

void inspect(const case_t& c, const generator_options_t& o, census_t& census)
{
  const model_t& b = c.base;
  ++census.cases;
  GPRS_REQUIRE(well_formed(b));
  GPRS_REQUIRE(well_formed(c.model));
  GPRS_CHECK(b.nrows <= o.max_rows);
  GPRS_CHECK(b.ncols <= o.max_enum_cols + o.max_line_cols);
  std::vector<uint8_t> line;
  GPRS_REQUIRE(oracle::classify(b, o.max_width, line));
  GPRS_CHECK(std::accumulate(line.begin(), line.end(), 0) <= o.max_line_cols);
  census.add("zero rows", b.nrows == 0);
  census.add("zero columns", b.ncols == 0);
  census.add("offset", b.objective_offset != 0.);
  GPRS_CHECK(small(b.objective_offset, o.max_side));

  // Scaling.
  GPRS_REQUIRE((int32_t)c.row_scale.size() == b.nrows);
  GPRS_REQUIRE((int32_t)c.col_scale.size() == b.ncols);
  GPRS_CHECK(same_model(c.model, oracle::scale_model(b, c.row_scale, c.col_scale)));
  bool scaled = false;
  for (double s : c.row_scale) {
    GPRS_CHECK(is_power_of_two(s) && s <= std::ldexp(1., o.max_scale_exp2) &&
               s >= std::ldexp(1., -o.max_scale_exp2));
    scaled = scaled || s != 1.;
  }
  for (int32_t j = 0; j < b.ncols; ++j) {
    double s = c.col_scale[j];
    GPRS_CHECK(is_power_of_two(s) && s <= std::ldexp(1., o.max_scale_exp2) &&
               s >= std::ldexp(1., -o.max_scale_exp2));
    GPRS_CHECK(b.col_type[j] == 0 || s == 1.);
    scaled = scaled || s != 1.;
  }
  census.add("scaled", scaled);
  if (!scaled) { GPRS_CHECK(same_model(c.model, b)); }

  // Columns.
  std::vector<int32_t> entries(b.ncols, 0), down_locks(b.ncols, 0), up_locks(b.ncols, 0);
  for (int32_t i = 0; i < b.nrows; ++i) {
    for (int64_t e = b.row_start[i]; e < b.row_start[i + 1]; ++e) {
      int32_t j = b.col_index[e];
      ++entries[j];
      bool positive = b.value[e] > 0;
      if (b.row_upper[i] != inf) { ++(positive ? up_locks : down_locks)[j]; }
      if (b.row_lower[i] != -inf) { ++(positive ? down_locks : up_locks)[j]; }
    }
  }
  bool crossing = false;
  for (int32_t j = 0; j < b.ncols; ++j) {
    double lower = b.col_lower[j], upper = b.col_upper[j];
    GPRS_CHECK(small(b.cost[j], o.max_coefficient));
    bool huge = false, fractional = false;
    for (double bound : {lower, upper}) {
      if (!std::isfinite(bound)) { continue; }
      if (std::fabs(bound) > o.max_side + 0.5) {
        GPRS_CHECK(line[j] && std::fabs(bound) >= 1e7 && std::fabs(bound) <= 1e22);
        GPRS_CHECK(bound == std::floor(bound));
        huge = true;
      } else {
        GPRS_CHECK(half_step(bound, o.max_side));
        GPRS_CHECK(b.col_type[j] == 1 || bound == std::floor(bound));
        fractional = fractional || bound != std::floor(bound);
      }
    }
    crossing = crossing || lower > upper;
    if (!line[j]) { GPRS_CHECK(upper - lower <= o.max_width); }
    census.add("empty column", entries[j] == 0);
    census.add("fixed column", lower == upper);
    census.add("fixed continuous column", lower == upper && b.col_type[j] == 0);
    census.add("free column", lower == -inf && upper == inf);
    census.add("column with one infinite bound", (lower == -inf) != (upper == inf));
    census.add("line column with two finite bounds", line[j] && lower != -inf && upper != inf);
    census.add("huge bound", huge);
    census.add("fractional bound", fractional);
    census.add("crossing column", lower > upper);
    census.add("continuous column", b.col_type[j] == 0);
    census.add("integer line column", line[j] && b.col_type[j] == 1);
    census.add("binary column", !line[j] && lower == 0. && upper == 1.);
    census.add("zero cost", b.cost[j] == 0.);
    census.add("empty column with cost", entries[j] == 0 && b.cost[j] != 0.);
    bool fixable = lower < upper && entries[j] > 0 &&
                   ((b.cost[j] >= 0 && down_locks[j] == 0) || (b.cost[j] <= 0 && up_locks[j] == 0));
    census.add("dual fixing candidate", fixable);
  }

  // Rows.
  for (int32_t i = 0; i < b.nrows; ++i) {
    int64_t begin = b.row_start[i], end = b.row_start[i + 1];
    double lower = b.row_lower[i], upper = b.row_upper[i];
    GPRS_CHECK(lower == -inf || small(lower, o.max_side));
    GPRS_CHECK(upper == inf || small(upper, o.max_side));
    crossing       = crossing || lower > upper;
    double act_min = 0., act_max = 0., factor = 0.;
    bool integer_row = true, infinite_term = false;
    for (int64_t e = begin; e < end; ++e) {
      int32_t j = b.col_index[e];
      double a  = b.value[e];
      GPRS_CHECK(a != 0. && small(a, o.max_coefficient));
      act_min += a * (a > 0 ? b.col_lower[j] : b.col_upper[j]);
      act_max += a * (a > 0 ? b.col_upper[j] : b.col_lower[j]);
      integer_row = integer_row && b.col_type[j] == 1;
      factor      = std::gcd((int64_t)factor, (int64_t)std::fabs(a));
      for (double bound : {b.col_lower[j], b.col_upper[j]}) {
        infinite_term = infinite_term || (std::isfinite(bound) && std::fabs(a * bound) > 1e20);
      }
    }
    int64_t size  = end - begin;
    bool equation = lower == upper;
    census.add("empty row", size == 0);
    census.add("singleton row", size == 1);
    census.add("doubleton equation", size == 2 && equation);
    census.add("equation", equation);
    census.add("ranged row", lower != -inf && upper != inf && lower < upper);
    census.add("free row", lower == -inf && upper == inf);
    census.add("row with one side", (lower == -inf) != (upper == inf));
    census.add("crossing row", lower > upper);
    census.add("row with a line column", [&] {
      for (int64_t e = begin; e < end; ++e) {
        if (line[b.col_index[e]]) { return true; }
      }
      return false;
    }());
    census.add("huge bound as a finite activity term", [&] {
      for (int64_t e = begin; e < end; ++e) {
        for (double bound : {b.col_lower[b.col_index[e]], b.col_upper[b.col_index[e]]}) {
          double term = std::fabs(b.value[e] * bound);
          if (term >= 1e7 && term <= 1e20) { return true; }
        }
      }
      return false;
    }());
    census.add("huge bound as an infinite activity term", infinite_term);
    if (size == 0 || lower > upper) { continue; }
    bool infeasible = act_min > upper || act_max < lower;
    bool redundant  = !infeasible && act_min >= lower && act_max <= upper;
    census.add("row infeasible by its activity", infeasible);
    census.add("row redundant by its activity", redundant && (lower != -inf || upper != inf));
    census.add("forcing row",
               size >= 2 && !infeasible && !redundant && (act_min == upper || act_max == lower));
    census.add("integer row with a common factor", size >= 2 && integer_row && factor > 1);
    // A bound of an integer column that the row implies from the bounds of its other columns
    // and that is tighter than the column's own bound; the row is not infeasible.
    census.add("row with a bound propagation candidate", [&] {
      for (int64_t e = begin; e < end && !infeasible && size >= 2; ++e) {
        int32_t j = b.col_index[e];
        double a  = b.value[e];
        if (b.col_type[j] == 0) { continue; }
        double others_min = act_min - a * (a > 0 ? b.col_lower[j] : b.col_upper[j]);
        double others_max = act_max - a * (a > 0 ? b.col_upper[j] : b.col_lower[j]);
        // NaN when the own term is infinite; not finite when another term is.
        if (upper != inf && std::fabs(others_min) <= 1e20) {
          double bound = (upper - others_min) / a;
          if (a > 0 ? std::floor(bound) < b.col_upper[j] : std::ceil(bound) > b.col_lower[j]) {
            return true;
          }
        }
        if (lower != -inf && std::fabs(others_max) <= 1e20) {
          double bound = (lower - others_max) / a;
          if (a > 0 ? std::ceil(bound) > b.col_lower[j] : std::floor(bound) < b.col_upper[j]) {
            return true;
          }
        }
      }
      return false;
    }());
    if (integer_row && !infeasible && !redundant && (lower == -inf) != (upper == inf)) {
      double gap      = upper != inf ? act_max - upper : lower - act_min;
      bool strengthen = false;
      for (int64_t e = begin; e < end; ++e) {
        strengthen = strengthen || (gap > 0 && std::isfinite(gap) && std::fabs(b.value[e]) > gap);
      }
      census.add("coefficient strengthening candidate", strengthen);
    }
    if (size == 2 && equation) {
      int32_t j = b.col_index[begin], k = b.col_index[begin + 1];
      double p = std::fabs(b.value[begin]), q = std::fabs(b.value[begin + 1]);
      bool both_integer = b.col_type[j] == 1 && b.col_type[k] == 1;
      census.add("integral doubleton equation of two integer columns",
                 both_integer && (std::fmod(p, q) == 0. || std::fmod(q, p) == 0.));
      census.add("doubleton equation with a continuous column", !both_integer);
    }
    for (int32_t r = 0; r < i; ++r) {
      int64_t other = b.row_start[r];
      if (size < 2 || b.row_start[r + 1] - other != size) { continue; }
      bool parallel = true;
      for (int64_t e = 0; e < size; ++e) {
        parallel = parallel && b.col_index[begin + e] == b.col_index[other + e] &&
                   b.value[begin + e] * b.value[other] == b.value[other + e] * b.value[begin];
      }
      census.add("parallel row pair", parallel);
      census.add("duplicate row pair",
                 parallel && b.value[begin] == b.value[other] && lower == b.row_lower[r] &&
                   upper == b.row_upper[r]);
      census.add("parallel row pair with a negative ratio",
                 parallel && b.value[begin] * b.value[other] < 0);
    }
  }
  census.add("crossing case", crossing);

  solution_t base = oracle::solve(b, line);
  census.add("optimal", base.outcome == outcome_t::optimal);
  census.add("infeasible", base.outcome == outcome_t::infeasible);
  census.add("unbounded", base.outcome == outcome_t::unbounded);
  census.add("infeasible without crossing", base.outcome == outcome_t::infeasible && !crossing);
}

bool at_least(const census_t& census, const char* name, double share)
{
  bool enough = census[name] >= share * census.cases;
  if (!enough) {
    test::fail(__FILE__,
               __LINE__,
               std::string(name) + ": " + std::to_string(census[name]) + " in " +
                 std::to_string(census.cases) + " cases");
  }
  return enough;
}

}  // namespace

GPRS_TEST(oracle_rng_is_splitmix64)
{
  oracle::rng_t zero(0);
  GPRS_CHECK_EQ(zero.next(), 0xe220a8397b1dcdafull);
  GPRS_CHECK_EQ(zero.next(), 0x6e789e6aa1b965f4ull);
  GPRS_CHECK_EQ(zero.next(), 0x06c45d188009454full);
  oracle::rng_t rng(1234567);
  GPRS_CHECK_EQ(rng.next(), 6457827717110365317ull);
  GPRS_CHECK_EQ(rng.next(), 3203168211198807973ull);
  GPRS_CHECK_EQ(rng.next(), 9817491932198370423ull);
  GPRS_CHECK_EQ(rng.next(), 4593380528125082431ull);
  GPRS_CHECK_EQ(rng.next(), 16408922859458223821ull);
}

GPRS_TEST(oracle_rng_uniform_and_chance)
{
  oracle::rng_t rng(42);
  std::vector<int32_t> hits(7, 0);
  for (int32_t k = 0; k < 70000; ++k) {
    int32_t v = rng.uniform(-3, 3);
    GPRS_REQUIRE(v >= -3 && v <= 3);
    ++hits[v + 3];
  }
  for (int32_t h : hits) {
    GPRS_CHECK(h > 9500 && h < 10500);
  }
  for (int32_t k = 0; k < 100; ++k) {
    GPRS_CHECK_EQ(rng.uniform(5, 5), 5);
    GPRS_CHECK_EQ(rng.uniform(-7, -7), -7);
    int32_t v = rng.uniform(INT32_MIN, INT32_MAX - 1);
    GPRS_CHECK(v < INT32_MAX);
    GPRS_CHECK(rng.uniform(INT32_MAX - 1, INT32_MAX) >= INT32_MAX - 1);
    GPRS_CHECK(rng.uniform(INT32_MIN, INT32_MIN + 1) <= INT32_MIN + 1);
    GPRS_CHECK(!rng.chance(0.));
    GPRS_CHECK(rng.chance(1.));
  }
  // Both ends of a range of two, and the whole int32 range without a hang.
  int32_t low = 0;
  for (int32_t k = 0; k < 1000; ++k) {
    low += rng.uniform(INT32_MIN, INT32_MAX) < 0;
  }
  GPRS_CHECK(low > 400 && low < 600);
  int32_t yes = 0;
  for (int32_t k = 0; k < 100000; ++k) {
    yes += rng.chance(0.25);
  }
  GPRS_CHECK(yes > 24000 && yes < 26000);
  // The draws of uniform and chance come from next() in order.
  oracle::rng_t a(7), b(7);
  uint64_t draw = a.next();
  GPRS_CHECK_EQ(b.uniform(0, 15), (int32_t)(draw % 16));
  draw = a.next();
  GPRS_CHECK_EQ(b.chance(0.5), (draw >> 63) == 0);
}

GPRS_TEST(oracle_generate_is_deterministic)
{
  generator_options_t o;
  int32_t different = 0;
  fnv_t fnv;
  for (uint64_t seed = 0; seed < 300; ++seed) {
    case_t a = oracle::generate(seed, o);
    case_t b = oracle::generate(seed, o);
    GPRS_CHECK(same_model(a.model, b.model) && same_model(a.base, b.base));
    GPRS_CHECK(a.row_scale == b.row_scale && a.col_scale == b.col_scale);
    case_t next = oracle::generate(seed + 1, o);
    different += !same_model(a.base, next.base);
    fnv.add(a.base);
    fnv.add(a.model);
    fnv.add(a.row_scale);
    fnv.add(a.col_scale);
  }
  GPRS_CHECK(different > 270);
  // The cases of seeds 0 to 299 with the default options, as first generated (clang, arm64).
  // A change of the generator changes this value; a platform must not.
  GPRS_CHECK_EQ(fnv.hash, 17151125249596340048ull);
}

GPRS_TEST(oracle_generate_stays_in_the_class)
{
  std::vector<generator_options_t> variants(6);
  variants[1].max_rows             = 9;
  variants[1].max_enum_cols        = 7;
  variants[1].max_line_cols        = 4;
  variants[1].density              = 0.9;
  variants[1].p_scaled             = 1.;
  variants[1].max_scale_exp2       = 30;
  variants[2].max_width            = 0;
  variants[2].max_coefficient      = 1;
  variants[2].max_side             = 1;
  variants[3].max_enum_cols        = 0;
  variants[3].p_continuous         = 0.;
  variants[3].p_infinite_bound     = 0.;
  variants[3].p_huge_bound         = 1.;
  variants[3].p_fractional_bound   = 1.;
  variants[4].max_line_cols        = 0;
  variants[4].max_width            = 7;
  variants[4].max_coefficient      = 6;
  variants[4].p_parallel_row       = 0.6;
  variants[4].p_common_factor      = 0.6;
  variants[4].p_fractional_bound   = 0.6;
  variants[4].p_crossing           = 0.5;
  variants[5].p_doubleton_equation = 1.;
  variants[5].p_parallel_row       = 0.;
  variants[5].p_empty_row          = 0.;
  variants[5].p_singleton_row      = 0.;
  variants[5].p_infinite_bound     = 1.;
  variants[5].p_fixed_col          = 0.;
  variants[5].p_forcing_side       = 1.;
  variants[5].max_scale_exp2       = 0;
  for (const generator_options_t& o : variants) {
    census_t census;
    for (uint64_t seed = 1000; seed < 4000; ++seed) {
      inspect(oracle::generate(seed, o), o, census);
    }
    GPRS_CHECK(census["optimal"] > 0);
    GPRS_CHECK(census["infeasible"] > 0);
  }
}

GPRS_TEST(oracle_generate_covers_every_special_case)
{
  generator_options_t o;
  census_t census;
  for (uint64_t seed = 0; seed < 20000; ++seed) {
    inspect(oracle::generate(seed, o), o, census);
  }
  if (std::getenv("GPRS_ORACLE_CENSUS")) {
    for (const auto& entry : census.count) {
      std::printf("    %-66s %7lld  %.4f\n",
                  entry.first.c_str(),
                  (long long)entry.second,
                  (double)entry.second / census.cases);
    }
  }
  // Occurrences per case, each at about half of what the default options give.
  GPRS_CHECK(at_least(census, "optimal", 0.35));
  GPRS_CHECK(at_least(census, "infeasible", 0.2));
  GPRS_CHECK(at_least(census, "unbounded", 0.07));
  GPRS_CHECK(at_least(census, "infeasible without crossing", 0.2));
  GPRS_CHECK(at_least(census, "zero rows", 0.1));
  GPRS_CHECK(at_least(census, "zero columns", 0.03));
  GPRS_CHECK(at_least(census, "scaled", 0.1));
  GPRS_CHECK(at_least(census, "crossing column", 0.008));
  GPRS_CHECK(at_least(census, "crossing row", 0.005));
  GPRS_CHECK(at_least(census, "empty row", 0.1));
  GPRS_CHECK(at_least(census, "singleton row", 0.3));
  GPRS_CHECK(at_least(census, "doubleton equation", 0.15));
  GPRS_CHECK(at_least(census, "equation", 0.3));
  GPRS_CHECK(at_least(census, "ranged row", 0.05));
  GPRS_CHECK(at_least(census, "free row", 0.05));
  GPRS_CHECK(at_least(census, "row with one side", 0.8));
  GPRS_CHECK(at_least(census, "row with a line column", 0.5));
  GPRS_CHECK(at_least(census, "parallel row pair", 0.1));
  GPRS_CHECK(at_least(census, "duplicate row pair", 0.015));
  GPRS_CHECK(at_least(census, "parallel row pair with a negative ratio", 0.05));
  GPRS_CHECK(at_least(census, "empty column", 0.3));
  GPRS_CHECK(at_least(census, "empty column with cost", 0.2));
  GPRS_CHECK(at_least(census, "fixed column", 0.15));
  GPRS_CHECK(at_least(census, "fixed continuous column", 0.03));
  GPRS_CHECK(at_least(census, "free column", 0.1));
  GPRS_CHECK(at_least(census, "column with one infinite bound", 0.2));
  GPRS_CHECK(at_least(census, "line column with two finite bounds", 0.15));
  GPRS_CHECK(at_least(census, "huge bound", 0.08));
  GPRS_CHECK(at_least(census, "huge bound as a finite activity term", 0.05));
  GPRS_CHECK(at_least(census, "huge bound as an infinite activity term", 0.04));
  GPRS_CHECK(at_least(census, "fractional bound", 0.05));
  GPRS_CHECK(at_least(census, "continuous column", 0.25));
  GPRS_CHECK(at_least(census, "integer line column", 0.2));
  GPRS_CHECK(at_least(census, "binary column", 0.2));
  GPRS_CHECK(at_least(census, "zero cost", 0.5));
  GPRS_CHECK(at_least(census, "offset", 0.3));
  GPRS_CHECK(at_least(census, "row infeasible by its activity", 0.1));
  GPRS_CHECK(at_least(census, "row redundant by its activity", 0.15));
  GPRS_CHECK(at_least(census, "forcing row", 0.1));
  GPRS_CHECK(at_least(census, "row with a bound propagation candidate", 0.3));
  GPRS_CHECK(at_least(census, "integer row with a common factor", 0.15));
  GPRS_CHECK(at_least(census, "coefficient strengthening candidate", 0.04));
  GPRS_CHECK(at_least(census, "integral doubleton equation of two integer columns", 0.1));
  GPRS_CHECK(at_least(census, "doubleton equation with a continuous column", 0.04));
  GPRS_CHECK(at_least(census, "dual fixing candidate", 0.3));
}

GPRS_TEST(oracle_solve_does_not_depend_on_the_classification_or_the_scaling)
{
  generator_options_t o;
  o.p_scaled           = 0.5;
  int32_t reclassified = 0, scaled = 0;
  for (uint64_t seed = 50000; seed < 62000; ++seed) {
    case_t c = oracle::generate(seed, o);
    std::vector<uint8_t> narrow, wide, model_line;
    GPRS_REQUIRE(oracle::classify(c.base, o.max_width, narrow));
    GPRS_REQUIRE(oracle::classify_widest(c.base, 1e5, wide));
    GPRS_REQUIRE(oracle::classify(c.model, o.max_width, model_line));
    GPRS_CHECK(model_line == narrow);
    solution_t a = oracle::solve(c.base, narrow, 4);
    bool same    = true;
    if (wide != narrow) {
      // An integer column with narrow finite bounds as a line column and enumerated.
      ++reclassified;
      solution_t b = oracle::solve(c.base, wide);
      same         = same && a.outcome == b.outcome;
      same         = same &&
             std::fabs(a.objective - b.objective) <= 1e-12 * std::max(1., std::fabs(a.objective));
    }
    solution_t m = oracle::solve(c.model, model_line, 4);
    scaled += !same_model(c.model, c.base);
    same = same && m.outcome == a.outcome && m.objective == a.objective;
    same = same && m.x.size() == a.x.size() && m.feasible.size() == a.feasible.size();
    for (size_t j = 0; same && j < m.x.size(); ++j) {
      same = m.x[j] * c.col_scale[j] == a.x[j];
    }
    for (size_t k = 0; same && k < m.feasible.size(); ++k) {
      for (size_t j = 0; same && j < m.feasible[k].size(); ++j) {
        same = m.feasible[k][j] * c.col_scale[j] == a.feasible[k][j];
      }
    }
    if (!same) {
      test::fail(__FILE__, __LINE__, "seed " + std::to_string(seed) + "\n" + oracle::to_code(c));
      break;
    }
  }
  GPRS_CHECK(reclassified > 300);
  GPRS_CHECK(scaled > 4000);
}

GPRS_TEST(oracle_scale_model_and_to_code)
{
  model_t base;
  base.nrows            = 2;
  base.ncols            = 3;
  base.row_start        = {0, 2, 3};
  base.col_index        = {0, 2, 1};
  base.value            = {1., -2., 3.};
  base.row_lower        = {-inf, 1.};
  base.row_upper        = {4., 1.};
  base.col_lower        = {0., -1e22, -inf};
  base.col_upper        = {1., 3., 6.};
  base.cost             = {1., 0., -1.};
  base.col_type         = {1, 1, 0};
  base.objective_offset = -2.;
  case_t c              = oracle::make_case(base, {0.5, 4.}, {1., 1., 8.});
  GPRS_CHECK(same_model(c.base, base));
  GPRS_CHECK(c.model.value == std::vector<double>({0.5, -8., 12.}));
  GPRS_CHECK(c.model.row_lower == std::vector<double>({-inf, 4.}));
  GPRS_CHECK(c.model.row_upper == std::vector<double>({2., 4.}));
  GPRS_CHECK(c.model.col_lower == std::vector<double>({0., -1e22, -inf}));
  GPRS_CHECK(c.model.col_upper == std::vector<double>({1., 3., 0.75}));
  GPRS_CHECK(c.model.cost == std::vector<double>({1., 0., -8.}));
  GPRS_CHECK_EQ(c.model.objective_offset, -2.);
  GPRS_CHECK(c.model.row_start == base.row_start && c.model.col_index == base.col_index);
  // A point of the model and its image x_base = col_scale * x_model have the same activities
  // up to the row factors: row 0 at x_model = (1, 2, 0.5) is 0.5 * (1 - 2 * 4) = -3.5.
  GPRS_CHECK_EQ(c.model.value[0] * 1. + c.model.value[1] * 0.5, 0.5 * (1. * 1. - 2. * 4.));

  case_t plain = oracle::make_case(base, {}, {});
  GPRS_CHECK(same_model(plain.model, base));
  GPRS_CHECK(plain.row_scale == std::vector<double>({1., 1.}));
  GPRS_CHECK(plain.col_scale == std::vector<double>({1., 1., 1.}));

  base.value[1]    = 0.1;
  std::string code = oracle::to_code(oracle::make_case(base, {0.5, 4.}, {1., 1., 8.}));
  const char* wanted =
    "  model_t base;\n"
    "  base.nrows = 2;\n"
    "  base.ncols = 3;\n"
    "  base.row_start = {0, 2, 3};\n"
    "  base.col_index = {0, 2, 1};\n"
    "  base.value = {1., 0.10000000000000001, 3.};\n"
    "  base.row_lower = {-inf, 1.};\n"
    "  base.row_upper = {4., 1.};\n"
    "  base.col_lower = {0., -1e+22, -inf};\n"
    "  base.col_upper = {1., 3., 6.};\n"
    "  base.cost = {1., 0., -1.};\n"
    "  base.col_type = {1, 1, 0};\n"
    "  base.objective_offset = -2.;\n"
    "  oracle::case_t c = oracle::make_case(\n"
    "    base,\n"
    "    {0.5, 4.},\n"
    "    {1., 1., 8.});\n";
  GPRS_CHECK_EQ(code, std::string(wanted));
  // Every number of the text reads back as the number it was written from.
  GPRS_CHECK_EQ(std::strtod("0.10000000000000001", nullptr), 0.1);
  GPRS_CHECK_EQ(std::strtod("-1e+22", nullptr), -1e22);
  // Long lists are broken into lines of at most 100 columns.
  model_t wide;
  wide.ncols = 40;
  wide.row_start.push_back(0);
  wide.col_lower.assign(40, -0.5);
  wide.col_upper.assign(40, 1e22);
  wide.cost.assign(40, 1.);
  wide.col_type.assign(40, 1);
  code        = oracle::to_code(oracle::make_case(wide, {}, {}));
  size_t line = 0;
  for (size_t k = 0; k < code.size(); ++k) {
    if (code[k] != '\n') { continue; }
    GPRS_CHECK(k - line <= 100);
    line = k + 1;
  }
  // Lists of one-digit numbers behind the longest prefix: among these lengths are those at
  // which the last number or the closing brace would pass column 100. to_code only prints.
  for (int32_t entries = 20; entries <= 40; ++entries) {
    case_t shown;
    shown.base.nrows     = 1;
    shown.base.ncols     = 9;
    shown.base.row_start = {0, entries};
    shown.base.col_index.assign(entries, 7);
    shown.base.value.assign(entries, 1.);
    code = oracle::to_code(shown);
    line = 0;
    for (size_t k = 0; k < code.size(); ++k) {
      if (code[k] != '\n') { continue; }
      GPRS_CHECK(k - line <= 100);
      line = k + 1;
    }
  }
}
