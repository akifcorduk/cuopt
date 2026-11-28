// Identity driver of milestone M1: prints, per model, what a presolve result of the base commit
// (the groundwork, 74350acb3) holds, so that the outputs of the base engine and of the engine
// with the singleton-column rules can be compared model by model. Compiles against both trees:
// IDENT_NEW is defined for the new one. The generator is the new tests/oracle.cpp in both.
//
//   ident dump <file.csr> <on|off>
//   ident random <first seed> <seeds per profile> <on|off> [--singletons] [--columns]
//                [--verbose] [--chunk <n>]
// on / off: the family singleton_cols switched on or off, in both engines (in the base engine
// the family is empty).
// random prints one line per chunk of seeds with a hash over the lines of the chunk, or with
// --verbose every line.

#include "oracle.hpp"

#include <gprs/io.hpp>
#include <gprs/presolve.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace gprs;
using oracle::case_t;
using oracle::generator_options_t;

namespace {

struct fnv_t {
  uint64_t h = 0xcbf29ce484222325ull;
  void bytes(const void* p, size_t n)
  {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; ++i) {
      h ^= b[i];
      h *= 0x100000001b3ull;
    }
  }
  template <typename T>
  void add(const T& v)
  {
    bytes(&v, sizeof(T));
  }
  void add_double(double v)
  {
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    add(bits);
  }
};

// Every record of the base engine, field by field. A stack of the new engine with the family
// off has no other record; one that has gets another hash.
uint64_t stack_hash(const postsolve_t& ps)
{
  fnv_t h;
  h.add(ps.nrows);
  h.add(ps.ncols);
  h.add(ps.order.size());
  for (record_kind_t k : ps.order) {
    h.add((uint8_t)k);
  }
  h.add(ps.fixed_cols.size());
  for (const fixed_col_record_t& r : ps.fixed_cols) {
    h.add(r.col);
    h.add_double(r.value);
  }
  h.add(ps.aliases.size());
  for (const alias_record_t& r : ps.aliases) {
    h.add(r.col);
    h.add(r.rep);
    h.add(r.row);
    h.add_double(r.scale);
    h.add_double(r.offset);
    h.add((uint8_t)r.origin);
  }
  h.add(ps.parallel_rows.size());
  for (const parallel_row_record_t& r : ps.parallel_rows) {
    h.add(r.removed_row);
    h.add(r.kept_row);
    h.add_double(r.ratio);
  }
  h.add(ps.row_additions.size());
  for (const row_addition_record_t& r : ps.row_additions) {
    h.add(r.row);
    h.add(r.equation_row);
    h.add_double(r.factor);
  }
#ifdef IDENT_NEW
  if (!ps.col_from_rows.empty()) { h.add(ps.col_from_rows.size()); }
#endif
  for (int32_t v : ps.row_origin) {
    h.add(v);
  }
  for (int32_t v : ps.col_origin) {
    h.add(v);
  }
  for (uint8_t v : ps.col_removed) {
    h.add(v);
  }
  return h.h;
}

void put_stats(std::string& line, const rule_stats_t& s)
{
  char buf[200];
  std::snprintf(buf,
                sizeof(buf),
                " [%lld %lld %lld %lld %lld %lld %lld]",
                (long long)s.calls,
                (long long)s.work,
                (long long)s.findings,
                (long long)s.records,
                (long long)s.rows_removed,
                (long long)s.cols_removed,
                (long long)s.nnz_removed);
  line += buf;
}

// The reduced-model hash, the record stack, the fixed-point flag, and with them the rounds, the
// rebuilds, the dual findings and the statistics of every family but the singleton columns.
std::string result_line(const model_t& model, const options_t& o, const result_t& r)
{
  char buf[400];
  int32_t mask = 0;
  for (int32_t k = 0; k < rule::count; ++k) {
    mask |= o.enabled[k] << k;
  }
  std::snprintf(buf,
                sizeof(buf),
                "in %016llx opt %03x p%d | %s %016llx fp %d rounds %d explicit %d rebuilds %d "
                "stack %016llx records %zu dual %lld red %d %d %lld |",
                (unsigned long long)model_hash(model),
                mask,
                (int)o.primal_only,
                status_name(r.status),
                (unsigned long long)r.hash,
                (int)r.fixed_point,
                r.rounds,
                r.explicit_rounds,
                r.rebuilds,
                (unsigned long long)stack_hash(r.postsolve),
                r.postsolve.order.size(),
                (long long)r.dual_findings,
                r.reduced.nrows,
                r.reduced.ncols,
                (long long)r.reduced.nnz());
  std::string line = buf;
  put_stats(line, r.start);
  for (int32_t k = 0; k < rule::count; ++k) {
    if (k != rule::singleton_cols) { put_stats(line, r.stats[k]); }
  }
  return line;
}

struct profile_t {
  const char* name;
  generator_options_t options;
};

// The profiles of tools/gprs_fuzz.cpp.
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

// with_columns of tools/gprs_fuzz.cpp.
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

// add_singletons of tools/gprs_fuzz.cpp.
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

// The option draw of tools/gprs_fuzz.cpp at the base commit.
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
      break;
    default: break;
  }
  return o;
}

int run_dump(const char* path, bool on)
{
  model_t model;
  std::string error;
  if (!read_csr_dump(path, model, error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  const char* base = std::strrchr(path, '/');
  base             = base ? base + 1 : path;
  for (int32_t v = 0; v < 8; ++v) {
    options_t o;
    const char* name = "default";
    if (v == 1) {
      o.primal_only = true;
      name          = "primal_only";
    } else if (v == 2) {
      o.rebuild_dead_fraction  = 0.;
      o.rebuild_fixed_fraction = 0.;
      name                     = "rebuild_always";
    } else if (v == 3) {
      o.rebuild_dead_fraction  = 2.;
      o.rebuild_fixed_fraction = 2.;
      name                     = "rebuild_never";
    } else if (v == 4) {
      for (budget_t& b : o.budget) {
        b = budget_t{0.25, 0.125, 0};
      }
      name = "tiny_budget";
    } else if (v == 5) {
      o.max_rounds = 2;
      name         = "two_rounds";
    } else if (v == 6) {
      o.enabled[rule::propagation] = false;
      name                         = "no_propagation";
    } else if (v == 7) {
      o.enabled[rule::row_rules]     = false;
      o.enabled[rule::propagation]   = false;
      o.enabled[rule::tightening]    = false;
      o.enabled[rule::parallel_rows] = false;
      name                           = "doubleton_dual_only";
    }
    o.enabled[rule::singleton_cols] = o.enabled[rule::singleton_cols] && on;
    result_t r                      = presolve(model, o);
    std::printf("%s %s %s\n", base, name, result_line(model, o, r).c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc >= 4 && !std::strcmp(argv[1], "dump")) {
    return run_dump(argv[2], !std::strcmp(argv[3], "on"));
  }
  if (argc < 5 || std::strcmp(argv[1], "random")) {
    std::fprintf(stderr, "usage: see the head of ident.cpp\n");
    return 2;
  }
  uint64_t first  = std::strtoull(argv[2], nullptr, 10);
  uint64_t count  = std::strtoull(argv[3], nullptr, 10);
  bool on         = !std::strcmp(argv[4], "on");
  bool singletons = false, columns = false, verbose = false;
  uint64_t chunk  = 1000;
  for (int i = 5; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--singletons")) { singletons = true; }
    if (!std::strcmp(argv[i], "--columns")) { columns = true; }
    if (!std::strcmp(argv[i], "--verbose")) { verbose = true; }
    if (!std::strcmp(argv[i], "--chunk") && i + 1 < argc) {
      chunk = std::strtoull(argv[++i], nullptr, 10);
    }
  }
  int64_t models = 0;
  int32_t p      = 0;
  for (const profile_t& profile : profiles()) {
    fnv_t h;
    generator_options_t generator = columns ? with_columns(profile.options) : profile.options;
    for (uint64_t seed = first; seed < first + count; ++seed) {
      case_t c = oracle::generate(seed, generator);
      if (singletons) { c = add_singletons(c, seed, profile.options); }
      options_t o                     = variant(seed);
      o.enabled[rule::singleton_cols] = o.enabled[rule::singleton_cols] && on;
      result_t r                      = presolve(c.model, o);
      std::string line                = result_line(c.model, o, r);
      ++models;
      if (verbose) {
        std::printf("%d %s %llu %s\n", p, profile.name, (unsigned long long)seed, line.c_str());
      } else {
        h.bytes(line.data(), line.size());
        if ((seed - first + 1) % chunk == 0 || seed + 1 == first + count) {
          std::printf("%d %s %llu %016llx\n",
                      p,
                      profile.name,
                      (unsigned long long)(seed - (seed - first) % chunk),
                      (unsigned long long)h.h);
          h = fnv_t();
        }
      }
    }
    ++p;
  }
  std::fprintf(stderr, "models %lld\n", (long long)models);
  return 0;
}
