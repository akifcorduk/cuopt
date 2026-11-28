// The profiles and add_singletons of tools/gprs_fuzz.cpp, for the tools of the M0 review.
#pragma once

#include "oracle.hpp"

#include <cmath>
#include <vector>

namespace ident {

using namespace gprs;
using oracle::case_t;
using oracle::generator_options_t;

struct profile_t {
  const char* name;
  generator_options_t options;
};

// The profiles of tools/gprs_fuzz.cpp.
inline std::vector<profile_t> profiles()
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

// add_singletons of tools/gprs_fuzz.cpp.
inline case_t add_singletons(const case_t& c, uint64_t seed, const generator_options_t& g)
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

}  // namespace ident
