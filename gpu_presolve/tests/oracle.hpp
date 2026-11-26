/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// Test oracle: tiny random MIPs, an exact solver for them, and the end-to-end check of a presolve.
//
// The class of models the solver is exact for ("oracle class"):
// - every column is either enumerated or a line column;
// - an enumerated column is integer with finite bounds and upper - lower <= max_width;
// - a line column is anything else: continuous or integer, with finite, huge or infinite bounds;
// - every row holds at most one line column.
// Given values of the enumerated columns, each row with a line column confines that column to an
// interval and each line column appears in the objective alone, so the best completion is found
// per line column on the ends of its interval (rounded inwards for an integer line column), or
// the model is unbounded along that column. Enumerating the enumerated columns therefore decides
// infeasible / unbounded / optimal without a linear program. On a model with integer data of the
// generator's size every sum is exact in double arithmetic; a comparison uses solve_tolerance
// only on models whose data are not integer (a reduced model after substitutions).
//
// The generator draws from this class. Stage 1 keeps a reduced model inside it: a line column
// that a substitution puts into the rows of an enumerated column inherits that column's bounds
// and is enumerated afterwards, which classify() re-derives on the reduced model.

#include <gprs/presolve.hpp>
#include <gprs/state.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace gprs {
namespace oracle {

constexpr double solve_tolerance = 1e-9;

// splitmix64; the same sequence on every platform.
struct rng_t {
  uint64_t state;

  explicit rng_t(uint64_t seed);
  uint64_t next();
  // Uniform in [lo, hi].
  int32_t uniform(int32_t lo, int32_t hi);
  bool chance(double probability);
};

struct generator_options_t {
  int32_t max_rows        = 5;
  int32_t max_enum_cols   = 5;
  int32_t max_line_cols   = 2;
  int32_t max_coefficient = 3;   // |a| and |cost| are integers up to this
  int32_t max_side        = 12;  // |side| and |bound| of enumerated columns
  int32_t max_width       = 3;   // upper - lower of an enumerated column
  double density          = 0.5;

  // Structure the Stage 1 rules act on, each drawn per row or column.
  double p_equation           = 0.2;
  double p_ranged_row         = 0.1;
  double p_free_row           = 0.05;
  double p_empty_row          = 0.05;
  double p_singleton_row      = 0.15;
  double p_doubleton_equation = 0.15;
  double p_parallel_row       = 0.15;  // a multiple of an earlier row with its own sides
  double p_empty_col          = 0.05;
  double p_fixed_col          = 0.1;
  double p_zero_cost          = 0.3;

  // Line columns.
  double p_continuous     = 0.5;
  double p_infinite_bound = 0.5;  // per bound; both infinite gives a free column
  double p_huge_bound     = 0.2;  // per finite bound: magnitude up to 1e22, so that a term can
                                  // count as infinite in an activity

  // Probability that the case is scaled, and the largest exponent of the power-of-two row and
  // column factors. Column factors apply to continuous columns only.
  double p_scaled        = 0.2;
  int32_t max_scale_exp2 = 10;
};

// A model for the presolve and the integer-data model it was scaled from.
// model row i = row_scale[i] * base row i; x_base[j] = col_scale[j] * x_model[j].
struct case_t {
  model_t model;
  model_t base;
  std::vector<double> row_scale;
  std::vector<double> col_scale;
};

// The same case for the same seed and options on every platform. Models with zero rows or zero
// columns occur.
case_t generate(uint64_t seed, const generator_options_t& options);

// Marks the line columns of a model. Returns false when a row holds two of them, that is when
// the model is outside the oracle class.
bool classify(const model_t& model, int32_t max_width, std::vector<uint8_t>& line_col);

enum class outcome_t : uint8_t { optimal, infeasible, unbounded };

struct solution_t {
  outcome_t outcome = outcome_t::infeasible;
  double objective  = 0.;  // with the offset; for optimal
  std::vector<double> x;   // an optimal point; for optimal
  // Feasible points, whatever the outcome: one per feasible assignment of the enumerated
  // columns, the first keep_points of them in enumeration order. Each line column sits at the
  // end of its interval that its cost prefers if that end is finite, else at the other end,
  // else at 0.
  std::vector<std::vector<double>> feasible;
};

// Exact on the oracle class (see the top of this file). The enumerated columns are enumerated
// in lexicographic order, column 0 slowest, values ascending. Work: product of the widths of
// the enumerated columns times the nonzeros.
solution_t solve(const model_t& model,
                 const std::vector<uint8_t>& line_col,
                 int32_t keep_points = 0);

struct verdict_t {
  bool ok = true;
  std::string message;
};

// Feasible points of the reduced model that check_presolve postsolves.
constexpr int32_t checked_points = 64;

// Presolves c.model, solves the reduced model and compares with solve() on c.base:
// - infeasible / unbounded_or_infeasible from the presolve agree with the base outcome;
// - otherwise the reduced model is well_formed and in the oracle class, has the outcome of the
//   base model, and for optimal its optimal value equals the base optimum and its optimal
//   point, postsolved, is feasible for c.model (check_point, options.tol) with that objective
//   value;
// - every point of solve(reduced, ..., checked_points).feasible, postsolved, is feasible for
//   c.model;
// - with options.primal_only and a feasible base model: the optimal point of the base model
//   (or its first feasible point when it is unbounded), taken to c.model by col_scale and
//   crushed, is feasible for the reduced model and has the same objective value there.
// Objective values are compared with a relative tolerance of 1e-9.
verdict_t check_presolve(const case_t& c, const options_t& options);

}  // namespace oracle
}  // namespace gprs
