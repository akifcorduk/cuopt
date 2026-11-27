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
// One reduction leaves the class: an equation subtracted from a row that is parallel to it up
// to singleton columns (rules.hpp (e)) can leave that row with two line columns, its own and
// that of the equation. into_class() takes such a row back: it subtracts from it a multiple of
// an equation of the model that holds one of the two as its only line column, which leaves the
// feasible points as they are, and the solver then sees a model of the class.
//
// Where the tolerance enters, row by row: a row whose sides and enumerated coefficients are all
// integers is tested exactly; any other row with solve_tolerance * max(1, sum |a * x|, |side|)
// for each side on its own (a huge side does not loosen the other one). The coefficient of an
// integer line column counts as an enumerated one here: the end that its row gives the column is
// rounded inwards, and a coefficient one rounding step away from an integer (coefficient
// strengthening with the error bound of an activity) would move that end by one.
// An end of the interval of a line column is exact when it is a bound that is a multiple of 1/2
// or comes from a row of the first kind; any other end gets solve_tolerance * max(1, |end|). A
// scaled case (factors that are powers of two) gets the tolerance in its scaled rows and is
// still decided as its base model, because distinct values of its data are much further apart
// than the tolerance.

#include <gprs/presolve.hpp>
#include <gprs/state.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace gprs {
namespace oracle {

constexpr double solve_tolerance = 1e-9;

// Distance of a point of solution_t::far from an end whose other end is infinite, in units of
// max(1, |end|). An integer, so that an integer line column stays integral.
constexpr double far_step = 1024.;

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

  // Per finite side: the side sits on an end of the row's activity range over the small
  // bounds of its columns (forcing or exactly redundant when no bound is infinite or huge).
  double p_forcing_side = 0.15;
  // Per finite side that is not on an end: the side lies one beyond an end of that range.
  double p_outside_side = 0.1;
  // Per row drawn entry by entry: every coefficient is a multiple of one factor above 1.
  double p_common_factor = 0.15;
  // Per case: one row or one column gets lower > upper.
  double p_crossing = 0.03;
  // Per small finite bound of an integer column: the bound moves outwards by 1/2, which leaves
  // the integer points of the column as they are.
  double p_fractional_bound = 0.05;

  // Singleton columns, behind the columns counted above: per row with an entry, the row gets a
  // column that no other row holds, up to max_singleton_cols per model. With p_parallel_row and
  // p_equation this gives rows that are parallel up to singleton columns, equations among
  // them. The column is a line column with p_singleton_line when the row has none, else an
  // integer column with bounds one apart; its cost is drawn as that of the other columns. With
  // p_singleton_col == 0 nothing is drawn for it and the cases are those of before.
  double p_singleton_col     = 0.;
  double p_singleton_line    = 0.4;
  int32_t max_singleton_cols = 3;
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
// The base model: rows, enumerated columns and line columns are counted uniformly from 0 to
// their maximum and the line columns sit at random positions. A row is, in this order of the
// draws, a multiple of an earlier row (with the sides of that row or its own), empty, a
// singleton, a doubleton equation, or drawn entry by entry with the density (at least two
// entries where the columns allow it); a column that no row took and that p_empty_col did not
// pick goes into a row drawn entry by entry. The sides of a row are, in this order, both
// infinite, an equation, a range, or one side, with values in the row's activity range over
// the small bounds of its columns, on its ends or one beyond. Every coefficient, cost, side
// and small bound is an integer inside the limits of the options, except bounds moved by
// p_fractional_bound. An integer line column with two small finite bounds is wider than
// max_width. Requires max_coefficient >= 1, max_width >= 0, max_side > max_width and
// 0 <= max_scale_exp2 <= 30.
// With p_singleton_col > 0 the singleton columns follow the columns counted above, in the
// order of their rows; a row that is a multiple of an earlier row copies that row without its
// singleton column and can get one of its own. The sides of a row are drawn with its singleton
// column in the activity range.
case_t generate(uint64_t seed, const generator_options_t& options);

// base with row i multiplied by row_scale[i] > 0 and column j divided by col_scale[j] > 0
// (coefficients and cost times col_scale[j], bounds divided by it). Exact for powers of two.
model_t scale_model(const model_t& base,
                    const std::vector<double>& row_scale,
                    const std::vector<double>& col_scale);

// The case of a base model and its factors; empty factors stand for ones.
case_t make_case(const model_t& base,
                 const std::vector<double>& row_scale,
                 const std::vector<double>& col_scale);

// C++ statements that build the case in a variable `c` (through a model_t `base` and
// make_case), every number with all its digits; they compile inside a test that has
// `using namespace gprs;`.
std::string to_code(const case_t& c);

// Marks the line columns of a model. Returns false when a row holds two of them, that is when
// the model is outside the oracle class.
bool classify(const model_t& model, int32_t max_width, std::vector<uint8_t>& line_col);

// Number of assignments of the enumerated columns that solve() visits, as a double; 0 when an
// enumerated column has no integer point.
double assignments(const model_t& model, const std::vector<uint8_t>& line_col);

// classify() with the largest max_width that keeps assignments() at or below max_assignments.
// Returns false when no max_width puts the model into the oracle class within that limit.
bool classify_widest(const model_t& model, double max_assignments, std::vector<uint8_t>& line_col);

// A model of the class with the feasible points and the objective of `model`, and its line
// columns: `model` itself with classify_widest() where that succeeds. Otherwise, for the largest
// max_width within the limit that allows it, `model` with every row that holds several line
// columns reduced to one: as long as such a row has a line column j that an equation of the
// model holds as its only line column, that equation times the ratio of the two coefficients
// of j is subtracted from the row (the first such column of the row, the first such equation).
// A coefficient that cancels within 1e-12 of the two terms is dropped. False when a row is left
// with two line columns for every max_width.
bool into_class(const model_t& model,
                double max_assignments,
                model_t& in_class,
                std::vector<uint8_t>& line_col);

enum class outcome_t : uint8_t { optimal, infeasible, unbounded };

struct solution_t {
  outcome_t outcome = outcome_t::infeasible;
  double objective  = 0.;  // with the offset; for optimal
  std::vector<double> x;   // an optimal point; for optimal
  // Feasible points, whatever the outcome: one per feasible assignment of the enumerated
  // columns, the first keep_points of them in enumeration order. Each line column sits at the
  // end of its interval that its cost prefers if that end is finite, else at the other end,
  // else at 0. A line column with zero cost prefers the lower end.
  std::vector<std::vector<double>> feasible;
  // far[k] is feasible[k] with every line column at the other end of its interval or, where
  // that end is infinite, far_step * max(1, |value|) further into the interval. It is feasible
  // as well: given the enumerated columns, the intervals of the line columns form a box.
  std::vector<std::vector<double>> far;
};

// Exact on the oracle class (see the top of this file). The enumerated columns are enumerated
// in lexicographic order, column 0 slowest, values ascending. Work: product of the widths of
// the enumerated columns times the nonzeros.
// The model is well_formed and line_col marks at most one column per row; a column that is not
// marked is integer with finite bounds. Among assignments of equal objective value the first
// in enumeration order gives x. unbounded means that a feasible assignment has a line column
// whose cost points to an infinite end of its interval.
solution_t solve(const model_t& model,
                 const std::vector<uint8_t>& line_col,
                 int32_t keep_points = 0);

struct verdict_t {
  bool ok = true;
  std::string message;
};

// Feasible points of the reduced model that check_presolve postsolves.
constexpr int32_t checked_points = 64;

// Assignments up to which check_presolve enumerates a model (classify_widest).
constexpr double check_max_assignments = 16384.;

// Presolves c.model, solves the reduced model and compares with solve() on c.base:
// - infeasible / unbounded_or_infeasible from the presolve agree with the base outcome;
// - otherwise the reduced model is well_formed and into_class() takes it into the oracle class,
//   it has the outcome of the base model, and for optimal its optimal value equals the base
//   optimum and its optimal point, postsolved, is feasible for c.model (check_point,
//   options.tol) with that objective value;
// - every point of solve(reduced, ..., checked_points).feasible, postsolved, is feasible for
//   c.model;
// - with options.primal_only and a feasible base model: the optimal point of the base model
//   (or its first feasible point when it is unbounded), taken to c.model by col_scale and
//   crushed, is feasible for the reduced model and has the same objective value there.
// Objective values are compared with a relative tolerance of 1e-9.
// In addition to the above:
// - the base model is classified by classify_widest(..., check_max_assignments) and the reduced
//   model by into_class(..., check_max_assignments), so that an integer column with narrow
//   finite bounds is enumerated wherever that is affordable;
// - the feasible points that are postsolved are checked_points of all those of the reduced
//   model, evenly spaced in enumeration order (all of them when there are fewer), each with
//   its point of solution_t::far: a constraint that the reduced model lost on the side of a
//   line column that its cost does not push against shows only there;
// - the crush test also runs when the dual-fixing family is disabled or reports no finding,
//   and it takes the optimal point and checked_points evenly spaced feasible points of the
//   base model, each with its point of solution_t::far;
// - status failed fails, and so does a postsolve stack whose sizes are not those of c.model or
//   whose row_origin / col_origin are not distinct rows / columns of c.model, one per row /
//   column of the reduced model.
verdict_t check_presolve(const case_t& c, const options_t& options);

// The checks of check_presolve on a result that is already there.
verdict_t check_result(const case_t& c, const options_t& options, const result_t& result);

}  // namespace oracle
}  // namespace gprs
