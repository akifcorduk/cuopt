/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// oracle::check_result on results written by hand: it accepts a model that was left as it is
// and rejects a wrong status, a changed optimum, a relaxed and a restricted reduced model and a
// wrong postsolve record. These tests need the postsolve of part B and no rule.

#include "harness.hpp"
#include "oracle.hpp"

#include <numeric>

using namespace gprs;
using oracle::case_t;
using oracle::outcome_t;
using oracle::verdict_t;

namespace {

result_t untouched(const model_t& model)
{
  result_t result;
  result.status  = status_t::unchanged;
  result.reduced = model;
  result.postsolve.init(model);
  result.postsolve.row_origin.resize(model.nrows);
  result.postsolve.col_origin.resize(model.ncols);
  std::iota(result.postsolve.row_origin.begin(), result.postsolve.row_origin.end(), 0);
  std::iota(result.postsolve.col_origin.begin(), result.postsolve.col_origin.end(), 0);
  return result;
}

outcome_t outcome_of(const model_t& base)
{
  std::vector<uint8_t> line;
  GPRS_CHECK(oracle::classify_widest(base, oracle::check_max_assignments, line));
  return oracle::solve(base, line).outcome;
}

// One integer column in [0, 5] with the given cost and, when `row` is set, the row x <= 2.
model_t one_column(double cost, bool row)
{
  model_t m;
  m.nrows     = row;
  m.ncols     = 1;
  m.row_start = {0};
  if (row) {
    m.row_start.push_back(1);
    m.col_index = {0};
    m.value     = {1.};
    m.row_lower = {-inf};
    m.row_upper = {2.};
  }
  m.col_lower = {0.};
  m.col_upper = {5.};
  m.cost      = {cost};
  m.col_type  = {1};
  return m;
}

options_t primal()
{
  options_t options;
  options.primal_only = true;
  return options;
}

}  // namespace

GPRS_TEST(e2e_check_accepts_an_untouched_model)
{
  oracle::generator_options_t variants[3];
  variants[1].p_scaled      = 1.;
  variants[2].max_line_cols = 3;
  variants[2].p_huge_bound  = 0.6;
  int32_t failures          = 0;
  for (const oracle::generator_options_t& o : variants) {
    for (uint64_t seed = 7000000; seed < 7001500 && failures < 3; ++seed) {
      case_t c = oracle::generate(seed, o);
      for (const options_t& options : {options_t(), primal()}) {
        verdict_t v = oracle::check_result(c, options, untouched(c.model));
        if (v.ok) { continue; }
        ++failures;
        test::fail(__FILE__,
                   __LINE__,
                   "seed " + std::to_string(seed) + ": " + v.message + "\n" + oracle::to_code(c));
      }
    }
  }
}

GPRS_TEST(e2e_check_follows_the_status)
{
  // Without huge bounds every objective value is small against a change of 1.
  oracle::generator_options_t o;
  o.p_huge_bound  = 0.;
  int32_t seen[3] = {0, 0, 0};
  for (uint64_t seed = 7100000; seed < 7100600; ++seed) {
    case_t c          = oracle::generate(seed, o);
    outcome_t outcome = outcome_of(c.base);
    ++seen[(int32_t)outcome];
    result_t result = untouched(c.model);
    GPRS_CHECK(oracle::check_result(c, options_t(), result).ok);
    result.status = status_t::failed;
    GPRS_CHECK(!oracle::check_result(c, options_t(), result).ok);
    // With these statuses the reduced model is not read.
    result.reduced = model_t();
    result.status  = status_t::infeasible;
    GPRS_CHECK_EQ(oracle::check_result(c, options_t(), result).ok,
                  outcome == outcome_t::infeasible);
    result.status = status_t::unbounded_or_infeasible;
    GPRS_CHECK_EQ(oracle::check_result(c, options_t(), result).ok, outcome != outcome_t::optimal);
    // Another outcome of the reduced model: a row that nothing satisfies, a free column that
    // the objective pushes down.
    result = untouched(c.model);
    if (c.model.nrows > 0 && outcome != outcome_t::infeasible) {
      result.reduced.row_lower[0] = 1.;
      result.reduced.row_upper[0] = 0.;
      GPRS_CHECK(!oracle::check_result(c, options_t(), result).ok);
      // The outcomes tell, with a dual finding as without one.
      result.dual_findings = 1;
      GPRS_CHECK(!oracle::check_result(c, options_t(), result).ok);
    }
    // A changed offset shows in the optimum, and in the objective of the crushed points: one
    // that is higher always, one that is lower when the result has no dual finding.
    result = untouched(c.model);
    result.reduced.objective_offset += 1.;
    GPRS_CHECK_EQ(oracle::check_result(c, options_t(), result).ok,
                  outcome == outcome_t::infeasible);
    result.status        = status_t::limit;
    result.dual_findings = 1;
    GPRS_CHECK_EQ(oracle::check_result(c, options_t(), result).ok,
                  outcome == outcome_t::infeasible);
    result.reduced.objective_offset -= 2.;
    GPRS_CHECK_EQ(oracle::check_result(c, options_t(), result).ok, outcome != outcome_t::optimal);
    result.dual_findings = 0;
    GPRS_CHECK_EQ(oracle::check_result(c, options_t(), result).ok,
                  outcome == outcome_t::infeasible);
  }
  GPRS_CHECK(seen[0] > 100 && seen[1] > 100 && seen[2] > 20);
}

GPRS_TEST(e2e_check_rejects_a_relaxed_model)
{
  // The reduced model lost the row x <= 2. With a cost the optimum moves; without one only
  // the feasible points 3, 4, 5 show it.
  for (double cost : {-1., 0., 1.}) {
    case_t c        = oracle::make_case(one_column(cost, true), {}, {});
    result_t result = untouched(c.model);
    GPRS_CHECK(oracle::check_result(c, options_t(), result).ok);
    result.reduced = one_column(cost, false);
    result.postsolve.row_origin.clear();
    verdict_t v = oracle::check_result(c, options_t(), result);
    GPRS_CHECK(!v.ok);
    GPRS_CHECK_EQ(v.message.find("feasible point") != std::string::npos, cost >= 0.);
  }
}

GPRS_TEST(e2e_check_rejects_a_restricted_model)
{
  // The reduced model has x <= 2 where the model allows 5; no optimum moves with zero cost.
  case_t c                    = oracle::make_case(one_column(0., false), {}, {});
  result_t result             = untouched(c.model);
  result.reduced.col_upper[0] = 2.;
  GPRS_CHECK(!oracle::check_result(c, primal(), result).ok);
  GPRS_CHECK(!oracle::check_result(c, options_t(), result).ok);
  // A dual finding does not excuse it: the point x = 5 has no image, and a dual reduction that
  // leaves its column in the model gives crush a record to move the point with.
  result.dual_findings = 1;
  GPRS_CHECK(!oracle::check_result(c, options_t(), result).ok);
  GPRS_CHECK(!oracle::check_result(c, primal(), result).ok);
  // The optimal point itself is cut off: caught in every mode by the optimum.
  c                           = oracle::make_case(one_column(-1., false), {}, {});
  result                      = untouched(c.model);
  result.reduced.col_upper[0] = 2.;
  result.dual_findings        = 1;
  GPRS_CHECK(!oracle::check_result(c, options_t(), result).ok);
}

GPRS_TEST(e2e_check_crushes_through_a_dual_fixing)
{
  // One integer column in [0, 5] and no row, fixed by its locks at the bound that its cost
  // prefers. The reduced model is empty and holds the cost of that bound as its offset; crush
  // drops the column, which moves every point to the bound.
  for (double cost : {1., 0., -1.}) {
    case_t c     = oracle::make_case(one_column(cost, false), {}, {});
    double bound = cost >= 0. ? 0. : 5.;
    auto fixed   = [&c](double value, double offset) {
      result_t result;
      result.status                   = status_t::reduced_to_empty;
      result.reduced.row_start        = {0};
      result.reduced.objective_offset = offset;
      result.postsolve.init(c.model);
      result.postsolve.push_fixed_col(0, value);
      result.dual_findings = 1;
      return result;
    };
    result_t result = fixed(bound, cost * bound);
    verdict_t v     = oracle::check_result(c, options_t(), result);
    GPRS_CHECK(v.ok);
    GPRS_CHECK(v.crushed_points > 1);
    GPRS_CHECK_EQ(v.skipped_points, 0);
    // No result of primal-only mode has a dual finding.
    v = oracle::check_result(c, primal(), result);
    GPRS_CHECK(!v.ok);
    GPRS_CHECK(v.message.find("dual findings") != std::string::npos);
    // Without the finding the points off the bound must keep their cost, in both modes.
    result.dual_findings = 0;
    GPRS_CHECK_EQ(oracle::check_result(c, options_t(), result).ok, cost == 0.);
    GPRS_CHECK_EQ(oracle::check_result(c, primal(), result).ok, cost == 0.);
    // Fixed at the other bound: wrong with a cost.
    result = fixed(5. - bound, cost * (5. - bound));
    GPRS_CHECK_EQ(oracle::check_result(c, options_t(), result).ok, cost == 0.);
  }

  // min -y with y >= 0 continuous and the row y - x <= 0, x integer in [0, 5] with cost 0: the
  // model is bounded through x. A wrong fixing of x at 0 by the lock that the row puts on it
  // leaves y <= 0: the optimum is another one, and the crushed point of (5, 5) is infeasible.
  model_t m;
  m.nrows     = 1;
  m.ncols     = 2;
  m.row_start = {0, 2};
  m.col_index = {0, 1};
  m.value     = {-1., 1.};
  m.row_lower = {-inf};
  m.row_upper = {0.};
  m.col_lower = {0., 0.};
  m.col_upper = {5., inf};
  m.cost      = {0., -1.};
  m.col_type  = {1, 0};
  case_t c    = oracle::make_case(m, {}, {});
  result_t result;
  result.status            = status_t::reduced;
  result.reduced.nrows     = 1;
  result.reduced.ncols     = 1;
  result.reduced.row_start = {0, 1};
  result.reduced.col_index = {0};
  result.reduced.value     = {1.};
  result.reduced.row_lower = {-inf};
  result.reduced.row_upper = {0.};
  result.reduced.col_lower = {0.};
  result.reduced.col_upper = {inf};
  result.reduced.cost      = {-1.};
  result.reduced.col_type  = {0};
  result.postsolve.init(c.model);
  result.postsolve.push_fixed_col(0, 0.);
  result.postsolve.row_origin = {0};
  result.postsolve.col_origin = {1};
  result.dual_findings        = 1;
  GPRS_CHECK(!oracle::check_result(c, options_t(), result).ok);
  // With x fixed at 5, where no row objects, the result is right.
  result.reduced.row_upper = {5.};
  result.postsolve.init(c.model);
  result.postsolve.push_fixed_col(0, 5.);
  result.postsolve.row_origin = {0};
  result.postsolve.col_origin = {1};
  GPRS_CHECK(oracle::check_result(c, options_t(), result).ok);
}

GPRS_TEST(e2e_check_sees_the_far_side_of_a_line_column)
{
  // min y with 0 <= y and the row y <= 5, y continuous. A reduced model without the row has
  // the same optimum and the same feasible point y = 0; only a point beyond 5 shows the loss.
  model_t m      = one_column(1., true);
  m.col_type     = {0};
  m.col_upper    = {inf};
  m.row_upper    = {5.};
  case_t c       = oracle::make_case(m, {}, {});
  model_t lost   = one_column(1., false);
  lost.col_type  = {0};
  lost.col_upper = {inf};
  for (const options_t& options : {options_t(), primal()}) {
    result_t result = untouched(c.model);
    GPRS_CHECK(oracle::check_result(c, options, result).ok);
    result.reduced = lost;
    result.postsolve.row_origin.clear();
    verdict_t v = oracle::check_result(c, options, result);
    GPRS_CHECK(!v.ok);
    GPRS_CHECK(v.message.find("far point") != std::string::npos);
  }
  // min y with 0 <= y <= 10 and a reduced model with y <= 4: the optimum and the point y = 0
  // stay, the point y = 10 has no image. Wrong with a dual finding as without one: the column
  // stays in the model and no record moves the point.
  m                           = one_column(1., false);
  m.col_type                  = {0};
  m.col_upper                 = {10.};
  c                           = oracle::make_case(m, {}, {});
  result_t result             = untouched(c.model);
  result.reduced.col_upper[0] = 4.;
  GPRS_CHECK(!oracle::check_result(c, primal(), result).ok);
  GPRS_CHECK(!oracle::check_result(c, options_t(), result).ok);
  result.dual_findings = 1;
  GPRS_CHECK(!oracle::check_result(c, options_t(), result).ok);
  // The same with an infinite upper bound: the far point lies far_step inside.
  m.col_upper                 = {inf};
  c                           = oracle::make_case(m, {}, {});
  result                      = untouched(c.model);
  result.reduced.col_upper[0] = 1000.;
  GPRS_CHECK(!oracle::check_result(c, primal(), result).ok);
}

GPRS_TEST(e2e_check_rejects_origins_that_are_no_rows_or_columns_of_the_model)
{
  model_t m = one_column(1., true);
  case_t c  = oracle::make_case(m, {}, {});
  GPRS_CHECK(oracle::check_result(c, options_t(), untouched(c.model)).ok);
  for (int32_t variant = 0; variant < 6; ++variant) {
    result_t result = untouched(c.model);
    if (variant == 0) { result.postsolve.col_origin = {1}; }
    if (variant == 1) { result.postsolve.col_origin = {-1}; }
    if (variant == 2) { result.postsolve.row_origin = {1}; }
    if (variant == 3) { result.postsolve.row_origin = {-1}; }
    if (variant == 4) { result.postsolve.ncols = 2; }
    if (variant == 5) { result.postsolve.nrows = 0; }
    verdict_t v = oracle::check_result(c, options_t(), result);
    GPRS_CHECK(!v.ok);
    GPRS_CHECK(v.message.find("postsolve stack") != std::string::npos ||
               v.message.find("row_origin / col_origin") != std::string::npos);
  }
  // Two columns of the reduced model from one column of the model.
  model_t two = one_column(1., false);
  two.ncols   = 2;
  two.col_lower.push_back(0.);
  two.col_upper.push_back(1.);
  two.cost.push_back(0.);
  two.col_type.push_back(1);
  c               = oracle::make_case(two, {}, {});
  result_t result = untouched(c.model);
  GPRS_CHECK(oracle::check_result(c, options_t(), result).ok);
  result.postsolve.col_origin = {0, 0};
  verdict_t v                 = oracle::check_result(c, options_t(), result);
  GPRS_CHECK(!v.ok);
  GPRS_CHECK(v.message.find("row_origin / col_origin") != std::string::npos);
}

GPRS_TEST(e2e_check_reads_the_postsolve_records)
{
  // min x0 + x1 with x0 + x1 <= 3, x0 in [0, 3], x1 in [2, 3]. No row stops x1 from moving
  // down, so its locks fix it at 2 and leave x0 <= 1 with the offset 2. A record with the
  // value 3 gives a feasible point with another objective value, one with the value 4 an
  // infeasible point.
  model_t m;
  m.nrows     = 1;
  m.ncols     = 2;
  m.row_start = {0, 2};
  m.col_index = {0, 1};
  m.value     = {1., 1.};
  m.row_lower = {-inf};
  m.row_upper = {3.};
  m.col_lower = {0., 2.};
  m.col_upper = {3., 3.};
  m.cost      = {1., 1.};
  m.col_type  = {1, 1};
  case_t c    = oracle::make_case(m, {}, {});

  model_t reduced;
  reduced.nrows            = 1;
  reduced.ncols            = 1;
  reduced.row_start        = {0, 1};
  reduced.col_index        = {0};
  reduced.value            = {1.};
  reduced.row_lower        = {-inf};
  reduced.row_upper        = {1.};
  reduced.col_lower        = {0.};
  reduced.col_upper        = {3.};
  reduced.cost             = {1.};
  reduced.col_type         = {1};
  reduced.objective_offset = 2.;

  for (double value : {2., 3., 4.}) {
    result_t result;
    result.status        = status_t::reduced;
    result.dual_findings = 1;
    result.reduced       = reduced;
    result.postsolve.init(m);
    result.postsolve.push_fixed_col(1, value);
    result.postsolve.row_origin = {0};
    result.postsolve.col_origin = {0};
    verdict_t v                 = oracle::check_result(c, options_t(), result);
    GPRS_CHECK_EQ(v.ok, value == 2.);
    GPRS_CHECK_EQ(v.message.find("infeasible for the model") != std::string::npos, value == 4.);
    // The point (0, 3) of the model costs 3 and its crushed point 2: no result of primal-only
    // mode and no result without a dual finding does that.
    GPRS_CHECK(!oracle::check_result(c, primal(), result).ok);
    result.dual_findings = 0;
    GPRS_CHECK(!oracle::check_result(c, options_t(), result).ok);
    result.dual_findings        = 1;
    result.postsolve.col_origin = {0, 1};
    GPRS_CHECK(!oracle::check_result(c, options_t(), result).ok);
  }

  // The same model with the row x0 + x1 >= 3, which locks x1 down: fixing x1 at 2 keeps the
  // optimum, but dropping x1 is no map of the feasible points. (0, 3) crushes to x0 = 0, which
  // the reduced row x0 >= 1 rejects.
  m.row_lower       = {3.};
  m.row_upper       = {inf};
  c                 = oracle::make_case(m, {}, {});
  reduced.row_lower = {1.};
  reduced.row_upper = {inf};
  result_t result;
  result.status        = status_t::reduced;
  result.dual_findings = 1;
  result.reduced       = reduced;
  result.postsolve.init(m);
  result.postsolve.push_fixed_col(1, 2.);
  result.postsolve.row_origin = {0};
  result.postsolve.col_origin = {0};
  verdict_t v                 = oracle::check_result(c, options_t(), result);
  GPRS_CHECK(!v.ok);
  GPRS_CHECK(v.message.find("infeasible for the reduced model") != std::string::npos);
}
