/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// The structure check (check/structure_check.hpp) on what it reads besides bounds and rows: the
// objective, the mode of the result, the origin of an alias and the kinds of the records. On
// results written by hand, right and wrong, and on the engine.

#include "../check/structure_check.hpp"
#include "harness.hpp"
#include "oracle.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace gprs;
using gprs_check::structure_check;
using gprs_check::structure_report_t;

namespace {

// Columns x, y, s, t in this order.
constexpr int32_t x = 0, y = 1, s = 2, t = 3;

// min 2 x + 3 y + 5 s + t + 10 with the row y - 2 x = 1, x and y in [0, 3], s in [-10, 10],
// t in [0, 1].
model_t cost_model()
{
  model_t m;
  m.nrows            = 1;
  m.ncols            = 4;
  m.row_start        = {0, 2};
  m.col_index        = {x, y};
  m.value            = {-2., 1.};
  m.row_lower        = {1.};
  m.row_upper        = {1.};
  m.col_lower        = {0., 0., -10., 0.};
  m.col_upper        = {3., 3., 10., 1.};
  m.cost             = {2., 3., 5., 1.};
  m.col_type         = {0, 0, 0, 0};
  m.objective_offset = 10.;
  return m;
}

// The reduced model of cost_model after y = 2 x + 1 and t fixed at 1: the columns x in [0, 1]
// and s with the given costs and offset, and no row. The right ones are 2 + 3 * 2 = 8, 5 and
// 10 + 3 * 1 + 1 * 1 = 14.
result_t substituted(const model_t& m, double cost_x, double cost_s, double offset)
{
  result_t result;
  result.status                   = status_t::reduced;
  result.reduced.ncols            = 2;
  result.reduced.row_start        = {0};
  result.reduced.col_lower        = {0., -10.};
  result.reduced.col_upper        = {1., 10.};
  result.reduced.cost             = {cost_x, cost_s};
  result.reduced.col_type         = {0, 0};
  result.reduced.objective_offset = offset;
  result.postsolve.init(m);
  result.postsolve.col_origin = {x, s};
  result.postsolve.push_alias(y, x, 0, 2., 1.);
  result.postsolve.push_fixed_col(t, 1.);
  return result;
}

bool silent(const structure_report_t& r)
{
  return r.stack_error.empty() && r.lost_integers == 0 && r.bound_failures == 0 &&
         r.row_failures == 0 && r.kept_failures == 0 && r.added_failures == 0 &&
         r.objective_failures == 0 && r.postsolve_failures == 0;
}

// The same, but for the objective.
bool only_objective(const structure_report_t& r)
{
  return r.stack_error.empty() && r.lost_integers == 0 && r.bound_failures == 0 &&
         r.row_failures == 0 && r.kept_failures == 0 && r.added_failures == 0 &&
         r.objective_failures > 0 && r.postsolve_failures == 0;
}

// An alias record as it is given, without the contract of push_alias.
void put_alias(result_t& result, const alias_record_t& record)
{
  result.postsolve.order.push_back(record_kind_t::alias);
  result.postsolve.aliases.push_back(record);
  result.postsolve.col_removed[record.col] = 1;
}

// cost_model with y aliased to x by the given record and t left in the reduced model.
result_t aliased(const model_t& m, const alias_record_t& record)
{
  result_t result;
  result.status                   = status_t::reduced;
  result.reduced.ncols            = 3;
  result.reduced.row_start        = {0};
  result.reduced.col_lower        = {0., -10., 0.};
  result.reduced.col_upper        = {1., 10., 1.};
  result.reduced.cost             = {8., 5., 1.};
  result.reduced.col_type         = {0, 0, 0};
  result.reduced.objective_offset = 13.;
  result.postsolve.init(m);
  result.postsolve.col_origin = {x, s, t};
  put_alias(result, record);
  return result;
}

}  // namespace

GPRS_TEST(e2e_structure_objective_of_a_substitution_and_a_fixing)
{
  model_t m = cost_model();
  GPRS_CHECK(silent(structure_check(m, substituted(m, 8., 5., 14.), false)));
  GPRS_CHECK(silent(structure_check(m, substituted(m, 8., 5., 14.), true)));

  // The cost of the substituted column reaches its rep without the scale.
  structure_report_t r = structure_check(m, substituted(m, 5., 5., 14.), false);
  GPRS_CHECK(only_objective(r));
  GPRS_CHECK_EQ(r.objective_failures, 1);
  GPRS_CHECK_EQ(r.objective_worst_col, x);
  GPRS_CHECK_EQ(r.objective_worst, 3. / (2. + 6. + 5.));
  GPRS_CHECK_EQ(r.objective_constant, 0.);
  // It does not reach it at all.
  r = structure_check(m, substituted(m, 2., 5., 14.), false);
  GPRS_CHECK(only_objective(r));
  GPRS_CHECK_EQ(r.objective_worst_col, x);
  // The cost of a column that stays is another one.
  r = structure_check(m, substituted(m, 8., 0., 14.), false);
  GPRS_CHECK(only_objective(r));
  GPRS_CHECK_EQ(r.objective_worst_col, s);
  GPRS_CHECK_EQ(r.objective_worst, 1.);
  // The offset of the alias, the value of the fixed column or both are not in the offset.
  for (double offset : {11., 13., 10.}) {
    r = structure_check(m, substituted(m, 8., 5., offset), false);
    GPRS_CHECK(only_objective(r));
    GPRS_CHECK_EQ(r.objective_failures, 1);
    GPRS_CHECK_EQ(r.objective_worst, 0.);
    GPRS_CHECK_EQ(r.objective_constant, (14. - offset) / (10. + 3. + 1. + offset));
  }
  // Both a cost and the offset.
  r = structure_check(m, substituted(m, 8., 4., 15.), false);
  GPRS_CHECK_EQ(r.objective_failures, 2);

  // What the rounding of a cost or of the offset can be passes, more does not.
  GPRS_CHECK(silent(structure_check(m, substituted(m, 8. * (1. + 1e-9), 5., 14.), false)));
  GPRS_CHECK(only_objective(structure_check(m, substituted(m, 8. * (1. + 1e-6), 5., 14.), false)));
  GPRS_CHECK(silent(structure_check(m, substituted(m, 8., 5., 14. * (1. + 1e-12)), false)));
  GPRS_CHECK(only_objective(structure_check(m, substituted(m, 8., 5., 14. + 1e-6), false)));

  // A cost that the substitution cancels to zero: y with cost -1 on x with cost 2.
  m.cost[y] = -1.;
  GPRS_CHECK(silent(structure_check(m, substituted(m, 0., 5., 10.), false)));
  GPRS_CHECK(only_objective(structure_check(m, substituted(m, 2., 5., 10.), false)));

  // A huge fixed value: the offset is tested relative to its terms.
  m           = cost_model();
  m.col_upper = {3., 3., 10., 1e20};
  m.cost[t]   = 3.;
  auto huge   = [&m](double offset) {
    result_t result = substituted(m, 8., 5., offset);
    result.postsolve.init(m);
    result.postsolve.col_origin = {x, s};
    result.postsolve.push_alias(y, x, 0, 2., 1.);
    result.postsolve.push_fixed_col(t, 1e20);
    return result;
  };
  GPRS_CHECK(silent(structure_check(m, huge(13. + 3e20), false)));
  GPRS_CHECK(only_objective(structure_check(m, huge(13. + 1.5e20), false)));
  GPRS_CHECK(only_objective(structure_check(m, huge(13.), false)));
}

// An alias of an alias: the cost and the offset compose along the chain.
GPRS_TEST(e2e_structure_objective_of_a_chain)
{
  // min x + 2 y + 4 s with y = 3 x - 1 and s = -2 y + 5 = -6 x + 7: on x alone the objective
  // is (1 + 6 - 24) x + (-2 + 28). The rows are the two equations; x in [1, 2].
  model_t m;
  m.nrows     = 2;
  m.ncols     = 3;
  m.row_start = {0, 2, 4};
  m.col_index = {0, 1, 1, 2};
  m.value     = {-3., 1., 2., 1.};
  m.row_lower = {-1., 5.};
  m.row_upper = {-1., 5.};
  m.col_lower = {1., -10., -10.};
  m.col_upper = {2., 10., 10.};
  m.cost      = {1., 2., 4.};
  m.col_type  = {0, 0, 0};
  auto chain  = [&m](double cost, double offset) {
    result_t result;
    result.status                   = status_t::reduced;
    result.reduced.ncols            = 1;
    result.reduced.row_start        = {0};
    result.reduced.col_lower        = {1.};
    result.reduced.col_upper        = {2.};
    result.reduced.cost             = {cost};
    result.reduced.col_type         = {0};
    result.reduced.objective_offset = offset;
    result.postsolve.init(m);
    result.postsolve.col_origin = {0};
    // The engine substitutes s by y first, then y by x.
    result.postsolve.push_alias(2, 1, 1, -2., 5.);
    result.postsolve.push_alias(1, 0, 0, 3., -1.);
    return result;
  };
  GPRS_CHECK(silent(structure_check(m, chain(-17., 26.), false)));
  GPRS_CHECK(only_objective(structure_check(m, chain(-17., 20.), false)));
  GPRS_CHECK(only_objective(structure_check(m, chain(7., 26.), false)));
  GPRS_CHECK(only_objective(structure_check(m, chain(-17. + 1e-4, 26.), false)));
}

// The mode of the result and the origin of an alias are part of the contract of the stack.
GPRS_TEST(e2e_structure_reads_the_mode_and_the_origin)
{
  model_t m  = cost_model();
  auto error = [&m](const alias_record_t& record, bool primal_only) {
    return structure_check(m, aliased(m, record), primal_only).stack_error;
  };
  const alias_origin_t doubleton = alias_origin_t::doubleton;
  const alias_origin_t probing   = alias_origin_t::probing;
  const alias_origin_t lock      = alias_origin_t::lock;
  for (bool primal_only : {false, true}) {
    // The alias of a doubleton equation names its row.
    GPRS_CHECK(silent(structure_check(m, aliased(m, {y, x, 0, 2., 1., doubleton}), primal_only)));
    GPRS_CHECK_EQ(error({y, x, -1, 2., 1., doubleton}, primal_only), "row of alias record 0");
    GPRS_CHECK_EQ(error({y, x, 1, 2., 1., doubleton}, primal_only), "row of alias record 0");
    // An alias of simple probing names its row or none.
    GPRS_CHECK_EQ(error({y, x, 0, 2., 1., probing}, primal_only), "");
    GPRS_CHECK_EQ(error({y, x, -1, 2., 1., probing}, primal_only), "");
    GPRS_CHECK_EQ(error({y, x, -2, 2., 1., probing}, primal_only), "row of alias record 0");
    GPRS_CHECK_EQ(error({y, x, 1, 2., 1., probing}, primal_only), "row of alias record 0");
    // An origin that the check does not know.
    GPRS_CHECK_EQ(error({y, x, 0, 2., 1., (alias_origin_t)3}, primal_only),
                  "origin of alias record 0");
  }
  // A lock alias names no row, and no result of primal-only mode has one.
  GPRS_CHECK_EQ(error({y, x, -1, 2., 1., lock}, false), "");
  GPRS_CHECK_EQ(error({y, x, 0, 2., 1., lock}, false), "row of alias record 0");
  GPRS_CHECK_EQ(error({y, x, -1, 2., 1., lock}, true), "lock alias in a primal-only result at 0");
  // With the row left in the reduced model the lock alias passes every part: the row
  // y - 2 x = 1 is 0 = 0 on the reduced columns.
  result_t result             = aliased(m, {y, x, -1, 2., 1., lock});
  result.reduced.nrows        = 1;
  result.reduced.row_start    = {0, 0};
  result.reduced.row_lower    = {0.};
  result.reduced.row_upper    = {0.};
  result.postsolve.row_origin = {0};
  result.dual_findings        = 1;
  GPRS_CHECK(silent(structure_check(m, result, false)));

  // A dual finding in a result of primal-only mode.
  result               = substituted(m, 8., 5., 14.);
  result.dual_findings = 2;
  GPRS_CHECK(silent(structure_check(m, result, false)));
  GPRS_CHECK_EQ(structure_check(m, result, true).stack_error,
                "dual findings in a primal-only result: 2");
}

// A row that an alias took out of the model cannot have an equation subtracted afterwards; the
// lock row of a lock alias stays in the model and can.
GPRS_TEST(e2e_structure_lock_row_stays_in_the_model)
{
  //   r0:  x + y + s <= 5
  //   r1:  x + y      = 3
  // with t, which no row holds, aliased to s.
  model_t m;
  m.nrows         = 2;
  m.ncols         = 4;
  m.row_start     = {0, 3, 5};
  m.col_index     = {x, y, s, x, y};
  m.value         = {1., 1., 1., 1., 1.};
  m.row_lower     = {-inf, 3.};
  m.row_upper     = {5., 3.};
  m.col_lower     = {0., 0., 0., 0.};
  m.col_upper     = {3., 3., 1., 1.};
  m.cost          = {0., 0., 0., 0.};
  m.col_type      = {0, 0, 0, 0};
  auto subtracted = [&m](int32_t row, alias_origin_t origin) {
    result_t result;
    result.status            = status_t::reduced;
    result.reduced.nrows     = 2;
    result.reduced.ncols     = 3;
    result.reduced.row_start = {0, 1, 3};
    result.reduced.col_index = {2, 0, 1};
    result.reduced.value     = {1., 1., 1.};
    result.reduced.row_lower = {-inf, 3.};
    result.reduced.row_upper = {2., 3.};
    result.reduced.col_lower = {0., 0., 0.};
    result.reduced.col_upper = {3., 3., 1.};
    result.reduced.cost      = {0., 0., 0.};
    result.reduced.col_type  = {0, 0, 0};
    result.postsolve.init(m);
    result.postsolve.row_origin = {0, 1};
    result.postsolve.col_origin = {x, y, s};
    put_alias(result, {t, s, row, 1., 0., origin});
    result.postsolve.push_row_addition(0, 1, 1.);
    result.dual_findings = origin == alias_origin_t::lock;
    return result;
  };
  GPRS_CHECK_EQ(structure_check(m, subtracted(1, alias_origin_t::doubleton), false).stack_error,
                "rows of row addition record 1");
  GPRS_CHECK_EQ(structure_check(m, subtracted(1, alias_origin_t::probing), false).stack_error,
                "rows of row addition record 1");
  GPRS_CHECK(silent(structure_check(m, subtracted(-1, alias_origin_t::probing), false)));
  GPRS_CHECK(silent(structure_check(m, subtracted(-1, alias_origin_t::lock), false)));
}

GPRS_TEST(e2e_structure_rejects_a_record_kind_it_does_not_know)
{
  model_t m       = cost_model();
  result_t result = substituted(m, 8., 5., 14.);
  result.postsolve.order.push_back((record_kind_t)4);
  GPRS_CHECK_EQ(structure_check(m, result, false).stack_error, "unknown record kind at 2");
  result.postsolve.order.back() = (record_kind_t)255;
  GPRS_CHECK_EQ(structure_check(m, result, true).stack_error, "unknown record kind at 2");
  result.postsolve.order.back() = record_kind_t::row_addition;
  GPRS_CHECK_EQ(structure_check(m, result, false).stack_error, "row addition records missing at 2");
}

// The engine on generated models with costs on every column: the objective part is silent in
// both modes, and it is what fails when the reduced objective is changed.
GPRS_TEST(e2e_structure_objective_of_the_engine)
{
  oracle::generator_options_t g;
  g.p_doubleton_equation = 0.4;
  g.p_equation           = 0.3;
  g.p_zero_cost          = 0.1;
  g.p_fixed_col          = 0.2;
  // Without huge bounds every term of the offset is small against a change of 1.
  g.p_huge_bound  = 0.;
  int64_t carried = 0, aliases = 0, fixed = 0, failed = 0, changed = 0, seen = 0;
  for (uint64_t seed = 9100000; seed < 9106000 && failed < 3; ++seed) {
    oracle::case_t c = oracle::generate(seed, g);
    for (bool primal_only : {false, true}) {
      options_t options;
      options.primal_only = primal_only;
      result_t result     = presolve(c.model, options);
      if (!gprs_check::carries_model(result.status)) { continue; }
      ++carried;
      aliases += !result.postsolve.aliases.empty();
      fixed += !result.postsolve.fixed_cols.empty();
      structure_report_t r = structure_check(c.model, result, primal_only);
      if (r.objective_failures != 0 || !r.stack_error.empty()) {
        ++failed;
        test::fail(__FILE__,
                   __LINE__,
                   "seed " + std::to_string(seed) + ": " + r.stack_error + " objective " +
                     std::to_string(r.objective_worst) + " constant " +
                     std::to_string(r.objective_constant) + "\n" + oracle::to_code(c));
        continue;
      }
      // One unit more on the offset, and twice the cost of the first reduced column with one.
      result.reduced.objective_offset += 1. + std::fabs(result.reduced.objective_offset);
      seen += structure_check(c.model, result, primal_only).objective_failures == 1;
      ++changed;
      for (double& cost : result.reduced.cost) {
        if (cost == 0.) { continue; }
        cost *= 2.;
        seen += structure_check(c.model, result, primal_only).objective_failures == 2;
        ++changed;
        break;
      }
    }
  }
  if (std::getenv("GPRS_ORACLE_CENSUS")) {
    std::printf("    results %lld, with an alias %lld, with a fixed column %lld, changed %lld\n",
                (long long)carried,
                (long long)aliases,
                (long long)fixed,
                (long long)changed);
  }
  GPRS_CHECK_EQ(failed, 0);
  GPRS_CHECK(carried > 5000 && aliases > 300 && fixed > 3500);
  GPRS_CHECK_EQ(seen, changed);
}
