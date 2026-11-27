/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// oracle::check_result on the results of a small reference presolve that is written here on a
// dense matrix and shares nothing with the engine: it must accept every result of the correct
// version, on plain, scaled and huge-bound cases, and reject each of the wrong versions on some
// case. These tests need the postsolve of part B and no rule.

#include "harness.hpp"
#include "oracle.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace gprs;
using oracle::case_t;
using oracle::generator_options_t;
using oracle::verdict_t;

namespace {

enum class bug_t : int32_t {
  none,
  singleton_sign,         // a singleton row with a negative coefficient gives its bounds unswapped
  fixed_sides,            // a fixed column leaves the sides of its rows as they are
  fixed_offset,           // a fixed column leaves the objective offset as it is
  rounding_outwards,      // the bounds of an integer column are rounded away from the column
  dual_ignores_lower,     // dual fixing counts no lock for a finite lower side
  dual_wrong_bound,       // dual fixing moves the column against its cost
  dual_infinite_is_zero,  // dual fixing towards an infinite bound fixes the column at 0
  alias_sides,            // a substitution leaves the sides of the other rows as they are
  alias_bounds,           // a substitution drops the bounds of the substituted column
  alias_cost,             // a substitution drops the cost of the substituted column
  alias_record,           // the alias record has the scale with the wrong sign
  parallel_sign,          // a parallel row with a negative ratio gives its sides unswapped
  parallel_drop,          // a parallel row is deleted without its sides
  empty_row,              // an empty row is deleted whatever its sides are
  crossing_equal,         // lower == upper is reported as infeasible
  relax_far_side,         // a singleton row of a continuous column loses the side that the cost
                          // of the column does not push against
  restrict_far_side,      // ... and that side is moved inwards instead (valid as a dual
                          // reduction, wrong in primal-only mode)
  count
};

const char* const bug_name[(int32_t)bug_t::count] = {"none",
                                                     "singleton_sign",
                                                     "fixed_sides",
                                                     "fixed_offset",
                                                     "rounding_outwards",
                                                     "dual_ignores_lower",
                                                     "dual_wrong_bound",
                                                     "dual_infinite_is_zero",
                                                     "alias_sides",
                                                     "alias_bounds",
                                                     "alias_cost",
                                                     "alias_record",
                                                     "parallel_sign",
                                                     "parallel_drop",
                                                     "empty_row",
                                                     "crossing_equal",
                                                     "relax_far_side",
                                                     "restrict_far_side"};

// Fixed columns, empty and singleton rows, dual fixing, doubleton equations and parallel rows,
// repeated until nothing changes. Every decision is taken with a tolerance far below the
// distance of two distinct values of a generated case.
struct reference_t {
  bug_t bug;
  bool dual;
  bool substitutions_only = false;
  int32_t nrows, ncols;
  std::vector<std::vector<double>> a;
  std::vector<double> row_lower, row_upper, col_lower, col_upper, cost;
  std::vector<uint8_t> col_type, row_live, col_live;
  double offset;
  postsolve_t ps;
  status_t status = status_t::reduced;
  int64_t fixed = 0, singletons = 0, dual_fixings = 0, aliases = 0, integer_aliases = 0;
  int64_t parallel = 0;

  reference_t(const model_t& m, bool dual_reductions, bug_t which)
    : bug(which),
      dual(dual_reductions),
      nrows(m.nrows),
      ncols(m.ncols),
      a(m.nrows, std::vector<double>(m.ncols, 0.)),
      row_lower(m.row_lower),
      row_upper(m.row_upper),
      col_lower(m.col_lower),
      col_upper(m.col_upper),
      cost(m.cost),
      col_type(m.col_type),
      row_live(m.nrows, 1),
      col_live(m.ncols, 1),
      offset(m.objective_offset)
  {
    for (int32_t i = 0; i < nrows; ++i) {
      for (int64_t e = m.row_start[i]; e < m.row_start[i + 1]; ++e) {
        a[i][m.col_index[e]] = m.value[e];
      }
    }
    ps.init(m);
  }

  static bool crossing(double lower, double upper)
  {
    double scale = 1.;
    if (std::isfinite(lower)) { scale = std::max(scale, std::fabs(lower)); }
    if (std::isfinite(upper)) { scale = std::max(scale, std::fabs(upper)); }
    return lower > upper + 1e-9 * scale;
  }

  void round_bounds(int32_t j)
  {
    if (!col_type[j]) { return; }
    if (bug == bug_t::rounding_outwards) {
      col_lower[j] = std::floor(col_lower[j] + 1e-9);
      col_upper[j] = std::ceil(col_upper[j] - 1e-9);
    } else {
      col_lower[j] = std::ceil(col_lower[j] - 1e-9);
      col_upper[j] = std::floor(col_upper[j] + 1e-9);
    }
  }

  // False when the bounds cross.
  bool settle(int32_t j)
  {
    round_bounds(j);
    if (crossing(col_lower[j], col_upper[j])) { return false; }
    if (bug == bug_t::crossing_equal && col_lower[j] == col_upper[j]) { return false; }
    if (col_lower[j] > col_upper[j]) { col_lower[j] = col_upper[j]; }
    return true;
  }

  bool tighten(int32_t j, double lower, double upper)
  {
    col_lower[j] = std::max(col_lower[j], lower);
    col_upper[j] = std::min(col_upper[j], upper);
    return settle(j);
  }

  void shift_sides(int32_t i, double shift)
  {
    row_lower[i] -= shift;
    row_upper[i] -= shift;
  }

  void fix(int32_t j, double value)
  {
    for (int32_t i = 0; i < nrows; ++i) {
      if (!row_live[i] || a[i][j] == 0.) { continue; }
      if (bug != bug_t::fixed_sides) { shift_sides(i, a[i][j] * value); }
      a[i][j] = 0.;
    }
    if (bug != bug_t::fixed_offset) { offset += cost[j] * value; }
    col_live[j] = 0;
    ps.push_fixed_col(j, value);
  }

  int32_t row_size(int32_t i, int32_t& first, int32_t& second) const
  {
    int32_t size = 0;
    for (int32_t j = 0; j < ncols; ++j) {
      if (a[i][j] == 0.) { continue; }
      (size == 0 ? first : second) = j;
      ++size;
    }
    return size;
  }

  bool fixed_columns()
  {
    bool changed = false;
    for (int32_t j = 0; j < ncols; ++j) {
      if (!col_live[j] || col_lower[j] != col_upper[j]) { continue; }
      fix(j, col_lower[j]);
      ++fixed;
      changed = true;
    }
    return changed;
  }

  bool short_rows()
  {
    bool changed = false;
    for (int32_t i = 0; i < nrows && status == status_t::reduced; ++i) {
      if (!row_live[i]) { continue; }
      int32_t j = -1, other = -1;
      int32_t size = row_size(i, j, other);
      if (size > 1) { continue; }
      if (size == 0) {
        if (bug != bug_t::empty_row && (crossing(row_lower[i], 0.) || crossing(0., row_upper[i]))) {
          status = status_t::infeasible;
        }
      } else {
        double divisor = bug == bug_t::singleton_sign ? std::fabs(a[i][j]) : a[i][j];
        double lower = row_lower[i] / divisor, upper = row_upper[i] / divisor;
        if (a[i][j] < 0 && bug != bug_t::singleton_sign) { std::swap(lower, upper); }
        if (!col_type[j] && bug == bug_t::relax_far_side) {
          (cost[j] >= 0 ? upper : lower) = cost[j] >= 0 ? inf : -inf;
        }
        if (!col_type[j] && bug == bug_t::restrict_far_side && !crossing(lower, upper)) {
          double from = std::max(lower, col_lower[j]), to = std::min(upper, col_upper[j]);
          if (cost[j] >= 0 && std::isfinite(from)) {
            upper = std::isfinite(to) ? from + 0.5 * (to - from) : from + 50.;
          } else if (cost[j] < 0 && std::isfinite(to)) {
            lower = std::isfinite(from) ? to - 0.5 * (to - from) : to - 50.;
          }
        }
        if (crossing(lower, upper) || !tighten(j, lower, upper)) { status = status_t::infeasible; }
        a[i][j] = 0.;
        ++singletons;
      }
      row_live[i] = 0;
      changed     = true;
    }
    return changed;
  }

  bool dual_fixing()
  {
    bool changed = false;
    for (int32_t j = 0; j < ncols && status == status_t::reduced; ++j) {
      if (!col_live[j] || col_lower[j] == col_upper[j]) { continue; }
      int32_t down = 0, up = 0;
      for (int32_t i = 0; i < nrows; ++i) {
        if (!row_live[i] || a[i][j] == 0.) { continue; }
        bool lower_side = row_lower[i] != -inf && bug != bug_t::dual_ignores_lower;
        bool upper_side = row_upper[i] != inf;
        if (a[i][j] > 0) {
          up += upper_side;
          down += lower_side;
        } else {
          up += lower_side;
          down += upper_side;
        }
      }
      double target;
      if (cost[j] >= 0 && down == 0) {
        target = col_lower[j];
        if (target == -inf && cost[j] == 0. && up == 0) { target = col_upper[j]; }
      } else if (cost[j] <= 0 && up == 0) {
        target = col_upper[j];
      } else {
        continue;
      }
      if (bug == bug_t::dual_wrong_bound && cost[j] != 0.) {
        target = target == col_lower[j] ? col_upper[j] : col_lower[j];
      }
      if (!std::isfinite(target)) {
        if (bug == bug_t::dual_infinite_is_zero || (cost[j] == 0. && down + up == 0)) {
          target = 0.;
        } else if (cost[j] != 0.) {
          status = status_t::unbounded_or_infeasible;
          continue;
        } else {
          continue;
        }
      }
      fix(j, target);
      ++dual_fixings;
      changed = true;
    }
    return changed;
  }

  // x[y] = scale * x[x] + shift from the equation `row`, and y leaves the model.
  void substitute(int32_t row, int32_t y, int32_t x, double scale, double shift)
  {
    double lower = (col_lower[y] - shift) / scale, upper = (col_upper[y] - shift) / scale;
    if (scale < 0) { std::swap(lower, upper); }
    if (bug != bug_t::alias_bounds && !tighten(x, lower, upper)) {
      status = status_t::infeasible;
      return;
    }
    row_live[row] = 0;
    for (int32_t i = 0; i < nrows; ++i) {
      double c = a[i][y];
      if (!row_live[i] || c == 0.) { continue; }
      double sum = a[i][x] + c * scale;
      a[i][x]    = std::fabs(sum) <= 1e-12 * (std::fabs(a[i][x]) + std::fabs(c * scale)) ? 0. : sum;
      a[i][y]    = 0.;
      if (bug != bug_t::alias_sides) { shift_sides(i, c * shift); }
    }
    if (bug != bug_t::alias_cost) {
      cost[x] += cost[y] * scale;
      offset += cost[y] * shift;
    }
    a[row][x] = a[row][y] = 0.;
    col_live[y]           = 0;
    ps.push_alias(y, x, row, bug == bug_t::alias_record ? -scale : scale, shift);
    ++aliases;
    integer_aliases += col_type[y];
  }

  bool doubletons()
  {
    bool changed = false;
    for (int32_t i = 0; i < nrows && status == status_t::reduced; ++i) {
      if (!row_live[i] || row_lower[i] != row_upper[i]) { continue; }
      int32_t first = -1, second = -1;
      if (row_size(i, first, second) != 2) { continue; }
      // The substituted column: a continuous one, else an integer one whose map is integral.
      for (int32_t turn = 0; turn < 2; ++turn) {
        int32_t y = turn == 0 ? second : first, x = turn == 0 ? first : second;
        double scale = -a[i][x] / a[i][y], shift = row_lower[i] / a[i][y];
        bool integral = scale == std::round(scale) && shift == std::round(shift);
        if (col_type[y] && !(col_type[x] && integral)) { continue; }
        substitute(i, y, x, scale, shift);
        changed = true;
        break;
      }
    }
    return changed;
  }

  bool parallel_rows()
  {
    bool changed = false;
    for (int32_t i = 0; i < nrows && status == status_t::reduced; ++i) {
      int32_t lead = -1, other = -1;
      if (!row_live[i] || row_size(i, lead, other) < 2) { continue; }
      for (int32_t k = i + 1; k < nrows && status == status_t::reduced; ++k) {
        if (!row_live[k] || a[k][lead] == 0.) { continue; }
        double ratio = a[k][lead] / a[i][lead];
        bool same    = true;
        for (int32_t j = 0; j < ncols; ++j) {
          same = same && std::fabs(a[k][j] - ratio * a[i][j]) <= 1e-9 * std::fabs(a[k][j]) &&
                 (a[k][j] == 0.) == (a[i][j] == 0.);
        }
        if (!same) { continue; }
        double divisor = bug == bug_t::parallel_sign ? std::fabs(ratio) : ratio;
        double lower = row_lower[k] / divisor, upper = row_upper[k] / divisor;
        if (ratio < 0 && bug != bug_t::parallel_sign) { std::swap(lower, upper); }
        if (bug != bug_t::parallel_drop) {
          row_lower[i] = std::max(row_lower[i], lower);
          row_upper[i] = std::min(row_upper[i], upper);
        }
        if (crossing(row_lower[i], row_upper[i])) { status = status_t::infeasible; }
        if (row_lower[i] > row_upper[i]) { row_lower[i] = row_upper[i]; }
        row_live[k] = 0;
        ps.push_parallel_row(k, i, ratio);
        ++parallel;
        changed = true;
      }
    }
    return changed;
  }

  result_t run()
  {
    for (int32_t i = 0; i < nrows; ++i) {
      if (crossing(row_lower[i], row_upper[i])) { status = status_t::infeasible; }
    }
    for (int32_t j = 0; j < ncols; ++j) {
      if (!settle(j)) { status = status_t::infeasible; }
    }
    bool changed = true;
    while (changed && status == status_t::reduced) {
      changed = fixed_columns();
      if (!substitutions_only) {
        changed = short_rows() || changed;
        if (status == status_t::reduced && !changed && dual) { changed = dual_fixing(); }
        if (status == status_t::reduced && !changed) { changed = parallel_rows(); }
      }
      if (status == status_t::reduced && !changed) { changed = doubletons(); }
    }

    result_t result;
    result.status                            = status;
    result.stats[rule::dual_fixing].findings = dual_fixings;
    if (status != status_t::reduced) { return result; }
    model_t& r = result.reduced;
    std::vector<int32_t> new_col(ncols, -1);
    for (int32_t j = 0; j < ncols; ++j) {
      if (!col_live[j]) { continue; }
      new_col[j] = r.ncols++;
      ps.col_origin.push_back(j);
      r.col_lower.push_back(col_lower[j]);
      r.col_upper.push_back(col_upper[j]);
      r.cost.push_back(cost[j]);
      r.col_type.push_back(col_type[j]);
    }
    r.row_start.push_back(0);
    for (int32_t i = 0; i < nrows; ++i) {
      if (!row_live[i]) { continue; }
      ++r.nrows;
      ps.row_origin.push_back(i);
      for (int32_t j = 0; j < ncols; ++j) {
        if (a[i][j] == 0.) { continue; }
        r.col_index.push_back(new_col[j]);
        r.value.push_back(a[i][j]);
      }
      r.row_start.push_back(r.value.size());
      r.row_lower.push_back(row_lower[i]);
      r.row_upper.push_back(row_upper[i]);
    }
    r.objective_offset = offset;
    result.postsolve   = ps;
    return result;
  }
};

struct profile_t {
  const char* name;
  generator_options_t options;
};

std::vector<profile_t> profiles()
{
  std::vector<profile_t> list;
  generator_options_t o;
  list.push_back({"default", o});
  o          = generator_options_t();
  o.p_scaled = 1.;
  list.push_back({"scaled", o});
  o                  = generator_options_t();
  o.max_line_cols    = 3;
  o.p_infinite_bound = 0.7;
  o.p_huge_bound     = 0.6;
  o.p_zero_cost      = 0.5;
  list.push_back({"open_ends", o});
  o                      = generator_options_t();
  o.p_doubleton_equation = 0.45;
  o.p_equation           = 0.4;
  o.p_singleton_row      = 0.05;
  o.p_empty_row          = 0.02;
  o.p_zero_cost          = 0.5;
  o.p_scaled             = 0.5;
  list.push_back({"equations", o});
  o                = generator_options_t();
  o.p_parallel_row = 0.5;
  o.max_rows       = 7;
  o.p_scaled       = 0.5;
  list.push_back({"parallel", o});
  return list;
}

options_t mode(bool primal_only)
{
  options_t options;
  options.primal_only = primal_only;
  return options;
}

constexpr int32_t cases_per_profile = 1500;

}  // namespace

GPRS_TEST(e2e_check_accepts_a_reference_presolve)
{
  int32_t failures = 0;
  int64_t fixed = 0, singletons = 0, dual = 0, aliases = 0, integer_aliases = 0, parallel = 0;
  int64_t with_model = 0, infeasible = 0, unbounded = 0;
  for (const profile_t& profile : profiles()) {
    for (uint64_t seed = 8000000; seed < 8000000 + cases_per_profile && failures < 3; ++seed) {
      case_t c = oracle::generate(seed, profile.options);
      for (bool primal_only : {false, true}) {
        reference_t reference(c.model, !primal_only, bug_t::none);
        result_t result = reference.run();
        fixed += reference.fixed;
        singletons += reference.singletons;
        dual += reference.dual_fixings;
        aliases += reference.aliases;
        integer_aliases += reference.integer_aliases;
        parallel += reference.parallel;
        with_model += result.status == status_t::reduced;
        infeasible += result.status == status_t::infeasible;
        unbounded += result.status == status_t::unbounded_or_infeasible;
        verdict_t v = oracle::check_result(c, mode(primal_only), result);
        if (v.ok) { continue; }
        ++failures;
        test::fail(__FILE__,
                   __LINE__,
                   std::string(profile.name) + ", seed " + std::to_string(seed) +
                     (primal_only ? ", primal only: " : ": ") + v.message + "\n" +
                     oracle::to_code(c));
      }
    }
  }
  if (std::getenv("GPRS_ORACLE_CENSUS")) {
    std::printf(
      "    fixed %lld, singleton rows %lld, dual fixings %lld, aliases %lld (integer "
      "%lld), parallel rows %lld; reduced %lld, infeasible %lld, unbounded %lld\n",
      (long long)fixed,
      (long long)singletons,
      (long long)dual,
      (long long)aliases,
      (long long)integer_aliases,
      (long long)parallel,
      (long long)with_model,
      (long long)infeasible,
      (long long)unbounded);
  }
  GPRS_CHECK(fixed > 2000 && singletons > 2000 && dual > 2000 && parallel > 500);
  GPRS_CHECK(aliases > 1000 && integer_aliases > 300);
  GPRS_CHECK(with_model > 5000 && infeasible > 1000 && unbounded > 200);
}

// Found with 30,000 cases of a profile with four line columns, huge bounds and scaling: dual
// fixing puts column 3 on 2e20, which leaves the scaled row 1 with the upper side 7.5e19 and
// makes it parallel to row 0. The tolerance of the lower side of the merged row must not
// follow its upper side.
GPRS_TEST(e2e_check_accepts_a_row_side_made_huge_by_a_fixed_column)
{
  model_t base;
  base.nrows            = 2;
  base.ncols            = 4;
  base.row_start        = {0, 3, 6};
  base.col_index        = {0, 1, 2, 0, 1, 3};
  base.value            = {2., -2., 2., 3., -3., -3.};
  base.row_lower        = {0., -inf};
  base.row_upper        = {inf, 3.};
  base.col_lower        = {0., 2., -1., -1e+20};
  base.col_upper        = {3., 3., 1., 2e+20};
  base.cost             = {1., 0., 0., 0.};
  base.col_type         = {1, 1, 1, 1};
  base.objective_offset = -8.;
  case_t c              = oracle::make_case(base, {0.00390625, 0.125}, {1., 1., 1., 1.});
  reference_t reference(c.model, true, bug_t::none);
  result_t result = reference.run();
  GPRS_CHECK_EQ(reference.dual_fixings, 2);
  GPRS_CHECK_EQ(reference.parallel, 1);
  GPRS_REQUIRE(result.reduced.nrows == 1);
  GPRS_CHECK_EQ(result.reduced.row_lower[0], -0.0078125);
  GPRS_CHECK(result.reduced.row_upper[0] > 1e18 && result.reduced.row_upper[0] < inf);
  verdict_t v = oracle::check_result(c, options_t(), result);
  if (!v.ok) { test::fail(__FILE__, __LINE__, v.message); }
}

// The model that the substitutions alone leave is a base model whose data are not integers:
// the oracle decides it with its tolerance, and the rest of the reference presolve must pass.
GPRS_TEST(e2e_check_accepts_a_reference_presolve_of_a_reduced_model)
{
  int32_t failures = 0, checked = 0, fractional = 0;
  for (const profile_t& profile : profiles()) {
    for (uint64_t seed = 8100000; seed < 8100000 + cases_per_profile && failures < 3; ++seed) {
      case_t c = oracle::generate(seed, profile.options);
      reference_t first(c.model, false, bug_t::none);
      first.substitutions_only = true;
      result_t stage           = first.run();
      if (stage.status != status_t::reduced || first.aliases == 0) { continue; }
      GPRS_REQUIRE(oracle::check_result(c, mode(true), stage).ok);
      case_t second = oracle::make_case(stage.reduced, {}, {});
      bool whole    = true;
      for (double v : stage.reduced.value) {
        whole = whole && v == std::floor(v);
      }
      fractional += !whole;
      for (bool primal_only : {false, true}) {
        result_t result = reference_t(second.model, !primal_only, bug_t::none).run();
        verdict_t v     = oracle::check_result(second, mode(primal_only), result);
        ++checked;
        if (v.ok) { continue; }
        ++failures;
        test::fail(__FILE__,
                   __LINE__,
                   std::string(profile.name) + ", seed " + std::to_string(seed) +
                     (primal_only ? ", primal only: " : ": ") + v.message + "\n" +
                     oracle::to_code(second));
      }
    }
  }
  if (std::getenv("GPRS_ORACLE_CENSUS")) {
    std::printf("    checked %d, with fractional coefficients %d\n", checked, fractional);
  }
  GPRS_CHECK(checked > 1000 && fractional > 100);
}

GPRS_TEST(e2e_check_rejects_every_wrong_reference_presolve)
{
  bool census = std::getenv("GPRS_ORACLE_CENSUS");
  for (int32_t b = 1; b < (int32_t)bug_t::count; ++b) {
    bug_t bug           = (bug_t)b;
    int32_t rejected[2] = {0, 0};
    for (const profile_t& profile : profiles()) {
      for (uint64_t seed = 8000000; seed < 8000000 + cases_per_profile / 2; ++seed) {
        case_t c = oracle::generate(seed, profile.options);
        for (bool primal_only : {false, true}) {
          result_t result = reference_t(c.model, !primal_only, bug).run();
          rejected[primal_only] += !oracle::check_result(c, mode(primal_only), result).ok;
        }
      }
    }
    if (census) {
      std::printf(
        "    %-22s rejected %5d, primal only %5d\n", bug_name[b], rejected[0], rejected[1]);
    }
    // Rejections wanted with and without dual reductions. The far side of a line column shows
    // in the points of solution_t::far: with them the two counts below are 206 and 64, without
    // them 120 and 7.
    int32_t wanted[2] = {20, 20};
    if (bug == bug_t::dual_ignores_lower || bug == bug_t::dual_wrong_bound ||
        bug == bug_t::dual_infinite_is_zero) {
      wanted[1] = 0;
    }
    if (bug == bug_t::relax_far_side) { wanted[1] = 160; }
    if (bug == bug_t::restrict_far_side) {
      wanted[0] = 0;
      wanted[1] = 30;
    }
    if (rejected[0] < wanted[0] || rejected[1] < wanted[1]) {
      test::fail(__FILE__,
                 __LINE__,
                 std::string("too few rejections of ") + bug_name[b] + ": " +
                   std::to_string(rejected[0]) + " and " + std::to_string(rejected[1]));
    }
  }
}
