/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gprs/assert.hpp>
#include <gprs/postsolve.hpp>

#include <cmath>

namespace gprs {

namespace {

bool integral(double value) { return value == std::round(value); }

// The contract of the stack with its column map, checked in full: the arrays agree with order,
// no record names a column that an earlier record removed, and every column is either left in
// the reduced model or removed by one record. O(columns + records).
bool stack_consistent(const postsolve_t& ps)
{
  size_t fixed = 0, alias = 0, parallel = 0, addition = 0;
  std::vector<uint8_t> removed(ps.ncols, 0);
  size_t n_removed = 0;
  for (record_kind_t kind : ps.order) {
    if (kind == record_kind_t::fixed_col) {
      if (fixed == ps.fixed_cols.size()) { return false; }
      int32_t col = ps.fixed_cols[fixed++].col;
      if (removed[col]) { return false; }
      removed[col] = 1;
      ++n_removed;
    } else if (kind == record_kind_t::alias) {
      if (alias == ps.aliases.size()) { return false; }
      const alias_record_t& record = ps.aliases[alias++];
      if (removed[record.col] || removed[record.rep]) { return false; }
      removed[record.col] = 1;
      ++n_removed;
    } else if (kind == record_kind_t::parallel_row) {
      ++parallel;
    } else {
      ++addition;
    }
  }
  if (fixed != ps.fixed_cols.size() || alias != ps.aliases.size() ||
      parallel != ps.parallel_rows.size() || addition != ps.row_additions.size()) {
    return false;
  }
  if (removed != ps.col_removed) { return false; }
  for (int32_t col : ps.col_origin) {
    if (col < 0 || col >= ps.ncols || removed[col]) { return false; }
    removed[col] = 1;
  }
  return n_removed + ps.col_origin.size() == (size_t)ps.ncols;
}

// max(lower - v, v - upper); NaN when v is.
double interval_violation(double lower, double v, double upper)
{
  double below = lower - v, above = v - upper;
  return below > above ? below : above;
}

// The same for a compensated sum.
double interval_violation(double lower, dd_t s, double upper)
{
  double below = std::isfinite(lower) ? dd_residual(lower, s) : lower;
  double above = std::isfinite(upper) ? -dd_residual(upper, s) : -upper;
  return std::isnan(below) || below > above ? below : above;
}

// worst = max(worst, v), and NaN once a v is NaN.
void raise(double& worst, double v)
{
  if (v > worst || std::isnan(v)) { worst = v; }
}

}  // namespace

void postsolve_t::init(const model_t& original)
{
  nrows    = original.nrows;
  ncols    = original.ncols;
  col_type = original.col_type;
  row_origin.clear();
  col_origin.clear();
  order.clear();
  fixed_cols.clear();
  aliases.clear();
  parallel_rows.clear();
  row_additions.clear();
  col_removed.assign(ncols, 0);
}

void postsolve_t::push_fixed_col(int32_t col, double value)
{
  gprs_assert(col >= 0 && col < ncols);
  gprs_assert(!col_removed[col]);
  gprs_assert(std::isfinite(value));
  gprs_assert(!col_type[col] || integral(value));
  col_removed[col] = 1;
  order.push_back(record_kind_t::fixed_col);
  fixed_cols.push_back(fixed_col_record_t{col, value});
}

void postsolve_t::push_alias(int32_t col, int32_t rep, int32_t row, double scale, double offset)
{
  gprs_assert(col >= 0 && col < ncols && rep >= 0 && rep < ncols && col != rep);
  gprs_assert(row >= 0 && row < nrows);
  gprs_assert(!col_removed[col] && !col_removed[rep]);
  gprs_assert(std::isfinite(scale) && scale != 0. && std::isfinite(offset));
  gprs_assert(!col_type[col] || (col_type[rep] && integral(scale) && integral(offset)));
  col_removed[col] = 1;
  order.push_back(record_kind_t::alias);
  aliases.push_back(alias_record_t{col, rep, row, scale, offset});
}

void postsolve_t::push_parallel_row(int32_t removed_row, int32_t kept_row, double ratio)
{
  gprs_assert(removed_row >= 0 && removed_row < nrows && kept_row >= 0 && kept_row < nrows);
  gprs_assert(removed_row != kept_row);
  gprs_assert(std::isfinite(ratio) && ratio != 0.);
  order.push_back(record_kind_t::parallel_row);
  parallel_rows.push_back(parallel_row_record_t{removed_row, kept_row, ratio});
}

void postsolve_t::push_row_addition(int32_t row, int32_t equation_row, double factor)
{
  gprs_assert(row >= 0 && row < nrows && equation_row >= 0 && equation_row < nrows);
  gprs_assert(row != equation_row);
  gprs_assert(std::isfinite(factor) && factor != 0.);
  order.push_back(record_kind_t::row_addition);
  row_additions.push_back(row_addition_record_t{row, equation_row, factor});
}

void postsolve_t::postsolve(const std::vector<double>& x_reduced,
                            std::vector<double>& x_original) const
{
  gprs_assert(&x_reduced != &x_original);
  gprs_assert(x_reduced.size() == col_origin.size());
  gprs_assert(stack_consistent(*this));
  x_original.assign(ncols, 0.);
  for (size_t k = 0; k < col_origin.size(); ++k) {
    int32_t col     = col_origin[k];
    x_original[col] = col_type[col] ? std::round(x_reduced[k]) : x_reduced[k];
  }
  size_t fixed = fixed_cols.size(), alias = aliases.size();
  for (size_t k = order.size(); k-- > 0;) {
    if (order[k] == record_kind_t::fixed_col) {
      const fixed_col_record_t& record = fixed_cols[--fixed];
      x_original[record.col]           = record.value;
    } else if (order[k] == record_kind_t::alias) {
      const alias_record_t& record = aliases[--alias];
      x_original[record.col]       = record.scale * x_original[record.rep] + record.offset;
    }
  }
}

void postsolve_t::crush(const std::vector<double>& x_original, std::vector<double>& x_reduced) const
{
  gprs_assert(&x_reduced != &x_original);
  gprs_assert(x_original.size() == (size_t)ncols);
  x_reduced.resize(col_origin.size());
  for (size_t k = 0; k < col_origin.size(); ++k) {
    x_reduced[k] = x_original[col_origin[k]];
  }
}

point_check_t check_point(const model_t& model, const std::vector<double>& x)
{
  gprs_assert(x.size() == (size_t)model.ncols);
  point_check_t check;
  for (int32_t i = 0; i < model.nrows; ++i) {
    dd_t activity;
    for (int64_t e = model.row_start[i]; e < model.row_start[i + 1]; ++e) {
      dd_add_product(activity, model.value[e], x[model.col_index[e]]);
    }
    raise(check.row_violation,
          interval_violation(model.row_lower[i], activity, model.row_upper[i]));
  }
  dd_t objective{model.objective_offset, 0.};
  for (int32_t j = 0; j < model.ncols; ++j) {
    // A value that is not finite is no point of the model, whatever the bounds are.
    raise(
      check.bound_violation,
      std::isfinite(x[j]) ? interval_violation(model.col_lower[j], x[j], model.col_upper[j]) : inf);
    if (model.col_type[j]) {
      raise(check.integrality_violation, std::fabs(x[j] - std::round(x[j])));
    }
    dd_add_product(objective, model.cost[j], x[j]);
  }
  check.objective = dd_value(objective);
  return check;
}

bool point_feasible(const point_check_t& check, const tolerances_t& tol)
{
  return check.row_violation <= tol.feasibility && check.bound_violation <= tol.feasibility &&
         check.integrality_violation <= tol.integrality;
}

}  // namespace gprs
