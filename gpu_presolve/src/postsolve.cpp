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

// The value an interval column starts from (record_col_t::prefer).
double preferred(double lower, double upper)
{
  return lower > -inf ? lower : upper < inf ? upper : 0.;
}

// A forcing record bounds its column from above: the column was free to go down. Read from its
// first row, since a column without any bound says nothing.
bool forcing_from_above(const postsolve_t& ps, const col_from_rows_record_t& r)
{
  return (ps.record_coefficients[r.coefficients] > 0) == (ps.saved_rows[r.rows_begin].upper < inf);
}

// The contract of one col_from_rows record whose columns are marked in `removed` already: its
// ranges, counts, coefficients, bounds, sides and types, and saved rows that name no removed
// column. `position` is its index in the order of the stack.
bool from_rows_consistent(const postsolve_t& ps,
                          const col_from_rows_record_t& r,
                          size_t position,
                          const std::vector<uint8_t>& removed)
{
  int32_t ncols = r.cols_end - r.cols_begin, nrows = r.rows_end - r.rows_begin;
  if (r.cols_begin < 0 || ncols < 1 || (size_t)r.cols_end > ps.record_cols.size() ||
      r.rows_begin < 0 || nrows < 1 || (size_t)r.rows_end > ps.saved_rows.size()) {
    return false;
  }
  bool defined  = r.kind == col_from_rows_kind_t::defined;
  bool interval = r.kind == col_from_rows_kind_t::interval;
  bool forcing  = r.kind == col_from_rows_kind_t::forcing;
  if (!defined && !interval && !forcing) { return false; }
  if ((!interval && ncols != 1) || (!forcing && nrows != 1) || (r.dual && !defined)) {
    return false;
  }
  int32_t count = ncols > nrows ? ncols : nrows;
  if (r.coefficients < 0 || (size_t)(r.coefficients + count) > ps.record_coefficients.size()) {
    return false;
  }
  for (int32_t k = 0; k < count; ++k) {
    double a = ps.record_coefficients[r.coefficients + k];
    if (!std::isfinite(a) || a == 0.) { return false; }
  }
  for (int32_t k = r.cols_begin; k < r.cols_end; ++k) {
    const record_col_t& c = ps.record_cols[k];
    if (!(c.lower <= c.upper) || c.prefer != preferred(c.lower, c.upper)) { return false; }
    if (interval && ps.integer_at(c.col, position)) { return false; }
    if (k > r.cols_begin && ps.record_cols[k - 1].col >= c.col) { return false; }
  }
  bool from_above       = forcing && forcing_from_above(ps, r);
  const record_col_t& c = ps.record_cols[r.cols_begin];
  if (forcing && (from_above ? c.lower != -inf : c.upper != inf)) { return false; }
  int64_t entries = ps.saved_entries.col.size();
  for (int32_t k = r.rows_begin; k < r.rows_end; ++k) {
    const saved_row_t& row = ps.saved_rows[k];
    if (row.row < 0 || row.row >= ps.nrows || row.begin < 0 || row.begin > row.end ||
        row.end > entries || !(row.lower <= row.upper)) {
      return false;
    }
    if (defined && !(row.lower == row.upper && std::isfinite(row.upper))) { return false; }
    if (interval && !(std::isfinite(row.lower) && std::isfinite(row.upper))) { return false; }
    if (forcing) {
      double a   = ps.record_coefficients[r.coefficients + (k - r.rows_begin)];
      bool upper = from_above == (a > 0);
      if (!std::isfinite(upper ? row.upper : row.lower) ||
          !std::isinf(upper ? row.lower : row.upper)) {
        return false;
      }
    }
    for (int64_t e = row.begin; e < row.end; ++e) {
      int32_t col = ps.saved_entries.col[e];
      if (col < 0 || col >= ps.ncols || removed[col]) { return false; }
      if (defined && ps.integer_at(c.col, position) && !ps.integer_at(col, position)) {
        return false;
      }
    }
  }
  return true;
}

// The contract of the stack with its column map, checked in full: the arrays agree with order,
// no record names a column that an earlier record removed, and every column is either left in
// the reduced model or removed by one record. O(columns + records + saved entries).
bool stack_consistent(const postsolve_t& ps)
{
  size_t fixed = 0, alias = 0, parallel = 0, addition = 0, from_rows = 0;
  int32_t cols_used = 0, rows_used = 0;
  int64_t coefficients_used = 0, entries_used = 0;
  std::vector<uint8_t> removed(ps.ncols, 0);
  size_t n_removed = 0;
  for (size_t position = 0; position < ps.order.size(); ++position) {
    record_kind_t kind = ps.order[position];
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
    } else if (kind == record_kind_t::row_addition) {
      ++addition;
    } else if (kind == record_kind_t::col_from_rows) {
      if (from_rows == ps.col_from_rows.size()) { return false; }
      const col_from_rows_record_t& r = ps.col_from_rows[from_rows++];
      // The parts of the records follow each other in the arrays.
      if (r.cols_begin < cols_used || r.rows_begin < rows_used ||
          r.coefficients < coefficients_used || r.cols_end > (int32_t)ps.record_cols.size() ||
          r.rows_end > (int32_t)ps.saved_rows.size()) {
        return false;
      }
      for (int32_t k = r.cols_begin; k < r.cols_end; ++k) {
        int32_t col = ps.record_cols[k].col;
        if (col < 0 || col >= ps.ncols || removed[col]) { return false; }
        removed[col] = 1;
        ++n_removed;
      }
      if (!from_rows_consistent(ps, r, position, removed)) { return false; }
      for (int32_t k = r.rows_begin; k < r.rows_end; ++k) {
        if (ps.saved_rows[k].begin < entries_used) { return false; }
        entries_used = ps.saved_rows[k].end;
      }
      int32_t ncols = r.cols_end - r.cols_begin, nrows = r.rows_end - r.rows_begin;
      cols_used         = r.cols_end;
      rows_used         = r.rows_end;
      coefficients_used = r.coefficients + (ncols > nrows ? ncols : nrows);
    } else {
      return false;
    }
  }
  if (fixed != ps.fixed_cols.size() || alias != ps.aliases.size() ||
      parallel != ps.parallel_rows.size() || addition != ps.row_additions.size() ||
      from_rows != ps.col_from_rows.size()) {
    return false;
  }
  if (removed != ps.col_removed) { return false; }
  for (int32_t col : ps.col_origin) {
    if (col < 0 || col >= ps.ncols || removed[col]) { return false; }
    removed[col] = 1;
  }
  return n_removed + ps.col_origin.size() == (size_t)ps.ncols;
}

// A pushed record keeps its contract; its columns are marked removed already.
bool pushed_consistent(const postsolve_t& ps)
{
  return from_rows_consistent(ps, ps.col_from_rows.back(), ps.order.size() - 1, ps.col_removed);
}

// Appends a col_from_rows record: `ncols` columns, `nrows` saved rows and the larger of the two
// counts in coefficients.
void push_from_rows(postsolve_t& ps,
                    col_from_rows_kind_t kind,
                    bool dual,
                    const record_col_t* cols,
                    int32_t ncols,
                    const double* coefficients,
                    const saved_row_t* rows,
                    int32_t nrows)
{
  gprs_assert(ncols >= 1 && nrows >= 1);
  col_from_rows_record_t r;
  r.kind         = kind;
  r.dual         = dual;
  r.cols_begin   = ps.record_cols.size();
  r.cols_end     = r.cols_begin + ncols;
  r.rows_begin   = ps.saved_rows.size();
  r.rows_end     = r.rows_begin + nrows;
  r.coefficients = ps.record_coefficients.size();
  for (int32_t k = 0; k < ncols; ++k) {
    int32_t col = cols[k].col;
    gprs_assert(col >= 0 && col < ps.ncols && !ps.col_removed[col]);
    ps.col_removed[col] = 1;
    ps.record_cols.push_back(cols[k]);
  }
  // The rows are the last ones saved, one after the other.
  gprs_assert(rows[nrows - 1].end == (int64_t)ps.saved_entries.col.size());
  for (int32_t k = 0; k < nrows; ++k) {
    gprs_assert(k == 0 || rows[k].begin == rows[k - 1].end);
    ps.saved_rows.push_back(rows[k]);
  }
  int32_t count = ncols > nrows ? ncols : nrows;
  ps.record_coefficients.insert(ps.record_coefficients.end(), coefficients, coefficients + count);
  ps.order.push_back(record_kind_t::col_from_rows);
  ps.col_from_rows.push_back(r);
  gprs_assert(pushed_consistent(ps));
}

// Compensated sum of the saved entries of a row at x; `size` gets the sum of their absolute
// values.
dd_t rest_of(const postsolve_t& ps,
             const saved_row_t& row,
             const std::vector<double>& x,
             double& size)
{
  dd_t rest;
  size = 0.;
  for (int64_t e = row.begin; e < row.end; ++e) {
    double a = ps.saved_entries.value[e], v = x[ps.saved_entries.col[e]];
    dd_add_product(rest, a, v);
    size += std::fabs(a * v);
  }
  return rest;
}

dd_t rest_of(const postsolve_t& ps, const saved_row_t& row, const std::vector<double>& x)
{
  double size;
  return rest_of(ps, row, x, size);
}

// The value that a row leaves an integer forcing column is rounded into the row, and one that
// lies within this many rounding units of the terms it was computed from of an integer counts
// as that integer: 9 - 3 * (7 / 3) is 1.9999999999999996 and stands for 2. Without it the
// value would hang on the last bit of a sum, which another summation order gives otherwise.
constexpr double forcing_rounding_units = 8.;

// The columns of an interval record at prefer, then moved until the row is inside its sides or
// every column that can help is at its bound. The column that ends between prefer and its bound
// gets (side - the other terms) / a and not prefer plus a step: a step from a prefer of 1e9
// keeps eight digits less of the value, which a row of another scale then reads.
void undo_interval(const postsolve_t& ps, const col_from_rows_record_t& r, std::vector<double>& x)
{
  const saved_row_t& row = ps.saved_rows[r.rows_begin];
  dd_t activity          = rest_of(ps, row, x);
  for (int32_t k = r.cols_begin; k < r.cols_end; ++k) {
    const record_col_t& c = ps.record_cols[k];
    x[c.col]              = c.prefer;
    dd_add_product(activity, ps.record_coefficients[r.coefficients + (k - r.cols_begin)], c.prefer);
  }
  double below = dd_residual(row.lower, activity);
  double above = -dd_residual(row.upper, activity);
  if (!(below > 0) && !(above > 0)) { return; }
  bool raise  = below > 0;
  double side = raise ? row.lower : row.upper;
  double need = raise ? below : above;
  for (int32_t k = r.cols_begin; k < r.cols_end && need > 0; ++k) {
    const record_col_t& c = ps.record_cols[k];
    double a              = ps.record_coefficients[r.coefficients + (k - r.cols_begin)];
    bool up               = (a > 0) == raise;
    double target         = up ? c.upper : c.lower;
    double room           = std::fabs(a) * std::fabs(target - c.prefer);
    dd_sub_product(activity, a, c.prefer);
    if (room <= need) {
      x[c.col] = target;
      dd_add_product(activity, a, target);
      need = raise ? dd_residual(side, activity) : -dd_residual(side, activity);
    } else {
      // On the way from prefer to the bound, whatever the last bits of the quotient say.
      double value = dd_residual(side, activity) / a;
      x[c.col]     = up ? std::fmin(std::fmax(value, c.prefer), target)
                        : std::fmax(std::fmin(value, c.prefer), target);
      need         = 0.;
    }
  }
}

void undo_from_rows(const postsolve_t& ps,
                    const col_from_rows_record_t& r,
                    size_t position,
                    std::vector<double>& x)
{
  const record_col_t& c = ps.record_cols[r.cols_begin];
  if (r.kind == col_from_rows_kind_t::defined) {
    const saved_row_t& row = ps.saved_rows[r.rows_begin];
    double value =
      dd_residual(row.upper, rest_of(ps, row, x)) / ps.record_coefficients[r.coefficients];
    x[c.col] = ps.integer_at(c.col, position) ? std::round(value) : value;
  } else if (r.kind == col_from_rows_kind_t::interval) {
    undo_interval(ps, r, x);
  } else {
    bool from_above = forcing_from_above(ps, r);
    double value    = from_above ? c.upper : c.lower;
    // What the value can be off by: the rounding of the row that gave it, none for the bound.
    double rounding = 0.;
    for (int32_t k = r.rows_begin; k < r.rows_end; ++k) {
      const saved_row_t& row = ps.saved_rows[k];
      double a               = ps.record_coefficients[r.coefficients + (k - r.rows_begin)];
      double side            = from_above == (a > 0) ? row.upper : row.lower;
      double size;
      double limit = dd_residual(side, rest_of(ps, row, x, size)) / a;
      if (from_above ? limit < value : limit > value) {
        value    = limit;
        rounding = forcing_rounding_units * rounding_unit * (std::fabs(side) + size) / std::fabs(a);
      }
    }
    if (ps.integer_at(c.col, position)) {
      value = from_above ? std::floor(value + rounding) : std::ceil(value - rounding);
    }
    x[c.col] = value;
  }
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
  nnz      = original.nnz();
  col_type = original.col_type;
  row_origin.clear();
  col_origin.clear();
  order.clear();
  fixed_cols.clear();
  aliases.clear();
  parallel_rows.clear();
  row_additions.clear();
  col_from_rows.clear();
  record_cols.clear();
  saved_rows.clear();
  record_coefficients.clear();
  saved_entries = row_pool_t();
  col_removed.assign(ncols, 0);
}

void postsolve_t::push_fixed_col(int32_t col, double value)
{
  gprs_assert(col >= 0 && col < ncols);
  gprs_assert(!col_removed[col]);
  gprs_assert(std::isfinite(value));
  gprs_assert(!integer_now(col) || integral(value));
  col_removed[col] = 1;
  order.push_back(record_kind_t::fixed_col);
  fixed_cols.push_back(fixed_col_record_t{col, value});
}

void postsolve_t::push_alias(
  int32_t col, int32_t rep, int32_t row, double scale, double offset, alias_origin_t origin)
{
  gprs_assert(col >= 0 && col < ncols && rep >= 0 && rep < ncols && col != rep);
  gprs_assert(origin == alias_origin_t::doubleton || origin == alias_origin_t::probing ||
              origin == alias_origin_t::lock);
  gprs_assert(row >= -1 && row < nrows);
  gprs_assert(origin != alias_origin_t::doubleton || row >= 0);
  gprs_assert(origin != alias_origin_t::lock || row == -1);
  gprs_assert(!col_removed[col] && !col_removed[rep]);
  gprs_assert(std::isfinite(scale) && scale != 0. && std::isfinite(offset));
  gprs_assert(!integer_now(col) || (integer_now(rep) && integral(scale) && integral(offset)));
  col_removed[col] = 1;
  order.push_back(record_kind_t::alias);
  aliases.push_back(alias_record_t{col, rep, row, scale, offset, origin});
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

void postsolve_t::push_defined_col(
  int32_t col, double coefficient, double lower, double upper, const saved_row_t& row, bool dual)
{
  record_col_t c{col, lower, upper, preferred(lower, upper)};
  push_from_rows(*this, col_from_rows_kind_t::defined, dual, &c, 1, &coefficient, &row, 1);
}

void postsolve_t::push_interval_cols(const std::vector<record_col_t>& cols,
                                     const std::vector<double>& coefficients,
                                     const saved_row_t& row)
{
  gprs_assert(!cols.empty() && cols.size() == coefficients.size());
  push_from_rows(*this,
                 col_from_rows_kind_t::interval,
                 false,
                 cols.data(),
                 cols.size(),
                 coefficients.data(),
                 &row,
                 1);
}

void postsolve_t::push_forcing_col(const record_col_t& col,
                                   const std::vector<double>& coefficients,
                                   const std::vector<saved_row_t>& rows)
{
  gprs_assert(!rows.empty() && rows.size() == coefficients.size());
  push_from_rows(*this,
                 col_from_rows_kind_t::forcing,
                 false,
                 &col,
                 1,
                 coefficients.data(),
                 rows.data(),
                 rows.size());
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
    x_original[col] = integer_at(col, order.size()) ? std::round(x_reduced[k]) : x_reduced[k];
  }
  size_t from_rows = col_from_rows.size();
  size_t fixed = fixed_cols.size(), alias = aliases.size();
  for (size_t k = order.size(); k-- > 0;) {
    if (order[k] == record_kind_t::fixed_col) {
      const fixed_col_record_t& record = fixed_cols[--fixed];
      x_original[record.col]           = record.value;
    } else if (order[k] == record_kind_t::alias) {
      const alias_record_t& record = aliases[--alias];
      x_original[record.col]       = record.scale * x_original[record.rep] + record.offset;
    } else if (order[k] == record_kind_t::col_from_rows) {
      undo_from_rows(*this, col_from_rows[--from_rows], k, x_original);
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
