/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// This file is compiled with -ffp-contract=off, as every file that includes the gprs headers.
// It therefore instantiates no floating-point template that the rest of the library shares.

#include <presolve/gprs_presolve.hpp>

#include <cuopt/mathematical_optimization/optimization_problem_utils.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>
#include <utilities/macros.cuh>

#include <gprs/presolve.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <type_traits>
#include <utility>

namespace cuopt::mathematical_optimization::mip {

namespace {

// Sorts the entries of the row that starts at `begin` and ends the matrix by column.
// False when a column appears twice.
bool sort_last_row(int64_t begin,
                   gprs::model_t& model,
                   std::vector<std::pair<int32_t, double>>& entries)
{
  entries.clear();
  for (int64_t p = begin; p < model.nnz(); ++p) {
    entries.emplace_back(model.col_index[p], model.value[p]);
  }
  std::sort(entries.begin(), entries.end());
  for (size_t k = 0; k < entries.size(); ++k) {
    if (k > 0 && entries[k].first == entries[k - 1].first) { return false; }
    model.col_index[begin + k] = entries[k].first;
    model.value[begin + k]     = entries[k].second;
  }
  return true;
}

// The problem as a gprs model: a minimisation with rows lower <= a x <= upper. A maximisation is
// negated; explicit zeros of the matrix are left out. Returns nullptr, or what gprs cannot carry.
template <typename i_t, typename f_t>
const char* build_gprs_model(io::mps_data_model_t<i_t, f_t> const& problem, gprs::model_t& model)
{
  if (!std::is_same_v<f_t, double>) { return "gprs works in double precision"; }
  if (problem.has_quadratic_objective() || problem.has_quadratic_constraints()) {
    return "the problem has quadratic terms";
  }

  const i_t n_rows       = problem.get_n_constraints();
  const i_t n_cols       = problem.get_n_variables();
  const auto& offsets    = problem.get_constraint_matrix_offsets();
  const auto& indices    = problem.get_constraint_matrix_indices();
  const auto& values     = problem.get_constraint_matrix_values();
  const auto& objective  = problem.get_objective_coefficients();
  const auto& var_lb     = problem.get_variable_lower_bounds();
  const auto& var_ub     = problem.get_variable_upper_bounds();
  const auto& var_types  = problem.get_variable_types();
  const bool maximize    = problem.get_sense();
  const f_t offset_value = problem.get_objective_offset();
  std::vector<f_t> constr_lb(problem.get_constraint_lower_bounds());
  std::vector<f_t> constr_ub(problem.get_constraint_upper_bounds());
  expand_rhs(problem, constr_lb, constr_ub);
  cuopt_assert(n_rows == 0 || offsets.size() == (size_t)n_rows + 1,
               "CSR offsets must cover every constraint");
  cuopt_assert(constr_lb.size() == (size_t)n_rows && constr_ub.size() == (size_t)n_rows,
               "constraint bounds must cover every constraint");
  cuopt_assert(var_types.empty() || var_types.size() == (size_t)n_cols,
               "variable types must cover every variable");
  if (var_lb.size() != (size_t)n_cols || var_ub.size() != (size_t)n_cols) {
    return "the variable bounds are not set for every variable";
  }

  model       = gprs::model_t{};
  model.nrows = n_rows;
  model.ncols = n_cols;
  model.row_start.reserve((size_t)n_rows + 1);
  model.col_index.reserve(values.size());
  model.value.reserve(values.size());
  model.row_start.push_back(0);
  std::vector<std::pair<int32_t, double>> entries;
  for (i_t i = 0; i < n_rows; ++i) {
    const int64_t begin = model.nnz();
    bool sorted         = true;
    for (i_t p = offsets[i]; p < offsets[i + 1]; ++p) {
      if (values[p] == 0) { continue; }
      sorted = sorted && (model.nnz() == begin || model.col_index.back() < indices[p]);
      model.col_index.push_back(indices[p]);
      model.value.push_back(values[p]);
    }
    if (!sorted && !sort_last_row(begin, model, entries)) {
      return "a constraint has two entries of the same variable";
    }
    model.row_start.push_back(model.nnz());
  }
  model.row_lower.assign(constr_lb.begin(), constr_lb.end());
  model.row_upper.assign(constr_ub.begin(), constr_ub.end());
  model.col_lower.assign(var_lb.begin(), var_lb.end());
  model.col_upper.assign(var_ub.begin(), var_ub.end());
  model.cost.resize(n_cols);
  for (i_t j = 0; j < n_cols; ++j) {
    model.cost[j] = maximize ? -objective[j] : objective[j];
  }
  model.objective_offset = maximize ? -offset_value : offset_value;
  model.col_type.assign(n_cols, 0);
  for (size_t j = 0; j < var_types.size(); ++j) {
    const var_t type = char_to_var_type(var_types[j]);
    if (type == var_t::SEMI_CONTINUOUS) { return "the problem has semi-continuous variables"; }
    model.col_type[j] = type == var_t::INTEGER;
  }
  if (!gprs::well_formed(model)) {
    return "the problem has a value gprs does not accept (NaN, a lower bound of +inf, an upper "
           "bound of -inf, or a coefficient or objective value that is not finite)";
  }
  return nullptr;
}

// The inverse of build_gprs_model for the reduced model. Like the reduced problem of the
// third-party presolve it has constraint lower and upper bounds, no row types and no names.
template <typename i_t, typename f_t>
void build_reduced_problem(const gprs::model_t& model,
                           io::mps_data_model_t<i_t, f_t> const& problem,
                           io::mps_data_model_t<i_t, f_t>& reduced_problem)
{
  // problem_t takes a problem without nonzeros as one without constraints.
  cuopt_assert(model.nnz() > 0 || model.nrows == 0,
               "a reduced problem without nonzeros must have no constraints");
  const bool maximize = problem.get_sense();
  reduced_problem     = io::mps_data_model_t<i_t, f_t>{};
  reduced_problem.set_maximize(maximize);
  reduced_problem.set_objective_offset(maximize ? -model.objective_offset : model.objective_offset);
  reduced_problem.set_objective_scaling_factor(problem.get_objective_scaling_factor());
  reduced_problem.set_problem_name(problem.get_problem_name());

  const std::vector<i_t> offsets(model.row_start.begin(), model.row_start.end());
  const std::vector<i_t> indices(model.col_index.begin(), model.col_index.end());
  const std::vector<f_t> values(model.value.begin(), model.value.end());
  reduced_problem.set_csr_constraint_matrix(std::span<const f_t>(values.data(), values.size()),
                                            std::span<const i_t>(indices.data(), indices.size()),
                                            std::span<const i_t>(offsets.data(), offsets.size()));
  if (model.nrows == 0 && model.ncols == 0) { return; }

  std::vector<f_t> objective(model.ncols);
  std::vector<char> var_types(model.ncols);
  for (int32_t j = 0; j < model.ncols; ++j) {
    objective[j] = maximize ? -model.cost[j] : model.cost[j];
    var_types[j] = model.col_type[j] ? 'I' : 'C';
  }
  const std::vector<f_t> constr_lb(model.row_lower.begin(), model.row_lower.end());
  const std::vector<f_t> constr_ub(model.row_upper.begin(), model.row_upper.end());
  const std::vector<f_t> var_lb(model.col_lower.begin(), model.col_lower.end());
  const std::vector<f_t> var_ub(model.col_upper.begin(), model.col_upper.end());
  reduced_problem.set_objective_coefficients(
    std::span<const f_t>(objective.data(), objective.size()));
  reduced_problem.set_constraint_lower_bounds(
    std::span<const f_t>(constr_lb.data(), constr_lb.size()));
  reduced_problem.set_constraint_upper_bounds(
    std::span<const f_t>(constr_ub.data(), constr_ub.size()));
  reduced_problem.set_variable_lower_bounds(std::span<const f_t>(var_lb.data(), var_lb.size()));
  reduced_problem.set_variable_upper_bounds(std::span<const f_t>(var_ub.data(), var_ub.size()));
  reduced_problem.set_variable_types(var_types);
}

gprs_presolve_status_t convert_status(gprs::status_t status)
{
  switch (status) {
    case gprs::status_t::reduced: return gprs_presolve_status_t::REDUCED;
    case gprs::status_t::unchanged: return gprs_presolve_status_t::UNCHANGED;
    case gprs::status_t::reduced_to_empty: return gprs_presolve_status_t::REDUCED_TO_EMPTY;
    case gprs::status_t::infeasible: return gprs_presolve_status_t::INFEASIBLE;
    case gprs::status_t::unbounded_or_infeasible:
      return gprs_presolve_status_t::UNBOUNDED_OR_INFEASIBLE;
    case gprs::status_t::limit:
    case gprs::status_t::failed:
      return gprs_presolve_status_t::NOT_RUN;
      // No default case: a new status of gprs must fail to compile here.
  }
  return gprs_presolve_status_t::NOT_RUN;
}

// Absolute plus relative tolerance of a constraint: the rule of get_cstr_tolerance.
double constraint_tolerance(double lower,
                            double upper,
                            double absolute_tolerance,
                            double relative_tolerance)
{
  double max_bound = 0.;
  if (std::isfinite(upper)) { max_bound = std::max(max_bound, std::abs(upper)); }
  if (std::isfinite(lower)) { max_bound = std::max(max_bound, std::abs(lower)); }
  return absolute_tolerance + max_bound * relative_tolerance;
}

double seconds_since(std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

template <typename i_t, typename f_t>
struct gprs_presolve_t<i_t, f_t>::impl_t {
  gprs::model_t original;
  gprs::postsolve_t postsolve;
  std::vector<i_t> reduced_to_original_map;
  std::vector<i_t> original_to_reduced_map;
  double seconds = 0.;
};

template <typename i_t, typename f_t>
gprs_presolve_t<i_t, f_t>::gprs_presolve_t() : impl_(std::make_unique<impl_t>())
{
}

template <typename i_t, typename f_t>
gprs_presolve_t<i_t, f_t>::~gprs_presolve_t() = default;

template <typename i_t, typename f_t>
gprs_presolve_status_t gprs_presolve_t<i_t, f_t>::apply(
  io::mps_data_model_t<i_t, f_t> const& problem,
  bool primal_only,
  double time_limit,
  io::mps_data_model_t<i_t, f_t>& reduced_problem)
{
  const auto start   = std::chrono::steady_clock::now();
  const char* reason = build_gprs_model(problem, impl_->original);
  if (reason != nullptr) {
    CUOPT_LOG_INFO("GPRS presolve skipped: %s", reason);
    impl_->seconds = seconds_since(start);
    return gprs_presolve_status_t::NOT_RUN;
  }

  gprs::options_t options;
  options.primal_only                 = primal_only;
  options.time_limit                  = time_limit;
  gprs::result_t result               = gprs::presolve(impl_->original, options);
  const gprs_presolve_status_t status = convert_status(result.status);

  const bool has_postsolve = status == gprs_presolve_status_t::UNCHANGED ||
                             status == gprs_presolve_status_t::REDUCED ||
                             status == gprs_presolve_status_t::REDUCED_TO_EMPTY;
  if (has_postsolve) {
    const auto& col_origin = result.postsolve.col_origin;
    impl_->reduced_to_original_map.assign(col_origin.begin(), col_origin.end());
    impl_->original_to_reduced_map.assign(impl_->original.ncols, -1);
    for (size_t k = 0; k < col_origin.size(); ++k) {
      impl_->original_to_reduced_map[col_origin[k]] = k;
    }
    if (status != gprs_presolve_status_t::UNCHANGED) {
      build_reduced_problem(result.reduced, problem, reduced_problem);
    }
  }
  impl_->seconds = seconds_since(start);

  CUOPT_LOG_INFO(
    "GPRS presolve: %d constraints, %d variables, %lld nonzeros -> %d constraints, %d variables, "
    "%lld nonzeros in %.3f seconds (engine %.3f), status %s%s",
    impl_->original.nrows,
    impl_->original.ncols,
    (long long)impl_->original.nnz(),
    result.reduced.nrows,
    result.reduced.ncols,
    (long long)result.reduced.nnz(),
    impl_->seconds,
    result.seconds,
    gprs::status_name(result.status),
    status == gprs_presolve_status_t::NOT_RUN ? ", the original problem is kept" : "");
  CUOPT_LOG_DEBUG("GPRS presolve: %d rounds, %d rebuilds, primal_only %d",
                  result.rounds,
                  result.rebuilds,
                  (int)primal_only);
  for (int32_t rule = 0; rule < gprs::rule::count; ++rule) {
    CUOPT_LOG_DEBUG(
      "GPRS %s: %lld calls, %lld work, %lld findings, %.3f seconds, removed %lld constraints, "
      "%lld variables, %lld nonzeros",
      gprs::rule_name(rule),
      (long long)result.stats[rule].calls,
      (long long)result.stats[rule].work,
      (long long)result.stats[rule].findings,
      result.stats[rule].seconds,
      (long long)result.stats[rule].rows_removed,
      (long long)result.stats[rule].cols_removed,
      (long long)result.stats[rule].nnz_removed);
  }

  if (has_postsolve) { impl_->postsolve = std::move(result.postsolve); }
  return status;
}

template <typename i_t, typename f_t>
void gprs_presolve_t<i_t, f_t>::postsolve(std::vector<f_t>& primal_solution) const
{
  cuopt_assert(primal_solution.size() == impl_->reduced_to_original_map.size(),
               "postsolve takes a point of the reduced problem");
  const std::vector<double> x_reduced(primal_solution.begin(), primal_solution.end());
  std::vector<double> x_original;
  impl_->postsolve.postsolve(x_reduced, x_original);
  primal_solution.assign(x_original.begin(), x_original.end());
}

template <typename i_t, typename f_t>
void gprs_presolve_t<i_t, f_t>::crush(const std::vector<f_t>& original_primal,
                                      std::vector<f_t>& reduced_primal) const
{
  cuopt_assert(original_primal.size() == impl_->original_to_reduced_map.size(),
               "crush takes a point of the original problem");
  const std::vector<double> x_original(original_primal.begin(), original_primal.end());
  std::vector<double> x_reduced;
  impl_->postsolve.crush(x_original, x_reduced);
  reduced_primal.assign(x_reduced.begin(), x_reduced.end());
}

template <typename i_t, typename f_t>
bool gprs_presolve_t<i_t, f_t>::is_original_primal_solution_feasible(
  const std::vector<f_t>& original_primal,
  f_t absolute_tolerance,
  f_t relative_tolerance,
  f_t integrality_tolerance) const
{
  const gprs::model_t& model = impl_->original;
  if (original_primal.size() != (size_t)model.ncols) { return false; }

  for (int32_t j = 0; j < model.ncols; ++j) {
    const double value = original_primal[j];
    if (!std::isfinite(value) || value < model.col_lower[j] - integrality_tolerance ||
        value > model.col_upper[j] + integrality_tolerance ||
        (model.col_type[j] && std::abs(value - std::round(value)) > integrality_tolerance)) {
      return false;
    }
  }

  for (int32_t i = 0; i < model.nrows; ++i) {
    gprs::dd_t sum;
    for (int64_t p = model.row_start[i]; p < model.row_start[i + 1]; ++p) {
      gprs::dd_add_product(sum, model.value[p], original_primal[model.col_index[p]]);
    }
    const double activity  = gprs::dd_value(sum);
    const double tolerance = constraint_tolerance(
      model.row_lower[i], model.row_upper[i], absolute_tolerance, relative_tolerance);
    if (!std::isfinite(activity) || activity < model.row_lower[i] - tolerance ||
        activity > model.row_upper[i] + tolerance) {
      return false;
    }
  }
  return true;
}

template <typename i_t, typename f_t>
void gprs_presolve_t<i_t, f_t>::compose_maps(std::vector<i_t>& reduced_to_original,
                                             std::vector<i_t>& original_to_reduced) const
{
  for (i_t& j : reduced_to_original) {
    cuopt_assert(j >= 0 && j < (i_t)impl_->reduced_to_original_map.size(),
                 "the map must point into the reduced problem of gprs");
    j = impl_->reduced_to_original_map[j];
  }
  original_to_reduced.assign(impl_->original_to_reduced_map.size(), -1);
  for (size_t k = 0; k < reduced_to_original.size(); ++k) {
    original_to_reduced[reduced_to_original[k]] = k;
  }
}

template <typename i_t, typename f_t>
const std::vector<i_t>& gprs_presolve_t<i_t, f_t>::get_reduced_to_original_map() const
{
  return impl_->reduced_to_original_map;
}

template <typename i_t, typename f_t>
const std::vector<i_t>& gprs_presolve_t<i_t, f_t>::get_original_to_reduced_map() const
{
  return impl_->original_to_reduced_map;
}

template <typename i_t, typename f_t>
double gprs_presolve_t<i_t, f_t>::get_presolve_time() const
{
  return impl_->seconds;
}

#if MIP_INSTANTIATE_FLOAT || PDLP_INSTANTIATE_FLOAT
template class gprs_presolve_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class gprs_presolve_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
