/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpu_submip_fp.hpp"

#include <cuopt/error.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <dual_simplex/presolve.hpp>
#include <mip_heuristics/gpu_submip_fixed_columns.hpp>
#include <mip_heuristics/local_search/feasibility_pump/feasibility_pump.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <mip_heuristics/utils.cuh>
#include <pdlp/utils.cuh>

#include <thrust/logical.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <utilities/logger.hpp>

namespace cuopt::mathematical_optimization::mip {

namespace {

template <typename f_t>
struct is_finite_value_t {
  __device__ bool operator()(f_t value) const { return isfinite(value); }
};

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> make_gpu_submip_problem(
  const raft::handle_t* handle,
  const simplex::lp_problem_t<i_t, f_t>& lp,
  const std::vector<simplex::variable_type_t>& variable_types)
{
  cuopt_expects(variable_types.size() <= static_cast<size_t>(lp.num_cols),
                error_type_t::ValidationError,
                "Sub-MIP variable type count exceeds its column count");
  csr_matrix_t<i_t, f_t> matrix(lp.num_rows, lp.num_cols, lp.A.nnz());
  lp.A.to_compressed_row(matrix);
  matrix.x.resize(lp.A.nnz());
  matrix.j.resize(lp.A.nnz());
  // Keep a stable final row for the improving objective cutoff. This avoids
  // changing row identities when a new incumbent tightens the cutoff.
  for (i_t j = 0; j < lp.num_cols; ++j) {
    if (lp.objective[j] != f_t(0)) {
      matrix.x.push_back(lp.objective[j]);
      matrix.j.push_back(j);
    }
  }
  matrix.row_start.push_back(static_cast<i_t>(matrix.x.size()));
  std::vector<f_t> row_lower = lp.rhs;
  std::vector<f_t> row_upper = lp.rhs;
  row_lower.push_back(-std::numeric_limits<f_t>::infinity());
  row_upper.push_back(std::numeric_limits<f_t>::infinity());
  std::vector<var_t> types(lp.num_cols, var_t::CONTINUOUS);
  for (size_t j = 0; j < variable_types.size(); ++j) {
    if (variable_types[j] != simplex::variable_type_t::CONTINUOUS) { types[j] = var_t::INTEGER; }
  }

  optimization_problem_t<i_t, f_t> model(handle);
  model.set_csr_constraint_matrix(matrix.x.data(),
                                  matrix.x.size(),
                                  matrix.j.data(),
                                  matrix.j.size(),
                                  matrix.row_start.data(),
                                  matrix.row_start.size());
  model.set_variable_lower_bounds(lp.lower.data(), lp.num_cols);
  model.set_variable_upper_bounds(lp.upper.data(), lp.num_cols);
  model.set_variable_types(types.data(), lp.num_cols);
  model.set_objective_coefficients(lp.objective.data(), lp.num_cols);
  model.set_objective_offset(0);
  model.set_objective_scaling_factor(1);
  model.set_maximize(false);
  model.set_constraint_lower_bounds(row_lower.data(), row_lower.size());
  model.set_constraint_upper_bounds(row_upper.data(), row_upper.size());
  handle->sync_stream();
  return model;
}

template <typename i_t, typename f_t>
mip_solver_settings_t<i_t, f_t> gpu_submip_settings(const mip_solver_settings_t<i_t, f_t>& settings,
                                                    const gpu_submip_fp_config_t& config)
{
  auto local_settings                                   = settings;
  local_settings.heuristic_params.relaxed_lp_time_limit = config.pdlp_seconds;
  return local_settings;
}

}  // namespace

template <typename i_t, typename f_t>
struct gpu_submip_fp_t<i_t, f_t>::impl_t {
  impl_t(const raft::handle_t* handle_,
         const simplex::lp_problem_t<i_t, f_t>& lp,
         const std::vector<simplex::variable_type_t>& variable_types,
         const mip_solver_settings_t<i_t, f_t>& settings,
         bool compact_fixed,
         bool fused_adaptive_step,
         const gpu_submip_fp_config_t& config_)
    : handle(handle_),
      config(config_),
      source_lp(lp),
      source_types(variable_types),
      columns(lp, variable_types, compact_fixed),
      model(make_gpu_submip_problem(handle, columns.reduced_lp, columns.reduced_types)),
      problem(model, settings.tolerances),
      context(handle, &problem, gpu_submip_settings(settings, config)),
      lp_solution(problem.n_variables, handle->get_stream()),
      solution(problem),
      fj(context),
      constraint_prop(context),
      line_segment_search(fj, constraint_prop),
      fp(context, fj, constraint_prop, line_segment_search, lp_solution)
  {
    cuopt_expects(std::isfinite(config.initial_alpha) && config.initial_alpha >= 0 &&
                    config.initial_alpha <= 1 && std::isfinite(config.pdlp_seconds) &&
                    config.pdlp_seconds > 0 && std::isfinite(config.rounding_seconds) &&
                    config.rounding_seconds > 0 && std::isfinite(config.propagation_seconds) &&
                    config.propagation_seconds >= config.rounding_seconds,
                  error_type_t::ValidationError,
                  "Invalid GPU FP configuration");
    fp.config.initial_alpha          = config.initial_alpha;
    fp.config.projection_time_limit  = config.pdlp_seconds;
    fp.config.rounding_time_limit    = config.rounding_seconds;
    fp.config.propagation_time_limit = config.propagation_seconds;
    CUOPT_LOG_INFO(
      "GpuSubmipFpConfig name=%s initial_alpha=%.9g pdlp_seconds=%.9g "
      "rounding_seconds=%.9g propagation_seconds=%.9g projection_cap=hard "
      "rounding_schedule=remaining_time_div_10",
      config.name.c_str(),
      config.initial_alpha,
      config.pdlp_seconds,
      config.rounding_seconds,
      config.propagation_seconds);
    // The retained simplex columns bypass GPU preprocessing. FP's integer-fixed
    // verification LP needs identity maps in these compact GPU coordinates.
    problem.presolve_data.initialize_var_mapping(problem, handle);
    problem.original_ids.resize(problem.n_variables);
    std::iota(problem.original_ids.begin(), problem.original_ids.end(), 0);
    problem.reverse_original_ids        = problem.original_ids;
    fp.config.use_projection_warm_start = true;
    fp.config.log_descent_diagnostics   = true;
    fp.config.fused_adaptive_step       = fused_adaptive_step;
    i_t free_integers                   = 0;
    for (i_t j = 0; j < static_cast<i_t>(columns.reduced_types.size()); ++j) {
      if (columns.reduced_types[j] != simplex::variable_type_t::CONTINUOUS) {
        integer_columns.push_back(j);
        if (columns.reduced_lp.upper[j] - columns.reduced_lp.lower[j] >
            problem.tolerances.integrality_tolerance) {
          ++free_integers;
        }
      }
    }
    fp.set_projection_host_metadata(columns.reduced_lp.objective, integer_columns);
    CUOPT_LOG_INFO(
      "GpuSubmipFpModel compact_fixed=%d full_cols=%d gpu_cols=%d "
      "full_lp_nnz=%zu gpu_lp_nnz=%zu fixed_cols=%zu free_integer=%d "
      "objective_shift=%.17g",
      int(compact_fixed),
      int(lp.num_cols),
      int(columns.reduced_lp.num_cols),
      static_cast<size_t>(lp.A.nnz()),
      static_cast<size_t>(columns.reduced_lp.A.nnz()),
      columns.fixed_columns.size(),
      int(free_integers),
      double(columns.objective_shift));
  }

  ~impl_t() noexcept
  {
    try {
      handle->sync_stream();
    } catch (const std::exception& e) {
      CUOPT_LOG_WARN("GPU subMIP FP cleanup synchronization failed: %s", e.what());
    } catch (...) {
      CUOPT_LOG_WARN("GPU subMIP FP cleanup synchronization failed with an unknown exception");
    }
  }

  bool can_reuse(const simplex::lp_problem_t<i_t, f_t>& node_lp) const
  {
    if (node_lp.num_rows != source_lp.num_rows || node_lp.num_cols != source_lp.num_cols ||
        node_lp.A.col_start != source_lp.A.col_start || node_lp.A.i != source_lp.A.i ||
        node_lp.A.x != source_lp.A.x || node_lp.rhs != source_lp.rhs ||
        node_lp.objective != source_lp.objective ||
        node_lp.lower.size() != source_lp.lower.size() ||
        node_lp.upper.size() != source_lp.upper.size()) {
      return false;
    }
    for (const auto j : columns.fixed_columns) {
      if (node_lp.lower[j] != columns.fixed_values[j] ||
          node_lp.upper[j] != columns.fixed_values[j]) {
        return false;
      }
    }
    return true;
  }

  bool run(const simplex::lp_problem_t<i_t, f_t>& node_lp,
           const std::vector<f_t>& node_primal,
           f_t cutoff,
           double remaining_time,
           std::atomic<int>* halt,
           std::vector<f_t>& candidate,
           bool use_cached_primal,
           const char* primal_seed_name,
           uint64_t job_id,
           int projection_limit)
  {
    run_info = {};
    candidate.clear();
    if (remaining_time <= 0 || (halt && halt->load(std::memory_order_acquire)) ||
        !std::isfinite(cutoff)) {
      return false;
    }
    cuopt_expects(can_reuse(node_lp),
                  error_type_t::ValidationError,
                  "GPU sub-MIP FP session must be recreated after model or fixing changes");
    if (problem.n_integer_vars == 0) { return false; }
    cuopt_expects(node_primal.size() == columns.full_to_free.size(),
                  error_type_t::ValidationError,
                  "GPU sub-MIP FP session must be recreated after model changes");
    if (!std::all_of(
          node_primal.begin(), node_primal.end(), [](f_t value) { return std::isfinite(value); })) {
      return false;
    }

    fp.config.projection_limit      = projection_limit;
    fp.timer                        = timer_t(remaining_time);
    fp.concurrent_halt              = halt;
    fj.concurrent_halt              = halt;
    constraint_prop.concurrent_halt = halt;
    const f_t local_cutoff          = columns.reduced_cutoff(cutoff);
    std::vector<typename type_2<f_t>::type> bounds(problem.n_variables);
    for (i_t j = 0; j < problem.n_variables; ++j) {
      const i_t full_j = columns.free_to_full[j];
      bounds[j]        = {node_lp.lower[full_j], node_lp.upper[full_j]};
    }
    fp.set_projection_host_bounds(bounds);
    raft::copy(problem.variable_bounds.data(), bounds.data(), bounds.size(), handle->get_stream());
    problem.constraint_upper_bounds.set_element_async(
      problem.n_constraints - 1, local_cutoff, handle->get_stream());
    pdlp::combine_constraint_bounds(problem, problem.combined_bounds);
    problem.compute_binary_var_table();
    fj.reset_weights(handle->get_stream());

    if (use_cached_primal && has_projection) {
      has_projection = thrust::all_of(handle->get_thrust_policy(),
                                      fp.last_projection.begin(),
                                      fp.last_projection.end(),
                                      is_finite_value_t<f_t>{});
    }
    const bool cached_seed = use_cached_primal && has_projection;
    std::vector<f_t> reduced_primal;
    if (cached_seed) {
      solution.copy_new_assignment(fp.last_projection);
    } else {
      reduced_primal = columns.gather(node_primal);
      solution.copy_new_assignment(reduced_primal);
    }
    solution.clamp_within_bounds();
    raft::copy(fp.last_projection.data(),
               solution.assignment.data(),
               solution.assignment.size(),
               handle->get_stream());
    raft::copy(lp_solution.data(),
               solution.assignment.data(),
               solution.assignment.size(),
               handle->get_stream());
    // The cutoff has the same row identity but a new RHS; start its dual at zero.
    const f_t zero = 0;
    solution.lp_state.prev_dual.set_element_async(
      problem.n_constraints - 1, zero, handle->get_stream());
    fp.cycle_queue.reset(solution);
    fp.reset();
    fp.diagnostics.cutoff = local_cutoff;
    fp.diagnostics.job_id = job_id;
    for (const auto j : integer_columns) {
      const i_t full_j = columns.free_to_full[j];
      if (node_lp.upper[full_j] - node_lp.lower[full_j] >
          problem.tolerances.integrality_tolerance) {
        ++fp.diagnostics.free_integers;
      }
    }
    const auto descent_begin      = fp.timer.elapsed_time();
    const auto projection_begin   = fp.total_lp_time;
    const auto projections_before = fp.n_lp_calls;
    const auto verification_begin = fp.total_verification_time;
    const auto verifies_before    = fp.n_verification_calls;
    run_info.started              = true;
    const bool found              = fp.run_single_fp_descent(solution);
    run_info.projections          = fp.n_lp_calls - projections_before;
    run_info.exit_reason          = fp.diagnostics.exit_reason;
    const double descent_seconds  = fp.timer.elapsed_time() - descent_begin;
    run_info.descent_seconds      = descent_seconds;
    run_info.projection_seconds   = fp.total_lp_time - projection_begin;
    has_projection                = true;
    // Refresh once after the descent: an unsuccessful FJ may have restored the
    // assignment without recomputing its cached row residuals.
    const bool final_feasible      = solution.compute_feasibility();
    const auto max_row_violation   = solution.compute_max_constraint_violation();
    const auto max_bound_violation = solution.compute_max_variable_violation();
    // Projection time includes constructing and solving PDLP; verification time
    // also includes the integer-fixed model and its bound propagation.
    CUOPT_LOG_INFO(
      "GpuSubmipFpDescent job=%llu config=%s seed=%s projections=%d projection_lp_seconds=%.6f "
      "verification_calls=%d verification_seconds=%.6f descent_seconds=%.6f "
      "rounding_prop_seconds=%.6f fj_seconds=%.6f exit=%s free_integer=%d "
      "last_projection_fractional=%d best_projection_fractional=%d last_checked_distance=%.9g "
      "end_fractional=%d end_total_excess=%.9g end_max_row_violation=%.9g "
      "end_max_bound_violation=%.9g end_cutoff_excess=%.9g fp_found=%d",
      (unsigned long long)job_id,
      config.name.c_str(),
      cached_seed ? "previous_projection" : primal_seed_name,
      int(fp.n_lp_calls - projections_before),
      double(fp.total_lp_time - projection_begin),
      int(fp.n_verification_calls - verifies_before),
      double(fp.total_verification_time - verification_begin),
      descent_seconds,
      fp.diagnostics.rounding_seconds,
      fp.diagnostics.fj_seconds,
      fp.diagnostics.exit_reason,
      fp.diagnostics.free_integers,
      fp.diagnostics.last_fractional,
      fp.diagnostics.projections > 0 ? fp.diagnostics.best_fractional : -1,
      fp.diagnostics.last_distance,
      int(problem.n_integer_vars - solution.n_assigned_integers),
      double(solution.h_infeasibility_cost),
      double(max_row_violation),
      double(max_bound_violation),
      std::max(0., double(solution.get_objective() - local_cutoff)),
      int(found));
    if (!found || !final_feasible || solution.get_objective() > local_cutoff ||
        max_bound_violation > problem.tolerances.absolute_tolerance) {
      return false;
    }
    candidate = columns.scatter(solution.get_host_assignment());
    if (!std::all_of(
          candidate.begin(), candidate.end(), [](f_t value) { return std::isfinite(value); })) {
      candidate.clear();
      return false;
    }
    return true;
  }

  bool get_restart_seed(std::vector<f_t>& seed)
  {
    seed.clear();
    if (!has_projection) { return false; }
    std::vector<f_t> reduced(fp.last_projection.size());
    raft::copy(reduced.data(), fp.last_projection.data(), reduced.size(), handle->get_stream());
    handle->sync_stream();
    const auto finite = [](f_t value) { return std::isfinite(value); };
    if (!std::all_of(reduced.begin(), reduced.end(), finite)) {
      reduced = solution.get_host_assignment();
      if (!std::all_of(reduced.begin(), reduced.end(), finite)) { return false; }
    }
    seed = columns.scatter(reduced);
    return true;
  }

  const raft::handle_t* handle;
  const gpu_submip_fp_config_t config;
  const simplex::lp_problem_t<i_t, f_t> source_lp;
  const std::vector<simplex::variable_type_t> source_types;
  gpu_submip_fixed_columns_t<i_t, f_t> columns;
  optimization_problem_t<i_t, f_t> model;
  problem_t<i_t, f_t> problem;
  mip_solver_context_t<i_t, f_t> context;
  rmm::device_uvector<f_t> lp_solution;
  solution_t<i_t, f_t> solution;
  fj_t<i_t, f_t> fj;
  constraint_prop_t<i_t, f_t> constraint_prop;
  line_segment_search_t<i_t, f_t> line_segment_search;
  feasibility_pump_t<i_t, f_t> fp;
  std::vector<i_t> integer_columns;
  bool has_projection{false};
  gpu_submip_fp_run_info_t run_info;
};

template <typename i_t, typename f_t>
gpu_submip_fp_t<i_t, f_t>::gpu_submip_fp_t(
  const raft::handle_t* handle,
  const simplex::lp_problem_t<i_t, f_t>& lp,
  const std::vector<simplex::variable_type_t>& variable_types,
  const mip_solver_settings_t<i_t, f_t>& settings,
  bool compact_fixed,
  bool fused_adaptive_step,
  const gpu_submip_fp_config_t& config)
  : impl_(std::make_unique<impl_t>(
      handle, lp, variable_types, settings, compact_fixed, fused_adaptive_step, config))
{
}

template <typename i_t, typename f_t>
gpu_submip_fp_t<i_t, f_t>::~gpu_submip_fp_t() = default;

template <typename i_t, typename f_t>
bool gpu_submip_fp_t<i_t, f_t>::can_reuse(
  const simplex::lp_problem_t<i_t, f_t>& node_lp,
  const std::vector<simplex::variable_type_t>& variable_types) const
{
  return variable_types == impl_->source_types && impl_->can_reuse(node_lp);
}

template <typename i_t, typename f_t>
bool gpu_submip_fp_t<i_t, f_t>::get_restart_seed(std::vector<f_t>& seed) const
{
  return impl_->get_restart_seed(seed);
}

template <typename i_t, typename f_t>
gpu_submip_fp_run_info_t gpu_submip_fp_t<i_t, f_t>::last_run_info() const
{
  return impl_->run_info;
}

template <typename i_t, typename f_t>
const gpu_submip_fp_config_t& gpu_submip_fp_t<i_t, f_t>::config() const
{
  return impl_->config;
}

template <typename i_t, typename f_t>
bool gpu_submip_fp_t<i_t, f_t>::run(const simplex::lp_problem_t<i_t, f_t>& node_lp,
                                    const std::vector<f_t>& node_primal,
                                    f_t improving_cutoff,
                                    double remaining_time,
                                    std::atomic<int>* halt,
                                    std::vector<f_t>& candidate,
                                    bool use_cached_primal,
                                    const char* primal_seed_name,
                                    uint64_t job_id,
                                    int projection_limit)
{
  return impl_->run(node_lp,
                    node_primal,
                    improving_cutoff,
                    remaining_time,
                    halt,
                    candidate,
                    use_cached_primal,
                    primal_seed_name,
                    job_id,
                    projection_limit);
}

#if MIP_INSTANTIATE_FLOAT
template class gpu_submip_fp_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class gpu_submip_fp_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
