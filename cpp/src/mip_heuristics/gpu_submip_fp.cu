/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpu_submip_fp.hpp"

#include <cuopt/error.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <dual_simplex/presolve.hpp>
#include <mip_heuristics/gpu_submip_experiment.hpp>
#include <mip_heuristics/gpu_submip_fixed_columns.hpp>
#include <mip_heuristics/local_search/feasibility_pump/feasibility_pump.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <mip_heuristics/utils.cuh>
#include <pdlp/utils.cuh>

#include <thrust/logical.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
  const std::vector<simplex::variable_type_t>& variable_types,
  bool cutoff_row)
{
  cuopt_expects(variable_types.size() <= static_cast<size_t>(lp.num_cols),
                error_type_t::ValidationError,
                "Sub-MIP variable type count exceeds its column count");
  csr_matrix_t<i_t, f_t> matrix(lp.num_rows, lp.num_cols, lp.A.nnz());
  lp.A.to_compressed_row(matrix);
  matrix.x.resize(lp.A.nnz());
  matrix.j.resize(lp.A.nnz());
  std::vector<f_t> row_lower = lp.rhs;
  std::vector<f_t> row_upper = lp.rhs;
  // Keep a stable final row for the improving objective cutoff. This avoids
  // changing row identities when a new incumbent tightens the cutoff. A search without an
  // incumbent has no such row: FJ needs a bound on every row.
  if (cutoff_row) {
    for (i_t j = 0; j < lp.num_cols; ++j) {
      if (lp.objective[j] != f_t(0)) {
        matrix.x.push_back(lp.objective[j]);
        matrix.j.push_back(j);
      }
    }
    matrix.row_start.push_back(static_cast<i_t>(matrix.x.size()));
    row_lower.push_back(-std::numeric_limits<f_t>::infinity());
    row_upper.push_back(std::numeric_limits<f_t>::infinity());
  }
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
  auto local_settings                                             = settings;
  local_settings.mip_hyper_params.heuristic_relaxed_lp_time_limit = config.pdlp_seconds;
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
         const gpu_submip_fp_config_t& config_,
         bool cutoff_row_)
    : handle(handle_),
      config(config_),
      source_lp(lp),
      source_types(variable_types),
      columns(lp, variable_types, compact_fixed),
      model(
        make_gpu_submip_problem(handle, columns.reduced_lp, columns.reduced_types, cutoff_row_)),
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
    cutoff_row                           = cutoff_row_;
    fp.config.initial_alpha              = config.initial_alpha;
    fp.config.projection_time_limit      = config.pdlp_seconds;
    fp.config.rounding_time_limit        = config.rounding_seconds;
    fp.config.propagation_time_limit     = config.propagation_seconds;
    fp.config.propagate_initial_rounding = config.initial_rounding;
    CUOPT_LOG_INFO(
      "GpuSubmipFpConfig name=%s initial_alpha=%.9g pdlp_seconds=%.9g "
      "rounding_seconds=%.9g propagation_seconds=%.9g projection_cap=hard "
      "rounding_schedule=remaining_time_div_10 initial_rounding=%d",
      config.name.c_str(),
      config.initial_alpha,
      config.pdlp_seconds,
      config.rounding_seconds,
      config.propagation_seconds,
      int(config.initial_rounding));
    // The retained simplex columns bypass GPU preprocessing. FP's integer-fixed
    // verification LP needs identity maps in these compact GPU coordinates.
    problem.presolve_data.initialize_var_mapping(problem, handle);
    problem.original_ids.resize(problem.n_variables);
    std::iota(problem.original_ids.begin(), problem.original_ids.end(), 0);
    problem.reverse_original_ids        = problem.original_ids;
    fp.config.use_projection_warm_start = true;
    fp.config.log_descent_diagnostics   = true;
    fp.config.fused_adaptive_step       = fused_adaptive_step;
    fp.config.use_pdfp_projection       = gpu_submip_experiment::pdfp();
    if (gpu_submip_experiment::projection_seconds() > 0) {
      fp.config.projection_time_limit = gpu_submip_experiment::projection_seconds();
    }
    screen_seconds = gpu_submip_experiment::screen_seconds();
    lp_start       = gpu_submip_experiment::lp_start();
    target_share   = gpu_submip_experiment::target_share();
    cuopt_expects(target_share >= 0 && target_share < 1,
                  error_type_t::ValidationError,
                  "The target share of the LP improvement lies in [0, 1)");
    CUOPT_LOG_INFO(
      "GpuSubmipFpProjector projector=%s projection_seconds=%.9g screen_seconds=%.9g lp_start=%d "
      "target_share=%.9g",
      fp.config.use_pdfp_projection ? "pdfp" : "pdlp_epigraph",
      fp.config.projection_time_limit,
      screen_seconds,
      int(lp_start),
      target_share);
    i_t free_integers = 0;
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
      "objective_shift=%.17g cutoff_row=%d",
      int(compact_fixed),
      int(lp.num_cols),
      int(columns.reduced_lp.num_cols),
      static_cast<size_t>(lp.A.nnz()),
      static_cast<size_t>(columns.reduced_lp.A.nnz()),
      columns.fixed_columns.size(),
      int(free_integers),
      double(columns.objective_shift),
      int(cutoff_row));
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
    // An infinite cutoff is a search without an incumbent: the cutoff row stays free.
    if (remaining_time <= 0 || (halt && halt->load(std::memory_order_acquire)) ||
        !(cutoff > -std::numeric_limits<f_t>::infinity())) {
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
    cuopt_assert(cutoff_row == std::isfinite(local_cutoff),
                 "A session has a cutoff row exactly when its job has a cutoff");
    if (cutoff_row) {
      problem.constraint_upper_bounds.set_element_async(
        problem.n_constraints - 1, local_cutoff, handle->get_stream());
    }
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
    if (cutoff_row) {
      solution.lp_state.prev_dual.set_element_async(
        problem.n_constraints - 1, zero, handle->get_stream());
    }
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
    // The objective level the descent has to reach; the job's cutoff unless the screen sets more.
    f_t fp_cutoff = local_cutoff;
    if (fp.config.use_pdfp_projection && screen_seconds > 0 && std::isfinite(local_cutoff)) {
      // The restricted LP without its cutoff row: can c'x get below the cutoff at all?
      const auto stream    = handle->get_stream();
      const auto row       = problem.n_constraints - 1;
      const f_t no_bound   = std::numeric_limits<f_t>::infinity();
      const auto unbounded = [](const auto& b) {
        return !std::isfinite(get_lower(b)) || !std::isfinite(get_upper(b));
      };
      // The dual bound is valid only on a finite box, so the screen runs on the implied one.
      const auto box   = implied_box(bounds);
      const bool empty = std::any_of(
        box.begin(), box.end(), [](const auto& b) { return get_lower(b) > get_upper(b); });
      pdfp_projection_result_t screen;
      // With an LP start or a target the screen goes on to the LP optimum and hands its point on.
      const bool solve = lp_start || target_share > 0;
      rmm::device_uvector<f_t> lp_point(0, stream);
      problem.constraint_upper_bounds.set_element_async(row, no_bound, stream);
      if (empty) {
        screen.reach_verdict = 2;
      } else {
        raft::copy(problem.variable_bounds.data(), box.data(), box.size(), stream);
        screen =
          fp.pdfp_reach(solution, local_cutoff, screen_seconds, 1e-4, solve ? &lp_point : nullptr);
        raft::copy(problem.variable_bounds.data(), bounds.data(), bounds.size(), stream);
      }
      if (screen.reach_verdict > 0) { cuopt_func_call(check_dropped(local_cutoff)); }
      // An LP point at or below the cutoff: ask the descent for a share of what the LP gains. The
      // dual bound corrects a last point that is below the optimum because it is not yet feasible.
      const bool reached = screen.reach_verdict < 0;
      if (reached && target_share > 0) {
        const f_t lp_value = std::max(screen.linear_objective, screen.dual_bound);
        fp_cutoff = local_cutoff - target_share * std::max(f_t(0), local_cutoff - lp_value);
      }
      problem.constraint_upper_bounds.set_element_async(row, fp_cutoff, stream);
      pdlp::combine_constraint_bounds(problem, problem.combined_bounds);
      if (reached && lp_start) {
        solution.copy_new_assignment(lp_point);
        solution.clamp_within_bounds();
        raft::copy(fp.last_projection.data(),
                   solution.assignment.data(),
                   solution.assignment.size(),
                   stream);
        raft::copy(
          lp_solution.data(), solution.assignment.data(), solution.assignment.size(), stream);
      }
      fp.diagnostics.cutoff = fp_cutoff;
      CUOPT_LOG_INFO(
        "GpuSubmipFpScreen job=%llu verdict=%d dual_bound=%.9g cutoff=%.9g linear=%.9g "
        "iterations=%d seconds=%.6f converged=%d unbounded=%d left_unbounded=%d target=%.9g "
        "lp_start=%d primal_residual=%.9g dual_residual=%.9g relative_gap=%.9g restarts=%d",
        (unsigned long long)job_id,
        screen.reach_verdict,
        screen.dual_bound,
        double(local_cutoff),
        screen.linear_objective,
        screen.iterations,
        screen.elapsed_time,
        int(screen.converged),
        int(std::count_if(bounds.begin(), bounds.end(), unbounded)),
        int(std::count_if(box.begin(), box.end(), unbounded)),
        double(fp_cutoff),
        int(reached && lp_start),
        screen.primal_residual,
        screen.dual_residual,
        screen.relative_gap,
        screen.restarts);
      if (screen.reach_verdict > 0) {
        run_info.exit_reason = "screened";
        return false;
      }
    }
    const auto descent_begin      = fp.timer.elapsed_time();
    const auto projection_begin   = fp.total_lp_time;
    const auto projections_before = fp.n_lp_calls;
    const auto verification_begin = fp.total_verification_time;
    const auto verifies_before    = fp.n_verification_calls;
    run_info.started              = true;
    const bool found              = fp.run_single_fp_descent(solution, fp_cutoff);
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
      "end_max_bound_violation=%.9g end_cutoff_excess=%.9g fp_found=%d "
      "initial_rounding_cutoff_rejected=%d",
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
      int(found),
      int(fp.diagnostics.initial_rounding_cutoff_rejected));
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

  // A box that holds every point of the restricted LP: each infinite bound is replaced by the one
  // its rows imply from the other columns' bounds, widened by the row tolerance and by the
  // rounding of the activity sums.
  std::vector<typename type_2<f_t>::type> implied_box(
    const std::vector<typename type_2<f_t>::type>& bounds) const
  {
    const auto& lp = columns.reduced_lp;
    const f_t inf  = std::numeric_limits<f_t>::infinity();
    std::vector<f_t> lowest(lp.num_rows, 0), highest(lp.num_rows, 0), magnitude(lp.num_rows, 0);
    std::vector<i_t> lowest_open(lp.num_rows, 0), highest_open(lp.num_rows, 0);
    // Least and largest value of a * x_j over the column's bounds.
    const auto ends = [&](i_t j, f_t a) {
      const f_t l = get_lower(bounds[j]), u = get_upper(bounds[j]);
      return a > 0 ? std::pair{a * l, a * u} : std::pair{a * u, a * l};
    };
    for (i_t j = 0; j < lp.num_cols; ++j) {
      for (i_t p = lp.A.col_start[j]; p < lp.A.col_start[j + 1]; ++p) {
        if (lp.A.x[p] == 0) { continue; }
        const i_t i         = lp.A.i[p];
        const auto [lo, hi] = ends(j, lp.A.x[p]);
        if (std::isfinite(lo)) {
          lowest[i] += lo;
          magnitude[i] += std::abs(lo);
        } else {
          ++lowest_open[i];
        }
        if (std::isfinite(hi)) {
          highest[i] += hi;
          magnitude[i] += std::abs(hi);
        } else {
          ++highest_open[i];
        }
      }
    }
    auto box = bounds;
    for (i_t j = 0; j < lp.num_cols; ++j) {
      f_t lower = get_lower(bounds[j]), upper = get_upper(bounds[j]);
      if (std::isfinite(lower) && std::isfinite(upper)) { continue; }
      f_t implied_lower = -inf, implied_upper = inf;
      for (i_t p = lp.A.col_start[j]; p < lp.A.col_start[j + 1]; ++p) {
        const f_t a = lp.A.x[p];
        if (a == 0) { continue; }
        const i_t i         = lp.A.i[p];
        const auto [lo, hi] = ends(j, a);
        const i_t lo_open = !std::isfinite(lo), hi_open = !std::isfinite(hi);
        const f_t slack = problem.tolerances.absolute_tolerance +
                          problem.tolerances.relative_tolerance * std::abs(lp.rhs[i]) +
                          1e-9 * (magnitude[i] + std::abs(lp.rhs[i]));
        // a * x_j is the right-hand side minus the other columns' activity.
        f_t least = -inf, most = inf;
        if (highest_open[i] == hi_open) {
          least = lp.rhs[i] - (highest[i] - (hi_open ? 0 : hi)) - slack;
        }
        if (lowest_open[i] == lo_open) {
          most = lp.rhs[i] - (lowest[i] - (lo_open ? 0 : lo)) + slack;
        }
        implied_lower = std::max(implied_lower, a > 0 ? least / a : most / a);
        implied_upper = std::min(implied_upper, a > 0 ? most / a : least / a);
      }
      if (!std::isfinite(lower)) { lower = implied_lower; }
      if (!std::isfinite(upper)) { upper = implied_upper; }
      box[j] = {lower, upper};
    }
    return box;
  }

  // Runtime test of a neighbourhood the screen dropped: its objective LP, solved on to a tight
  // tolerance without the early stop, must end above the cutoff. Runs with the cutoff row relaxed
  // and on the node's own bounds, so it also tests the implied box.
  void check_dropped(f_t local_cutoff)
  {
    const f_t no_level = std::numeric_limits<f_t>::infinity();
    const auto solved  = fp.pdfp_reach(solution, no_level, 10., 1e-7);
    CUOPT_LOG_INFO("GpuSubmipFpScreenCheck converged=%d linear=%.12g cutoff=%.12g seconds=%.6f",
                   int(solved.converged),
                   solved.linear_objective,
                   double(local_cutoff),
                   solved.elapsed_time);
    cuopt_assert(!solved.converged || solved.linear_objective >
                                        local_cutoff - 1e-5 * std::max(1., std::abs(local_cutoff)),
                 "The screen dropped a neighbourhood whose LP reaches the cutoff");
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
  // Seconds for the objective-only screen before a PDFP descent; zero skips it.
  bool cutoff_row{true};
  double screen_seconds{0.};
  bool lp_start{false};
  double target_share{0.};
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
  const gpu_submip_fp_config_t& config,
  bool cutoff_row)
  : impl_(std::make_unique<impl_t>(
      handle, lp, variable_types, settings, compact_fixed, fused_adaptive_step, config, cutoff_row))
{
}

template <typename i_t, typename f_t>
bool gpu_submip_fp_t<i_t, f_t>::has_cutoff_row() const
{
  return impl_->cutoff_row;
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
