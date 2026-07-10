/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "vanilla_fp.cuh"

#include <cuopt/error.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/solve.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>

#include <mip_heuristics/diversity/diversity_manager.cuh>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <mip_heuristics/solver.cuh>
#include <pdlp/utilities/problem_checking.cuh>
#include <utilities/seed_generator.cuh>

#include <argparse/argparse.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization {
namespace {

using metrics_t = mip::fp_run_metrics_t<int, double>;

enum class fp_variant_t { single, batched, vanilla };

const char* variant_name(fp_variant_t variant)
{
  if (variant == fp_variant_t::single) { return "single"; }
  if (variant == fp_variant_t::batched) { return "batched"; }
  return "vanilla";
}

struct run_result_t {
  int run;
  int seed;
  std::string path;
  int diversity_climber;
  int diversity_seed;
  std::string rounding_mode;
  std::string projection_backend;
  int n_integer_vars;
  bool feasible;
  metrics_t metrics;
  double total_run_time;
};

double quantile(std::vector<double> values, double q)
{
  if (values.empty()) { return 0.; }
  std::sort(values.begin(), values.end());
  double position = q * (values.size() - 1);
  size_t lower    = (size_t)position;
  size_t upper    = std::min(lower + 1, values.size() - 1);
  double fraction = position - lower;
  return values[lower] + fraction * (values[upper] - values[lower]);
}

void print_distribution(const std::string& name, const std::vector<double>& values)
{
  std::cerr << "  " << name << ": median=" << quantile(values, 0.5)
            << " p10=" << quantile(values, 0.1) << " p90=" << quantile(values, 0.9) << '\n';
}

run_result_t run_fp(const std::string& mps_path,
                    int run,
                    int seed,
                    double time_limit,
                    fp_variant_t variant,
                    int batch_size,
                    int diversity_climber,
                    int diversity_seed)
{
  using clock_t = std::chrono::steady_clock;

  seed_generator::set_seed(seed);
  const raft::handle_t handle;
  auto mps_problem = io::read_mps<int, double>(mps_path, false);
  handle.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle, mps_problem);
  problem_checking_t<int, double>::check_problem_representation(op_problem);
  mip::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();
  cuopt_assert(problem.n_integer_vars > 0, "FP regression requires integer variables");

  mip_solver_settings_t<int, double> settings;
  settings.time_limit     = time_limit;
  settings.log_to_console = false;
  settings.presolver      = presolver_t::None;
  settings.seed           = seed;
  problem.tolerances      = settings.get_tolerances();
  timer_t solver_timer(time_limit);
  mip::mip_solver_t<int, double> solver(problem, settings, solver_timer);
  mip::diversity_manager_t<int, double> dm(solver.context);
  solver.context.diversity_manager_ptr = &dm;

  dm.ls.resize_vectors(problem, problem.handle_ptr);
  dm.ls.constraint_prop.bounds_update.resize(problem);
  problem.compute_integer_fixed_problem();
  cuopt_func_call(
    dm.ls.constraint_prop.bounds_update.calculate_activity_on_problem_bounds(problem));
  cuopt_assert(
    dm.ls.constraint_prop.bounds_update.calculate_infeasible_redundant_constraints(problem),
    "The FP regression problem must not be integer infeasible");

  mip::solution_t<int, double> solution(problem);
  mip::relaxed_lp_settings_t lp_settings;
  lp_settings.time_limit         = time_limit;
  lp_settings.tolerance          = problem.tolerances.absolute_tolerance;
  lp_settings.has_initial_primal = false;
  auto lp_result                 = mip::get_relaxed_lp_solution(problem, solution, lp_settings);
  cuopt_assert(lp_result.get_primal_solution().size() == (size_t)problem.n_variables,
               "Initial relaxed LP did not return a complete primal solution");
  raft::copy(dm.lp_optimal_solution.data(),
             solution.assignment.data(),
             solution.assignment.size(),
             problem.handle_ptr->get_stream());
  dm.ls.lp_optimal_exists = true;
  dm.ls.fj.reset_weights(problem.handle_ptr->get_stream());

  metrics_t metrics;
  dm.ls.fp.metrics = &metrics;
  dm.ls.fp.timer   = timer_t(time_limit);
  dm.ls.fp.cycle_queue.reset(solution);
  dm.ls.fp.reset();
  dm.ls.fp.resize_vectors(problem, problem.handle_ptr);
  if (variant == fp_variant_t::batched && batch_size > 0) {
    dm.ls.fp.batch_config.target_min_batch_size  = batch_size;
    dm.ls.fp.batch_config.target_max_batch_size  = batch_size;
    dm.ls.fp.batch_config.latency_max_batch_size = batch_size;
    dm.ls.fp.batch_config.fallback_threshold     = 1;
  }
  auto wall_begin = clock_t::now();
  bool feasible   = false;
  if (variant == fp_variant_t::single) {
    feasible = dm.ls.fp.run_single_fp_descent(solution);
  } else if (variant == fp_variant_t::batched) {
    feasible = dm.ls.fp.run_batched_fp_cloud(solution);
  } else {
    feasible = mip::run_vanilla_fp_descent(
      dm.ls.fp, solution, mip::vanilla_fp_config_t{diversity_climber, diversity_seed});
  }
  problem.handle_ptr->sync_stream();

  double total_run_time = std::chrono::duration<double>(clock_t::now() - wall_begin).count();
  metrics.total_time    = total_run_time;
  bool vanilla          = variant == fp_variant_t::vanilla;
  bool batched          = variant == fp_variant_t::batched;
  return {run,
          seed,
          variant_name(variant),
          vanilla ? diversity_climber : -1,
          vanilla ? diversity_seed : -1,
          vanilla ? "nearest" : "cp",
          batched ? "unified_batch" : "dynamic",
          problem.n_integer_vars,
          feasible,
          std::move(metrics),
          total_run_time};
}

void write_csv(std::ostream& out, const std::vector<run_result_t>& results)
{
  out << "run,seed,path,diversity_climber,diversity_seed,rounding_mode,projection_backend,"
         "iteration,batch_size,projected_integers,projected_ratio,"
         "projection_violated_constraints,projection_l1_distance,projection_total_violation,"
         "projection_time,rounded_integers,rounding_violated_constraints,rounding_l1_movement,"
         "rounding_total_violation,rounding_time,projection_feasible,rounding_feasible,cycle,"
         "cycle_kind,perturbed,timed_out,feasible,feasible_events,total_run_time\n";
  out << std::setprecision(17);
  for (const auto& result : results) {
    for (const auto& metric : result.metrics.iterations) {
      double projected_ratio = (double)metric.projected_integers / result.n_integer_vars;
      out << result.run << ',' << result.seed << ',' << result.path << ','
          << result.diversity_climber << ',' << result.diversity_seed << ',' << result.rounding_mode
          << ',' << result.projection_backend << ',' << metric.iteration << ',' << metric.batch_size
          << ',' << metric.projected_integers << ',' << projected_ratio << ','
          << metric.projection_violated_constraints << ',' << metric.projection_l1_distance << ','
          << metric.projection_total_violation << ',' << metric.projection_time << ','
          << metric.rounded_integers << ',' << metric.rounding_violated_constraints << ','
          << metric.rounding_l1_movement << ',' << metric.rounding_total_violation << ','
          << metric.rounding_time << ',' << metric.projection_feasible << ','
          << metric.rounding_feasible << ',' << metric.cycle << ',' << metric.cycle_kind << ','
          << metric.perturbed << ',' << metric.timed_out << ',' << metric.feasible << ','
          << result.metrics.feasible_events << ',' << result.total_run_time << '\n';
    }
  }
}

void print_summary(const std::vector<run_result_t>& results, const std::string& path)
{
  std::vector<double> projected_integers;
  std::vector<double> projected_ratio;
  std::vector<double> projection_l1;
  std::vector<double> projection_violated;
  std::vector<double> projection_total_violation;
  std::vector<double> rounded_integers;
  std::vector<double> rounding_l1;
  std::vector<double> rounding_violated;
  std::vector<double> rounding_total_violation;
  std::vector<double> projection_time;
  std::vector<double> normalized_projection_time;
  std::vector<double> rounding_time;
  std::vector<double> total_time;
  int feasible_runs   = 0;
  int feasible_events = 0;
  int distance_cycles = 0;
  int integer_cycles  = 0;
  int perturbations   = 0;
  size_t count        = 0;

  for (const auto& result : results) {
    if (result.path != path) { continue; }
    feasible_runs += result.feasible;
    feasible_events += result.metrics.feasible_events;
    total_time.push_back(result.total_run_time);
    for (const auto& metric : result.metrics.iterations) {
      ++count;
      projected_integers.push_back(metric.projected_integers);
      projected_ratio.push_back((double)metric.projected_integers / result.n_integer_vars);
      projection_l1.push_back(metric.projection_l1_distance);
      projection_violated.push_back(metric.projection_violated_constraints);
      projection_total_violation.push_back(metric.projection_total_violation);
      rounded_integers.push_back(metric.rounded_integers);
      rounding_l1.push_back(metric.rounding_l1_movement);
      rounding_violated.push_back(metric.rounding_violated_constraints);
      rounding_total_violation.push_back(metric.rounding_total_violation);
      projection_time.push_back(metric.projection_time);
      normalized_projection_time.push_back(metric.projection_time / metric.batch_size);
      rounding_time.push_back(metric.rounding_time);
      distance_cycles += metric.cycle_kind == 1;
      integer_cycles += metric.cycle_kind == 2;
      perturbations += metric.perturbed;
    }
  }

  std::cerr << path << ": count=" << count << " feasible_runs=" << feasible_runs << '/'
            << total_time.size() << " feasible_events=" << feasible_events
            << " distance_cycles=" << distance_cycles << " integer_cycles=" << integer_cycles
            << " perturbations=" << perturbations << '\n';
  print_distribution("projected_integers", projected_integers);
  print_distribution("projected_ratio", projected_ratio);
  print_distribution("projection_l1", projection_l1);
  print_distribution("projection_violated_constraints", projection_violated);
  print_distribution("projection_total_violation", projection_total_violation);
  print_distribution("rounded_integers", rounded_integers);
  print_distribution("rounding_l1", rounding_l1);
  print_distribution("rounding_violated_constraints", rounding_violated);
  print_distribution("rounding_total_violation", rounding_total_violation);
  print_distribution("projection_time", projection_time);
  if (path == "batched") {
    print_distribution("projection_time_per_climber", normalized_projection_time);
  }
  print_distribution("rounding_time", rounding_time);
  print_distribution("total_time", total_time);
}

std::vector<fp_variant_t> parse_variants(const std::string& value)
{
  std::vector<fp_variant_t> variants;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item == "single") {
      variants.push_back(fp_variant_t::single);
    } else if (item == "batched") {
      variants.push_back(fp_variant_t::batched);
    } else if (item == "vanilla") {
      variants.push_back(fp_variant_t::vanilla);
    } else {
      throw std::runtime_error("Unknown FP variant: " + item);
    }
  }
  if (variants.empty()) { throw std::runtime_error("--variants must not be empty"); }
  return variants;
}

}  // namespace
}  // namespace cuopt::mathematical_optimization

int main(int argc, char** argv)
{
  argparse::ArgumentParser program("fp_regression");
  program.add_argument("mps_path").help("MPS input path");
  program.add_argument("--runs").scan<'i', int>().default_value(10);
  program.add_argument("--seed").scan<'i', int>().default_value(42);
  program.add_argument("--time-limit").scan<'g', double>().default_value(30.);
  program.add_argument("--batch-size").scan<'i', int>().default_value(0);
  program.add_argument("--variants").default_value(std::string("single,batched"));
  program.add_argument("--diversity-climber").scan<'i', int>().default_value(1);
  program.add_argument("--diversity-seed").scan<'i', int>().default_value(-1);
  program.add_argument("--summary-only").default_value(false).implicit_value(true);
  program.add_argument("--output");

  try {
    program.parse_args(argc, argv);
    const auto mps_path  = program.get<std::string>("mps_path");
    const int runs       = program.get<int>("--runs");
    const int base_seed  = program.get<int>("--seed");
    const double limit   = program.get<double>("--time-limit");
    const int batch_size = program.get<int>("--batch-size");
    const auto variants =
      cuopt::mathematical_optimization::parse_variants(program.get<std::string>("--variants"));
    const int diversity_climber = program.get<int>("--diversity-climber");
    const int diversity_seed    = program.get<int>("--diversity-seed");
    const bool summary_only     = program.get<bool>("--summary-only");
    const auto output           = program.present<std::string>("--output");
    if (runs <= 0) { throw std::runtime_error("--runs must be positive"); }
    if (limit <= 0.) { throw std::runtime_error("--time-limit must be positive"); }
    if (batch_size < 0) { throw std::runtime_error("--batch-size must not be negative"); }
    if (diversity_climber < 1) { throw std::runtime_error("--diversity-climber must be positive"); }
    std::vector<cuopt::mathematical_optimization::run_result_t> results;
    results.reserve(variants.size() * runs);
    for (int run = 0; run < runs; ++run) {
      int seed         = base_seed + run;
      int vanilla_seed = diversity_seed >= 0 ? diversity_seed : seed;
      for (auto variant : variants) {
        results.push_back(cuopt::mathematical_optimization::run_fp(
          mps_path, run, seed, limit, variant, batch_size, diversity_climber, vanilla_seed));
      }
    }

    if (!summary_only) {
      if (output.has_value()) {
        std::ofstream file(*output);
        if (!file) { throw std::runtime_error("Failed to open output file: " + *output); }
        cuopt::mathematical_optimization::write_csv(file, results);
      } else {
        cuopt::mathematical_optimization::write_csv(std::cout, results);
      }
    }
    for (auto variant : variants) {
      cuopt::mathematical_optimization::print_summary(
        results, cuopt::mathematical_optimization::variant_name(variant));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fp_regression: " << error.what() << '\n';
    return 1;
  }
}
