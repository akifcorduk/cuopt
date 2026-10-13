/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/gpu_submip_worker.hpp>
#include <branch_and_bound/post_root_snapshot.hpp>
#include <math_optimization/tic_toc.hpp>
#include <mip_heuristics/gpu_submip_full_device.hpp>
#include <mip_heuristics/utils.hpp>
#include <utilities/logger.hpp>
#include <utilities/mip_green_context.cuh>
#include <utilities/scope_guard.hpp>

#include <argparse/argparse.hpp>
#include <raft/core/handle.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using i_t         = int;
using f_t         = double;
namespace opt     = cuopt::mathematical_optimization;
namespace mip     = opt::mip;
namespace simplex = opt::simplex;
using clock_type  = std::chrono::steady_clock;

// Recompute residuals directly on the frozen augmented parent LP; a worker's declaration of
// feasibility and its reported objective are not used to accept an improvement.
class replay_audit_t {
 public:
  replay_audit_t(const simplex::lp_problem_t<i_t, f_t>& lp,
                 const std::vector<simplex::variable_type_t>& types,
                 const opt::mip_solver_settings_t<i_t, f_t>& settings,
                 f_t initial_cutoff)
    : lp_(lp),
      types_(types),
      tolerances_(settings.tolerances),
      rows_(0, 0, 0),
      best_(initial_cutoff)
  {
    lp.A.to_compressed_row(rows_);
  }

  void start() { start_ = clock_type::now(); }

  void publish(const std::vector<f_t>& candidate, f_t reported_objective)
  {
    audit(candidate, reported_objective, false);
  }

  bool validate_initial(const std::vector<f_t>& candidate, f_t cutoff)
  {
    return audit(candidate, cutoff, true);
  }

  f_t best_user_objective()
  {
    std::lock_guard<std::mutex> guard(mutex_);
    return lp_.obj_scale * (best_ + lp_.obj_constant);
  }

  int finish()
  {
    std::cout << "REPLAY_RESULT seconds=" << elapsed() << " improvements=" << improvements_
              << " first_improvement_seconds=" << first_improvement_ << " invalid=" << invalid_
              << " best_objective=" << best_
              << " best_user_objective=" << lp_.obj_scale * (best_ + lp_.obj_constant) << '\n';
    return invalid_ == 0 ? 0 : 3;
  }

 private:
  bool audit(const std::vector<f_t>& candidate, f_t reported_objective, bool initial)
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (candidate.size() != static_cast<size_t>(lp_.num_cols) ||
        !std::all_of(candidate.begin(), candidate.end(), [](f_t x) { return std::isfinite(x); })) {
      if (!initial) { ++invalid_; }
      std::cout << (initial ? "REPLAY_INITIAL valid=0" : "REPLAY_REJECT")
                << " reason=invalid_assignment\n";
      return false;
    }
    f_t max_row_error = 0, max_row_ratio = 0, max_bound_error = 0, max_integer_error = 0;
    for (i_t row = 0; row < lp_.num_rows; ++row) {
      const f_t activity = mip::compensated_dot2_csr(
        rows_.row_start.data(), rows_.j.data(), rows_.x.data(), candidate.data(), row);
      const f_t error = std::abs(activity - lp_.rhs[row]);
      const f_t tolerance =
        tolerances_.absolute_tolerance +
        tolerances_.relative_tolerance * std::max(std::abs(activity), std::abs(lp_.rhs[row]));
      if (!std::isfinite(activity)) {
        if (!initial) { ++invalid_; }
        std::cout << (initial ? "REPLAY_INITIAL valid=0" : "REPLAY_REJECT")
                  << " reason=nonfinite_activity\n";
        return false;
      }
      max_row_error = std::max(max_row_error, error);
      max_row_ratio = std::max(max_row_ratio, error / tolerance);
    }
    for (i_t col = 0; col < lp_.num_cols; ++col) {
      max_bound_error = std::max(
        {max_bound_error, lp_.lower[col] - candidate[col], candidate[col] - lp_.upper[col]});
      if (types_[col] != simplex::variable_type_t::CONTINUOUS) {
        max_integer_error =
          std::max(max_integer_error, std::abs(candidate[col] - std::round(candidate[col])));
      }
    }
    const f_t objective =
      mip::compensated_dot2(lp_.objective.data(), candidate.data(), candidate.size());
    const f_t objective_error = std::abs(objective - reported_objective);
    const f_t objective_tolerance =
      tolerances_.absolute_tolerance +
      tolerances_.relative_tolerance *
        std::max({f_t{1}, std::abs(objective), std::abs(reported_objective)});
    const bool objective_consistent =
      std::isfinite(reported_objective) && objective_error <= objective_tolerance;
    const bool valid = max_row_ratio <= 1 && max_bound_error <= tolerances_.absolute_tolerance &&
                       max_integer_error <= tolerances_.integrality_tolerance &&
                       std::isfinite(objective) && (!initial || objective_consistent);
    if (!valid && !initial) { ++invalid_; }
    if (valid && initial) { best_ = objective; }
    // Repeated CPU/GPU assignments can differ only in floating-point summation noise.
    // Count an improvement only beyond the same tolerance used to audit its objective.
    const f_t improvement_tolerance =
      std::max(f_t(tolerances_.absolute_mip_gap), objective_tolerance);
    const bool improved = !initial && valid && objective < best_ - improvement_tolerance;
    if (improved) {
      best_ = objective;
      ++improvements_;
      if (first_improvement_ < 0) { first_improvement_ = elapsed(); }
    }
    std::cout << (initial ? "REPLAY_INITIAL" : "REPLAY_CANDIDATE")
              << " seconds=" << (initial ? 0.0 : elapsed()) << " valid=" << valid
              << " improved=" << improved << " objective=" << objective
              << " reported_objective=" << reported_objective
              << " objective_consistent=" << objective_consistent
              << " objective_error=" << objective_error
              << " objective_tolerance=" << objective_tolerance
              << " improvement_tolerance=" << improvement_tolerance
              << " user_objective=" << lp_.obj_scale * (objective + lp_.obj_constant)
              << " max_row_error=" << max_row_error << " max_row_ratio=" << max_row_ratio
              << " max_bound_error=" << max_bound_error
              << " max_integer_error=" << max_integer_error << '\n';
    return valid;
  }

  double elapsed() const
  {
    return std::chrono::duration<double>(clock_type::now() - start_).count();
  }
  const simplex::lp_problem_t<i_t, f_t>& lp_;
  const std::vector<simplex::variable_type_t>& types_;
  opt::mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances_;
  opt::csr_matrix_t<i_t, f_t> rows_;
  clock_type::time_point start_;
  f_t best_;
  double first_improvement_{-1};
  int improvements_{0}, invalid_{0};
  std::mutex mutex_;
};

int run_worker_replay(const argparse::ArgumentParser& program,
                      const mip::post_root_snapshot_t<i_t, f_t>& snapshot,
                      const mip::gpu_submip_options_t& options,
                      const raft::handle_t* worker_handle,
                      clock_type::time_point replay_start)
{
  simplex::lp_problem_t<i_t, f_t> base_lp(worker_handle, 0, 0, 0);
  snapshot.restore_lp(base_lp);
  std::vector<simplex::variable_type_t> types;
  for (auto type : snapshot.var_types) {
    types.push_back(static_cast<simplex::variable_type_t>(type));
  }
  opt::mip_solver_settings_t<i_t, f_t> mip_settings;
  mip_settings.seed       = program.get<int>("--seed");
  mip_settings.time_limit = options.time_limit;
  std::atomic<int> halt{0};
  std::cout << std::setprecision(17);
  // A node may have a global objective cutoff but no feasible incumbent within its
  // bounds. The captured incumbent is an optional warm-start seed, not a proof.
  std::cout << "REPLAY_INITIAL cutoff=" << snapshot.upper_bound
            << " incumbent_seed=" << !snapshot.incumbent.empty() << '\n';
  auto job = mip::make_gpu_submip_job<i_t, f_t>(base_lp,
                                                types,
                                                snapshot.root_primal,
                                                snapshot.incumbent,
                                                snapshot.upper_bound,
                                                options,
                                                mip_settings,
                                                program.get<unsigned>("--job-id"),
                                                options.fix_rate > 0);
  // Audit against the exact executed job, including optional additional RINS fixings.
  replay_audit_t node_audit(job.lp, job.var_types, mip_settings, job.cutoff);
  std::cout << std::setprecision(17) << "REPLAY_START mode=direct_fp_job partition="
            << (program.get<bool>("--full-gpu") ? "full" : "40_percent_requested")
            << " regular_partition=idle rows=" << snapshot.num_rows << " cols=" << snapshot.num_cols
            << " nnz=" << snapshot.coefficients.size() << " fix_rate=" << options.fix_rate
            << " time_limit=" << options.time_limit << " cached_primal=" << options.cached_primal
            << " fp=" << options.enable_fp << " fused_pdlp=" << options.fused_pdlp
            << " compact_fixed=" << options.compact_fixed
            << " incumbent_blend=" << options.incumbent_blend
            << " fp_config=" << options.fp_config.name
            << " continuous=" << program.get<bool>("--continuous")
            << " initial_cutoff=" << snapshot.upper_bound << " setup_seconds="
            << std::chrono::duration<double>(clock_type::now() - replay_start).count()
            << " budget_domain="
            << (program.get<bool>("--continuous") ? "paired_session_including_construction"
                                                  : "single_descent_including_session_construction")
            << " cpu_node_solves=0\n";
  node_audit.start();
  const auto publish = [&node_audit](const std::vector<f_t>& candidate, f_t objective) {
    node_audit.publish(candidate, objective);
  };
  if (program.get<bool>("--continuous")) {
    job.deadline = opt::tic() + options.time_limit;
    std::atomic<int> pair_halt{0};
    mip::run_gpu_submip_job_full_device_continuous<i_t, f_t>(
      job,
      mip_settings,
      &halt,
      &pair_halt,
      options,
      [&node_audit] { return node_audit.best_user_objective(); },
      publish,
      mip::gpu_submip_device_id(worker_handle));
  } else {
    mip::run_gpu_submip_job<i_t, f_t>(
      job, mip_settings, worker_handle, &halt, options, {}, publish);
  }
  return node_audit.finish();
}

int run_replay(const argparse::ArgumentParser& program)
{
  const auto replay_start = clock_type::now();
  const auto snapshot =
    mip::read_post_root_snapshot<i_t, f_t>(program.get<std::string>("snapshot"));
  mip::gpu_submip_options_t options;
  options.fix_rate        = program.get<double>("--fix-rate");
  options.time_limit      = program.get<double>("--time-limit");
  options.cached_primal   = program.get<bool>("--cached-primal");
  options.enable_fp       = !program.get<bool>("--disable-fp");
  options.compact_fixed   = !program.get<bool>("--no-compact-fixed");
  options.incumbent_blend = program.get<double>("--incumbent-blend");
  options.fused_pdlp      = !program.get<bool>("--no-fused-pdlp");
  options.fp_config =
    program.is_used("--fp-config")
      ? mip::gpu_submip_fp_config_from_name(program.get<std::string>("--fp-config"))
      : mip::gpu_submip_fp_config_from_environment();
  if (program.get<bool>("--continuous") && !program.get<bool>("--full-gpu")) {
    throw std::runtime_error("Continuous paired-session replay requires --full-gpu");
  }
  if (!std::isfinite(options.fix_rate) || options.fix_rate < 0 || options.fix_rate >= 1 ||
      !std::isfinite(options.time_limit) || options.time_limit <= 0 ||
      !std::isfinite(options.incumbent_blend) || options.incumbent_blend < 0 ||
      options.incumbent_blend > 1) {
    throw std::runtime_error("Invalid replay fixing fraction, time limit, or incumbent blend");
  }
  cuopt::init_logger_t log_guard("", true);
  auto previous_resource =
    rmm::mr::set_current_device_resource(rmm::mr::cuda_async_memory_resource{});
  cuopt::scope_guard restore_resource([&previous_resource]() noexcept {
    try {
      rmm::mr::set_current_device_resource(std::move(previous_resource));
    } catch (const std::exception& error) {
      std::cerr << "REPLAY_ERROR resource_cleanup=" << error.what() << '\n';
    } catch (...) {
      std::cerr << "REPLAY_ERROR resource_cleanup=unknown_exception\n";
    }
  });
  raft::handle_t regular_handle;
  const auto device_sms = regular_handle.get_device_properties().multiProcessorCount;
  cuopt::mip_green_context_t partition(&regular_handle);
  cuopt::scope_guard release_context([&partition]() noexcept {
    try {
      partition.release_submip();
    } catch (const std::exception& error) {
      std::cerr << "REPLAY_ERROR context_cleanup=" << error.what() << '\n';
    } catch (...) {
      std::cerr << "REPLAY_ERROR context_cleanup=unknown_exception\n";
    }
  });
  const bool full_gpu                 = program.get<bool>("--full-gpu");
  const raft::handle_t* worker_handle = &regular_handle;
  if (!full_gpu) {
    partition.request();
    if (!partition.activate_regular()) {
      throw std::runtime_error("Cannot activate 40% GPU partition for replay");
    }
    partition.publish_ready();
    worker_handle = partition.submip_handle();
    if (!worker_handle) {
      throw std::runtime_error("GPU partition did not publish its worker handle");
    }
  }
  const auto worker_sms = worker_handle->get_device_properties().multiProcessorCount;
  const auto regular_sms =
    full_gpu ? 0 : regular_handle.get_device_properties().multiProcessorCount;
  if (worker_sms <= 0 || worker_sms + regular_sms != device_sms ||
      (!full_gpu && regular_sms <= 0)) {
    throw std::runtime_error("Replay handles report inconsistent green-context SM counts");
  }
  std::cout << std::setprecision(17) << "REPLAY_RESOURCES device_sms=" << device_sms
            << " worker_sms=" << worker_sms << " regular_sms=" << regular_sms
            << " worker_share=" << static_cast<double>(worker_sms) / device_sms
            << " allocator=cuda_async\n";
  const int result = run_worker_replay(program, snapshot, options, worker_handle, replay_start);
  // The worker and all its device allocations are gone before either green context is released.
  const auto cleanup_start = clock_type::now();
  partition.release_submip();
  partition.restore_regular();
  std::cout << "REPLAY_CLEANUP seconds="
            << std::chrono::duration<double>(clock_type::now() - cleanup_start).count()
            << " total_seconds="
            << std::chrono::duration<double>(clock_type::now() - replay_start).count() << '\n';
  return result;
}

}  // namespace

int main(int argc, char** argv)
{
  argparse::ArgumentParser program("solve_GPU_SUBMIP_REPLAY");
  program.add_description("Run GPU FP on a frozen node job, without CPU LP or tree solves.");
  program.add_argument("snapshot").help("binary node-job or post-root snapshot");
  program.add_argument("--fix-rate").scan<'g', double>().default_value(0.0);
  program.add_argument("--time-limit").scan<'g', double>().default_value(60.0);
  program.add_argument("--seed").scan<'i', int>().default_value(0);
  program.add_argument("--job-id").scan<'u', unsigned>().default_value(0u);
  program.add_argument("--cached-primal").flag();
  program.add_argument("--disable-fp").flag();
  program.add_argument("--fp-config")
    .help("override CUOPT_CONFIG_ID: baseline, round_long, pdlp_long, distance")
    .default_value(std::string{"baseline"});
  program.add_argument("--continuous")
    .help(
      "restart descents until --time-limit, simulating a paired CPU lifetime; requires --full-gpu")
    .flag();
  program.add_argument("--no-compact-fixed").flag();
  program.add_argument("--incumbent-blend").scan<'g', double>().default_value(0.0);
  program.add_argument("--no-fused-pdlp")
    .help("use the original separate tentative-update and step-decision CUDA graphs")
    .flag();
  program.add_argument("--full-gpu")
    .help("exploratory ablation without the 40% SM partition")
    .flag();
  try {
    program.parse_args(argc, argv);
    return run_replay(program);
  } catch (const std::exception& error) {
    std::cerr << "REPLAY_ERROR " << error.what() << '\n';
    return 2;
  }
}
