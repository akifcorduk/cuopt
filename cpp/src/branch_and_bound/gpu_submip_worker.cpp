/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <branch_and_bound/gpu_submip_worker.hpp>
#include <branch_and_bound/post_root_snapshot.hpp>
#include <dual_simplex/solve.hpp>
#include <math_optimization/tic_toc.hpp>
#include <mip_heuristics/gpu_submip_fp.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/utils.hpp>
#include <utilities/logger.hpp>
#include <utilities/pcgenerator.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>

namespace cuopt::mathematical_optimization::mip {
namespace {
constexpr std::array<std::string_view, 4> fp_config_names{
  "baseline", "round_long", "pdlp_long", "distance"};

int experiment_integer(const char* name, int fallback, int lower)
{
  const char* text = std::getenv(name);
  if (text == nullptr) { return fallback; }
  const std::string_view input{text};
  int value               = 0;
  const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
  if (error != std::errc{} || end != input.data() + input.size() || value < lower) {
    throw std::invalid_argument(std::string("Invalid GPU FP benchmark option: ") + name);
  }
  return value;
}

double experiment_option(const char* name, double fallback, double lower, double upper)
{
  const char* text = std::getenv(name);
  if (text == nullptr) { return fallback; }
  char* end          = nullptr;
  const double value = std::strtod(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(value) || value < lower || value > upper) {
    throw std::invalid_argument(std::string("Invalid GPU sub-MIP option: ") + name);
  }
  return value;
}

}  // namespace

const char* gpu_submip_source_name(gpu_submip_source_t source)
{
  switch (source) {
    case gpu_submip_source_t::MAIN_TREE: return "main_tree";
    case gpu_submip_source_t::CPU_SUBMIP: return "cpu_submip";
    case gpu_submip_source_t::DIVE: return "dive";
  }
  return "unknown";
}

std::string gpu_submip_job_snapshot_path(gpu_submip_source_t source,
                                         const char* path,
                                         const char* requested_source,
                                         const char* directory)
{
  // Directory mode records each source once, independently of the legacy single-source filter.
  if (directory && *directory) {
    return (std::filesystem::path(directory) /
            (std::string(gpu_submip_source_name(source)) + ".bin"))
      .string();
  }
  if (!path || !*path) { return {}; }
  const std::string filter = requested_source ? requested_source : "";
  if (!filter.empty() && filter != "main" && filter != "submip" && filter != "dive") {
    throw std::invalid_argument("GPU job snapshot source must be main, submip, or dive");
  }
  const bool matches = filter.empty() ||
                       (filter == "main" && source == gpu_submip_source_t::MAIN_TREE) ||
                       (filter == "submip" && source == gpu_submip_source_t::CPU_SUBMIP) ||
                       (filter == "dive" && source == gpu_submip_source_t::DIVE);
  return matches ? path : "";
}

gpu_submip_fp_config_t gpu_submip_fp_config_from_name(std::string_view name)
{
  gpu_submip_fp_config_t config;
  if (name == "baseline") { return config; }
  config.name = name;
  if (name == "round_long") {
    config.rounding_seconds    = 2.;
    config.propagation_seconds = 2.;
  } else if (name == "pdlp_long") {
    config.pdlp_seconds = 3.;
  } else if (name == "distance") {
    config.initial_alpha = 0.;
  } else {
    throw std::invalid_argument(
      "GPU FP config must be baseline, round_long, pdlp_long, or distance");
  }
  return config;
}

gpu_submip_fp_config_t gpu_submip_fp_config_from_environment()
{
  if (std::getenv("CUOPT_CONFIG_ID") != nullptr) {
    const int config_id  = experiment_integer("CUOPT_CONFIG_ID", 0, 0);
    const int max_config = experiment_integer("CUOPT_MAX_CONFIG", fp_config_names.size(), 1);
    if (config_id >= max_config) {
      throw std::invalid_argument("CUOPT_CONFIG_ID must be in [0, CUOPT_MAX_CONFIG)");
    }
    // Benchmark repetitions cycle through the fixed preset order without changing the RNG seed.
    const auto preset_id = config_id % fp_config_names.size();
    auto config          = gpu_submip_fp_config_from_name(fp_config_names[preset_id]);
    CUOPT_LOG_INFO("GpuSubmipFpSelection config_id=%d max_config=%d preset_id=%zu name=%s",
                   config_id,
                   max_config,
                   preset_id,
                   config.name.c_str());
    return config;
  }
  // Keep saved named-config experiments reproducible; numeric benchmark IDs take precedence.
  if (const char* name = std::getenv("CUOPT_GPU_SUBMIP_FP_CONFIG")) {
    return gpu_submip_fp_config_from_name(name);
  }
  return {};
}

gpu_submip_options_t gpu_submip_options_from_environment()
{
  gpu_submip_options_t options;
  const char* mode = std::getenv("CUOPT_GPU_SUBMIP_MODE");
  if (mode != nullptr) {
    if (std::string(mode) == "cpufj") {
      options.mode = gpu_submip_mode_t::CPUFJ_FULL_GPU;
    } else if (std::string(mode) != "node") {
      throw std::invalid_argument("GPU sub-MIP mode must be node or cpufj");
    }
  }
  if (const char* lifetime = std::getenv("CUOPT_GPU_SUBMIP_LIFETIME")) {
    const std::string value(lifetime);
    if (value != "independent" && value != "paired") {
      throw std::invalid_argument("GPU sub-MIP lifetime must be independent or paired");
    }
    options.independent = value == "independent";
  }
  options.max_projections = experiment_integer("CUOPT_GPU_SUBMIP_MAX_PROJECTIONS", 20, 0);
  options.max_descents    = experiment_integer("CUOPT_GPU_SUBMIP_MAX_DESCENTS", 0, 0);
  options.fix_rate        = experiment_option("CUOPT_GPU_SUBMIP_FIX_RATE", 0.25, 0, 0.99);
  options.time_limit      = experiment_option("CUOPT_GPU_SUBMIP_SECONDS", 60, 0.01, 86400);
  options.node_stride     = experiment_option("CUOPT_GPU_SUBMIP_STRIDE", 8, 1, 1000000);
  options.min_depth       = experiment_option("CUOPT_GPU_SUBMIP_MIN_DEPTH", 0, 0, 1000000);
  options.max_fractional  = experiment_option("CUOPT_GPU_SUBMIP_MAX_FRACTIONAL",
                                             std::numeric_limits<int>::max(),
                                             0,
                                             std::numeric_limits<int>::max());
  options.initial_calls   = experiment_option("CUOPT_GPU_SUBMIP_INITIAL_CALLS", 4, 0, 1000000);
  options.cached_primal   = experiment_option("CUOPT_GPU_SUBMIP_CACHED_PRIMAL", 0, 0, 1) != 0;
  options.enable_fp       = experiment_option("CUOPT_GPU_SUBMIP_DISABLE_FP", 0, 0, 1) == 0;
  options.compact_fixed   = experiment_option("CUOPT_GPU_SUBMIP_COMPACT", 1, 0, 1) != 0;
  options.fused_pdlp      = experiment_option("CUOPT_GPU_SUBMIP_FUSED_PDLP", 1, 0, 1) != 0;
  options.incumbent_blend = experiment_option("CUOPT_GPU_SUBMIP_INCUMBENT_BLEND", 0, 0, 1);
  options.fp_config       = gpu_submip_fp_config_from_environment();
  return options;
}

template <typename i_t, typename f_t>
bool gpu_submip_candidate_is_valid(const simplex::lp_problem_t<i_t, f_t>& lp,
                                   const std::vector<simplex::variable_type_t>& types,
                                   const std::vector<f_t>& candidate,
                                   const mip_solver_settings_t<i_t, f_t>& settings)
{
  if (candidate.size() != static_cast<size_t>(lp.num_cols) || types.size() > candidate.size()) {
    return false;
  }
  const f_t tolerance = settings.tolerances.absolute_tolerance;
  for (i_t j = 0; j < lp.num_cols; ++j) {
    if (!std::isfinite(candidate[j]) || candidate[j] < lp.lower[j] - tolerance ||
        candidate[j] > lp.upper[j] + tolerance ||
        (static_cast<size_t>(j) < types.size() &&
         types[j] != simplex::variable_type_t::CONTINUOUS &&
         std::abs(candidate[j] - std::round(candidate[j])) >
           settings.tolerances.integrality_tolerance)) {
      return false;
    }
  }
  csr_matrix_t<i_t, f_t> rows(lp.num_rows, lp.num_cols, lp.A.nnz());
  lp.A.to_compressed_row(rows);
  for (i_t i = 0; i < lp.num_rows; ++i) {
    const f_t activity = compensated_dot2_csr(
      rows.row_start.data(), rows.j.data(), rows.x.data(), candidate.data(), i);
    const f_t allowed = tolerance + settings.tolerances.relative_tolerance *
                                      std::max(std::abs(activity), std::abs(lp.rhs[i]));
    if (!std::isfinite(activity) || std::abs(activity - lp.rhs[i]) > allowed) { return false; }
  }
  return true;
}

template <typename i_t, typename f_t>
gpu_submip_job_t<i_t, f_t> make_gpu_submip_job(
  const simplex::lp_problem_t<i_t, f_t>& lp,
  const std::vector<simplex::variable_type_t>& var_types,
  const std::vector<f_t>& primal,
  const std::vector<f_t>& incumbent,
  f_t cutoff,
  const gpu_submip_options_t& options,
  const mip_solver_settings_t<i_t, f_t>& settings,
  uint64_t job_id,
  bool apply_rins)
{
  if (primal.size() != static_cast<size_t>(lp.num_cols) || var_types.size() > primal.size() ||
      (!incumbent.empty() && incumbent.size() != primal.size()) || std::isnan(cutoff) ||
      !std::all_of(primal.begin(), primal.end(), [](f_t x) { return std::isfinite(x); }) ||
      !std::all_of(incumbent.begin(), incumbent.end(), [](f_t x) { return std::isfinite(x); })) {
    throw std::invalid_argument("GPU job requires finite seeds in exact LP coordinates");
  }
  gpu_submip_job_t<i_t, f_t> job{
    lp, var_types, primal, incumbent, cutoff, simplex::compute_objective(lp, primal)};
  job.job_id = job_id;
  job.var_types.resize(primal.size(), simplex::variable_type_t::CONTINUOUS);
  if (!apply_rins || incumbent.empty() || options.fix_rate == 0) { return job; }
  std::vector<i_t> candidates;
  i_t free_integers = 0;
  for (i_t j = 0; j < static_cast<i_t>(var_types.size()); ++j) {
    if (var_types[j] == simplex::variable_type_t::CONTINUOUS || lp.lower[j] == lp.upper[j]) {
      continue;
    }
    ++free_integers;
    const f_t value = std::round(incumbent[j]);
    if (std::isfinite(value) &&
        std::abs(primal[j] - value) <= settings.tolerances.integrality_tolerance &&
        value >= lp.lower[j] && value <= lp.upper[j]) {
      candidates.push_back(j);
    }
  }
  const auto seed = get_base_seed(settings.seed);
  pcgenerator_t rng(derive_seed(seed, rng_id_t::gpu_submip_worker, job_id),
                    derive_stream(seed, rng_id_t::gpu_submip_worker, job_id));
  rng.shuffle(candidates);
  job.fixed = std::min(i_t(candidates.size()), i_t(options.fix_rate * free_integers));
  for (i_t k = 0; k < job.fixed; ++k) {
    const i_t j     = candidates[k];
    job.lp.lower[j] = job.lp.upper[j] = std::round(incumbent[j]);
  }
  return job;
}

template <typename i_t, typename f_t>
i_t perturb_gpu_submip_restart_seed(const simplex::lp_problem_t<i_t, f_t>& lp,
                                    const std::vector<simplex::variable_type_t>& types,
                                    std::vector<f_t>& seed,
                                    pcgenerator_t& rng)
{
  if (seed.size() != static_cast<size_t>(lp.num_cols) || types.size() > seed.size() ||
      lp.lower.size() != seed.size() || lp.upper.size() != seed.size() ||
      !std::all_of(seed.begin(), seed.end(), [](f_t value) { return std::isfinite(value); })) {
    throw std::invalid_argument("Invalid GPU sub-MIP restart seed");
  }
  std::vector<i_t> eligible;
  for (i_t j = 0; j < lp.num_cols; ++j) {
    seed[j] = std::clamp(seed[j], lp.lower[j], lp.upper[j]);
    if (j >= static_cast<i_t>(types.size()) || types[j] == simplex::variable_type_t::CONTINUOUS ||
        lp.lower[j] == lp.upper[j]) {
      continue;
    }
    const f_t lower = std::ceil(lp.lower[j]);
    const f_t upper = std::floor(lp.upper[j]);
    if (lower >= upper) { continue; }
    const f_t base = std::clamp(std::round(seed[j]), lower, upper);
    if ((base - 1 >= lower && base - 1 < base) || (base + 1 <= upper && base + 1 > base)) {
      eligible.push_back(j);
    }
  }
  // A sparse perturbation changes the next rounding while preserving most of
  // the previous projection. Fixed columns and continuous coordinates stay put.
  const size_t count = std::min(size_t(8), (eligible.size() + 99) / 100);
  for (size_t k = 0; k < count; ++k) {
    const size_t selected = rng.uniform(k, eligible.size());
    std::swap(eligible[k], eligible[selected]);
    const i_t j     = eligible[k];
    const f_t lower = std::ceil(lp.lower[j]);
    const f_t upper = std::floor(lp.upper[j]);
    const f_t base  = std::clamp(std::round(seed[j]), lower, upper);
    const bool down = base - 1 >= lower && base - 1 < base;
    const bool up   = base + 1 <= upper && base + 1 > base;
    seed[j]         = down && (!up || (rng.next_u32() & 1)) ? base - 1 : base + 1;
  }
  return static_cast<i_t>(count);
}

template <typename i_t, typename f_t>
bool run_gpu_submip_job(const gpu_submip_job_t<i_t, f_t>& job,
                        const mip_solver_settings_t<i_t, f_t>& settings,
                        const raft::handle_t* handle,
                        std::atomic<int>* halt,
                        const gpu_submip_options_t& options,
                        std::function<f_t()> current_user_cutoff,
                        std::function<void(const std::vector<f_t>&, f_t)> publish,
                        std::unique_ptr<gpu_submip_fp_t<i_t, f_t>>* session_cache,
                        const std::vector<f_t>* primal_seed_override,
                        const char* primal_seed_name_override,
                        gpu_submip_fp_run_info_t* run_info,
                        int projection_limit)
{
  if (run_info) { *run_info = {}; }
  if (!options.enable_fp || (halt && halt->load()) || tic() >= job.deadline) { return false; }
  const double started = tic();
  const double budget  = std::min(options.time_limit, job.deadline - started);
  f_t cutoff           = job.cutoff;
  if (current_user_cutoff) {
    cutoff = std::min(cutoff, simplex::compute_presolved_objective(job.lp, current_user_cutoff()));
  }
  if (cutoff == std::numeric_limits<f_t>::infinity()) {
    if (run_info) { run_info->exit_reason = "no_incumbent"; }
    return false;
  }
  if (std::isnan(cutoff)) { return false; }
  const f_t margin = std::max(f_t(settings.tolerances.absolute_tolerance),
                              f_t(1e-8) * std::max(f_t(1), std::abs(cutoff)));
  f_t target       = std::isfinite(cutoff) ? cutoff - margin : cutoff;
  if (std::isfinite(cutoff) && (job.lp.objective_step.has_step() || job.lp.objective_is_integral)) {
    const f_t step = job.lp.objective_step.has_step() ? job.lp.objective_step.step_size : f_t(1);
    const f_t bias = job.lp.objective_step.has_step() ? job.lp.objective_step.bias : f_t(0);
    const f_t k    = std::ceil((cutoff - bias) / step - settings.tolerances.integrality_tolerance);
    target         = (k - 1) * step + bias;
  }
  const auto snapshot_path =
    gpu_submip_job_snapshot_path(job.source,
                                 std::getenv("CUOPT_GPU_SUBMIP_JOB_SNAPSHOT"),
                                 std::getenv("CUOPT_GPU_SUBMIP_JOB_SNAPSHOT_SOURCE"),
                                 std::getenv("CUOPT_GPU_SUBMIP_JOB_SNAPSHOT_DIR"));
  if (!snapshot_path.empty()) {
    static std::mutex snapshot_mutex;
    static std::set<std::string> captured_paths;
    std::lock_guard lock(snapshot_mutex);
    if (!captured_paths.contains(snapshot_path)) {
      const auto parent = std::filesystem::path(snapshot_path).parent_path();
      if (!parent.empty()) { std::filesystem::create_directories(parent); }
      // This is a warm seed and may violate the selected node's bounds. Retain it so replay
      // applies the same incumbent blend and the session's identical bound clamping.
      write_post_root_snapshot(
        snapshot_path,
        capture_gpu_submip_job_snapshot(job.lp, job.var_types, job.primal, job.incumbent, cutoff));
      captured_paths.insert(snapshot_path);
      CUOPT_LOG_INFO(
        "GpuSubmipJobSnapshot path=%s job=%llu source=%s strategy=%d tree_node=%d "
        "depth=%d fractional=%d fixed=%d",
        snapshot_path.c_str(),
        (unsigned long long)job.job_id,
        gpu_submip_source_name(job.source),
        static_cast<int>(job.strategy),
        job.tree_node,
        job.depth,
        job.fractional,
        job.fixed);
    }
  }
  std::unique_ptr<gpu_submip_fp_t<i_t, f_t>> local_session;
  auto& session = session_cache ? *session_cache : local_session;
  const bool reused =
    session && session->can_reuse(job.lp, job.var_types) && session->config() == options.fp_config;
  if (!reused) {
    session = std::make_unique<gpu_submip_fp_t<i_t, f_t>>(handle,
                                                          job.lp,
                                                          job.var_types,
                                                          settings,
                                                          options.compact_fixed,
                                                          options.fused_pdlp,
                                                          options.fp_config);
  }
  const double setup_seconds = toc(started);
  std::vector<f_t> seed      = primal_seed_override ? *primal_seed_override : job.primal;
  if (!primal_seed_override && options.incumbent_blend > 0 && job.incumbent.size() == seed.size()) {
    for (size_t j = 0; j < seed.size(); ++j) {
      seed[j] = std::lerp(seed[j], job.incumbent[j], f_t(options.incumbent_blend));
    }
  }
  CUOPT_LOG_INFO(
    "GpuSubmipJobStart job=%llu source=%s source_id=%llu tree_node=%d depth=%d "
    "fractional=%d fixed=%d seconds=%.6f session_reused=%d strategy=%d cpu_lp_solves=0 "
    "fp_config=%s cutoff=%.17g target_cutoff=%.17g",
    (unsigned long long)job.job_id,
    gpu_submip_source_name(job.source),
    (unsigned long long)job.source_id,
    job.tree_node,
    job.depth,
    job.fractional,
    job.fixed,
    budget,
    int(reused),
    static_cast<int>(job.strategy),
    options.fp_config.name.c_str(),
    double(cutoff),
    double(target));
  std::vector<f_t> candidate;
  const bool found = session->run(job.lp,
                                  seed,
                                  target,
                                  budget - toc(started),
                                  halt,
                                  candidate,
                                  !primal_seed_override && options.cached_primal,
                                  primal_seed_override          ? primal_seed_name_override
                                  : options.incumbent_blend > 0 ? "incumbent_blend"
                                                                : "node_primal",
                                  job.job_id,
                                  projection_limit);
  if (run_info) { *run_info = session->last_run_info(); }
  bool valid = found && gpu_submip_candidate_is_valid(job.lp, job.var_types, candidate, settings);
  f_t objective = std::numeric_limits<f_t>::infinity();
  if (valid) {
    objective = simplex::compute_objective(job.lp, candidate);
    valid     = std::isfinite(objective) && objective < cutoff;
  }
  std::vector<f_t> user_candidate;
  if (valid && job.to_user) { valid = job.to_user(candidate, user_candidate); }
  if (valid && current_user_cutoff) {
    valid = objective < simplex::compute_presolved_objective(job.lp, current_user_cutoff());
  }
  if (valid && !(halt && halt->load()) && tic() < job.deadline && publish) {
    publish(job.to_user ? user_candidate : candidate, objective);
  } else {
    valid = false;
  }
  CUOPT_LOG_INFO(
    "GpuSubmipJobEnd job=%llu source=%s seconds=%.6f setup_seconds=%.6f "
    "fp_found=%d valid=%d tree_node=%d depth=%d fractional=%d cpu_lp_solves=0",
    (unsigned long long)job.job_id,
    gpu_submip_source_name(job.source),
    toc(started),
    setup_seconds,
    int(found),
    int(valid),
    job.tree_node,
    job.depth,
    job.fractional);
  return valid;
}

template bool gpu_submip_candidate_is_valid<int, double>(
  const simplex::lp_problem_t<int, double>&,
  const std::vector<simplex::variable_type_t>&,
  const std::vector<double>&,
  const mip_solver_settings_t<int, double>&);

template gpu_submip_job_t<int, double> make_gpu_submip_job<int, double>(
  const simplex::lp_problem_t<int, double>&,
  const std::vector<simplex::variable_type_t>&,
  const std::vector<double>&,
  const std::vector<double>&,
  double,
  const gpu_submip_options_t&,
  const mip_solver_settings_t<int, double>&,
  uint64_t,
  bool);

template bool run_gpu_submip_job<int, double>(
  const gpu_submip_job_t<int, double>&,
  const mip_solver_settings_t<int, double>&,
  const raft::handle_t*,
  std::atomic<int>*,
  const gpu_submip_options_t&,
  std::function<double()>,
  std::function<void(const std::vector<double>&, double)>,
  std::unique_ptr<gpu_submip_fp_t<int, double>>*,
  const std::vector<double>*,
  const char*,
  gpu_submip_fp_run_info_t*,
  int);

template int perturb_gpu_submip_restart_seed<int, double>(
  const simplex::lp_problem_t<int, double>&,
  const std::vector<simplex::variable_type_t>&,
  std::vector<double>&,
  pcgenerator_t&);

}  // namespace cuopt::mathematical_optimization::mip
