/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <mip_heuristics/deterministic_calibrator/linear_work_model.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace cuopt::linear_programming::detail::calib {

// One calibration observation: the (median) feature vector and (median) measured per-iteration
// wall time for a single instance / leaf algorithm.
struct calibration_sample_t {
  std::string instance;
  std::vector<double> features;        // same layout as the linear model
  double measured_time_per_iter{0.0};  // seconds
};

struct calibration_result_t {
  std::vector<double> coeffs;
  double cv{0.0};  // coefficient of variation of (measured_time / predicted_work)
  int iterations_used{0};
  bool converged{false};
};

// Generic, leaf-agnostic work-unit calibrator. Fits non-negative coefficients of a linear model so
// that predicted_work matches measured per-iteration time across instances *as uniformly as
// possible*: it minimizes the coefficient of variation (std/mean) of the ratio
//   r_i = measured_time_i / predict_work(coeffs, features_i)
// across instances. CV is scale invariant, so after fitting we rescale the coefficients so the mean
// ratio equals 1 (i.e. wups == 1: one work unit ~ one second). The optimizer is a simulated-anneal
// / hill-climb with multiplicative moves (work contributions are non-negative), run up to
// `max_iters` or until CV < `target_cv`.
class work_calibrator_t {
 public:
  struct options_t {
    int max_iters      = 100000;
    double target_cv   = 0.10;
    double init_sigma  = 0.75;  // initial log-scale step for multiplicative moves
    double final_sigma = 0.02;  // final log-scale step
    double init_temp   = 0.20;  // initial SA temperature on CV
    unsigned seed      = 12345;
  };

  static double coeff_of_variation(const std::vector<double>& coeffs,
                                   const std::vector<calibration_sample_t>& samples)
  {
    const std::size_t n = samples.size();
    if (n == 0) { return 0.0; }
    std::vector<double> ratios(n);
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double pred = predict_work(coeffs, samples[i].features);
      ratios[i]         = samples[i].measured_time_per_iter / pred;
      mean += ratios[i];
    }
    mean /= static_cast<double>(n);
    if (mean <= 0.0) { return 1e30; }
    double var = 0.0;
    for (double r : ratios) {
      var += (r - mean) * (r - mean);
    }
    var /= static_cast<double>(n);
    return std::sqrt(var) / mean;
  }

  // Rescale coefficients so mean_i(predict/measured) == 1  =>  mean ratio (measured/predict) == 1,
  // pinning the work unit to seconds (wups == 1).
  static void scale_to_wups_one(std::vector<double>& coeffs,
                                const std::vector<calibration_sample_t>& samples)
  {
    const std::size_t n = samples.size();
    if (n == 0) { return; }
    double mean_pred_over_meas = 0.0;
    for (const auto& s : samples) {
      const double pred = predict_work(coeffs, s.features);
      mean_pred_over_meas += pred / s.measured_time_per_iter;
    }
    mean_pred_over_meas /= static_cast<double>(n);
    if (mean_pred_over_meas <= 0.0) { return; }
    const double s = 1.0 / mean_pred_over_meas;
    for (double& c : coeffs) {
      c *= s;
    }
  }

  calibration_result_t fit(const std::vector<calibration_sample_t>& samples,
                           std::size_t n_features,
                           options_t opt) const
  {
    std::mt19937 rng(opt.seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> pick(0, n_features - 1);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    std::vector<double> cur(n_features, 1.0);
    double cur_cv            = coeff_of_variation(cur, samples);
    std::vector<double> best = cur;
    double best_cv           = cur_cv;

    int it = 0;
    for (; it < opt.max_iters && best_cv > opt.target_cv; ++it) {
      const double t     = static_cast<double>(it) / static_cast<double>(opt.max_iters);
      const double sigma = opt.init_sigma * std::pow(opt.final_sigma / opt.init_sigma, t);
      const double temp  = opt.init_temp * (1.0 - t);

      std::vector<double> cand = cur;
      const std::size_t k      = pick(rng);
      cand[k] *= std::exp(sigma * gauss(rng));
      if (cand[k] < 0.0) { cand[k] = 0.0; }

      const double cand_cv = coeff_of_variation(cand, samples);
      const double delta   = cand_cv - cur_cv;
      if (delta < 0.0 || (temp > 0.0 && unit(rng) < std::exp(-delta / temp))) {
        cur    = cand;
        cur_cv = cand_cv;
        if (cur_cv < best_cv) {
          best    = cur;
          best_cv = cur_cv;
        }
      }
    }

    scale_to_wups_one(best, samples);

    calibration_result_t res;
    res.coeffs          = best;
    res.cv              = best_cv;
    res.iterations_used = it;
    res.converged       = best_cv <= opt.target_cv;
    return res;
  }
};

// Emit a generated C++ header holding the fitted coefficients for one leaf algorithm, plus the
// per-instance fit diagnostics as comments.
inline void emit_coeffs_header(const std::string& path,
                               const std::string& algo_name,
                               const std::vector<std::string>& feature_names,
                               const calibration_result_t& result,
                               const std::vector<calibration_sample_t>& samples)
{
  std::ofstream f(path);
  f << "/* clang-format off */\n";
  f << "/*\n";
  f << " * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All "
       "rights reserved.\n";
  f << " * SPDX-License-Identifier: Apache-2.0\n";
  f << " *\n";
  f << " * AUTO-GENERATED by mip_work_calibrator. Do not edit by hand.\n";
  f << " * Algorithm: " << algo_name << "\n";
  f << " * Fit CV (coeff of variation of measured/predicted): " << result.cv << "\n";
  f << " * Converged: " << (result.converged ? "yes" : "no") << " in " << result.iterations_used
    << " iters\n";
  f << " */\n";
  f << "/* clang-format on */\n";
  f << "#pragma once\n\n";
  f << "#include <array>\n\n";
  f << "namespace cuopt::linear_programming::detail::calib {\n\n";
  f << "// Per-instance fit diagnostics (measured s/iter, predicted work, ratio):\n";
  for (const auto& s : samples) {
    const double pred  = predict_work(result.coeffs, s.features);
    const double ratio = s.measured_time_per_iter / pred;
    f << "//   " << s.instance << ": measured=" << s.measured_time_per_iter << " pred=" << pred
      << " ratio=" << ratio << "\n";
  }
  f << "\nconstexpr std::array<double, " << result.coeffs.size() << "> " << algo_name
    << "_work_coeffs = {\n";
  for (std::size_t i = 0; i < result.coeffs.size(); ++i) {
    const std::string fname = i < feature_names.size() ? feature_names[i] : "feature";
    f << "  " << result.coeffs[i] << ",  // " << fname << "\n";
  }
  f << "};\n\n";
  f << "}  // namespace cuopt::linear_programming::detail::calib\n";
}

}  // namespace cuopt::linear_programming::detail::calib
