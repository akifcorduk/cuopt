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
  // Longest constraint row (max nnz over rows). Recorded separately from `features` (so the legacy
  // per-GPU model is untouched); the device-aware model uses it for a clock-bound serialization
  // term (the longest row is processed by ~one warp, so its cost scales with clock, not SM
  // count/bw).
  double max_row_nnz{0.0};
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
    int max_iters    = 100000;  // max coordinate-descent passes
    double target_cv = 0.10;    // reporting target (CV of measured/predicted)
    double tol       = 1e-15;   // convergence tolerance on coefficient change
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

  // Fit non-negative coefficients by coordinate-descent NNLS on the relative-error system:
  //   minimize  sum_i ( dot(c, features_i) / measured_i  -  1 )^2     subject to c >= 0
  // i.e. drive every predicted/measured ratio toward 1 (the CV objective's intent), weighting each
  // sample equally regardless of its absolute time. The objective is convex and bounded below, so
  // exact coordinate descent on the non-negative orthant converges to the global optimum (no random
  // restarts / local-minimum issues). With wups == 1 the resulting work unit ~ predicted seconds.
  calibration_result_t fit(const std::vector<calibration_sample_t>& samples,
                           std::size_t n_features,
                           options_t opt) const
  {
    const std::size_t n = samples.size();
    std::vector<double> c(n_features, 0.0);
    if (n == 0) { return {c, 0.0, 0, false}; }

    // A[i][j] = features_i[j] / measured_i ; target b_i = 1. Keep pred[i] = dot(c, A[i]).
    std::vector<std::vector<double>> A(n, std::vector<double>(n_features, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
      const double inv_m = 1.0 / samples[i].measured_time_per_iter;
      for (std::size_t j = 0; j < n_features && j < samples[i].features.size(); ++j) {
        A[i][j] = samples[i].features[j] * inv_m;
      }
    }
    std::vector<double> denom(n_features, 0.0);
    for (std::size_t j = 0; j < n_features; ++j) {
      for (std::size_t i = 0; i < n; ++i) {
        denom[j] += A[i][j] * A[i][j];
      }
    }
    std::vector<double> pred(n, 0.0);  // dot(c, A[i]) == 0 initially

    int pass = 0;
    for (; pass < opt.max_iters; ++pass) {
      double max_change = 0.0;
      for (std::size_t j = 0; j < n_features; ++j) {
        if (denom[j] <= 0.0) { continue; }
        // optimal c_j (others fixed): residual excluding j is (1 - (pred_i - c_j A_ij)).
        double num = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          num += A[i][j] * (1.0 - (pred[i] - c[j] * A[i][j]));
        }
        double new_cj = num / denom[j];
        if (new_cj < 0.0) { new_cj = 0.0; }
        const double delta = new_cj - c[j];
        if (delta != 0.0) {
          for (std::size_t i = 0; i < n; ++i) {
            pred[i] += delta * A[i][j];
          }
          c[j]       = new_cj;
          max_change = std::max(max_change, std::abs(delta));
        }
      }
      if (max_change <= opt.tol) {
        ++pass;
        break;
      }
    }

    scale_to_wups_one(c, samples);

    calibration_result_t res;
    res.coeffs          = c;
    res.cv              = coeff_of_variation(c, samples);
    res.iterations_used = pass;
    res.converged       = res.cv <= opt.target_cv;
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
