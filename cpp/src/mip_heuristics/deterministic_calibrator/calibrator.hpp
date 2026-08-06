/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <mip_heuristics/deterministic_calibrator/linear_work_model.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace cuopt::mathematical_optimization::mip::calib {

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
// possible*. The system is the relative-error one
//   minimize  sum_i w_i ( dot(c, features_i)/measured_i - 1 )^2
//             + l2 * sum_j denom_j c_j^2 + l1 * sum_j sqrt(denom_j) c_j     subject to c >= 0
// i.e. drive every predicted/measured ratio toward 1, with THREE ingredients over the plain NNLS:
//   (1) ROBUST loss: w_i are Huber IRLS weights recomputed from the current ratio residual, so the
//       heavy tail of badly-mispredicted instances (the fj 10-60x outliers) cannot dominate the fit
//       and drag the bulk off 1.0. huber_delta is the residual (in ratio space) beyond which a row
//       is down-weighted; huber_delta<=0 recovers plain least squares.
//   (2) ELASTIC NET: a non-negative L1+L2 penalty, scaled per column by that column's own data
//       energy (denom_j) so it is scale-invariant. L1 drives the collinear device_terms (the same
//       raw count offered as /mem_bw, /(sm*clock), /clock) to a SPARSE, physically-readable subset;
//       L2 stabilizes the survivors. This is what makes the coefficients "understandable".
//   (3) MEDIAN pinning: after fitting we rescale so the MEDIAN ratio == 1 (not the mean), matching
//       the actual success metric (median wall/budget within +-10%) and staying robust to the tail.
// The optimizer is exact non-negative coordinate descent on the (weighted, penalized) convex inner
// problem, wrapped in an IRLS outer loop; it converges without random restarts.
class work_calibrator_t {
 public:
  struct options_t {
    int max_iters      = 100000;  // max coordinate-descent passes per IRLS step
    double target_cv   = 0.10;    // reporting target (CV of measured/predicted)
    double tol         = 1e-15;   // convergence tolerance on coefficient change
    double huber_delta = 0.5;     // robust threshold in ratio-residual space (<=0 => plain L2)
    int irls_iters     = 15;      // outer IRLS reweighting passes
    double l2          = 1e-3;    // ridge strength (fraction of per-column energy)
    double l1          = 1e-3;    // lasso strength (fraction of per-column energy) for sparsity
    bool pin_median    = true;    // pin median ratio to 1 (else mean)
  };

  // Median / dispersion / within-band diagnostics of the ratio r_i = measured/predicted. Reports
  // the metric we actually ship against (median deviation + fraction within +-band).
  struct ratio_stats_t {
    double median{0.0};
    double within_band{0.0};  // fraction with |r-1| <= band
    double p10{0.0}, p90{0.0};
  };
  static ratio_stats_t ratio_stats(const std::vector<double>& coeffs,
                                   const std::vector<calibration_sample_t>& samples,
                                   double band = 0.10)
  {
    ratio_stats_t st;
    const std::size_t n = samples.size();
    if (n == 0) { return st; }
    std::vector<double> r(n);
    std::size_t nin = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const double pred = predict_work(coeffs, samples[i].features);
      r[i]              = samples[i].measured_time_per_iter / pred;
      if (std::abs(r[i] - 1.0) <= band) { ++nin; }
    }
    std::sort(r.begin(), r.end());
    st.median      = r[n / 2];
    st.p10         = r[(std::size_t)(0.10 * (n - 1))];
    st.p90         = r[(std::size_t)(0.90 * (n - 1))];
    st.within_band = static_cast<double>(nin) / static_cast<double>(n);
    return st;
  }

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

  // Rescale coefficients so the work unit is pinned to seconds (wups == 1). With pin_median the
  // MEDIAN ratio (measured/predict) is set to 1 (robust to the heavy tail, matches the median
  // success metric); otherwise the mean ratio is set to 1 (legacy behaviour).
  static void scale_to_wups_one(std::vector<double>& coeffs,
                                const std::vector<calibration_sample_t>& samples,
                                bool pin_median = true)
  {
    const std::size_t n = samples.size();
    if (n == 0) { return; }
    if (pin_median) {
      // scale s so median(pred/meas) == 1 => median(meas/pred) == 1.
      std::vector<double> pom;
      pom.reserve(n);
      for (const auto& s : samples) {
        const double pred = predict_work(coeffs, s.features);
        pom.push_back(pred / s.measured_time_per_iter);
      }
      std::sort(pom.begin(), pom.end());
      const double med = pom[n / 2];
      if (med <= 0.0) { return; }
      const double s = 1.0 / med;
      for (double& c : coeffs) {
        c *= s;
      }
      return;
    }
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

  // Fit non-negative coefficients on the robust, elastic-net-penalized relative-error system (see
  // the class comment). Inner solver: exact non-negative coordinate descent with fixed IRLS weights
  // w_i and per-column elastic-net penalties; outer loop: recompute the Huber weights from the
  // current ratio residual (irls_iters times). Both the inner problem (convex, bounded below) and
  // the IRLS iteration converge without random restarts. Median-pinned so wups == 1.
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
    // Unweighted per-column energy: the scale used to make the elastic-net penalty scale-invariant
    // (penalize c_j in units of its own column contribution, so no single device_term is favoured
    // just because its raw counts are numerically larger).
    std::vector<double> col_energy(n_features, 0.0);
    for (std::size_t j = 0; j < n_features; ++j) {
      for (std::size_t i = 0; i < n; ++i) {
        col_energy[j] += A[i][j] * A[i][j];
      }
    }

    std::vector<double> w(n, 1.0);     // IRLS robustness weights
    std::vector<double> pred(n, 0.0);  // dot(c, A[i])
    std::vector<double> wdenom(n_features, 0.0);
    int total_passes = 0;

    const int outer = opt.huber_delta > 0.0 ? std::max(1, opt.irls_iters) : 1;
    for (int it = 0; it < outer; ++it) {
      // (Re)compute Huber weights from the current fit's ratio residual (predict/measured - 1).
      if (it > 0 && opt.huber_delta > 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
          const double a = std::abs(pred[i] - 1.0);
          w[i]           = (a <= opt.huber_delta) ? 1.0 : opt.huber_delta / a;
        }
      }
      // Weighted per-column energy for this IRLS step.
      for (std::size_t j = 0; j < n_features; ++j) {
        double d = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          d += w[i] * A[i][j] * A[i][j];
        }
        wdenom[j] = d;
      }
      // Refresh pred for current c (c carries over across IRLS steps as a warm start).
      for (std::size_t i = 0; i < n; ++i) {
        double p = 0.0;
        for (std::size_t j = 0; j < n_features; ++j) {
          p += c[j] * A[i][j];
        }
        pred[i] = p;
      }

      int pass = 0;
      for (; pass < opt.max_iters; ++pass) {
        double max_change = 0.0;
        for (std::size_t j = 0; j < n_features; ++j) {
          if (wdenom[j] <= 0.0) { continue; }
          // Weighted numerator; residual excluding j is (1 - (pred_i - c_j A_ij)).
          double num = 0.0;
          for (std::size_t i = 0; i < n; ++i) {
            num += w[i] * A[i][j] * (1.0 - (pred[i] - c[j] * A[i][j]));
          }
          // Elastic net (scale-invariant): ridge adds l2*col_energy to the denominator; lasso
          // subtracts l1*sqrt(col_energy) from the numerator (soft-threshold at 0 for c_j >= 0).
          const double l1_pen = opt.l1 * std::sqrt(col_energy[j]);
          const double den    = wdenom[j] + opt.l2 * col_energy[j];
          double new_cj       = (num - l1_pen) / den;
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
      total_passes += pass;
    }

    scale_to_wups_one(c, samples, opt.pin_median);

    calibration_result_t res;
    res.coeffs          = c;
    res.cv              = coeff_of_variation(c, samples);
    res.iterations_used = total_passes;
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
  f << "namespace cuopt::mathematical_optimization::mip::calib {\n\n";
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
  f << "}  // namespace cuopt::mathematical_optimization::mip::calib\n";
}

}  // namespace cuopt::mathematical_optimization::mip::calib
