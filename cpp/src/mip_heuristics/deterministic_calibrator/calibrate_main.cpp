/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/deterministic_calibrator/bp_harness.hpp>
#include <mip_heuristics/deterministic_calibrator/calibrator.hpp>
#include <mip_heuristics/deterministic_calibrator/fj_harness.hpp>
#include <mip_heuristics/deterministic_calibrator/work_features.hpp>

#include <cstdio>
#include <string>
#include <vector>

using namespace cuopt::linear_programming::detail::calib;

namespace {

// The user-approved calibration set (datasets/mip/miplib2017): spread over size / sparsity /
// row-size variance / integer fraction.
const std::vector<std::string> kInstances = {
  "gen-ip054",
  "gmu-35-50",
  "cvs16r128-89",
  "sct2",
  "tr12-30",
  "seymour1",
  "nw04",
  "uccase9",
  "ns1208400",
  "neos-3004026-krka",
  "bab2",
  "nursesched-medium-hint03",
};

void report_and_emit(const std::string& algo,
                     const std::vector<std::string>& feature_names,
                     const std::vector<calibration_sample_t>& samples,
                     const std::string& out_header)
{
  if (samples.size() < 2) {
    std::printf("\nNot enough samples to calibrate (%zu).\n", samples.size());
    return;
  }
  work_calibrator_t calibrator;
  work_calibrator_t::options_t opt;
  auto result = calibrator.fit(samples, feature_names.size(), opt);

  std::printf("\n=== %s fit ===\n", algo.c_str());
  std::printf("samples=%zu  CV (measured/predicted) = %.4f  converged=%s  iters=%d\n",
              samples.size(),
              result.cv,
              result.converged ? "yes" : "no",
              result.iterations_used);
  for (std::size_t i = 0; i < result.coeffs.size(); ++i) {
    std::printf("  coeff[%s] = %.6g\n", feature_names[i].c_str(), result.coeffs[i]);
  }
  std::printf("\nper-sample ratio (measured / predicted, want ~1):\n");
  for (const auto& s : samples) {
    const double pred = predict_work(result.coeffs, s.features);
    std::printf("  %-32s measured=%.3e pred=%.3e ratio=%.3f\n",
                s.instance.c_str(),
                s.measured_time_per_iter,
                pred,
                s.measured_time_per_iter / pred);
  }

  emit_coeffs_header(out_header, algo, feature_names, result, samples);
  std::printf("\nWrote %s\n", out_header.c_str());
}

}  // namespace

int main(int argc, char** argv)
{
  std::string algo        = argc > 1 ? argv[1] : "fj";
  std::string dataset_dir = argc > 2 ? argv[2] : "datasets/mip/miplib2017";
  std::string out_header  = argc > 3 ? argv[3] : std::string();

  std::printf("=== deterministic work-unit calibration: %s ===\n", algo.c_str());
  std::printf("dataset_dir: %s\n", dataset_dir.c_str());

  std::vector<calibration_sample_t> samples;
  for (const auto& name : kInstances) {
    const std::string path = dataset_dir + "/" + name + ".mps";
    std::printf("\n[instance] %s\n", name.c_str());
    std::fflush(stdout);
    try {
      if (algo == "fj") {
        samples.push_back(run_fj_calibration_sample(path, name));
      } else if (algo == "bp") {
        auto s = run_bp_calibration_samples(path, name);
        samples.insert(samples.end(), s.begin(), s.end());
      } else {
        std::printf("Unknown algo '%s' (expected fj|bp)\n", algo.c_str());
        return 2;
      }
    } catch (const std::exception& e) {
      std::printf("  SKIP (%s)\n", e.what());
    }
    std::fflush(stdout);
  }

  if (algo == "fj") {
    if (out_header.empty()) {
      out_header = "cpp/src/mip_heuristics/deterministic_calibrator/generated/fj_work_coeffs.hpp";
    }
    report_and_emit("fj", fj_feature_names(), samples, out_header);
  } else {  // bp
    if (out_header.empty()) {
      out_header = "cpp/src/mip_heuristics/deterministic_calibrator/generated/bp_work_coeffs.hpp";
    }
    report_and_emit("bp", bp_feature_names(), samples, out_header);
  }
  return 0;
}
