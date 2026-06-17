/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

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

}  // namespace

int main(int argc, char** argv)
{
  std::string dataset_dir = "datasets/mip/miplib2017";
  std::string out_header =
    "cpp/src/mip_heuristics/deterministic_calibrator/generated/fj_work_coeffs.hpp";
  if (argc > 1) { dataset_dir = argv[1]; }
  if (argc > 2) { out_header = argv[2]; }

  std::printf("=== FJ deterministic work-unit calibration ===\n");
  std::printf("dataset_dir: %s\n", dataset_dir.c_str());

  std::vector<calibration_sample_t> samples;
  for (const auto& name : kInstances) {
    const std::string path = dataset_dir + "/" + name + ".mps";
    std::printf("\n[instance] %s\n", name.c_str());
    std::fflush(stdout);
    try {
      auto s = run_fj_calibration_sample(path, name);
      std::printf("  median %.3e s/step  features:", s.measured_time_per_iter);
      for (double f : s.features) {
        std::printf(" %.4g", f);
      }
      std::printf("\n");
      samples.push_back(std::move(s));
    } catch (const std::exception& e) {
      std::printf("  SKIP (%s)\n", e.what());
    }
    std::fflush(stdout);
  }

  if (samples.size() < 2) {
    std::printf("\nNot enough samples to calibrate (%zu).\n", samples.size());
    return 1;
  }

  const auto feature_names = fj_feature_names();
  work_calibrator_t calibrator;
  work_calibrator_t::options_t opt;
  auto result = calibrator.fit(samples, feature_names.size(), opt);

  std::printf("\n=== FJ fit ===\n");
  std::printf("CV (measured/predicted) = %.4f  converged=%s  iters=%d\n",
              result.cv,
              result.converged ? "yes" : "no",
              result.iterations_used);
  for (std::size_t i = 0; i < result.coeffs.size(); ++i) {
    std::printf("  coeff[%s] = %.6g\n", feature_names[i].c_str(), result.coeffs[i]);
  }
  std::printf("\nper-instance ratio (measured / predicted, want ~1):\n");
  for (const auto& s : samples) {
    const double pred = predict_work(result.coeffs, s.features);
    std::printf("  %-26s measured=%.3e pred=%.3e ratio=%.3f\n",
                s.instance.c_str(),
                s.measured_time_per_iter,
                pred,
                s.measured_time_per_iter / pred);
  }

  emit_coeffs_header(out_header, "fj", feature_names, result, samples);
  std::printf("\nWrote %s\n", out_header.c_str());
  return 0;
}
