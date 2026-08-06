/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace cuopt::mathematical_optimization::mip::calib {

// Predicted per-iteration work (== predicted seconds, since wups == 1) for a leaf algorithm:
// a plain dot product of the calibrated coefficients with the feature vector. Clamped to a small
// positive floor so a work clock built on top of it always advances.
inline double predict_work(const std::vector<double>& coeffs, const std::vector<double>& features)
{
  const std::size_t n = std::min(coeffs.size(), features.size());
  double w            = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    w += coeffs[i] * features[i];
  }
  constexpr double kMinWork = 1e-12;
  return w > kMinWork ? w : kMinWork;
}

}  // namespace cuopt::mathematical_optimization::mip::calib
