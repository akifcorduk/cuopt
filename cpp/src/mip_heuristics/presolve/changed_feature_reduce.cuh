/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <mip_heuristics/problem/problem.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

namespace cuopt::linear_programming::detail {

// Dynamic features of a changed-constraint set used by the bound-presolve / multi-probe work model:
// (#changed, nnz of changed rows, warp-loads = sum ceil(rowsize/32) of changed rows).
struct changed_feat_t {
  double n{0.0};
  double nnz{0.0};
  double warp{0.0};
  __host__ __device__ changed_feat_t operator+(const changed_feat_t& o) const
  {
    return {n + o.n, nnz + o.nnz, warp + o.warp};
  }
};

template <typename i_t, typename f_t>
inline changed_feat_t reduce_changed_features(problem_t<i_t, f_t>& pb,
                                              const rmm::device_uvector<i_t>& changed)
{
  auto offsets = pb.offsets.data();
  auto cmask   = changed.data();
  return thrust::transform_reduce(
    pb.handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator<i_t>(0),
    thrust::make_counting_iterator<i_t>(pb.n_constraints),
    [offsets, cmask] __device__(i_t c) -> changed_feat_t {
      if (cmask[c] == 0) { return changed_feat_t{}; }
      const double rs = (double)(offsets[c + 1] - offsets[c]);
      return changed_feat_t{1.0, rs, (double)(((long)rs + 31) / 32)};
    },
    changed_feat_t{},
    thrust::plus<changed_feat_t>{});
}

}  // namespace cuopt::linear_programming::detail
