/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/solution/solution.cuh>

namespace cuopt {
namespace mathematical_optimization {
namespace mip {

// Boost-style hash combine. Host+device callable so the same recurrence is shared between the
// device solution hasher (assignment_hash_map_t) and host-side integer-assignment hashing.
struct combine_hash {
  HDI size_t operator()(size_t hash_1, size_t hash_2) const
  {
    const size_t magic_constant = 0x9e3779b97f4a7c15;
    hash_1 ^= hash_2 + magic_constant + (hash_1 << 12) + (hash_1 >> 4);
    return hash_1;
  }
};

template <typename i_t, typename f_t>
class assignment_hash_map_t {
 public:
  assignment_hash_map_t(const problem_t<i_t, f_t>& problem);
  void fill_integer_assignment(solution_t<i_t, f_t>& solution);
  size_t hash_solution(solution_t<i_t, f_t>& solution);
  void insert(solution_t<i_t, f_t>& solution);
  bool check_skip_solution(solution_t<i_t, f_t>& solution, i_t max_occurance);

  // keep the hash to encounter count of solution hash
  std::unordered_map<size_t, i_t> solution_hash_count;
  rmm::device_uvector<size_t> reduction_buffer;
  rmm::device_uvector<size_t> integer_assignment;
  rmm::device_scalar<size_t> hash_sum;
  rmm::device_buffer temp_storage;
};

}  // namespace mip
}  // namespace mathematical_optimization
}  // namespace cuopt
