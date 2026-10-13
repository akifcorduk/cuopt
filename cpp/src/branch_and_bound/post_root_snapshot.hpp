/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Frozen input for heuristic-worker experiments, captured after the outer root's strong
// branching. This is NOT a checkpoint of the complete solver: it does not restore the search
// tree, other workers, RNGs, presolve/postsolve mappings, or pseudocost histories. All vectors
// use the augmented post-root LP's coordinates, including its cuts and slack columns.
template <typename i_t, typename f_t>
struct post_root_snapshot_t {
  static_assert(std::is_integral_v<i_t> && std::is_signed_v<i_t>);
  static_assert(std::is_floating_point_v<f_t> && std::numeric_limits<f_t>::is_iec559);

  i_t num_rows{0}, num_cols{0};
  std::vector<i_t> col_start, row_indices;
  std::vector<f_t> coefficients, rhs, objective, lower, upper;
  std::vector<int8_t> var_types, basis_status;
  std::vector<f_t> root_primal, root_dual, root_reduced_cost, edge_norms, incumbent;
  std::vector<i_t> new_slacks;
  f_t obj_constant{0}, obj_scale{1}, objective_step_size{0}, objective_step_bias{0};
  f_t max_abs_obj_coeff{0}, min_abs_obj_coeff{0};
  f_t root_objective{0}, upper_bound{std::numeric_limits<f_t>::infinity()};
  uint8_t objective_is_integral{0};

  void validate() const
  {
    auto require = [](bool ok, const char* message) {
      if (!ok) { throw std::runtime_error(std::string("Invalid post-root snapshot: ") + message); }
    };
    require(num_rows >= 0 && num_cols >= 0, "negative dimensions");
    const auto m = static_cast<size_t>(num_rows);
    const auto n = static_cast<size_t>(num_cols);
    require(col_start.size() == n + 1, "CSC column offsets");
    require(!col_start.empty() && col_start.front() == 0, "CSC initial offset");
    require(std::is_sorted(col_start.begin(), col_start.end()), "CSC offset order");
    require(col_start.back() >= 0 && static_cast<size_t>(col_start.back()) == coefficients.size(),
            "CSC terminal offset");
    require(row_indices.size() == coefficients.size(), "CSC row indices");
    for (i_t row : row_indices) {
      require(row >= 0 && row < num_rows, "CSC row out of range");
    }
    require(rhs.size() == m, "RHS dimensions");
    require(objective.size() == n && lower.size() == n && upper.size() == n, "column dimensions");
    require(var_types.size() == n && basis_status.size() == n, "type/basis dimensions");
    require(root_primal.size() == n && root_dual.size() == m && root_reduced_cost.size() == n,
            "root solution dimensions");
    require(incumbent.empty() || incumbent.size() == n, "incumbent dimensions");
    require(std::isfinite(obj_scale) && obj_scale != 0 && std::isfinite(obj_constant),
            "objective conversion");
    require(objective_is_integral <= 1, "objective integrality flag");
    for (size_t j = 0; j < n; ++j) {
      require(!std::isnan(lower[j]) && !std::isnan(upper[j]) && lower[j] <= upper[j],
              "variable bounds");
      require(var_types[j] >= 0 && var_types[j] <= 2, "variable type");
      require(basis_status[j] >= -1 && basis_status[j] <= 4, "basis status");
    }
    for (i_t col : new_slacks) {
      require(col >= 0 && col < num_cols, "slack column out of range");
    }
  }

  // LP and solution types are template parameters so serialization tests need no CUDA runtime.
  template <typename LP>
  void restore_lp(LP& lp) const
  {
    validate();
    restore_problem_data(lp);
    lp.Q                 = std::decay_t<decltype(lp.Q)>(0, 0, 0);
    lp.max_abs_obj_coeff = max_abs_obj_coeff;
    lp.min_abs_obj_coeff = min_abs_obj_coeff;
  }

  template <typename UserProblem>
  void restore_user_problem(UserProblem& problem) const
  {
    validate();
    restore_problem_data(problem);
    problem.row_sense.assign(num_rows, 'E');
    problem.range_rows.clear();
    problem.range_value.clear();
    problem.num_range_rows = 0;
    problem.problem_name   = "post_root_worker_replay";
    problem.row_names.clear();
    problem.col_names.clear();
    using variable_type = typename std::decay_t<decltype(problem.var_types)>::value_type;
    problem.var_types.resize(num_cols);
    std::transform(var_types.begin(), var_types.end(), problem.var_types.begin(), [](int8_t type) {
      return static_cast<variable_type>(type);
    });
    problem.Q_offsets.clear();
    problem.Q_indices.clear();
    problem.Q_values.clear();
    problem.original_num_cols = 0;
    problem.original_num_rows = 0;
    problem.original_col_to_expanded_col.clear();
    problem.cone_head_bounds.clear();
  }

  template <typename Solution>
  void restore_root_solution(Solution& solution) const
  {
    solution.resize(num_rows, num_cols);
    solution.x              = root_primal;
    solution.y              = root_dual;
    solution.z              = root_reduced_cost;
    solution.objective      = root_objective;
    solution.user_objective = obj_scale * (root_objective + obj_constant);
  }

  // Field order is part of the versioned file format. Scalars are serialized individually,
  // without struct padding; native floating-point bits, including infinities, are preserved.
  auto fields()
  {
    return std::tie(num_rows,
                    num_cols,
                    col_start,
                    row_indices,
                    coefficients,
                    rhs,
                    objective,
                    lower,
                    upper,
                    var_types,
                    basis_status,
                    root_primal,
                    root_dual,
                    root_reduced_cost,
                    edge_norms,
                    incumbent,
                    new_slacks,
                    obj_constant,
                    obj_scale,
                    objective_step_size,
                    objective_step_bias,
                    max_abs_obj_coeff,
                    min_abs_obj_coeff,
                    root_objective,
                    upper_bound,
                    objective_is_integral);
  }
  auto fields() const
  {
    return std::tie(num_rows,
                    num_cols,
                    col_start,
                    row_indices,
                    coefficients,
                    rhs,
                    objective,
                    lower,
                    upper,
                    var_types,
                    basis_status,
                    root_primal,
                    root_dual,
                    root_reduced_cost,
                    edge_norms,
                    incumbent,
                    new_slacks,
                    obj_constant,
                    obj_scale,
                    objective_step_size,
                    objective_step_bias,
                    max_abs_obj_coeff,
                    min_abs_obj_coeff,
                    root_objective,
                    upper_bound,
                    objective_is_integral);
  }

 private:
  template <typename Problem>
  void restore_problem_data(Problem& problem) const
  {
    problem.num_rows = num_rows;
    problem.num_cols = num_cols;
    problem.A.resize(num_rows, num_cols, static_cast<i_t>(coefficients.size()));
    problem.A.col_start              = col_start;
    problem.A.i                      = row_indices;
    problem.A.x                      = coefficients;
    problem.rhs                      = rhs;
    problem.objective                = objective;
    problem.lower                    = lower;
    problem.upper                    = upper;
    problem.obj_constant             = obj_constant;
    problem.obj_scale                = obj_scale;
    problem.objective_is_integral    = objective_is_integral != 0;
    problem.objective_step.step_size = objective_step_size;
    problem.objective_step.bias      = objective_step_bias;
    problem.cone_var_start           = 0;
    problem.second_order_cone_dims.clear();
  }
};

template <typename LP, typename VarType, typename Solution, typename BasisStatus>
auto capture_post_root_snapshot(
  const LP& lp,
  const std::vector<VarType>& var_types,
  const Solution& root_solution,
  const std::vector<BasisStatus>& basis_status,
  const std::vector<typename std::decay_t<decltype(lp.objective)>::value_type>& edge_norms,
  const std::vector<typename std::decay_t<decltype(lp.objective)>::value_type>& incumbent,
  typename std::decay_t<decltype(lp.objective)>::value_type root_objective,
  typename std::decay_t<decltype(lp.objective)>::value_type upper_bound,
  const std::vector<std::decay_t<decltype(lp.num_cols)>>& new_slacks)
{
  using i_t = std::decay_t<decltype(lp.num_cols)>;
  using f_t = typename std::decay_t<decltype(lp.objective)>::value_type;
  if (!lp.Q.x.empty() || !lp.second_order_cone_dims.empty()) {
    throw std::runtime_error("Post-root worker snapshots support linear MIPs only");
  }
  post_root_snapshot_t<i_t, f_t> snapshot;
  snapshot.num_rows  = lp.num_rows;
  snapshot.num_cols  = lp.num_cols;
  snapshot.col_start = lp.A.col_start;
  snapshot.row_indices.assign(lp.A.i.begin(), lp.A.i.begin() + lp.A.nnz());
  snapshot.coefficients.assign(lp.A.x.begin(), lp.A.x.begin() + lp.A.nnz());
  snapshot.rhs       = lp.rhs;
  snapshot.objective = lp.objective;
  snapshot.lower     = lp.lower;
  snapshot.upper     = lp.upper;
  snapshot.var_types.reserve(var_types.size());
  for (auto type : var_types) {
    snapshot.var_types.push_back(static_cast<int8_t>(type));
  }
  snapshot.basis_status.reserve(basis_status.size());
  for (auto status : basis_status) {
    snapshot.basis_status.push_back(static_cast<int8_t>(status));
  }
  snapshot.root_primal           = root_solution.x;
  snapshot.root_dual             = root_solution.y;
  snapshot.root_reduced_cost     = root_solution.z;
  snapshot.edge_norms            = edge_norms;
  snapshot.incumbent             = incumbent;
  snapshot.new_slacks            = new_slacks;
  snapshot.obj_constant          = lp.obj_constant;
  snapshot.obj_scale             = lp.obj_scale;
  snapshot.objective_is_integral = lp.objective_is_integral;
  snapshot.objective_step_size   = lp.objective_step.step_size;
  snapshot.objective_step_bias   = lp.objective_step.bias;
  snapshot.max_abs_obj_coeff     = lp.max_abs_obj_coeff;
  snapshot.min_abs_obj_coeff     = lp.min_abs_obj_coeff;
  snapshot.root_objective        = root_objective;
  snapshot.upper_bound           = upper_bound;
  snapshot.validate();
  return snapshot;
}

// A selected FP job uses the same matrix/vector serialization as a post-root snapshot,
// but has no saved simplex basis. It can be replayed directly by FP; it cannot resume
// simplex or the CPU tree. A finite cutoff does not require a feasible node incumbent.
template <typename LP, typename VarType>
auto capture_gpu_submip_job_snapshot(
  const LP& lp,
  const std::vector<VarType>& var_types,
  const std::vector<typename std::decay_t<decltype(lp.objective)>::value_type>& primal,
  const std::vector<typename std::decay_t<decltype(lp.objective)>::value_type>& incumbent,
  typename std::decay_t<decltype(lp.objective)>::value_type cutoff)
{
  using i_t = std::decay_t<decltype(lp.num_cols)>;
  using f_t = typename std::decay_t<decltype(lp.objective)>::value_type;
  struct seed_t {
    std::vector<f_t> x, y, z;
  } seed{primal, std::vector<f_t>(lp.num_rows, 0), std::vector<f_t>(lp.num_cols, 0)};
  return capture_post_root_snapshot(lp,
                                    var_types,
                                    seed,
                                    std::vector<int8_t>(lp.num_cols, -1),
                                    std::vector<f_t>{},
                                    incumbent,
                                    f_t{0},
                                    cutoff,
                                    std::vector<i_t>{});
}

namespace post_root_snapshot_detail {

class writer_t {
 public:
  explicit writer_t(std::ostream& stream) : stream_(stream) {}
  template <typename T>
  void operator()(const T& value)
  {
    static_assert(std::is_arithmetic_v<T>);
    stream_.write(reinterpret_cast<const char*>(&value), sizeof(T));
  }
  template <typename T>
  void operator()(const std::vector<T>& values)
  {
    (*this)(static_cast<uint64_t>(values.size()));
    if (!values.empty()) {
      stream_.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    }
  }

 private:
  std::ostream& stream_;
};

class reader_t {
 public:
  reader_t(std::istream& stream, uint64_t remaining) : stream_(stream), remaining_(remaining) {}
  template <typename T>
  void operator()(T& value)
  {
    static_assert(std::is_arithmetic_v<T>);
    read(reinterpret_cast<char*>(&value), sizeof(T));
  }
  template <typename T>
  void operator()(std::vector<T>& values)
  {
    uint64_t count;
    (*this)(count);
    if (count > remaining_ / sizeof(T) || count > values.max_size()) {
      throw std::runtime_error("Truncated or invalid post-root snapshot vector");
    }
    values.resize(static_cast<size_t>(count));
    if (count != 0) { read(reinterpret_cast<char*>(values.data()), count * sizeof(T)); }
  }
  uint64_t remaining() const { return remaining_; }

 private:
  void read(char* destination, uint64_t size)
  {
    if (size > remaining_) { throw std::runtime_error("Truncated post-root snapshot"); }
    stream_.read(destination, size);
    remaining_ -= size;
  }
  std::istream& stream_;
  uint64_t remaining_;
};

inline constexpr std::array<char, 8> magic{'C', 'U', 'O', 'P', 'T', 'R', 'P', '1'};
inline constexpr uint32_t endian_tag = 0x01020304;

}  // namespace post_root_snapshot_detail

template <typename i_t, typename f_t>
void write_post_root_snapshot(const std::string& path,
                              const post_root_snapshot_t<i_t, f_t>& snapshot)
{
  snapshot.validate();
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.exceptions(std::ios::badbit | std::ios::failbit);
  stream.write(post_root_snapshot_detail::magic.data(), post_root_snapshot_detail::magic.size());
  post_root_snapshot_detail::writer_t write(stream);
  write(post_root_snapshot_detail::endian_tag);
  write(uint32_t{1});
  write(static_cast<uint32_t>(sizeof(i_t)));
  write(static_cast<uint32_t>(sizeof(f_t)));
  std::apply([&write](const auto&... fields) { (write(fields), ...); }, snapshot.fields());
  stream.close();
}

template <typename i_t, typename f_t>
post_root_snapshot_t<i_t, f_t> read_post_root_snapshot(const std::string& path)
{
  const auto file_size = std::filesystem::file_size(path);
  if (file_size < post_root_snapshot_detail::magic.size()) {
    throw std::runtime_error("Truncated post-root snapshot header");
  }
  std::ifstream stream(path, std::ios::binary);
  stream.exceptions(std::ios::badbit | std::ios::failbit);
  std::array<char, 8> magic;
  stream.read(magic.data(), magic.size());
  if (magic != post_root_snapshot_detail::magic) {
    throw std::runtime_error("Unrecognized post-root snapshot format");
  }
  post_root_snapshot_detail::reader_t read(stream, file_size - magic.size());
  uint32_t endian, version, index_size, value_size;
  read(endian);
  read(version);
  read(index_size);
  read(value_size);
  if (endian != post_root_snapshot_detail::endian_tag || version != 1 ||
      index_size != sizeof(i_t) || value_size != sizeof(f_t)) {
    throw std::runtime_error("Unsupported post-root snapshot version, endianness, or scalar types");
  }
  post_root_snapshot_t<i_t, f_t> snapshot;
  std::apply([&read](auto&... fields) { (read(fields), ...); }, snapshot.fields());
  if (read.remaining() != 0) { throw std::runtime_error("Trailing data in post-root snapshot"); }
  snapshot.validate();
  return snapshot;
}

}  // namespace cuopt::mathematical_optimization::mip
