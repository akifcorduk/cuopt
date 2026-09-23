#pragma once
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace cuopt::hive_lns {
// Repair a private neighborhood: equal lower/upper bounds fix a variable.
// Bounds must lie within the model's domains. Start may be infeasible.
struct repair_request_t {
  std::vector<double> start, lower, upper;
  double time_limit_seconds = 0.1;
  uint64_t seed             = 0;
  int max_iterations        = 10000;
  int max_nodes             = 500;
};
struct repair_result_t {
  bool feasible = false;
  std::vector<double> assignment;  // full model coordinates; empty on failure/timeout
  double objective       = std::numeric_limits<double>::infinity();
  double elapsed_seconds = 0;
};
using repair_fn = std::function<repair_result_t(const repair_request_t&)>;
struct repair_tools_t {
  // Synchronous, single-threaded backends on the LNS worker. They never publish
  // incumbents or mutate population. Time includes setup and is capped by the
  // remaining solve time. An empty result is normal if no repair is found.
  // Every returned assignment passes the fixed model and neighborhood checks.
  repair_fn cpufj;
  repair_fn submip;
};

// Frozen, owning CPU view. All objective coefficients use minimization convention.
struct model_t {
  std::vector<int> offsets, columns;
  std::vector<double> coefficients, lower, upper, row_lower, row_upper, objective;
  std::vector<bool> integer;
  repair_tools_t repair;
  double feasibility_tolerance = 1e-6;
  double integrality_tolerance = 1e-5;

  double cost(const std::vector<double>& x) const
  {
    double value = 0, correction = 0;
    for (size_t j = 0; j < x.size(); ++j) {
      double term = objective[j] * x[j] - correction;
      double next = value + term;
      correction  = (next - value) - term;
      value       = next;
    }
    return value;
  }
  bool feasible(const std::vector<double>& x) const
  {
    if (x.size() != lower.size()) return false;
    for (size_t j = 0; j < x.size(); ++j) {
      if (!std::isfinite(x[j]) || x[j] < lower[j] - feasibility_tolerance ||
          x[j] > upper[j] + feasibility_tolerance ||
          (integer[j] && std::abs(x[j] - std::round(x[j])) > integrality_tolerance))
        return false;
    }
    for (size_t r = 0; r < row_lower.size(); ++r) {
      double value = 0, correction = 0;
      for (int p = offsets[r]; p < offsets[r + 1]; ++p) {
        double term = coefficients[p] * x[columns[p]] - correction;
        double next = value + term;
        correction  = (next - value) - term;
        value       = next;
      }
      if (!std::isfinite(value) || value < row_lower[r] - feasibility_tolerance ||
          value > row_upper[r] + feasibility_tolerance)
        return false;
    }
    return std::isfinite(cost(x));
  }
};
using population_t = std::vector<std::vector<double>>;
using snapshot_fn  = std::function<population_t()>;
using submit_fn    = std::function<void(const std::vector<double>&)>;
using stop_fn      = std::function<bool()>;
}  // namespace cuopt::hive_lns
