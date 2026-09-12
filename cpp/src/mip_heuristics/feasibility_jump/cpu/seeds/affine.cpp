/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../internal.hpp"

namespace cuopt::mathematical_optimization::mip {

// Minimize residuals in a whitened block of equality directions. Whitening prevents correlated
// rows from overwhelming independent residual directions; all accepted points are still checked
// against the complete, unchanged model.
template <typename i_t, typename f_t>
void apply_affine_equality_seed(fj_cpu_climber_t<i_t, f_t>& c, double budget)
{
  phase_timer_t timer(c.t_seed);
  const auto started = std::chrono::steady_clock::now();
  auto expired = [&] {
    return c.preemption_flag.load(std::memory_order_relaxed) ||
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= budget;
  };
  if (budget <= 0 || c.feasible_found) return;
  const auto& p = *c.problem;
  const i_t n = p.n_variables;
  if (n <= 0) return;
  const size_t block_size = std::min<size_t>(64, (4u << 20) / (size_t)n);
  if (!block_size) return;
  std::vector<i_t> rows;
  for (i_t r = 0; r < p.n_constraints && rows.size() < block_size; ++r) {
    if (!std::isfinite(p.cstr_lb[r]) || p.cstr_lb[r] != p.cstr_ub[r]) continue;
    for (i_t k = p.offsets[r]; k < p.offsets[r + 1]; ++k) {
      const auto b = c.h_var_bounds[p.variables[k]].get();
      if (get_lower(b) < get_upper(b) && p.coefficients[k] != 0) {
        rows.push_back(r);
        break;
      }
    }
  }
  const size_t q = rows.size();
  if (!q || expired()) return;

  std::vector<double> x(n), lower(n), upper(n);
  std::vector<uint8_t> integer(n);
  for (i_t v = 0; v < n; ++v) {
    const auto b = c.h_var_bounds[v].get();
    integer[v] = is_integer_var(c, v);
    lower[v] = integer[v] ? std::ceil(get_lower(b)) : get_lower(b);
    upper[v] = integer[v] ? std::floor(get_upper(b)) : get_upper(b);
    if (lower[v] > upper[v] || !std::isfinite(c.h_assignment[v])) return;
    x[v] = c.h_assignment[v];
  }

  std::vector<double> columns((size_t)n * q, 0), scale(q, 0), chol(q * q, 0);
  for (size_t r = 0; r < q; ++r)
    for (i_t k = p.offsets[rows[r]]; k < p.offsets[rows[r] + 1]; ++k) {
      const i_t v = p.variables[k];
      if (lower[v] < upper[v]) columns[(size_t)v * q + r] += p.coefficients[k];
    }
  for (i_t v = 0; v < n; ++v)
    for (size_t r = 0; r < q; ++r) {
      const double a = columns[(size_t)v * q + r];
      scale[r] += a * a;
    }
  for (double& s : scale) {
    if (!(s > 0) || !std::isfinite(s)) return;
    s = 1 / std::sqrt(s);
  }
  for (i_t v = 0; v < n; ++v) {
    if ((v & 255) == 0 && expired()) return;
    double* a = columns.data() + (size_t)v * q;
    for (size_t r = 0; r < q; ++r) a[r] *= scale[r];
    for (size_t r = 0; r < q; ++r)
      for (size_t s = 0; s <= r; ++s) chol[r * q + s] += a[r] * a[s];
  }
  for (size_t r = 0; r < q; ++r) {
    chol[r * q + r] += 1e-9;
    for (size_t s = 0; s <= r; ++s) {
      double a = chol[r * q + s];
      for (size_t k = 0; k < s; ++k) a -= chol[r * q + k] * chol[s * q + k];
      if (r == s) {
        if (!(a > 0) || !std::isfinite(a)) return;
        chol[r * q + s] = std::sqrt(a);
      } else {
        chol[r * q + s] = a / chol[s * q + s];
      }
    }
  }
  auto whiten = [&](double* a) {
    for (size_t r = 0; r < q; ++r) {
      for (size_t s = 0; s < r; ++s) a[r] -= chol[r * q + s] * a[s];
      a[r] /= chol[r * q + r];
    }
  };

  std::vector<double> norm(n, 0), gradient(n), error(q), multiplier(q, 0);
  std::vector<i_t> active;
  for (i_t v = 0; v < n; ++v) {
    double* a = columns.data() + (size_t)v * q;
    whiten(a);
    for (size_t r = 0; r < q; ++r) norm[v] += a[r] * a[r];
    if (!std::isfinite(norm[v])) return;
    if (norm[v] > 0) {
      active.push_back(v);
      x[v] = std::clamp(0.0, lower[v], upper[v]);
    }
  }
  if (active.empty() || expired()) return;

  // Cache only Gram rows actually used by coordinate and exchange moves; a full C'C matrix would
  // be quadratic in the number of columns.
  const size_t cache_size = std::min<size_t>(256, std::max<size_t>(1, (2u << 20) / (size_t)n));
  std::vector<std::vector<double>> cache(cache_size);
  std::vector<i_t> cache_column(cache_size, -1), column_slot(n, -1);
  std::vector<uint64_t> age(cache_size, 0);
  uint64_t clock = 0;
  auto gram_row = [&](i_t u) -> const double* {
    i_t slot = column_slot[u];
    if (slot < 0) {
      slot = std::min_element(age.begin(), age.end()) - age.begin();
      if (cache_column[slot] >= 0) column_slot[cache_column[slot]] = -1;
      cache_column[slot] = u;
      column_slot[u] = slot;
      auto& row = cache[slot];
      row.assign(n, 0);
      const double* a = columns.data() + (size_t)u * q;
      for (i_t v : active) {
        const double* b = columns.data() + (size_t)v * q;
        double cross = 0;
        for (size_t r = 0; r < q; ++r) cross += a[r] * b[r];
        row[v] = cross;
      }
      row[u] = norm[u];
    }
    age[slot] = ++clock;
    return cache[slot].data();
  };
  std::vector<double> residual_gradient(n, 0), multiplier_gradient(n, 0);
  auto update_gradient = [&](i_t u, double delta) {
    const double* gram = gram_row(u);
    for (i_t v : active) residual_gradient[v] += delta * gram[v];
  };
  auto refresh = [&] {
    for (size_t r = 0; r < q; ++r) {
      long double residual = -(long double)p.cstr_lb[rows[r]];
      for (i_t k = p.offsets[rows[r]]; k < p.offsets[rows[r] + 1]; ++k)
        residual += (long double)p.coefficients[k] * x[p.variables[k]];
      error[r] = (double)residual * scale[r];
    }
    whiten(error.data());
    for (i_t v : active) {
      const double* a = columns.data() + (size_t)v * q;
      double primal = 0, dual = 0;
      for (size_t r = 0; r < q; ++r) {
        primal += a[r] * error[r];
        dual += a[r] * multiplier[r];
      }
      residual_gradient[v] = primal;
      multiplier_gradient[v] = dual;
    }
  };

  recompute_lhs(c);
  std::vector<f_t> best(c.h_assignment.begin(), c.h_assignment.end());
  i_t best_count = c.violated_constraints.size();
  f_t best_severity = -c.total_violations;
  double best_metric = std::numeric_limits<double>::infinity();
  cuopt::pcgenerator_t rng((uint64_t)c.settings.seed);
  int stalls = 0;
  refresh();
  for (int iteration = 0; !expired(); ++iteration) {
    if (iteration && iteration % 256 == 0) refresh();
    double metric = 0;
    for (double e : error) metric += e * e;
    if (!std::isfinite(metric)) break;
    if (metric < best_metric) {
      best_metric = metric;
      for (i_t v = 0; v < n; ++v) c.h_assignment[v] = (f_t)x[v];
      recompute_lhs(c);
      const i_t count = c.violated_constraints.size();
      const f_t severity = -c.total_violations;
      if (count < best_count || (count == best_count && severity < best_severity)) {
        best_count = count;
        best_severity = severity;
        best.assign(c.h_assignment.begin(), c.h_assignment.end());
      }
      if (!count && check_variable_feasibility<i_t, f_t>(c)) {
        best.assign(c.h_assignment.begin(), c.h_assignment.end());
        break;
      }
    }
    if (metric < 1e-26) break;

    i_t chosen = -1;
    double chosen_delta = 0, improvement = -1e-12;
    for (i_t v : active) {
      const double g = residual_gradient[v] + multiplier_gradient[v];
      gradient[v] = g;
      double delta = -g / norm[v];
      if (integer[v]) delta = std::round(delta);
      delta = std::clamp(delta, lower[v] - x[v], upper[v] - x[v]);
      const double cost = delta * (2 * g + norm[v] * delta);
      if (std::isfinite(delta) && cost < improvement) {
        improvement = cost;
        chosen = v;
        chosen_delta = delta;
      }
    }
    if (chosen >= 0) {
      x[chosen] += chosen_delta;
      const double* a = columns.data() + (size_t)chosen * q;
      for (size_t r = 0; r < q; ++r) error[r] += chosen_delta * a[r];
      update_gradient(chosen, chosen_delta);
      continue;
    }

    std::vector<i_t> donors;
    for (i_t v : active)
      if (integer[v] && x[v] - 1 >= lower[v]) donors.push_back(v);
    i_t donor = -1, receiver = -1;
    const size_t draws = std::min<size_t>(12, donors.size());
    for (size_t k = 0; k < draws && !expired(); ++k) {
      const size_t j = k + rng.next_u32() % (donors.size() - k);
      std::swap(donors[k], donors[j]);
      const i_t u = donors[k];
      const double* gram = gram_row(u);
      const double removal = norm[u] - 2 * gradient[u];
      for (i_t v : active) {
        if (v == u || !integer[v] || x[v] + 1 > upper[v]) continue;
        const double cost = removal + norm[v] + 2 * (gradient[v] - gram[v]);
        if (cost < improvement) {
          improvement = cost;
          donor = u;
          receiver = v;
        }
      }
      if (donor >= 0) break;
    }
    if (donor >= 0) {
      x[donor] -= 1;
      x[receiver] += 1;
      const double* a = columns.data() + (size_t)donor * q;
      const double* b = columns.data() + (size_t)receiver * q;
      for (size_t r = 0; r < q; ++r) error[r] += b[r] - a[r];
      update_gradient(donor, -1);
      update_gradient(receiver, 1);
    } else {
      ++stalls;
      for (size_t r = 0; r < q; ++r) {
        multiplier[r] += 0.1 * error[r];
        if (stalls % 100 == 0) multiplier[r] *= 0.5;
      }
      for (i_t v : active) {
        multiplier_gradient[v] += 0.1 * residual_gradient[v];
        if (stalls % 100 == 0) multiplier_gradient[v] *= 0.5;
      }
    }
  }

  std::copy(best.begin(), best.end(), c.h_assignment.begin());
  recompute_lhs(c);
  c.h_best_assignment = c.h_assignment;
  if (c.violated_constraints.empty() && check_variable_feasibility<i_t, f_t>(c)) {
    c.h_best_objective = c.h_incumbent_objective - c.settings.parameters.breakthrough_move_epsilon;
    c.feasible_found = true;
    report_cpu_incumbent(c);
    if (c.shared_incumbent)
      c.shared_incumbent->publish(c.h_incumbent_objective,
                                  c.get_user_objective(c.h_incumbent_objective),
                                  c.h_assignment);
  }
}

template <typename i_t, typename f_t>
void apply_structural_completion_seed(fj_cpu_climber_t<i_t, f_t>& fj_cpu)
{
  apply_lock_weighted_seed<i_t, f_t>(fj_cpu);
  apply_exact_k_seed<i_t, f_t>(fj_cpu);
  apply_greedy_covering_seed<i_t, f_t>(fj_cpu);
  repair_difficult_anchor<i_t, f_t>(fj_cpu);
}

}  // namespace cuopt::mathematical_optimization::mip


