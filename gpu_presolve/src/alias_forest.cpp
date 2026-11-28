/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "integral.hpp"

#include <gprs/assert.hpp>
#include <gprs/postsolve.hpp>

#include <cmath>
#include <utility>

namespace gprs {

namespace {

int32_t find_root(std::vector<int32_t>& link, int32_t col)
{
  while (link[col] != col) {
    link[col] = link[link[col]];
    col       = link[col];
  }
  return col;
}

bool usable(double scale, double offset)
{
  return std::isfinite(scale) && scale != 0. && std::isfinite(offset);
}

// x[child] = scale * x[parent] + offset.
struct edge_map_t {
  bool admissible;
  double scale;
  double offset;
};

edge_map_t read_edge(const state_t& st, const alias_edge_t& edge, bool child_is_a)
{
  int32_t child      = child_is_a ? edge.col_a : edge.col_b;
  int32_t parent     = child_is_a ? edge.col_b : edge.col_a;
  double child_coef  = child_is_a ? edge.a : edge.b;
  double parent_coef = child_is_a ? edge.b : edge.a;
  edge_map_t map;
  map.scale  = -parent_coef / child_coef;
  map.offset = edge.rhs / child_coef;
  if (st.is_integer(child)) {
    double scale = std::round(map.scale), offset = std::round(map.offset);
    // The rounded map leaves (child_coef * scale + parent_coef) * x[parent] =
    // rhs - child_coef * offset of the row, which has to be 0 = 0 in the units of the row too: a
    // large child_coef turns a deviation below tol.zero into a violated row, and so does a
    // parent that reaches far (the drift). The child leaves the model, so the units of the row
    // are enough.
    integral_form_t form;
    integral_term(
      st.tol, child_coef, parent_coef, st.col_lower[parent], st.col_upper[parent], form);
    integral_side(st.tol, child_coef, edge.rhs, form);
    map.admissible =
      st.is_integer(parent) && usable(scale, offset) && form.integral &&
      drift_within(st.tol, st.tol.integral_drift_share, child_coef, form, drift_units_t::row);
    map.scale  = scale;
    map.offset = offset;
  } else {
    map.admissible = usable(map.scale, map.offset);
  }
  return map;
}

}  // namespace

void alias_forest_t::init(int32_t ncols)
{
  gprs_assert(ncols >= 0);
  link.assign(ncols, -1);
  position.assign(ncols, -1);
  touched.clear();
  edges.clear();
  set_size.clear();
  depth.clear();
}

bool alias_forest_t::add_edge(const alias_edge_t& edge)
{
  gprs_assert(edge.col_a >= 0 && (size_t)edge.col_a < link.size());
  gprs_assert(edge.col_b >= 0 && (size_t)edge.col_b < link.size());
  gprs_assert(edge.col_a != edge.col_b && edge.row >= 0);
  gprs_assert(std::isfinite(edge.a) && edge.a != 0. && std::isfinite(edge.b) && edge.b != 0.);
  gprs_assert(std::isfinite(edge.rhs));
  for (int32_t col : {edge.col_a, edge.col_b}) {
    if (link[col] != -1) { continue; }
    link[col]     = col;
    position[col] = touched.size();
    touched.push_back(col);
    set_size.push_back(1);
  }
  int32_t root_a = find_root(link, edge.col_a), root_b = find_root(link, edge.col_b);
  if (root_a == root_b) { return false; }
  if (set_size[position[root_a]] < set_size[position[root_b]]) { std::swap(root_a, root_b); }
  link[root_b] = root_a;
  set_size[position[root_a]] += set_size[position[root_b]];
  edges.push_back(edge);
  return true;
}

void alias_forest_t::freeze(const state_t& st, std::vector<alias_map_t>& maps)
{
  gprs_assert(link.size() == (size_t)st.ncols);
  maps.clear();
  int32_t n = touched.size(), m = edges.size();

  // Neighbour k of a column is the code 2 * edge + side: the column is col_a of the edge for
  // side 0 and col_b for side 1. code ^ 1 is the same edge at the other column.
  std::vector<int32_t> first(n + 1, 0), neighbour(2 * (size_t)m);
  for (const alias_edge_t& edge : edges) {
    ++first[position[edge.col_a] + 1];
    ++first[position[edge.col_b] + 1];
  }
  for (int32_t u = 0; u < n; ++u) {
    gprs_assert(st.is_live(touched[u]));
    first[u + 1] += first[u];
  }
  // admissible[code]: the edge with the column of the code as the child.
  std::vector<uint8_t> admissible(2 * (size_t)m);
  {
    std::vector<int32_t> fill(first.begin(), first.end() - 1);
    for (int32_t e = 0; e < m; ++e) {
      neighbour[fill[position[edges[e].col_a]]++] = 2 * e;
      neighbour[fill[position[edges[e].col_b]]++] = 2 * e + 1;
      admissible[2 * e]                           = read_edge(st, edges[e], true).admissible;
      admissible[2 * e + 1]                       = read_edge(st, edges[e], false).admissible;
    }
  }
  auto column_at = [&](int32_t code) {
    return position[code & 1 ? edges[code >> 1].col_b : edges[code >> 1].col_a];
  };

  std::vector<uint8_t> seen(n, 0);
  // Code of the edge towards the parent in the current walk, at the column; -1 for a root.
  std::vector<int32_t> up(n);
  std::vector<int32_t> score(n);
  std::vector<double> scale(n), offset(n);
  std::vector<int32_t> walk, roots;
  depth.assign(n, 0);
  for (int32_t start = 0; start < n; ++start) {
    if (seen[start]) { continue; }

    // score[u]: admissible edges of the tree when it hangs from u, less that number for the
    // first column. Along an edge only that edge changes direction.
    walk.assign(1, start);
    seen[start]  = 1;
    up[start]    = -1;
    score[start] = 0;
    for (size_t head = 0; head < walk.size(); ++head) {
      int32_t u = walk[head];
      for (int32_t k = first[u]; k < first[u + 1]; ++k) {
        int32_t code = neighbour[k];
        if (code == up[u]) { continue; }
        int32_t v = column_at(code ^ 1);
        seen[v]   = 1;
        up[v]     = code ^ 1;
        score[v]  = score[u] - admissible[code ^ 1] + admissible[code];
        walk.push_back(v);
      }
    }
    int32_t best = start;
    for (int32_t v : walk) {
      int32_t col = touched[v], best_col = touched[best];
      if (st.is_integer(col) != st.is_integer(best_col)) {
        if (st.is_integer(col)) { best = v; }
      } else if (score[v] != score[best]) {
        if (score[v] > score[best]) { best = v; }
      } else if (st.csc.size[col] != st.csc.size[best_col]) {
        if (st.csc.size[col] > st.csc.size[best_col]) { best = v; }
      } else if (col < best_col) {
        best = v;
      }
    }

    roots.assign(1, best);
    up[best] = -1;
    for (size_t r = 0; r < roots.size(); ++r) {
      int32_t root = roots[r];
      scale[root]  = 1.;
      offset[root] = 0.;
      walk.assign(1, root);
      for (size_t head = 0; head < walk.size(); ++head) {
        int32_t u = walk[head];
        for (int32_t k = first[u]; k < first[u + 1]; ++k) {
          int32_t code = neighbour[k];
          if (code == up[u]) { continue; }
          int32_t v                = column_at(code ^ 1);
          const alias_edge_t& edge = edges[code >> 1];
          edge_map_t map           = read_edge(st, edge, code & 1);
          up[v]                    = code ^ 1;
          scale[v]                 = map.scale * scale[u];
          offset[v]                = map.scale * offset[u] + map.offset;
          if (!map.admissible || !usable(scale[v], offset[v])) {
            roots.push_back(v);
            continue;
          }
          maps.push_back(alias_map_t{touched[v], touched[root], edge.row, scale[v], offset[v]});
          depth[v] = depth[u] + 1;
          walk.push_back(v);
        }
      }
    }
  }
}

void alias_forest_t::clear()
{
  for (int32_t col : touched) {
    link[col]     = -1;
    position[col] = -1;
  }
  touched.clear();
  edges.clear();
  set_size.clear();
}

}  // namespace gprs
