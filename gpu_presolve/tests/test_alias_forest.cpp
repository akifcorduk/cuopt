/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "harness.hpp"

#include <gprs/postsolve.hpp>

#include <cmath>
#include <cstdint>
#include <deque>
#include <tuple>
#include <vector>

using namespace gprs;

namespace {

// splitmix64.
struct rng_t {
  uint64_t state;

  uint64_t next()
  {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  int32_t uniform(int32_t lo, int32_t hi)
  {
    return lo + (int32_t)(next() % (uint64_t)(hi - lo + 1));
  }
  bool chance(double probability) { return (next() >> 11) * 0x1p-53 < probability; }
};

// Column j has entries[j] entries, each in a free singleton row of its own.
model_t column_model(const std::vector<uint8_t>& col_type, const std::vector<int32_t>& entries)
{
  model_t m;
  m.ncols = col_type.size();
  m.row_start.push_back(0);
  for (int32_t j = 0; j < m.ncols; ++j) {
    for (int32_t k = 0; k < entries[j]; ++k) {
      m.col_index.push_back(j);
      m.value.push_back(1.);
      m.row_start.push_back(m.col_index.size());
    }
  }
  m.nrows = m.row_start.size() - 1;
  m.row_lower.assign(m.nrows, -inf);
  m.row_upper.assign(m.nrows, inf);
  m.col_lower.assign(m.ncols, -inf);
  m.col_upper.assign(m.ncols, inf);
  m.cost.assign(m.ncols, 0.);
  m.col_type = col_type;
  return m;
}

state_t column_state(const std::vector<uint8_t>& col_type, const std::vector<int32_t>& entries)
{
  return state_t(column_model(col_type, entries), tolerances_t{});
}

void check_maps(const std::vector<alias_map_t>& maps, const std::vector<alias_map_t>& expected)
{
  GPRS_REQUIRE(maps.size() == expected.size());
  for (size_t k = 0; k < maps.size(); ++k) {
    GPRS_CHECK_EQ(maps[k].col, expected[k].col);
    GPRS_CHECK_EQ(maps[k].root, expected[k].root);
    GPRS_CHECK_EQ(maps[k].row, expected[k].row);
    GPRS_CHECK_EQ(maps[k].scale, expected[k].scale);
    GPRS_CHECK_EQ(maps[k].offset, expected[k].offset);
  }
}

std::vector<alias_map_t> frozen(const state_t& st, const std::vector<alias_edge_t>& edges)
{
  alias_forest_t forest;
  forest.init(st.ncols);
  for (const alias_edge_t& edge : edges) {
    GPRS_CHECK(forest.add_edge(edge));
  }
  std::vector<alias_map_t> maps = {alias_map_t{-1, -1, -1, 0., 0.}};
  forest.freeze(st, maps);
  return maps;
}

// ---- Reference: the header's rules without any shortcut ----

// x[child] = scale * x[other column] + offset; false when the header calls the edge
// inadmissible.
bool reference_edge(
  const state_t& st, const alias_edge_t& edge, int32_t child, double& scale, double& offset)
{
  int32_t parent  = edge.col_a == child ? edge.col_b : edge.col_a;
  double own      = edge.col_a == child ? edge.a : edge.b;
  double other    = edge.col_a == child ? edge.b : edge.a;
  scale           = -other / own;
  offset          = edge.rhs / own;
  bool admissible = true;
  if (st.is_integer(child)) {
    admissible = st.is_integer(parent) && std::fabs(scale - std::round(scale)) <= st.tol.zero &&
                 std::fabs(offset - std::round(offset)) <= st.tol.zero;
    scale  = std::round(scale);
    offset = std::round(offset);
    // What the rounded map leaves of the row: coefficient * x[parent] = side.
    double coefficient = own * scale + other, side = edge.rhs - own * offset;
    admissible =
      admissible && std::fabs(coefficient) <= st.tol.zero && std::fabs(side) <= st.tol.feasibility;
  }
  return admissible && std::isfinite(scale) && scale != 0. && std::isfinite(offset);
}

struct reference_t {
  const state_t& st;
  const std::vector<alias_edge_t>& edges;
  std::vector<std::vector<int32_t>> incident;  // per column, in the order the edges were added

  reference_t(const state_t& state, const std::vector<alias_edge_t>& tree_edges)
    : st(state), edges(tree_edges), incident(state.ncols)
  {
    for (size_t e = 0; e < edges.size(); ++e) {
      incident[edges[e].col_a].push_back(e);
      incident[edges[e].col_b].push_back(e);
    }
  }

  int32_t other(int32_t e, int32_t col) const
  {
    return edges[e].col_a == col ? edges[e].col_b : edges[e].col_a;
  }

  void collect(int32_t col, int32_t from, std::vector<int32_t>& tree) const
  {
    tree.push_back(col);
    for (int32_t e : incident[col]) {
      if (e != from) { collect(other(e, col), e, tree); }
    }
  }

  // Admissible edges below col when the tree hangs from it.
  int32_t admissible_below(int32_t col, int32_t from) const
  {
    int32_t count = 0;
    for (int32_t e : incident[col]) {
      if (e == from) { continue; }
      double scale, offset;
      count += reference_edge(st, edges[e], other(e, col), scale, offset);
      count += admissible_below(other(e, col), e);
    }
    return count;
  }

  std::vector<alias_map_t> freeze(const std::vector<int32_t>& touched) const
  {
    struct visit_t {
      int32_t col;
      int32_t from;
      double scale;
      double offset;
    };
    std::vector<alias_map_t> maps;
    std::vector<uint8_t> done(st.ncols, 0);
    for (int32_t first : touched) {
      if (done[first]) { continue; }
      std::vector<int32_t> tree;
      collect(first, -1, tree);
      int32_t root = -1;
      std::tuple<int32_t, int32_t, int32_t, int32_t> best;
      for (int32_t col : tree) {
        done[col] = 1;
        // Smaller is better.
        std::tuple<int32_t, int32_t, int32_t, int32_t> key(
          !st.is_integer(col), -admissible_below(col, -1), -st.csc.size[col], col);
        if (root == -1 || key < best) {
          root = col;
          best = key;
        }
      }
      std::deque<std::pair<int32_t, int32_t>> roots = {{root, -1}};
      for (; !roots.empty(); roots.pop_front()) {
        std::deque<visit_t> queue = {{roots.front().first, roots.front().second, 1., 0.}};
        for (; !queue.empty(); queue.pop_front()) {
          visit_t u = queue.front();
          for (int32_t e : incident[u.col]) {
            if (e == u.from) { continue; }
            int32_t child = other(e, u.col);
            double scale, offset;
            bool admissible = reference_edge(st, edges[e], child, scale, offset);
            offset          = scale * u.offset + offset;
            scale           = scale * u.scale;
            if (!admissible || !std::isfinite(scale) || scale == 0. || !std::isfinite(offset)) {
              roots.push_back({child, e});
              continue;
            }
            maps.push_back(alias_map_t{child, roots.front().first, edges[e].row, scale, offset});
            queue.push_back({child, e, scale, offset});
          }
        }
      }
    }
    return maps;
  }
};

// What a caller relies on, tested without the reference: the maps go on the postsolve stack as
// alias records in the order they come (push_alias asserts an integer column's map), and the
// point that postsolve returns satisfies the row of every map. Parents come first, the maps of
// a root are contiguous, and a root has no map.
void check_map_properties(const state_t& st,
                          const alias_forest_t& forest,
                          const std::vector<alias_map_t>& maps,
                          int32_t nrows,
                          rng_t& rng)
{
  model_t shell;
  shell.nrows = nrows;
  shell.ncols = st.ncols;
  for (int32_t j = 0; j < st.ncols; ++j) {
    shell.col_type.push_back(st.is_integer(j));
  }
  postsolve_t ps;
  ps.init(shell);

  std::vector<int32_t> map_of(st.ncols, -1);
  for (size_t k = 0; k < maps.size(); ++k) {
    const alias_map_t& map = maps[k];
    GPRS_REQUIRE(forest.position[map.col] >= 0 && forest.position[map.root] >= 0);
    GPRS_REQUIRE(map_of[map.col] == -1);
    map_of[map.col] = k;
    GPRS_REQUIRE(std::isfinite(map.scale) && map.scale != 0. && std::isfinite(map.offset));
    if (st.is_integer(map.col)) {
      GPRS_REQUIRE(st.is_integer(map.root));
      GPRS_REQUIRE(map.scale == std::round(map.scale) && map.offset == std::round(map.offset));
    }
    if (k > 0 && maps[k - 1].root != map.root) {
      for (size_t l = 0; l + 1 < k; ++l) {
        GPRS_CHECK(maps[l].root != map.root);
      }
    }
  }
  std::vector<double> x_reduced;
  for (int32_t j = 0; j < st.ncols; ++j) {
    if (map_of[j] != -1) { continue; }
    ps.col_origin.push_back(j);
    x_reduced.push_back(st.is_integer(j) ? rng.uniform(-9, 9) : rng.uniform(-24, 24) / 8.);
  }
  for (const alias_map_t& map : maps) {
    GPRS_REQUIRE(map_of[map.root] == -1);
    ps.push_alias(map.col, map.root, map.row, map.scale, map.offset);
  }
  std::vector<double> x;
  ps.postsolve(x_reduced, x);
  // size[col] bounds the terms that x[col] is the sum of: the rounding error of an equation
  // is relative to them, not to x[col], which can be a difference of large terms.
  std::vector<double> size(st.ncols);
  for (int32_t j = 0; j < st.ncols; ++j) {
    size[j] = std::fabs(x[j]);
    if (map_of[j] == -1) { continue; }
    const alias_map_t& map = maps[map_of[j]];
    size[j]                = std::fabs(map.scale * x[map.root]) + std::fabs(map.offset);
    if (st.is_integer(j)) { GPRS_CHECK_EQ(x[j], std::round(x[j])); }
  }
  for (size_t k = 0; k < maps.size(); ++k) {
    const alias_map_t& map   = maps[k];
    const alias_edge_t* edge = nullptr;
    for (const alias_edge_t& candidate : forest.edges) {
      if (candidate.row == map.row) { edge = &candidate; }
    }
    GPRS_REQUIRE(edge != nullptr);
    GPRS_CHECK(edge->col_a == map.col || edge->col_b == map.col);
    int32_t parent = edge->col_a == map.col ? edge->col_b : edge->col_a;
    // The parent is the root or has its own map to the same root earlier in the list.
    if (parent != map.root) {
      GPRS_REQUIRE(map_of[parent] != -1);
      GPRS_CHECK((size_t)map_of[parent] < k);
      GPRS_CHECK_EQ(maps[map_of[parent]].root, map.root);
    }
    double left  = edge->a * x[edge->col_a] + edge->b * x[edge->col_b];
    double range = std::fabs(edge->a) * size[edge->col_a] + std::fabs(edge->b) * size[edge->col_b] +
                   std::fabs(edge->rhs) + 1.;
    GPRS_CHECK(std::fabs(left - edge->rhs) <= 1e-9 * range);
  }
}

}  // namespace

GPRS_TEST(alias_init_and_touched_order)
{
  alias_forest_t forest;
  forest.init(6);
  GPRS_CHECK(forest.link == std::vector<int32_t>(6, -1));
  GPRS_CHECK(forest.position == std::vector<int32_t>(6, -1));
  GPRS_CHECK(forest.touched.empty() && forest.edges.empty() && forest.set_size.empty());

  GPRS_CHECK(forest.add_edge(alias_edge_t{10, 4, 2, 1., -1., 0.}));
  GPRS_CHECK(forest.add_edge(alias_edge_t{11, 5, 4, 2., 1., 3.}));
  GPRS_CHECK(forest.touched == std::vector<int32_t>({4, 2, 5}));
  GPRS_CHECK(forest.position == std::vector<int32_t>({-1, -1, 1, -1, 0, 2}));
  GPRS_CHECK_EQ(forest.link[0], -1);
  GPRS_CHECK_EQ(forest.link[1], -1);
  GPRS_CHECK_EQ(forest.link[3], -1);
  GPRS_CHECK(forest.link[2] != -1 && forest.link[4] != -1 && forest.link[5] != -1);
  GPRS_REQUIRE(forest.edges.size() == 2);
  GPRS_CHECK_EQ(forest.edges[0].row, 10);
  GPRS_CHECK_EQ(forest.edges[1].row, 11);
  GPRS_CHECK_EQ(forest.edges[1].col_a, 5);
  GPRS_CHECK_EQ(forest.edges[1].col_b, 4);
  GPRS_CHECK_EQ(forest.edges[1].a, 2.);
  GPRS_CHECK_EQ(forest.edges[1].b, 1.);
  GPRS_CHECK_EQ(forest.edges[1].rhs, 3.);
}

GPRS_TEST(alias_add_edge_refuses_cycles)
{
  alias_forest_t forest;
  forest.init(8);
  GPRS_CHECK(forest.add_edge(alias_edge_t{0, 0, 1, 1., 1., 0.}));
  GPRS_CHECK(forest.add_edge(alias_edge_t{1, 1, 2, 1., 1., 0.}));
  // A second equation on the same two columns, and the edge that closes the triangle.
  GPRS_CHECK(!forest.add_edge(alias_edge_t{2, 1, 0, 1., 2., 0.}));
  GPRS_CHECK(!forest.add_edge(alias_edge_t{3, 2, 0, 1., 1., 0.}));
  // A second tree, joined to the first; then every further edge between the five columns closes
  // a cycle.
  GPRS_CHECK(forest.add_edge(alias_edge_t{4, 5, 6, 1., 1., 0.}));
  GPRS_CHECK(!forest.add_edge(alias_edge_t{5, 6, 5, 1., 1., 0.}));
  GPRS_CHECK(forest.add_edge(alias_edge_t{6, 6, 0, 1., 1., 0.}));
  GPRS_CHECK(!forest.add_edge(alias_edge_t{7, 5, 2, 1., 1., 0.}));
  GPRS_CHECK(!forest.add_edge(alias_edge_t{8, 1, 6, 1., 1., 0.}));
  GPRS_CHECK(forest.add_edge(alias_edge_t{9, 7, 2, 1., 1., 0.}));
  GPRS_CHECK(!forest.add_edge(alias_edge_t{10, 7, 5, 1., 1., 0.}));

  // The refused edges are not stored.
  GPRS_REQUIRE(forest.edges.size() == 5);
  const int32_t rows[] = {0, 1, 4, 6, 9};
  for (int32_t k = 0; k < 5; ++k) {
    GPRS_CHECK_EQ(forest.edges[k].row, rows[k]);
  }
  GPRS_CHECK(forest.touched == std::vector<int32_t>({0, 1, 2, 5, 6, 7}));

  // One tree of six columns: five maps, and no row of a refused edge among them.
  state_t st = column_state({0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1, 1});
  std::vector<alias_map_t> maps;
  forest.freeze(st, maps);
  GPRS_REQUIRE(maps.size() == 5);
  for (const alias_map_t& map : maps) {
    GPRS_CHECK_EQ(map.root, 0);
    GPRS_CHECK(map.row == 0 || map.row == 1 || map.row == 4 || map.row == 6 || map.row == 9);
  }
}

// x0 = 2 x1 between integers: x1 = x0 / 2 would lose that x0 is even, so x1 is the root
// whatever the entries and the indices say.
GPRS_TEST(alias_integer_pair_has_one_direction)
{
  state_t st = column_state({1, 1}, {5, 1});
  check_maps(frozen(st, {alias_edge_t{7, 0, 1, 1., -2., 0.}}), {alias_map_t{0, 1, 7, 2., 0.}});
  // The same equation written from the other column and scaled.
  check_maps(frozen(st, {alias_edge_t{7, 1, 0, 6., -3., 0.}}), {alias_map_t{0, 1, 7, 2., 0.}});
  // x1 = 2 x0 + 4: now column 0 is the root although column 1 would win on the entries.
  state_t other = column_state({1, 1}, {1, 5});
  check_maps(frozen(other, {alias_edge_t{3, 0, 1, -2., 1., 4.}}), {alias_map_t{1, 0, 3, 2., 4.}});
  // A ratio of -1 is admissible both ways: the entries decide, then the index.
  check_maps(frozen(st, {alias_edge_t{4, 0, 1, 1., 1., 3.}}), {alias_map_t{1, 0, 4, -1., 3.}});
  check_maps(frozen(other, {alias_edge_t{4, 0, 1, 1., 1., 3.}}), {alias_map_t{0, 1, 4, -1., 3.}});
  state_t tie = column_state({1, 1}, {2, 2});
  check_maps(frozen(tie, {alias_edge_t{4, 1, 0, 1., 1., 3.}}), {alias_map_t{1, 0, 4, -1., 3.}});
}

GPRS_TEST(alias_integer_edge_needs_integral_scale_and_offset)
{
  state_t st = column_state({1, 1}, {1, 1});
  // x0 = x1 + 1 / 2; 3 x0 = 2 x1; x0 = 2 x1 + 1 / 2 and x1 = x0 / 2 - 1 / 4.
  GPRS_CHECK(frozen(st, {alias_edge_t{0, 0, 1, 1., -1., 0.5}}).empty());
  GPRS_CHECK(frozen(st, {alias_edge_t{0, 0, 1, 3., -2., 0.}}).empty());
  GPRS_CHECK(frozen(st, {alias_edge_t{0, 0, 1, 1., -2., 0.5}}).empty());

  // Within tol.zero the scale and the offset are rounded.
  double zero = st.tol.zero;
  check_maps(frozen(st, {alias_edge_t{0, 0, 1, 1., -(2. + 0.5 * zero), 3. - 0.5 * zero}}),
             {alias_map_t{0, 1, 0, 2., 3.}});
  GPRS_CHECK(frozen(st, {alias_edge_t{0, 0, 1, 1., -(2. + 5. * zero), 3.}}).empty());
  GPRS_CHECK(frozen(st, {alias_edge_t{0, 0, 1, 1., -2., 3. + 5. * zero}}).empty());

  // A scale that rounds to zero is no map: x0 = 1e-10 x1 is refused, x1 = 1e10 x0 is taken,
  // also when the entries speak for column 1 as the root.
  check_maps(frozen(st, {alias_edge_t{0, 0, 1, 1e10, -1., 0.}}), {alias_map_t{1, 0, 0, 1e10, 0.}});
  state_t heavy = column_state({1, 1}, {1, 5});
  check_maps(frozen(heavy, {alias_edge_t{0, 0, 1, 1e10, -1., 0.}}),
             {alias_map_t{1, 0, 0, 1e10, 0.}});

  // The same with a coefficient below tol.zero, which the row test alone would let through:
  // x0 + 1e-10 x1 = 0 is x1 = -1e10 x0 and never x0 = 0 x1.
  check_maps(frozen(heavy, {alias_edge_t{0, 0, 1, 1., 1e-10, 0.}}),
             {alias_map_t{1, 0, 0, -1e10, 0.}});
  // The scale is tested as a ratio as well: 1e-3 x0 = (2e-3 + 5e-12) x1 is off by 5e-9 there
  // and by 5e-12 in the units of the row.
  GPRS_CHECK(frozen(st, {alias_edge_t{0, 0, 1, 1e-3, -(2e-3 + 5e-12), 0.}}).empty());

  // Integral within tol.zero, but not in the units of the row. 1e10 x0 + 1e10 x1 = 3 has the
  // offset 3e-10 and no integer solution; with x0 = -x1 the row would be off by 3.
  GPRS_CHECK(frozen(st, {alias_edge_t{0, 0, 1, 1e10, 1e10, 3.}}).empty());
  check_maps(frozen(st, {alias_edge_t{0, 0, 1, 1e10, 1e10, 0.}}), {alias_map_t{1, 0, 0, -1., 0.}});
  // 1e6 x0 = (2e6 + 1 / 2048) x1: the scale is 2 + 4.9e-10, and x0 = 2 x1 leaves x1 / 2048.
  GPRS_CHECK(frozen(st, {alias_edge_t{0, 0, 1, 1e6, -(2e6 + 0x1p-11), 0.}}).empty());
  // The side of the row is judged by the feasibility tolerance: 1e3 x0 - 1e3 x1 = 1e3 + 5e-7
  // is x0 = x1 + 1 within it; 1e4 x0 - 1e4 x1 = 1e4 + 5e-6 has the same offset and is not.
  check_maps(frozen(st, {alias_edge_t{0, 0, 1, 1e3, -1e3, 1e3 + 5e-7}}),
             {alias_map_t{1, 0, 0, 1., -1.}});
  GPRS_CHECK(frozen(st, {alias_edge_t{0, 0, 1, 1e4, -1e4, 1e4 + 5e-6}}).empty());

  // A continuous column takes any scale and offset.
  state_t continuous = column_state({0, 0}, {1, 1});
  check_maps(frozen(continuous, {alias_edge_t{0, 0, 1, 3., -2., 0.75}}),
             {alias_map_t{1, 0, 0, 1.5, -0.375}});
}

// The integer column is the root whatever the entries and the indices say, and the continuous
// column follows it with any scale.
GPRS_TEST(alias_integer_column_is_the_root)
{
  state_t st = column_state({0, 1}, {9, 1});
  check_maps(frozen(st, {alias_edge_t{2, 0, 1, 1., 3., 2.}}), {alias_map_t{0, 1, 2, -3., 2.}});
  check_maps(frozen(st, {alias_edge_t{2, 1, 0, 4., 8., 1.}}), {alias_map_t{0, 1, 2, -0.5, 0.125}});
  state_t swapped = column_state({1, 0}, {1, 9});
  check_maps(frozen(swapped, {alias_edge_t{2, 0, 1, 4., 8., 1.}}),
             {alias_map_t{1, 0, 2, -0.5, 0.125}});
}

// The type comes before the number of admissible edges. Integer x0, continuous x1 and x2 with
//   row 0: 1e-200 x1 + 1e200 x0 = 0  (no map either way: x1 = -1e400 x0),  row 1: x2 = 2 x1.
// Every root keeps one edge; column 0 is the root, row 0 is cut and column 1 roots x2 = 2 x1,
// although column 2 has the entries and would give x1 = x2 / 2.
GPRS_TEST(alias_type_comes_before_admissible_edges)
{
  state_t st = column_state({1, 0, 0}, {1, 1, 9});
  check_maps(
    frozen(st, {alias_edge_t{0, 1, 0, 1e-200, 1e200, 0.}, alias_edge_t{1, 2, 1, 1., -2., 0.}}),
    {alias_map_t{2, 1, 1, 2., 0.}});
}

// A star around column 2, all continuous: every root makes every edge admissible, columns 1 and
// 3 have the most entries, 1 is the smaller index.
//   row 10: x0 - 2 x2 = 1,  row 11: x1 + x2 = 4,  row 12: 2 x3 - x2 = 0.
GPRS_TEST(alias_root_by_entries_then_index_and_composed_maps)
{
  std::vector<alias_edge_t> edges = {alias_edge_t{10, 0, 2, 1., -2., 1.},
                                     alias_edge_t{11, 1, 2, 1., 1., 4.},
                                     alias_edge_t{12, 3, 2, 2., -1., 0.}};
  state_t st                      = column_state({0, 0, 0, 0}, {1, 4, 2, 4});
  // x2 = -x1 + 4 first, then the neighbours of column 2 in the order of their edges:
  // x0 = 2 x2 + 1 = -2 x1 + 9 and x3 = x2 / 2 = -x1 / 2 + 2.
  check_maps(frozen(st, edges),
             {alias_map_t{2, 1, 11, -1., 4.},
              alias_map_t{0, 1, 10, -2., 9.},
              alias_map_t{3, 1, 12, -0.5, 2.}});

  // More entries on column 3: x2 = 2 x3, x0 = 4 x3 + 1, x1 = -2 x3 + 4.
  state_t heavy = column_state({0, 0, 0, 0}, {1, 4, 2, 5});
  check_maps(
    frozen(heavy, edges),
    {alias_map_t{2, 3, 12, 2., 0.}, alias_map_t{0, 3, 10, 4., 1.}, alias_map_t{1, 3, 11, -2., 4.}});

  // The centre as the root: its neighbours in the order of their edges.
  state_t centre = column_state({0, 0, 0, 0}, {1, 4, 6, 5});
  check_maps(frozen(centre, edges),
             {alias_map_t{0, 2, 10, 2., 1.},
              alias_map_t{1, 2, 11, -1., 4.},
              alias_map_t{3, 2, 12, 0.5, 0.}});
}

// Integer columns, x0 = 2 x1 and x2 = 3 x1: only the middle column makes both edges
// admissible. Then x0 = 2 x1 and x1 = 2 x2: only the far end does.
GPRS_TEST(alias_root_makes_the_most_edges_admissible)
{
  state_t st = column_state({1, 1, 1}, {5, 1, 5});
  check_maps(frozen(st, {alias_edge_t{0, 0, 1, 1., -2., 0.}, alias_edge_t{1, 2, 1, 1., -3., 0.}}),
             {alias_map_t{0, 1, 0, 2., 0.}, alias_map_t{2, 1, 1, 3., 0.}});
  check_maps(frozen(st, {alias_edge_t{0, 0, 1, 1., -2., 0.}, alias_edge_t{1, 1, 2, 1., -2., 6.}}),
             {alias_map_t{1, 2, 1, 2., 6.}, alias_map_t{0, 2, 0, 4., 12.}});
  state_t light = column_state({1, 1, 1}, {5, 5, 1});
  check_maps(
    frozen(light, {alias_edge_t{0, 0, 1, 1., -2., 0.}, alias_edge_t{1, 1, 2, 1., -2., 6.}}),
    {alias_map_t{1, 2, 1, 2., 6.}, alias_map_t{0, 2, 0, 4., 12.}});
}

// A path  i0 - c1 - i2 - i3  (i integer, c continuous):
//   row 20: x1 = x0 / 2,  row 21: x2 = 4 x1,  row 22: x3 = x2 + 1.
// An integer column cannot follow the continuous one, so whichever end is the root, one of the
// edges at column 1 is cut; every integer root keeps two edges.
GPRS_TEST(alias_cut_edge_starts_a_subtree)
{
  std::vector<alias_edge_t> edges = {alias_edge_t{20, 1, 0, 1., -0.5, 0.},
                                     alias_edge_t{21, 2, 1, 1., -4., 0.},
                                     alias_edge_t{22, 3, 2, 1., -1., 1.}};
  // Root 0 by the entries: x1 = x0 / 2; row 21 is cut and column 2 roots x3 = x2 + 1, after
  // the tree of column 0.
  state_t st = column_state({1, 0, 1, 1}, {3, 9, 1, 2});
  check_maps(frozen(st, edges), {alias_map_t{1, 0, 20, 0.5, 0.}, alias_map_t{3, 2, 22, 1., 1.}});

  // Root 3 by the entries: x2 = x3 - 1, x1 = x2 / 4 = x3 / 4 - 1 / 4; row 20 is cut.
  state_t other = column_state({1, 0, 1, 1}, {1, 9, 1, 5});
  check_maps(frozen(other, edges),
             {alias_map_t{2, 3, 22, 1., -1.}, alias_map_t{1, 3, 21, 0.25, -0.25}});

  // Equal entries: the smallest index among the integer columns.
  state_t tie = column_state({1, 0, 1, 1}, {2, 9, 2, 2});
  check_maps(frozen(tie, edges), {alias_map_t{1, 0, 20, 0.5, 0.}, alias_map_t{3, 2, 22, 1., 1.}});
}

// Two cut edges below one root: their subtrees are walked after the tree, in the order of the
// cuts, and the subtree behind a cut inside a subtree after those. The cut rows have no integer
// map in either direction, so every root keeps four edges and the entries pick column 0.
//   row 0: x1 = x0 + 1 / 2 (cut),  row 1: x2 = x0 + 1 / 2 (cut),  row 2: x3 = x0 + 1,
//   row 3: x4 = x1 + 2,  row 4: x5 = x2 - 1,  row 5: x6 = x4 + 1 / 2 (cut),  row 6: x7 = x6 + 3.
GPRS_TEST(alias_cut_subtrees_come_after_the_tree)
{
  std::vector<alias_edge_t> edges = {alias_edge_t{0, 1, 0, 1., -1., 0.5},
                                     alias_edge_t{1, 2, 0, 1., -1., 0.5},
                                     alias_edge_t{2, 3, 0, 1., -1., 1.},
                                     alias_edge_t{3, 4, 1, 1., -1., 2.},
                                     alias_edge_t{4, 5, 2, 1., -1., -1.},
                                     alias_edge_t{5, 6, 4, 1., -1., 0.5},
                                     alias_edge_t{6, 7, 6, 1., -1., 3.}};
  state_t st = column_state({1, 1, 1, 1, 1, 1, 1, 1}, {9, 1, 1, 1, 1, 1, 1, 1});
  check_maps(frozen(st, edges),
             {alias_map_t{3, 0, 2, 1., 1.},
              alias_map_t{4, 1, 3, 1., 2.},
              alias_map_t{5, 2, 4, 1., -1.},
              alias_map_t{7, 6, 6, 1., 3.}});
}

// Finite scales whose product is not: the composed map is refused where it stops being a
// number, and the column starts its own tree.
GPRS_TEST(alias_composed_scale_must_stay_finite_and_nonzero)
{
  state_t st = column_state({0, 0, 0, 0}, {5, 1, 1, 1});
  // x1 = 1e200 x0, x2 = 1e200 x1, x3 = 2 x2.
  check_maps(frozen(st,
                    {alias_edge_t{0, 1, 0, 1., -1e200, 0.},
                     alias_edge_t{1, 2, 1, 1., -1e200, 0.},
                     alias_edge_t{2, 3, 2, 1., -2., 0.}}),
             {alias_map_t{1, 0, 0, 1e200, 0.}, alias_map_t{3, 2, 2, 2., 0.}});
  // x1 = 1e-200 x0, x2 = 1e-200 x1.
  check_maps(
    frozen(st, {alias_edge_t{0, 1, 0, 1e200, -1., 0.}, alias_edge_t{1, 2, 1, 1e200, -1., 0.}}),
    {alias_map_t{1, 0, 0, 1. / 1e200, 0.}});
  // An offset that overflows on the way: x1 = x0 + 1e200, x2 = 1e200 x1.
  check_maps(
    frozen(st, {alias_edge_t{0, 1, 0, 1., -1., 1e200}, alias_edge_t{1, 2, 1, 1., -1e200, 0.}}),
    {alias_map_t{1, 0, 0, 1., 1e200}});
}

GPRS_TEST(alias_clear_ends_the_batch)
{
  state_t st                       = column_state({0, 1, 0, 1, 0, 0}, {1, 2, 3, 1, 2, 3});
  std::vector<alias_edge_t> first  = {alias_edge_t{0, 0, 1, 1., -2., 0.},
                                      alias_edge_t{1, 2, 3, 1., 1., 1.},
                                      alias_edge_t{2, 1, 3, 1., -1., 0.}};
  std::vector<alias_edge_t> second = {alias_edge_t{3, 4, 0, 2., 1., 0.},
                                      alias_edge_t{4, 1, 0, 1., 3., 2.},
                                      alias_edge_t{5, 5, 4, 1., -1., 1.}};
  alias_forest_t forest;
  forest.init(st.ncols);
  std::vector<alias_map_t> maps;
  for (const alias_edge_t& edge : first) {
    GPRS_CHECK(forest.add_edge(edge));
  }
  forest.freeze(st, maps);
  check_maps(maps, frozen(st, first));
  // freeze does not end the batch: a second call returns the same maps.
  std::vector<alias_map_t> again;
  forest.freeze(st, again);
  check_maps(again, maps);

  forest.clear();
  GPRS_CHECK(forest.link == std::vector<int32_t>(6, -1));
  GPRS_CHECK(forest.position == std::vector<int32_t>(6, -1));
  GPRS_CHECK(forest.touched.empty() && forest.edges.empty() && forest.set_size.empty());
  forest.freeze(st, maps);
  GPRS_CHECK(maps.empty());

  // Columns 0 and 1 were connected in the first batch and are not in the second.
  for (const alias_edge_t& edge : second) {
    GPRS_CHECK(forest.add_edge(edge));
  }
  GPRS_CHECK(forest.touched == std::vector<int32_t>({4, 0, 1, 5}));
  forest.freeze(st, maps);
  check_maps(maps, frozen(st, second));
  GPRS_CHECK(!maps.empty());
}

GPRS_TEST(alias_random_forests_against_reference)
{
  const double coefficient[] = {1., -1., 2., -2., 3., 0.5, -0.5, 4., 1.5, 1e10};
  const double rhs[]         = {0., 0., 1., -2., 3., 0.5, 6.};
  int64_t n_maps = 0, n_cut = 0, n_refused = 0, n_integer_maps = 0;
  for (uint64_t seed = 1; seed <= 1500; ++seed) {
    rng_t rng{seed};
    int32_t ncols       = rng.uniform(2, 14);
    double p_integer    = 0.5 * rng.uniform(0, 2);
    int32_t max_entries = rng.uniform(0, 3);
    std::vector<uint8_t> col_type(ncols);
    std::vector<int32_t> entries(ncols);
    for (int32_t j = 0; j < ncols; ++j) {
      col_type[j] = rng.chance(p_integer);
      entries[j]  = rng.uniform(0, max_entries);
    }
    state_t st = column_state(col_type, entries);
    alias_forest_t forest;
    forest.init(ncols);

    int32_t next_row = 0;
    for (int32_t batch = 0; batch < 2; ++batch) {
      // Naive connectivity: label[j] is the smallest column connected to j.
      std::vector<int32_t> label(ncols), touched;
      std::vector<uint8_t> in_batch(ncols, 0);
      for (int32_t j = 0; j < ncols; ++j) {
        label[j] = j;
      }
      std::vector<alias_edge_t> tree_edges;
      int32_t attempts = rng.uniform(1, 2 * ncols);
      for (int32_t k = 0; k < attempts; ++k) {
        alias_edge_t edge;
        edge.row   = next_row++;
        edge.col_a = rng.uniform(0, ncols - 1);
        edge.col_b = rng.uniform(0, ncols - 2);
        edge.col_b += edge.col_b >= edge.col_a;
        edge.a    = coefficient[rng.uniform(0, 9)];
        edge.b    = coefficient[rng.uniform(0, 9)];
        edge.rhs  = rhs[rng.uniform(0, 6)];
        bool tree = label[edge.col_a] != label[edge.col_b];
        GPRS_CHECK_EQ(forest.add_edge(edge), tree);
        n_refused += !tree;
        for (int32_t col : {edge.col_a, edge.col_b}) {
          if (!in_batch[col]) { touched.push_back(col); }
          in_batch[col] = 1;
        }
        if (!tree) { continue; }
        tree_edges.push_back(edge);
        int32_t from = label[edge.col_b], to = label[edge.col_a];
        for (int32_t j = 0; j < ncols; ++j) {
          if (label[j] == from) { label[j] = to; }
        }
      }
      GPRS_CHECK(forest.touched == touched);
      GPRS_REQUIRE(forest.edges.size() == tree_edges.size());

      std::vector<alias_map_t> maps;
      forest.freeze(st, maps);
      reference_t reference(st, tree_edges);
      check_maps(maps, reference.freeze(touched));
      check_map_properties(st, forest, maps, next_row, rng);
      n_maps += maps.size();
      n_cut += tree_edges.size() - maps.size();
      for (const alias_map_t& map : maps) {
        n_integer_maps += st.is_integer(map.col);
      }
      forest.clear();
    }
  }
  GPRS_CHECK(n_maps > 5000);
  GPRS_CHECK(n_cut > 1000);
  GPRS_CHECK(n_refused > 1000);
  GPRS_CHECK(n_integer_maps > 500);
}
