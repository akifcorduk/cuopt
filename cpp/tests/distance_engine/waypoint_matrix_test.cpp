/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "waypoint_matrix_test.hpp"

#include <cuopt/error.hpp>
#include <cuopt/routing/distance_engine/waypoint_matrix.hpp>
#include <rmm/device_uvector.hpp>
#include <utilities/base_fixture.hpp>
#include <utilities/common_utils.hpp>
#include "utilities/data_model.hpp"

namespace cuopt {
namespace distance_engine {
namespace test {
template <typename i_t, typename f_t>
class waypoint_matrix_waypoints_sequence_test_t
  : public base_test_t<i_t, f_t>,
    public ::testing::TestWithParam<
      waypoint_matrix_params_t<waypoint_sequence_params_t<i_t, f_t>, i_t, f_t>> {
 public:
  waypoint_matrix_waypoints_sequence_test_t() = default;

  void SetUp() override
  {
    auto param    = this->GetParam();
    this->offsets = param.offsets;
    this->indices = param.indices;
    this->weights = param.weights;

    this->waypoint_matrix = waypoint_matrix_t<i_t, f_t>(this->handle,
                                                        this->offsets.data(),
                                                        this->offsets.size() - 1,
                                                        this->indices.data(),
                                                        this->weights.data());

    this->target_locations          = param.target_locations;
    this->locations                 = param.locations;
    this->expected_cost_matrix      = param.cost_matrix;
    this->expected_full_path        = param.full_path;
    this->expected_sequence_offsets = param.sequence_offsets;
  }

  void TearDown() override { /* no-op */ }

  void test_compute_waypoint_sequence()
  {
    auto stream = this->handle.get_stream();

    rmm::device_uvector<f_t> d_cost_matrix(
      this->target_locations.size() * this->target_locations.size(), stream);

    this->waypoint_matrix.compute_cost_matrix(
      d_cost_matrix.data(), this->target_locations.data(), this->target_locations.size());

    std::vector<f_t> h_cost_matrix(this->target_locations.size() * this->target_locations.size());

    raft::copy(h_cost_matrix.data(), d_cost_matrix.data(), h_cost_matrix.size(), stream);
    stream.sync();

    for (size_t i = 0; i != h_cost_matrix.size(); ++i)
      EXPECT_EQ(h_cost_matrix[i], expected_cost_matrix[i]);

    auto [d_sequence_offsets, d_full_path] =
      this->waypoint_matrix.compute_waypoint_sequence(this->target_locations.data(),
                                                      this->target_locations.size(),
                                                      this->locations.data(),
                                                      this->locations.size());

    std::vector<i_t> h_sequence_offsets(this->expected_sequence_offsets.size());
    std::vector<i_t> h_full_path(this->expected_full_path.size());

    raft::copy(h_sequence_offsets.data(),
               (i_t*)d_sequence_offsets.get()->data(),
               h_sequence_offsets.size(),
               stream);
    raft::copy(h_full_path.data(), (i_t*)d_full_path.get()->data(), h_full_path.size(), stream);
    stream.sync();

    for (size_t i = 0; i != h_sequence_offsets.size(); ++i)
      EXPECT_EQ(h_sequence_offsets[i], expected_sequence_offsets[i]);
    for (size_t i = 0; i != h_full_path.size(); ++i)
      EXPECT_EQ(h_full_path[i], expected_full_path[i]);
  }

  void test_compute_waypoint_sequence_no_matrix_call()
  {
    auto stream = this->handle.get_stream();

    EXPECT_ANY_THROW(this->waypoint_matrix.compute_waypoint_sequence(this->target_locations.data(),
                                                                     this->target_locations.size(),
                                                                     this->locations.data(),
                                                                     this->locations.size()));
  }

 private:
  std::vector<i_t> offsets;
  std::vector<i_t> indices;
  std::vector<f_t> weights;
  std::vector<i_t> locations{};
  std::vector<f_t> expected_cost_matrix{};
  std::vector<i_t> expected_full_path{};
  std::vector<i_t> expected_sequence_offsets{};
};
template <typename i_t, typename f_t>
class waypoint_matrix_shortest_path_cost_t
  : public base_test_t<i_t, f_t>,
    public ::testing::TestWithParam<
      waypoint_matrix_params_t<shortest_path_cost_params_t<f_t>, i_t, f_t>> {
 public:
  waypoint_matrix_shortest_path_cost_t() = default;

  void SetUp() override
  {
    auto param = this->GetParam();

    this->offsets           = param.offsets;
    this->indices           = param.indices;
    this->weights           = param.weights;
    this->target_locations  = param.target_locations;
    this->ref_custom_matrix = param.cost_matrix;
    this->custom_weights    = param.custom_weights;

    this->waypoint_matrix = waypoint_matrix_t<i_t, f_t>(this->handle,
                                                        this->offsets.data(),
                                                        this->offsets.size() - 1,
                                                        this->indices.data(),
                                                        this->weights.data());
  }

  void TearDown() override { /* no-op */ }

  void test_compute_shortest_path_costs()
  {
    auto stream = this->handle.get_stream();

    rmm::device_uvector<f_t> d_cost_matrix(
      this->target_locations.size() * this->target_locations.size(), stream);

    this->waypoint_matrix.compute_cost_matrix(
      d_cost_matrix.data(), this->target_locations.data(), this->target_locations.size());

    rmm::device_uvector<f_t> d_custom_matrix(
      this->target_locations.size() * this->target_locations.size(), stream);

    this->waypoint_matrix.compute_shortest_path_costs(d_custom_matrix.data(),
                                                      this->target_locations.data(),
                                                      this->target_locations.size(),
                                                      this->custom_weights.data());

    std::vector<f_t> h_custom_matrix(this->target_locations.size() * this->target_locations.size());

    raft::copy(h_custom_matrix.data(), d_custom_matrix.data(), h_custom_matrix.size(), stream);
    stream.sync();

    for (size_t i = 0; i != h_custom_matrix.size(); ++i)
      EXPECT_EQ(h_custom_matrix[i], ref_custom_matrix[i]);
  }

 private:
  std::vector<i_t> offsets;
  std::vector<i_t> indices;
  std::vector<f_t> weights;
  std::vector<f_t> custom_weights;
  std::vector<f_t> ref_custom_matrix{};
};

template <typename i_t, typename f_t>
class waypoint_matrix_cost_matrix_test_t
  : public base_test_t<i_t, f_t>,
    public ::testing::TestWithParam<waypoint_matrix_params_t<cost_matrix_params_t<f_t>, i_t, f_t>> {
 public:
  waypoint_matrix_cost_matrix_test_t() = default;

  void SetUp() override
  {
    auto param = this->GetParam();

    this->offsets          = param.offsets;
    this->indices          = param.indices;
    this->weights          = param.weights;
    this->target_locations = param.target_locations;
    this->ref_cost_matrix  = param.cost_matrix;

    this->waypoint_matrix = waypoint_matrix_t<i_t, f_t>(this->handle,
                                                        this->offsets.data(),
                                                        this->offsets.size() - 1,
                                                        this->indices.data(),
                                                        this->weights.data());
  }

  void TearDown() override { /* no-op */ }

  void test_compute_cost_matrix()
  {
    auto stream = this->handle.get_stream();

    rmm::device_uvector<f_t> d_cost_matrix(
      this->target_locations.size() * this->target_locations.size(), stream);

    this->waypoint_matrix.compute_cost_matrix(
      d_cost_matrix.data(), this->target_locations.data(), this->target_locations.size());

    std::vector<f_t> h_cost_matrix(this->target_locations.size() * this->target_locations.size());

    raft::copy(h_cost_matrix.data(), d_cost_matrix.data(), h_cost_matrix.size(), stream);
    stream.sync();

    for (size_t i = 0; i != h_cost_matrix.size(); ++i)
      EXPECT_NEAR(h_cost_matrix[i], this->ref_cost_matrix[i], 0.001f);
  }

 private:
  std::vector<f_t> ref_cost_matrix{};
  std::vector<i_t> offsets;
  std::vector<i_t> indices;
  std::vector<f_t> weights;
};

typedef waypoint_matrix_waypoints_sequence_test_t<int, float>
  float_waypoint_matrix_waypoints_sequence_test_t;
typedef waypoint_matrix_shortest_path_cost_t<int, float> float_waypoint_matrix_shortest_path_cost_t;
typedef waypoint_matrix_cost_matrix_test_t<int, float> float_waypoint_matrix_cost_matrix_test_t;

TEST_P(float_waypoint_matrix_waypoints_sequence_test_t, compute_waypoint_sequence)
{
  test_compute_waypoint_sequence();
}

TEST_P(float_waypoint_matrix_waypoints_sequence_test_t, compute_waypoint_sequence_no_matrix_call)
{
  test_compute_waypoint_sequence_no_matrix_call();
}

INSTANTIATE_TEST_SUITE_P(test_waypoint_sequence,
                         float_waypoint_matrix_waypoints_sequence_test_t,
                         ::testing::ValuesIn(parse_data_models(first_input_, second_input_)));

TEST_P(float_waypoint_matrix_shortest_path_cost_t, compute_shortest_path_costs)
{
  test_compute_shortest_path_costs();
}

INSTANTIATE_TEST_SUITE_P(test_shortest_path_cost,
                         float_waypoint_matrix_shortest_path_cost_t,
                         ::testing::ValuesIn(parse_data_models_custom_weight(first_input_)));

TEST_P(float_waypoint_matrix_cost_matrix_test_t, compute_cost_matrix)
{
  test_compute_cost_matrix();
}

INSTANTIATE_TEST_SUITE_P(
  test_waypoint_matrix,
  float_waypoint_matrix_cost_matrix_test_t,
  ::testing::Values(parse_tests(
    parse_waypoint_matrix_file<int, int, float>(
      cuopt::test::read_waypoint_matrix_file("datasets/distance_engine/waypoint_matrix.txt")),
    parse_vector<int>(
      cuopt::test::read_target_file("datasets/distance_engine/target_locations_id.txt")),
    parse_vector<float>(
      cuopt::test::read_matrix_file("datasets/distance_engine/ref_cost_matrix.csv")))));

// Chain 0 - 1 - 2 - 3 with a distinct custom weight on each directed edge
struct chain_graph_t {
  std::vector<int> offsets              = {0, 1, 3, 5, 6};
  std::vector<int> indices              = {1, 0, 2, 1, 3, 2};
  std::vector<float> weights            = {1, 1, 1, 1, 1, 1};
  std::vector<float> custom_weights     = {1, 10, 100, 1000, 10000, 100000};
  std::vector<int> all_target_locations = {0, 1, 2, 3};
};

static std::vector<float> chain_shortest_path_costs(std::vector<int> const& cached_targets,
                                                    std::vector<int> const& targets)
{
  chain_graph_t graph;
  raft::handle_t handle;
  auto stream = handle.get_stream();

  waypoint_matrix_t<int, float> waypoint_matrix(handle,
                                                graph.offsets.data(),
                                                graph.offsets.size() - 1,
                                                graph.indices.data(),
                                                graph.weights.data());

  rmm::device_uvector<float> d_cost_matrix(cached_targets.size() * cached_targets.size(), stream);
  waypoint_matrix.compute_cost_matrix(
    d_cost_matrix.data(), cached_targets.data(), cached_targets.size());

  rmm::device_uvector<float> d_custom_matrix(targets.size() * targets.size(), stream);
  waypoint_matrix.compute_shortest_path_costs(
    d_custom_matrix.data(), targets.data(), targets.size(), graph.custom_weights.data());

  std::vector<float> h_custom_matrix(d_custom_matrix.size());
  raft::copy(h_custom_matrix.data(), d_custom_matrix.data(), h_custom_matrix.size(), stream);
  stream.sync();
  return h_custom_matrix;
}

TEST(waypoint_matrix_target_rows, shortest_path_costs_of_target_subset)
{
  chain_graph_t graph;
  std::vector<float> expected = {0, 10000, 100000, 0};
  EXPECT_EQ(chain_shortest_path_costs(graph.all_target_locations, {2, 3}), expected);
}

TEST(waypoint_matrix_target_rows, shortest_path_costs_of_reordered_targets)
{
  chain_graph_t graph;
  std::vector<float> expected = {0, 101000, 101010, 10100, 0, 10, 10101, 1, 0};
  EXPECT_EQ(chain_shortest_path_costs(graph.all_target_locations, {3, 1, 0}), expected);
}

TEST(waypoint_matrix_target_rows, shortest_path_costs_of_uncached_target)
{
  EXPECT_THROW(chain_shortest_path_costs({0, 1}, {2, 3}), cuopt::logic_error);
}

TEST(waypoint_matrix_target_rows, shortest_path_costs_no_matrix_call)
{
  chain_graph_t graph;
  raft::handle_t handle;
  auto stream = handle.get_stream();

  waypoint_matrix_t<int, float> waypoint_matrix(handle,
                                                graph.offsets.data(),
                                                graph.offsets.size() - 1,
                                                graph.indices.data(),
                                                graph.weights.data());

  auto const& targets = graph.all_target_locations;
  rmm::device_uvector<float> d_custom_matrix(targets.size() * targets.size(), stream);
  EXPECT_THROW(
    waypoint_matrix.compute_shortest_path_costs(
      d_custom_matrix.data(), targets.data(), targets.size(), graph.custom_weights.data()),
    cuopt::logic_error);
}

TEST(waypoint_matrix_target_rows, waypoint_sequence_of_reordered_targets)
{
  chain_graph_t graph;
  raft::handle_t handle;
  auto stream = handle.get_stream();

  waypoint_matrix_t<int, float> waypoint_matrix(handle,
                                                graph.offsets.data(),
                                                graph.offsets.size() - 1,
                                                graph.indices.data(),
                                                graph.weights.data());

  rmm::device_uvector<float> d_cost_matrix(16, stream);
  waypoint_matrix.compute_cost_matrix(
    d_cost_matrix.data(), graph.all_target_locations.data(), graph.all_target_locations.size());

  std::vector<int> targets   = {3, 1};
  std::vector<int> locations = {0, 1, 0};
  rmm::device_uvector<int> d_locations(locations.size(), stream);
  raft::copy(d_locations.data(), locations.data(), locations.size(), stream);

  auto [d_sequence_offsets, d_full_path] = waypoint_matrix.compute_waypoint_sequence(
    targets.data(), targets.size(), d_locations.data(), locations.size());

  std::vector<int> expected_sequence_offsets = {0, 3, 6};
  std::vector<int> expected_full_path        = {3, 2, 1, 1, 2, 3};

  ASSERT_EQ(d_sequence_offsets->size(), expected_sequence_offsets.size() * sizeof(int));
  ASSERT_EQ(d_full_path->size(), expected_full_path.size() * sizeof(int));

  std::vector<int> h_sequence_offsets(expected_sequence_offsets.size());
  std::vector<int> h_full_path(expected_full_path.size());
  raft::copy(
    h_sequence_offsets.data(), (int*)d_sequence_offsets->data(), h_sequence_offsets.size(), stream);
  raft::copy(h_full_path.data(), (int*)d_full_path->data(), h_full_path.size(), stream);
  stream.sync();

  EXPECT_EQ(h_sequence_offsets, expected_sequence_offsets);
  EXPECT_EQ(h_full_path, expected_full_path);

  std::vector<int> uncached_targets = {3, 1, 2};
  waypoint_matrix.compute_cost_matrix(d_cost_matrix.data(), targets.data(), targets.size());
  EXPECT_THROW(
    waypoint_matrix.compute_waypoint_sequence(
      uncached_targets.data(), uncached_targets.size(), d_locations.data(), locations.size()),
    cuopt::logic_error);
}

CUOPT_TEST_PROGRAM_MAIN()

}  // namespace test
}  // namespace distance_engine
}  // namespace cuopt
