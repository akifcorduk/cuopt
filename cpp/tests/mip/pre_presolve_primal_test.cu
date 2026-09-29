/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <mip_heuristics/pre_presolve_primal.cuh>

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace cuopt::mathematical_optimization::mip::test {

namespace {

using optional_view = std::optional<std::string_view>;

void expect_validation_error(optional_view config_id, optional_view max_config)
{
  try {
    (void)resolve_pre_presolve_config(config_id, max_config);
    FAIL() << "Expected a cuOpt validation error";
  } catch (const cuopt::logic_error& error) {
    EXPECT_EQ(error.get_error_type(), cuopt::error_type_t::ValidationError);
  } catch (...) {
    FAIL() << "Expected cuopt::logic_error";
  }
}

}  // namespace

TEST(pre_presolve_config, both_unset_preserves_stock_path)
{
  const auto config = resolve_pre_presolve_config(std::nullopt, std::nullopt);
  EXPECT_FALSE(config.enabled);
  EXPECT_EQ(config.raw_id, -1);
  EXPECT_EQ(config.max_config, -1);
  EXPECT_EQ(config.repeat, -1);
  EXPECT_EQ(config.config_id, -1);
  EXPECT_EQ(config.diving_workers(), 0);
  EXPECT_EQ(config.task_slots(), 0);
  EXPECT_EQ(config.cpufj_reserved_threads(), 0);
  EXPECT_FALSE(config.cooperative());
  EXPECT_STREQ(config.name(), "disabled");
}

TEST(pre_presolve_config, interleaves_four_profiles_across_repeats)
{
  for (int raw_id = 0; raw_id < 8; ++raw_id) {
    const std::string raw_id_text = std::to_string(raw_id);
    const auto config =
      resolve_pre_presolve_config(std::string_view{raw_id_text}, std::string_view{"8"});
    EXPECT_TRUE(config.enabled);
    EXPECT_EQ(config.raw_id, raw_id);
    EXPECT_EQ(config.max_config, 8);
    EXPECT_EQ(config.repeat, raw_id / 4);
    EXPECT_EQ(config.config_id, raw_id % 4);
  }
}

TEST(pre_presolve_config, profile_properties)
{
  constexpr std::array<int, 4> workers{3, 3, 3, 2};
  constexpr std::array<int, 4> task_slots{4, 4, 4, 3};
  constexpr std::array<int, 4> cpufj_reserved{8, 8, 8, 7};
  constexpr std::array<bool, 4> cooperative{false, true, true, true};
  constexpr std::array<const char*, 4> names{
    "independent-w3-all", "cooperative-w3-all", "refinement-w3-restricted", "cooperative-w2-all"};

  for (int id = 0; id < 4; ++id) {
    const std::string id_text = std::to_string(id);
    const auto config =
      resolve_pre_presolve_config(std::string_view{id_text}, std::string_view{"4"});
    EXPECT_EQ(config.diving_workers(), workers[id]);
    EXPECT_EQ(config.task_slots(), task_slots[id]);
    EXPECT_EQ(config.cpufj_reserved_threads(), cpufj_reserved[id]);
    EXPECT_EQ(config.cooperative(), cooperative[id]);
    EXPECT_STREQ(config.name(), names[id]);
  }
}

TEST(pre_presolve_config, requires_both_environment_values)
{
  expect_validation_error(std::string_view{"0"}, std::nullopt);
  expect_validation_error(std::nullopt, std::string_view{"4"});
}

TEST(pre_presolve_config, rejects_non_strict_integers)
{
  constexpr std::array<std::string_view, 8> invalid_ids{
    "", " 0", "0 ", "+0", "0x", "1.0", "--1", "999999999999999999999999"};
  for (const auto value : invalid_ids) {
    expect_validation_error(value, std::string_view{"4"});
  }

  constexpr std::array<std::string_view, 7> invalid_max{
    "", " 4", "4 ", "+4", "4x", "4.0", "999999999999999999999999"};
  for (const auto value : invalid_max) {
    expect_validation_error(std::string_view{"0"}, value);
  }
}

TEST(pre_presolve_config, validates_range_and_multiple_of_four)
{
  expect_validation_error(std::string_view{"0"}, std::string_view{"0"});
  expect_validation_error(std::string_view{"0"}, std::string_view{"-4"});
  expect_validation_error(std::string_view{"0"}, std::string_view{"2"});
  expect_validation_error(std::string_view{"0"}, std::string_view{"6"});
  expect_validation_error(std::string_view{"-1"}, std::string_view{"4"});
  expect_validation_error(std::string_view{"4"}, std::string_view{"4"});
  expect_validation_error(std::string_view{"8"}, std::string_view{"8"});
}

TEST(pre_presolve_eligibility, accepts_exact_inclusive_boundaries)
{
  EXPECT_TRUE(is_pre_presolve_primal_eligible(pre_presolve_max_rows,
                                              pre_presolve_max_cols,
                                              pre_presolve_max_nnz,
                                              false,
                                              false,
                                              true,
                                              true,
                                              true,
                                              pre_presolve_min_team_threads));
  EXPECT_TRUE(is_pre_presolve_primal_eligible(
    1, 1, 0, false, false, true, true, true, pre_presolve_min_team_threads));
}

TEST(pre_presolve_eligibility, rejects_out_of_range_or_empty_dimensions)
{
  const auto eligible = [](int rows, int cols, int nnz) {
    return is_pre_presolve_primal_eligible(
      rows, cols, nnz, false, false, true, true, true, pre_presolve_min_team_threads);
  };

  EXPECT_FALSE(eligible(0, 1, 0));
  EXPECT_FALSE(eligible(1, 0, 0));
  EXPECT_FALSE(eligible(-1, 1, 0));
  EXPECT_FALSE(eligible(1, -1, 0));
  EXPECT_FALSE(eligible(1, 1, -1));
  EXPECT_FALSE(eligible(pre_presolve_max_rows + 1, 1, 0));
  EXPECT_FALSE(eligible(1, pre_presolve_max_cols + 1, 0));
  EXPECT_FALSE(eligible(1, 1, pre_presolve_max_nnz + 1));
}

TEST(pre_presolve_eligibility, rejects_unsupported_modes)
{
  const auto eligible = [](bool quadratic_objective,
                           bool quadratic_constraints,
                           bool is_mip,
                           bool presolve,
                           bool opportunistic,
                           int threads) {
    return is_pre_presolve_primal_eligible(1,
                                           1,
                                           0,
                                           quadratic_objective,
                                           quadratic_constraints,
                                           is_mip,
                                           presolve,
                                           opportunistic,
                                           threads);
  };

  EXPECT_FALSE(eligible(true, false, true, true, true, pre_presolve_min_team_threads));
  EXPECT_FALSE(eligible(false, true, true, true, true, pre_presolve_min_team_threads));
  EXPECT_FALSE(eligible(false, false, false, true, true, pre_presolve_min_team_threads));
  EXPECT_FALSE(eligible(false, false, true, false, true, pre_presolve_min_team_threads));
  EXPECT_FALSE(eligible(false, false, true, true, false, pre_presolve_min_team_threads));
  EXPECT_FALSE(eligible(false, false, true, true, true, pre_presolve_min_team_threads - 1));
}

}  // namespace cuopt::mathematical_optimization::mip::test
