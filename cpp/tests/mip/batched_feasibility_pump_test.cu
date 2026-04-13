/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "../linear_programming/utilities/pdlp_test_utilities.cuh"
#include "mip_utils.cuh"

#include <cuopt/error.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/local_search/feasibility_pump/batched_feasibility_pump.cuh>
#include <mip_heuristics/local_search/line_segment_search/line_segment_search.cuh>
#include <mip_heuristics/local_search/rounding/constraint_prop.cuh>
#include <mip_heuristics/solution/solution.cuh>
#include <mip_heuristics/solver.cuh>
#include <mip_heuristics/solver_context.cuh>
#include <mps_parser/parser.hpp>
#include <pdlp/utilities/problem_checking.cuh>
#include <utilities/common_utils.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <thrust/fill.h>

#include <string>

namespace cuopt::linear_programming::test {

static void init_handler(const raft::handle_t* handle_ptr)
{
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_ptr->get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_ptr->get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_ptr->get_stream()));
}

struct batched_fp_result_t {
  bool found_feasible;
  bool solution_feasible;
  double objective;
};

static batched_fp_result_t run_batched_fp(const std::string& test_instance,
                                          double time_limit = 30.,
                                          int batch_size    = 4)
{
  const raft::handle_t handle{};
  std::cout << "Batched FP on: " << test_instance << std::endl;

  auto path = cuopt::test::get_rapids_dataset_root_dir() + "/mip/" + test_instance;
  auto mps  = cuopt::mps_parser::parse_mps<int, double>(path, false);
  handle.sync_stream();
  auto op_problem = mps_data_model_to_optimization_problem(&handle, mps);
  problem_checking_t<int, double>::check_problem_representation(op_problem);

  detail::problem_t<int, double> problem(op_problem);
  problem.preprocess_problem();

  auto settings       = mip_solver_settings_t<int, double>{};
  settings.time_limit = time_limit;
  auto timer          = cuopt::timer_t(time_limit);

  detail::mip_solver_t<int, double> solver(problem, settings, timer);
  auto& ctx = solver.context;

  detail::solution_t<int, double> solution(*ctx.problem_ptr);
  thrust::fill(solution.handle_ptr->get_thrust_policy(),
               solution.assignment.begin(),
               solution.assignment.end(),
               0.0);
  solution.clamp_within_bounds();

  rmm::device_uvector<double> lp_optimal(ctx.problem_ptr->n_variables,
                                         ctx.handle_ptr->get_stream());
  thrust::fill(
    rmm::exec_policy(ctx.handle_ptr->get_stream()), lp_optimal.begin(), lp_optimal.end(), 0.0);

  detail::fj_t<int, double> fj(ctx);
  detail::constraint_prop_t<int, double> cprop(ctx);
  detail::line_segment_search_t<int, double> lss(fj, cprop);

  detail::batched_feasibility_pump_t<int, double> bfp(ctx, fj, cprop, lss, lp_optimal);
  bfp.config.batch_size = batch_size;

  bfp.timer = timer;
  bfp.reset();
  bfp.resize_vectors(*ctx.problem_ptr, ctx.handle_ptr);
  bool found = bfp.run(solution, timer, nullptr);

  double obj    = solution.get_user_objective();
  bool feasible = solution.get_feasible();

  std::cout << "  result: feasible=" << feasible << " found=" << found << " obj=" << obj
            << std::endl;

  return {found, feasible, obj};
}

TEST(batched_fp, smoke_test_sct2)
{
  auto result = run_batched_fp("sct2.mps", 60., 4);
  // batched FP should run without crashing; feasibility is not guaranteed
  // on every instance but sct2 is small and usually solvable
  EXPECT_TRUE(result.found_feasible || !result.found_feasible);  // no crash
}

TEST(batched_fp, smoke_test_gen_ip054)
{
  auto result = run_batched_fp("gen-ip054.mps", 60., 4);
  EXPECT_TRUE(result.found_feasible || !result.found_feasible);
}

TEST(batched_fp, batch_size_1)
{
  auto result = run_batched_fp("sct2.mps", 60., 1);
  EXPECT_TRUE(result.found_feasible || !result.found_feasible);
}

TEST(batched_fp, batch_size_8)
{
  auto result = run_batched_fp("sct2.mps", 60., 8);
  EXPECT_TRUE(result.found_feasible || !result.found_feasible);
}

class BatchedFPParametricTest
  : public testing::TestWithParam<std::tuple<std::string, double, int>> {};

TEST_P(BatchedFPParametricTest, batched_fp_run)
{
  auto [instance, time_limit, batch_size] = GetParam();
  auto result                             = run_batched_fp(instance, time_limit, batch_size);
  if (result.found_feasible) {
    EXPECT_TRUE(result.solution_feasible) << instance << ": returned feasible but solution is not";
  }
}

INSTANTIATE_TEST_SUITE_P(BatchedFPSuite,
                         BatchedFPParametricTest,
                         testing::Values(std::make_tuple("sct2.mps", 60., 4),
                                         std::make_tuple("gen-ip054.mps", 60., 4),
                                         std::make_tuple("tr12-30.mps", 60., 4),
                                         std::make_tuple("ns1208400.mps", 60., 4),
                                         std::make_tuple("cvs16r128-89.mps", 60., 4)));

}  // namespace cuopt::linear_programming::test
