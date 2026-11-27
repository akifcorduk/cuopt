# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

project(gpu_presolve_baseline LANGUAGES CXX)

set(HIGHS_ROOT "/home/scratch.acoerduek_sw/gpu_presolve_deps/highs_stock_install"
    CACHE PATH "HiGHS install prefix the tool is built against")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(highs REQUIRED CONFIG PATHS "${HIGHS_ROOT}" NO_DEFAULT_PATH)

add_executable(baseline_tool baseline_tool.cpp)
target_compile_features(baseline_tool PRIVATE cxx_std_17)
target_compile_options(baseline_tool PRIVATE -Wall -Wextra)
target_compile_definitions(baseline_tool PRIVATE BASELINE_HIGHS_ROOT="${HIGHS_ROOT}")
target_link_libraries(baseline_tool PRIVATE highs::highs ${CMAKE_DL_LIBS})
set_target_properties(baseline_tool PROPERTIES BUILD_RPATH "${HIGHS_ROOT}/lib")

# Converts solutions that name their variables into the solution files of the correctness check.
add_executable(convert_solution convert_solution.cpp)
target_compile_features(convert_solution PRIVATE cxx_std_17)
target_compile_options(convert_solution PRIVATE -Wall -Wextra)
target_link_libraries(convert_solution PRIVATE highs::highs)
set_target_properties(convert_solution PROPERTIES BUILD_RPATH "${HIGHS_ROOT}/lib")
