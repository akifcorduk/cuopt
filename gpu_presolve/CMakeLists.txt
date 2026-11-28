# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Host engine of the GPU MIP presolve: standard library only, no CUDA.
#   cmake -S gpu_presolve -B <build dir> -G Ninja [-DGPRS_ASSERT=ON]
#   cmake --build <build dir> && ctest --test-dir <build dir>
# baseline/ and microbench/ are separate projects with their own CMakeLists.txt.

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(gprs LANGUAGES CXX)

option(GPRS_ASSERT "Compile gprs_assert in" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Sources are found by pattern so that a part adds files without editing this file.
file(GLOB GPRS_HOST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB GPRS_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

# The warnings are private to the targets; what the inline numerics of the headers need
# is public and reaches every target that links the library.
set(GPRS_WARNINGS -Wall -Wextra -Werror)

add_library(gprs_host STATIC ${GPRS_HOST_SOURCES})
target_include_directories(gprs_host PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(gprs_host PRIVATE ${GPRS_WARNINGS})
# -ffp-contract=off: the compensated sums rely on products and sums that are not fused.
target_compile_options(gprs_host PUBLIC -ffp-contract=off)
# two_prod calls std::fma once per activity term. Without -mfma an x86-64 build makes that a
# call into libm; with it, one instruction. The result is the same bits either way.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(gprs_host PUBLIC -mfma)
endif()
if(GPRS_ASSERT)
  target_compile_definitions(gprs_host PUBLIC GPRS_ASSERT=1)
endif()

add_executable(gprs_tests ${GPRS_TEST_SOURCES})
target_compile_options(gprs_tests PRIVATE ${GPRS_WARNINGS})
target_link_libraries(gprs_tests PRIVATE gprs_host)

add_executable(gprs_presolve tools/gprs_presolve.cpp)
target_compile_options(gprs_presolve PRIVATE ${GPRS_WARNINGS})
target_link_libraries(gprs_presolve PRIVATE gprs_host)

# Long random runs (tools/run_campaign.sh): the generator and the exact oracle of the tests, and
# the structure check of check/.
add_executable(gprs_fuzz tools/gprs_fuzz.cpp tests/oracle.cpp)
target_compile_options(gprs_fuzz PRIVATE ${GPRS_WARNINGS})
target_link_libraries(gprs_fuzz PRIVATE gprs_host)

# The numbers of a reduced model (analysis/quality_compare.py).
add_executable(gprs_quality tools/gprs_quality.cpp)
target_compile_options(gprs_quality PRIVATE ${GPRS_WARNINGS})
target_link_libraries(gprs_quality PRIVATE gprs_host)

# What a presolve result holds, model by model, to compare two builds (tools/run_ident.sh).
add_executable(gprs_ident tools/gprs_ident.cpp tests/oracle.cpp)
target_compile_options(gprs_ident PRIVATE ${GPRS_WARNINGS})
target_link_libraries(gprs_ident PRIVATE gprs_host)

enable_testing()
add_test(NAME gprs_tests COMMAND gprs_tests)
