# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Correctness check of the host engine with HiGHS as the solver (crush check with solution
# files, LP relaxations, MIP solve of the reduced model, postsolve; solution files from a MIP
# solve of the original model). Separate from the engine, which has no dependency.
#   cmake -S gpu_presolve/check -B <build dir> -G Ninja [-DHIGHS_ROOT=<install prefix>]
#         [-DGPRS_ENGINE_SRC=<src directory of a mutant, see make_mutants.py>]

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(gprs_check LANGUAGES CXX)

set(HIGHS_ROOT "/home/scratch.acoerduek_sw/gpu_presolve_deps/highs_stock_install"
    CACHE PATH "HiGHS install prefix the tool is built against")
set(GPRS_ENGINE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    CACHE PATH "Engine sources the tool is built with")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(highs REQUIRED CONFIG PATHS "${HIGHS_ROOT}" NO_DEFAULT_PATH)
find_package(Threads REQUIRED)

file(GLOB GPRS_HOST_SOURCES CONFIGURE_DEPENDS "${GPRS_ENGINE_SRC}/*.cpp")

add_executable(gprs_check gprs_check.cpp ${GPRS_HOST_SOURCES})
target_include_directories(gprs_check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_compile_options(gprs_check PRIVATE -Wall -Wextra -ffp-contract=off)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(gprs_check PRIVATE -mfma)
endif()
target_link_libraries(gprs_check PRIVATE highs::highs Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(gprs_check PROPERTIES BUILD_RPATH "${HIGHS_ROOT}/lib")
