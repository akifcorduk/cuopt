# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Device side of the GPU MIP presolve: CUDA, cuSPARSE and CCCL (Thrust, CUB); no cuOpt, no rmm.
# It builds against the headers of the host engine and links its library, which is built by
# the host project (../CMakeLists.txt) and stays free of CUDA.
#   cmake -S gpu_presolve/device -B <build dir> -G Ninja \
#     [-DGPRS_HOST_LIBRARY=<libgprs_host.a>] [-DCCCL_DIR=<dir of cccl-config.cmake>] \
#     [-DGPRS_ASSERT=ON]
#   cmake --build <build dir> && ctest --test-dir <build dir>
# Without CCCL_DIR the CCCL that ships with the CUDA toolkit is used.
# With GPRS_ASSERT the gprs_assert of this library and of the host headers are compiled in, and
# GPRS_HOST_LIBRARY must be the host library built with GPRS_ASSERT as well.

cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(gprs_device LANGUAGES CXX CUDA)

option(GPRS_ASSERT "Compile gprs_assert in" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# As the host project: the host loops of the bench tool are compiled like the engine.
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(GPRS_HOST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    CACHE PATH "Headers of the host engine")
if(GPRS_ASSERT)
  set(GPRS_HOST_BUILD gprs_host_assert)
else()
  set(GPRS_HOST_BUILD gprs_host)
endif()
set(GPRS_HOST_LIBRARY
    "/home/scratch.acoerduek_sw/gpu_presolve_deps/build/${GPRS_HOST_BUILD}/libgprs_host.a"
    CACHE FILEPATH "Library of the host engine, built with the same GPRS_ASSERT")

find_package(CUDAToolkit REQUIRED)
find_package(CCCL CONFIG REQUIRED
             HINTS "${CUDAToolkit_TARGET_DIR}" "${CUDAToolkit_LIBRARY_DIR}/cmake"
                   "${CUDAToolkit_LIBRARY_ROOT}/lib/cmake")
find_package(Threads REQUIRED)

add_library(gprs_host STATIC IMPORTED)
set_target_properties(gprs_host PROPERTIES
                      IMPORTED_LOCATION "${GPRS_HOST_LIBRARY}"
                      INTERFACE_INCLUDE_DIRECTORIES "${GPRS_HOST_INCLUDE_DIR}")
if(GPRS_ASSERT)
  set_target_properties(gprs_host PROPERTIES INTERFACE_COMPILE_DEFINITIONS GPRS_ASSERT=1)
endif()

add_library(gprs_device STATIC src/state_build.cu)
target_include_directories(gprs_device PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(gprs_device PRIVATE --fmad=false -Xcompiler=-ffp-contract=off)
target_link_libraries(gprs_device PUBLIC gprs_host PRIVATE CCCL::CCCL CUDA::cusparse CUDA::cudart)

# -ffp-contract=off and -mfma are the public options of the host library (../CMakeLists.txt).
add_executable(state_build_bench bench/state_build_bench.cpp)
target_compile_options(state_build_bench PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(state_build_bench PRIVATE -mfma)
endif()
target_link_libraries(state_build_bench PRIVATE gprs_device CUDA::cudart Threads::Threads)

add_executable(state_build_test test/state_build_test.cpp)
target_compile_options(state_build_test PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(state_build_test PRIVATE -mfma)
endif()
target_link_libraries(state_build_test PRIVATE gprs_device CUDA::cudart)

enable_testing()
add_test(NAME state_build_test COMMAND state_build_test)
