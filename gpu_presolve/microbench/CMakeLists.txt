# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(gpu_presolve_microbench LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# rmm brings CCCL (Thrust, CUB, libcudacxx) and the CUDA runtime as dependencies.
find_package(rmm REQUIRED)

# Each tool is one independent target; add new targets below without touching the others.

add_executable(fixed_costs fixed_costs.cu)
target_link_libraries(fixed_costs PRIVATE rmm::rmm)

# sweeps: the CPU reference is compiled twice, with and without hardware FMA, because the cost
# of the compensated host arithmetic depends on it. Contraction is off everywhere.
add_library(sweeps_cpu_base OBJECT sweeps_cpu.cpp)
target_compile_definitions(sweeps_cpu_base PRIVATE SWEEPS_CPU_NS=cpu_base)
target_compile_options(sweeps_cpu_base PRIVATE -ffp-contract=off)
add_library(sweeps_cpu_fma OBJECT sweeps_cpu.cpp)
target_compile_definitions(sweeps_cpu_fma PRIVATE SWEEPS_CPU_NS=cpu_fma)
target_compile_options(sweeps_cpu_fma PRIVATE -ffp-contract=off)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(sweeps_cpu_fma PRIVATE -mfma)
endif()
add_executable(sweeps sweeps.cu $<TARGET_OBJECTS:sweeps_cpu_base> $<TARGET_OBJECTS:sweeps_cpu_fma>)
target_compile_options(sweeps PRIVATE --fmad=false -Xcompiler=-ffp-contract=off)
target_link_libraries(sweeps PRIVATE rmm::rmm)
