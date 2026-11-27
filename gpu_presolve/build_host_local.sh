#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds libgprs_host.a, gprs_tests, gprs_presolve and gprs_fuzz with the system compiler, without CMake,
# once with assertions (-DGPRS_ASSERT=1) and once without, and runs the tests of both builds.
#
#   ./build_host_local.sh [test name filter]
#
# Environment: CXX (default c++), JOBS (default 8), GPRS_BUILD_DIR (default
# ${TMPDIR:-/tmp}/gprs_host_local; kept outside the source tree, which is synced).
# Every src/*.cpp goes into the library and every tests/*.cpp into gprs_tests.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cxx="${CXX:-c++}"
jobs="${JOBS:-8}"
out="${GPRS_BUILD_DIR:-${TMPDIR:-/tmp}/gprs_host_local}"
filter="${1:-}"
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off "-I$here/include")
# As in CMakeLists.txt: std::fma as one instruction on x86-64.
if [[ "$(uname -m)" == "x86_64" ]]; then
  flags+=(-mfma)
fi

# compile <object dir> <define or empty> <source>...
compile() {
  local dir="$1" define="$2"
  shift 2
  local pids=() src obj
  for src in "$@"; do
    obj="$dir/$(basename "$(dirname "$src")")_$(basename "$src" .cpp).o"
    "$cxx" "${flags[@]}" ${define:+"$define"} -c "$src" -o "$obj" &
    pids+=("$!")
    if ((${#pids[@]} >= jobs)); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
  if ((${#pids[@]} > 0)); then
    for pid in "${pids[@]}"; do wait "$pid"; done
  fi
}

# build <name> <define or empty>
build() {
  local name="$1" define="$2"
  local dir="$out/$name"
  rm -rf "$dir"
  mkdir -p "$dir/obj"
  compile "$dir/obj" "$define" "$here"/src/*.cpp "$here"/tests/*.cpp "$here"/tools/*.cpp
  ar rcs "$dir/libgprs_host.a" "$dir"/obj/src_*.o
  "$cxx" "${flags[@]}" "$dir"/obj/tests_*.o "$dir/libgprs_host.a" -o "$dir/gprs_tests"
  "$cxx" "${flags[@]}" "$dir/obj/tools_gprs_presolve.o" "$dir/libgprs_host.a" -o "$dir/gprs_presolve"
  "$cxx" "${flags[@]}" "$dir/obj/tools_gprs_fuzz.o" "$dir/obj/tests_oracle.o" "$dir/libgprs_host.a" \
    -o "$dir/gprs_fuzz"
  echo "== $name: built $dir/libgprs_host.a, gprs_tests, gprs_presolve, gprs_fuzz"
  if [[ -n "$filter" ]]; then
    "$dir/gprs_tests" "$filter"
  else
    "$dir/gprs_tests"
  fi
}

build assert -DGPRS_ASSERT=1
build release ""
echo "== both builds passed"
