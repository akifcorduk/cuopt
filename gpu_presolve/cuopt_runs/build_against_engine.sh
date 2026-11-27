#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# After an engine change: compiles what depends on the engine's headers besides the engine
# itself, and runs its tests.
#   1. The cuOpt adapter: the target NUMOPT_INTERNAL_TEST in the cuOpt build tree of this
#      worktree (cpp/build), which takes the engine sources from gpu_presolve/, and the gprs
#      tests of that binary.
#   2. The device code (gpu_presolve/device) of the engine version <tag>, against the release
#      library that tools/stage1c_engine_runs.sh built for that tag, and its tests.
# The synced tree must hold the engine of <tag>.
#
# Start it detached from the worktree root, inside the dev container:
#   setsid nohup bash gpu_presolve/cuopt_runs/build_against_engine.sh <tag> \
#     > /dev/null 2>&1 < /dev/null &
# Logs: $DEPS/results/stage1c/dependents/<tag>/<step>.log; steps.log has one line per step;
# ENDED lists the failed steps (empty: none).

set -u -o pipefail

TAG=${1:?usage: build_against_engine.sh <tag>}
DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
OUT=$DEPS/results/stage1c/dependents/$TAG
SRC=$DEPS/src/$TAG
BUILD=$DEPS/build/$TAG
mkdir -p "$OUT"
rm -f "$OUT/ENDED"
FAILED=""

# run <step> <command>...
run() {
  local step_name=$1 code
  shift
  echo "$(date +%F\ %T) $step_name: start" >> "$OUT/steps.log"
  "$@" > "$OUT/$step_name.log" 2>&1
  code=$?
  [ "$code" -eq 0 ] || FAILED="$FAILED $step_name"
  echo "$(date +%F\ %T) $step_name: end, exit $code" >> "$OUT/steps.log"
  return "$code"
}

if run adapter_build ninja -C cpp/build -j 24 NUMOPT_INTERNAL_TEST; then
  TEST=$(find cpp/build -name NUMOPT_INTERNAL_TEST -type f -perm -u+x | head -1)
  run adapter_tests "$TEST" --gtest_filter='*gprs*'
fi
if run device_configure cmake -S "$SRC/gpu_presolve/device" -B "$BUILD/device" -G Ninja \
  -DGPRS_HOST_LIBRARY="$BUILD/release/libgprs_host.a"; then
  if run device_build cmake --build "$BUILD/device" -j 16; then
    run device_tests ctest --test-dir "$BUILD/device" --output-on-failure
  fi
fi

echo "$FAILED" > "$OUT/ENDED"
[ -z "$FAILED" ]
