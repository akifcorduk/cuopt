#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs the release build of the host engine on the 40 testbed instances with one rule family
# switched off at a time, the six runs side by side (one core each, one repetition; the times
# are indicative only). Results: gpu_presolve/results/stage1/host/<tag>/off_<rule>.
#   setsid nohup bash gpu_presolve/tools/stage1_ablation.sh <tag> > /dev/null 2>&1 < /dev/null &
# The exit code is 1 when a runner failed; the log names it.

set -u -o pipefail

TAG=${1:?usage: stage1_ablation.sh <tag>}
DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
LOG=$DEPS/results/stage1/ablation_$TAG.log
OUT=gpu_presolve/results/stage1/host/$TAG

pids=()
for rule in row_rules propagation tightening doubleton parallel_rows dual_fixing; do
  (
    failed=0
    for set in dev large; do
      if ! python3 gpu_presolve/tools/run_engine.py --tool "$DEPS/build/gprs_host/gprs_presolve" \
        --dumps "$DEPS/csr_dumps_dev" --dumps "$DEPS/csr_dumps" --out "$OUT/off_$rule/$set" \
        --set "$set" --reps 1 --jobs 1 --kill-after 3600 --skip-done \
        --tool-arg=--off --tool-arg="$rule" | tail -1; then
        echo "$(date +%T) off_$rule $set: the runner failed"
        failed=1
      fi
    done
    echo "$(date +%T) off_$rule done"
    exit "$failed"
  ) >> "$LOG" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
echo "$(date +%T) all done, failed: $failed" >> "$LOG"
exit "$failed"
