#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs the release build of the host engine on the 40 testbed instances with one rule family
# switched off at a time, the six runs side by side (one core each, one repetition; the times
# are indicative only). Results: gpu_presolve/results/stage1/host/<tag>/off_<rule>.
#   setsid nohup bash gpu_presolve/tools/stage1_ablation.sh <tag> > /dev/null 2>&1 < /dev/null &

set -u

TAG=${1:?usage: stage1_ablation.sh <tag>}
DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
LOG=$DEPS/results/stage1/ablation_$TAG.log
OUT=gpu_presolve/results/stage1/host/$TAG

for rule in row_rules propagation tightening doubleton parallel_rows dual_fixing; do
  (
    for set in dev large; do
      python3 gpu_presolve/tools/run_engine.py --tool "$DEPS/build/gprs_host/gprs_presolve" \
        --dumps "$DEPS/csr_dumps_dev" --dumps "$DEPS/csr_dumps" --out "$OUT/off_$rule/$set" \
        --set "$set" --reps 1 --jobs 1 --kill-after 3600 --skip-done \
        --tool-arg=--off --tool-arg="$rule" | tail -1
    done
    echo "$(date +%T) off_$rule done"
  ) >> "$LOG" 2>&1 &
done
wait
echo "$(date +%T) all done" >> "$LOG"
