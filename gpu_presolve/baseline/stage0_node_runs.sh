#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs the remaining Stage 0 measurements on the node, one step after the other, and can be
# started again after a disconnect or a node change: a step that ended with exit code 0 leaves a
# marker and is skipped, a failed step is tried again at the next start, and inside a step the
# runner skips the instances that already have a result.
# Everything it needs or writes is on the scratch (builds, downloads, dumps, markers, log) or in
# the synced worktree (results).
#
# Start it detached from the worktree root, inside the dev container:
#   setsid nohup bash gpu_presolve/baseline/stage0_node_runs.sh > /dev/null 2>&1 < /dev/null &
# Progress: tail $DEPS/results/stage0/node_runs.log. When the driver ends it writes $STATE/ENDED
# with the names of the steps that failed (empty: all done).

set -u

DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
STATE=$DEPS/results/stage0/node_runs
LOG=$DEPS/results/stage0/node_runs.log
DUMPS=$DEPS/csr_dumps
RUN=gpu_presolve/baseline/run_baseline.py
OUT=gpu_presolve/results/stage0
STOCK=$DEPS/build/baseline_stock/baseline_tool
DIAG=$DEPS/build/baseline_diag/baseline_tool
STAGE1_OFF=probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns,parallel_columns
# HiGHS rule switches 5, 7, 8, 18, 19, 23: dominated column, forcing column, free column
# substitution, zero-cost singleton, column stuffing, weakly dominated column.
COLUMN_RULES_OFF=9175456

mkdir -p "$STATE"
rm -f "$STATE/ENDED"
FAILED=""
say() { echo "$(date +%F\ %T) $(hostname) $*" >> "$LOG"; }

# One driver at a time. The lock holds a heartbeat; a lock without a heartbeat in the last five
# minutes is taken over (its driver died with its node).
LOCK=$STATE/lock
if ! mkdir "$LOCK" 2> /dev/null; then
  age=$(($(date +%s) - $(stat -c %Y "$LOCK/heartbeat" 2> /dev/null || echo 0)))
  if [ "$age" -lt 300 ]; then
    say "another driver holds the lock ($(cat "$LOCK/owner" 2> /dev/null)), heartbeat ${age}s old: not starting"
    exit 3
  fi
  say "taking over a stale lock ($(cat "$LOCK/owner" 2> /dev/null), heartbeat ${age}s old)"
fi
echo "$(hostname) $$" > "$LOCK/owner"
touch "$LOCK/heartbeat"
(while [ -d "$LOCK" ] && [ "$(cat "$LOCK/owner" 2> /dev/null)" = "$(hostname) $$" ]; do
  touch "$LOCK/heartbeat"
  sleep 60
done) &
HEARTBEAT=$!
trap 'kill $HEARTBEAT 2> /dev/null; rm -rf "$LOCK"' EXIT

say "driver start, pid $$, load $(cut -d' ' -f1-3 /proc/loadavg), gpu apps $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null | wc -l)"

step() {
  name=$1
  shift
  if [ -f "$STATE/$name.done" ]; then
    say "step $name: already done"
    return
  fi
  say "step $name: start, load $(cut -d' ' -f1-3 /proc/loadavg), gpu apps $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null | wc -l)"
  "$@" >> "$LOG" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    touch "$STATE/$name.done"
  else
    FAILED="$FAILED $name"
  fi
  say "step $name: end, exit $code"
}

build_tools() {
  for tree in baseline_stock baseline_diag microbench; do
    cmake --build "$DEPS/build/$tree" -j 8 | tail -1 || return 1
  done
}

sweeps_large() {
  # Fails at once when the GPU this slot is pinned to is not visible in the container.
  if ! nvidia-smi -L | grep -q "${CUDA_VISIBLE_DEVICES:-GPU}"; then
    echo "pinned GPU ${CUDA_VISIBLE_DEVICES:-} is not visible here: $(nvidia-smi -L | head -2)"
    return 1
  fi
  mkdir -p "$OUT/sweeps_large" "$DEPS/results/stage0/sweeps_txt"
  status=0
  for dump in $(stat -c '%s %n' "$DUMPS"/*.csr | sort -n | cut -d' ' -f2-); do
    name=$(basename "$dump" .csr)
    if [ -f "$OUT/sweeps/$name.json" ] || [ -f "$OUT/sweeps_large/$name.json" ]; then
      echo "$name: sweep result present"
      continue
    fi
    "$DEPS/build/microbench/sweeps" "$dump" "$OUT/sweeps_large/$name.json.part" \
      > "$DEPS/results/stage0/sweeps_txt/$name.txt" 2>&1
    code=$?
    if [ -f "$OUT/sweeps_large/$name.json.part" ]; then
      mv "$OUT/sweeps_large/$name.json.part" "$OUT/sweeps_large/$name.json"
    fi
    echo "$name: sweeps exit $code $(date +%T)"
    [ "$code" -eq 0 ] || status=1
  done
  return $status
}

probing_without_limits() {
  # The clique table can grow without bound in this mode: cap the address space of each run.
  (
    ulimit -v 209715200
    python3 "$RUN" --set dev --variant diag_nostop --tool "$DIAG" --jobs 4 --time-limit 600 \
      --kill-after 1800 --env HIGHS_DIAG_PROBING_NOSTOP=1 --skip-done
  )
}

step build build_tools
# Timing check against the runs made before the container changed (same machine, other cores).
step stock_recheck python3 "$RUN" --set dev --variant stock_recheck --tool "$STOCK" --jobs 1 \
  --only wachplan,unitcal_7,uccase9,supportcase7,neos-4647030-tutaki --skip-done
# Stage 1 reference with the column rules inside HiGHS's row and column presolve switched off.
step diag_stage1b python3 "$RUN" --set dev --variant diag_stage1b --tool "$DIAG" --jobs 1 \
  --env "HIGHS_DIAG_OFF=$STAGE1_OFF" --presolve-rule-off "$COLUMN_RULES_OFF" --skip-done
step diag_nofm python3 "$RUN" --set dev --variant diag_nofm --tool "$DIAG" --jobs 1 \
  --only satellites2-60-fs --env HIGHS_DIAG_OFF=fourier_motzkin --skip-done
step sweeps_large sweeps_large
step diag_large python3 "$RUN" --set large --variant diag_large --tool "$DIAG" --jobs 1 \
  --time-limit 3600 --skip-done
step stock_large python3 "$RUN" --set large --variant stock_large --tool "$STOCK" --jobs 1 \
  --time-limit 3600 --skip-done
step diag_nostop probing_without_limits

echo "$FAILED" > "$STATE/ENDED"
say "driver end; failed steps:${FAILED:- none}"
