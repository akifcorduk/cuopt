#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The timing of stage1_timing_runs.sh on the 240 benchmark instances: the HiGHS Stage 1 reference
# (build diag2, the switch set of variant stage1c) and the host engine in three passes over
# testbed/benchmark240.json, each pass first the reference and then the engine, one process at a
# time. analysis/stage1_time_gate.py <name> applies the time rule to the minimum over the passes.
#
# Start it detached from the worktree root, inside the dev container:
#   setsid nohup bash gpu_presolve/baseline/bench240_timing_runs.sh <name> > /dev/null 2>&1 < /dev/null &
# Results: gpu_presolve/results/stage1/timing/<name>/ref_all_<k> and engine_all_<k>; markers in
# $DEPS/results/stage1/timing_runs/<name>; progress in $DEPS/results/stage1/timing_runs.log. It
# shares the lock of stage1_timing_runs.sh, so two timings never run side by side. The engine
# binary is $GPRS_ENGINE_TOOL (default: the release build of tools/stage1_host_runs.sh).

set -u -o pipefail

NAME=${1:?usage: bench240_timing_runs.sh <name>}
DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
STATE=$DEPS/results/stage1/timing_runs/$NAME
LOG=$DEPS/results/stage1/timing_runs.log
OUT=gpu_presolve/results/stage1/timing/$NAME
TESTBED=gpu_presolve/testbed/benchmark240.json
DIAG2=$DEPS/build/baseline_diag2/baseline_tool
ENGINE=${GPRS_ENGINE_TOOL:-$DEPS/build/gprs_host/gprs_presolve}
NEW=simple_probing,dual_fixing_substitution,implied_equations,implied_integer_conversion
STAGE1_OFF=probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns,parallel_columns,$NEW
COLUMN_RULES_OFF=9175456

mkdir -p "$STATE"
rm -f "$STATE/ENDED"
FAILED=""
say() { echo "$(date +%F\ %T) $(hostname) [$NAME] $*" >> "$LOG"; }

LOCK=$DEPS/results/stage1/timing_runs/lock
if ! mkdir "$LOCK" 2> /dev/null; then
  age=$(($(date +%s) - $(stat -c %Y "$LOCK/heartbeat" 2> /dev/null || echo 0)))
  if [ "$age" -lt 300 ]; then
    say "another run holds the lock ($(cat "$LOCK/owner" 2> /dev/null)), heartbeat ${age}s old: not starting"
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

say "start, pid $$, load $(cut -d' ' -f1-3 /proc/loadavg), engine $ENGINE $(sha256sum "$ENGINE" | cut -c1-12)"

step() {
  local step_name=$1 code
  shift
  if [ -f "$STATE/$step_name.done" ]; then
    say "step $step_name: already done"
    return 0
  fi
  say "step $step_name: start, load $(cut -d' ' -f1-3 /proc/loadavg)"
  "$@" >> "$LOG" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    touch "$STATE/$step_name.done"
  else
    FAILED="$FAILED $step_name"
  fi
  say "step $step_name: end, exit $code"
  return "$code"
}

for pass in 1 2 3; do
  step "ref_all_$pass" python3 gpu_presolve/baseline/run_baseline.py --testbed "$TESTBED" \
    --set all --variant stage1c --tool "$DIAG2" --jobs 1 --time-limit 900 \
    --out "$DEPS/results/stage1/timing_runs/$NAME/ref_all_$pass" --skip-done \
    --env "HIGHS_DIAG_OFF=$STAGE1_OFF" --presolve-rule-off "$COLUMN_RULES_OFF"
  mkdir -p "$OUT/ref_all_$pass"
  cp "$DEPS/results/stage1/timing_runs/$NAME/ref_all_$pass/summary.json" "$OUT/ref_all_$pass/" ||
    FAILED="$FAILED copy_ref_$pass"
  step "engine_all_$pass" python3 gpu_presolve/tools/run_engine.py --tool "$ENGINE" \
    --dumps "$DEPS/csr_dumps_bench" --testbed "$TESTBED" --out "$OUT/engine_all_$pass" \
    --set all --reps 1 --jobs 1 --kill-after 1800 --skip-done
done

echo "$FAILED" > "$STATE/ENDED"
say "end; failed steps:${FAILED:- none}"
[ -z "$FAILED" ]
