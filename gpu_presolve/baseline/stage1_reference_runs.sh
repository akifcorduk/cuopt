#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# HiGHS references for the Stage 1 and Stage 2 gates, made with the extended diagnostic patch
# (build name diag2). One step after the other; a step that ended with exit code 0 leaves a
# marker and is skipped at the next start, and the runner skips instances that have a result.
#
# Start it detached from the worktree root, inside the dev container:
#   setsid nohup bash gpu_presolve/baseline/stage1_reference_runs.sh > /dev/null 2>&1 < /dev/null &
# Progress: tail $DEPS/results/stage1/ref_runs.log. At the end $STATE/ENDED holds the failed steps.

set -u

DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
STATE=$DEPS/results/stage1/ref_runs
LOG=$DEPS/results/stage1/ref_runs.log
RUN=gpu_presolve/baseline/run_baseline.py
OUT=gpu_presolve/results/stage1/refs
DIAG2=$DEPS/build/baseline_diag2/baseline_tool
NEW=simple_probing,dual_fixing_substitution,implied_equations,implied_integer_conversion
STAGE1_OFF=probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns,parallel_columns,$NEW
STAGE12_OFF=implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns,parallel_columns,$NEW
# HiGHS rule switches 5, 7, 8, 18, 19, 23; with 17 (dual fixing) as well for the bracket run.
COLUMN_RULES_OFF=9175456
COLUMN_RULES_AND_DUAL_FIXING_OFF=9306528

mkdir -p "$STATE"
rm -f "$STATE/ENDED"
FAILED=""
say() { echo "$(date +%F\ %T) $(hostname) $*" >> "$LOG"; }

LOCK=$STATE/lock
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

say "start, pid $$, load $(cut -d' ' -f1-3 /proc/loadavg)"

step() {
  local step_name=$1
  shift
  if [ -f "$STATE/$step_name.done" ]; then
    say "step $step_name: already done"
    return
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
}

ref() {
  variant=$1
  jobs=$2
  shift 2
  python3 "$RUN" --set dev --variant "$variant" --tool "$DIAG2" --jobs "$jobs" \
    --out "$OUT/$variant" --skip-done "$@"
}

# The Stage 1 reference of the plan: Stage 1 switch set, HiGHS's column rules off, the four
# rule families of Stage 3 inside row and column presolve off.
step stage1c ref stage1c 1 --env "HIGHS_DIAG_OFF=$STAGE1_OFF" --presolve-rule-off "$COLUMN_RULES_OFF"
# The same with HiGHS's dual-fixing rule off as well (also removes fixing by locks).
step stage1d ref stage1d 1 --env "HIGHS_DIAG_OFF=$STAGE1_OFF" \
  --presolve-rule-off "$COLUMN_RULES_AND_DUAL_FIXING_OFF"
# Row redundancy tested on the sums over explicit bounds.
step stage1c_redexp ref stage1c_redexp 1 --env "HIGHS_DIAG_OFF=$STAGE1_OFF" \
  --env HIGHS_DIAG_REDUNDANCY_EXPLICIT=1 --presolve-rule-off "$COLUMN_RULES_OFF"
# The Stage 2 reference: the same with enumeration, probing and the clique rules on.
step stage12c ref stage12c 1 --env "HIGHS_DIAG_OFF=$STAGE12_OFF" --env HIGHS_DIAG_PROBE_ROWS=0 \
  --presolve-rule-off "$COLUMN_RULES_OFF"
# Full HiGHS with every counter of the extended patch (fixings by rule, rows touched per probe,
# pair tests of dominated columns). Not a timing run.
step full_counters ref full_counters 4

# The Stage 1 reference on the nine instances above 1e7 nonzeros.
step stage1c_large python3 "$RUN" --set large --variant stage1c_large --tool "$DIAG2" --jobs 1 \
  --time-limit 3600 --out "$OUT/stage1c_large" --skip-done --env "HIGHS_DIAG_OFF=$STAGE1_OFF" \
  --presolve-rule-off "$COLUMN_RULES_OFF"

echo "$FAILED" > "$STATE/ENDED"
say "end; failed steps:${FAILED:- none}"
