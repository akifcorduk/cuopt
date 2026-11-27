#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# HiGHS references for Stage 1c (the cheap column rules), made with the instrumented build diag2.
# Presolve only, unscaled models. One step after the other; a step that ended with exit code 0
# leaves a marker and is skipped at the next start, and the runner skips instances that have a
# result. These runs give sizes and counters; they are not timing runs (several jobs at a time).
#
# Start it detached from the worktree root, inside the dev container:
#   setsid nohup bash gpu_presolve/baseline/stage1c_reference_runs.sh > /dev/null 2>&1 < /dev/null &
# Progress: tail $DEPS/results/stage1c/ref_runs.log. At the end $STATE/ENDED holds the failed
# steps. Results: gpu_presolve/results/bench240/stage1c_refs/<step> and, for the testbed,
# gpu_presolve/results/stage1c/refs/<step> (summary.csv and summary.json of each step).
#
# The steps:
#   bobin_test    Stage 1 switch set with the zero-cost singleton rule on (rule bits 5, 7, 8, 19,
#                 23 off), on neos-3402294-bobin and neos-3402454-bohle: does the lock rule after
#                 that rule reach bobin's reduction?
#   m1_a .. m3_a  cumulative ladder on the 240 benchmark instances with rule bits 5, 7, 19, 23 off
#                 (no dominated, weakly dominated or forcing column, no stuffing): the singleton
#                 column rules on (m1), simple probing on equations as well (m2), parallel
#                 columns as well (m3). The next step of this ladder, with the dual-fixing
#                 extensions, is results/bench240/cheap_ablation/off_leftout.csv.
#   m1_b .. m3_b  the same ladder with only rule bit 19 off, which is how HiGHS has the forcing
#                 column: what the dominated-column tests add at each step. Its next step is
#                 results/bench240/cheap_ablation/off_leftout_minor.csv.
#   cheap_dev, cheap_large
#                 all cheap rules on (HiGHS's expensive stages off) on the testbed: the reference
#                 of the Stage 1c gate there. For the 240 it is results/bench240/highs_cheap_all.
#   tb_<step>_dev, tb_<step>_large
#                 the ladder on the testbed, where two instances above 1e7 nonzeros
#                 (neos-4332810-sesia, dlr2) carry the distance to all cheap rules: m1_a to m3_a
#                 as above, m4_a (the dual-fixing extensions as well; the switch set of
#                 off_leftout.csv), m4_b (the same with the dominated-column tests; the switch
#                 set of off_leftout_minor.csv) and off5 (all cheap rules without the dominated,
#                 weakly dominated and forcing column).

set -u -o pipefail

DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
STATE=$DEPS/results/stage1c/ref_runs
LOG=$DEPS/results/stage1c/ref_runs.log
RUN=gpu_presolve/baseline/run_baseline.py
BENCH=gpu_presolve/testbed/benchmark240.json
SCRATCH=$DEPS/results/stage1c/refs
OUT_BENCH=gpu_presolve/results/bench240/stage1c_refs
OUT_TESTBED=gpu_presolve/results/stage1c/refs
DIAG2=$DEPS/build/baseline_diag2/baseline_tool
HEAVY=probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns
MINOR=implied_equations,implied_integer_conversion
M1_OFF=$HEAVY,$MINOR,dual_fixing_substitution,parallel_columns,simple_probing
M2_OFF=$HEAVY,$MINOR,dual_fixing_substitution,parallel_columns
M3_OFF=$HEAVY,$MINOR,dual_fixing_substitution
# Rule bits 5 (dominated column), 7 (forcing column), 19 (stuffing), 23 (weakly dominated).
LADDER_A=8913056
# Rule bit 19 only.
LADDER_B=524288
# Rule bits 5, 7, 8 (free column substitution), 19, 23.
BOBIN=8913312

mkdir -p "$STATE" "$SCRATCH"
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

# step <name> <directory of the copy> <runner arguments>: run, then copy the two summaries.
step() {
  local step_name=$1 copy_to=$2 code
  shift 2
  if [ -f "$STATE/$step_name.done" ]; then
    say "step $step_name: already done"
    return 0
  fi
  say "step $step_name: start, load $(cut -d' ' -f1-3 /proc/loadavg)"
  python3 "$RUN" --variant "$step_name" --tool "$DIAG2" --out "$SCRATCH/$step_name" --skip-done \
    "$@" >> "$LOG" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    mkdir -p "$copy_to/$step_name" &&
      cp "$SCRATCH/$step_name/summary.csv" "$SCRATCH/$step_name/summary.json" \
        "$copy_to/$step_name/" || code=90
  fi
  if [ "$code" -eq 0 ]; then
    touch "$STATE/$step_name.done"
  else
    FAILED="$FAILED $step_name"
  fi
  say "step $step_name: end, exit $code"
  return 0
}

bench() {
  local step_name=$1 off=$2 rules=$3
  shift 3
  step "$step_name" "$OUT_BENCH" --testbed "$BENCH" --set all --jobs 8 --time-limit 600 \
    --env "HIGHS_DIAG_OFF=$off" --presolve-rule-off "$rules" "$@"
}

bench bobin_test "$M1_OFF" "$BOBIN" --only neos-3402294-bobin,neos-3402454-bohle
bench m1_a "$M1_OFF" "$LADDER_A"
bench m2_a "$M2_OFF" "$LADDER_A"
bench m3_a "$M3_OFF" "$LADDER_A"
step cheap_dev "$OUT_TESTBED" --set dev --jobs 4 --time-limit 1800 --env "HIGHS_DIAG_OFF=$HEAVY"
bench m1_b "$M1_OFF" "$LADDER_B"
bench m2_b "$M2_OFF" "$LADDER_B"
bench m3_b "$M3_OFF" "$LADDER_B"
step cheap_large "$OUT_TESTBED" --set large --jobs 1 --time-limit 3600 \
  --env "HIGHS_DIAG_OFF=$HEAVY"

testbed() {
  local step_name=$1 off=$2 rules=$3
  step "tb_${step_name}_dev" "$OUT_TESTBED" --set dev --jobs 4 --time-limit 1800 \
    --env "HIGHS_DIAG_OFF=$off" --presolve-rule-off "$rules"
  step "tb_${step_name}_large" "$OUT_TESTBED" --set large --jobs 3 --time-limit 3600 \
    --env "HIGHS_DIAG_OFF=$off" --presolve-rule-off "$rules"
}

testbed m1_a "$M1_OFF" "$LADDER_A"
testbed m2_a "$M2_OFF" "$LADDER_A"
testbed m3_a "$M3_OFF" "$LADDER_A"
testbed m4_a "$HEAVY,$MINOR" "$LADDER_A"
testbed m4_b "$HEAVY,$MINOR" "$LADDER_B"
testbed off5 "$HEAVY" 32

echo "$FAILED" > "$STATE/ENDED"
say "end; failed steps:${FAILED:- none}"
[ -z "$FAILED" ]
