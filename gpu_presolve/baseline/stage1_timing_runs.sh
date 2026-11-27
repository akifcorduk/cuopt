#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Times the Stage 1 reference (HiGHS, build diag2, variant stage1c) and the host engine on the
# same machine, in turn: three passes over the instances below 1e7 nonzeros, then three over the
# nine above, each pass first the reference and then the engine, one process at a time. The time
# rule of the plan (t <= 1.2 * t(reference) + 0.02 s on the minimum of three alternating runs) is
# applied to the result by analysis/stage1_time_gate.py <name>.
#
# Nothing else of this project should run on the machine meanwhile. Start it detached from the
# worktree root, inside the dev container:
#   setsid nohup bash gpu_presolve/baseline/stage1_timing_runs.sh <name> > /dev/null 2>&1 < /dev/null &
# <name> names the measurement (machine and engine version): results go to
# gpu_presolve/results/stage1/timing/<name>, markers to $DEPS/results/stage1/timing_runs/<name>.
# Progress: tail $DEPS/results/stage1/timing_runs.log. A step that ended with exit code 0 is
# skipped at the next start. The engine binary is $GPRS_ENGINE_TOOL (default: the release build
# of tools/stage1_host_runs.sh); its checksum is in every engine summary.

set -u -o pipefail

NAME=${1:?usage: stage1_timing_runs.sh <name>}
DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
STATE=$DEPS/results/stage1/timing_runs/$NAME
LOG=$DEPS/results/stage1/timing_runs.log
OUT=gpu_presolve/results/stage1/timing/$NAME
DIAG2=$DEPS/build/baseline_diag2/baseline_tool
ENGINE=${GPRS_ENGINE_TOOL:-$DEPS/build/gprs_host/gprs_presolve}
RUN_REF=gpu_presolve/baseline/run_baseline.py
RUN_ENGINE=gpu_presolve/tools/run_engine.py
# The switch set of the Stage 1 reference, as in stage1_reference_runs.sh.
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

reference() {
  local set=$1 variant=$2 out=$3
  shift 3
  python3 "$RUN_REF" --set "$set" --variant "$variant" --tool "$DIAG2" --jobs 1 --out "$out" \
    --skip-done --env "HIGHS_DIAG_OFF=$STAGE1_OFF" --presolve-rule-off "$COLUMN_RULES_OFF" "$@"
}

engine() {
  local set=$1 dumps=$2 out=$3 kill_after=$4
  python3 "$RUN_ENGINE" --tool "$ENGINE" --dumps "$dumps" --out "$out" --set "$set" --reps 1 \
    --jobs 1 --kill-after "$kill_after" --skip-done
}

for pass in 1 2 3; do
  step "ref_dev_$pass" reference dev stage1c "$OUT/ref_dev_$pass"
  step "engine_dev_$pass" engine dev "$DEPS/csr_dumps_dev" "$OUT/engine_dev_$pass" 1800
done
for pass in 1 2 3; do
  step "ref_large_$pass" reference large stage1c_large "$OUT/ref_large_$pass" --time-limit 3600
  step "engine_large_$pass" engine large "$DEPS/csr_dumps" "$OUT/engine_large_$pass" 3600
done

echo "$FAILED" > "$STATE/ENDED"
say "end; failed steps:${FAILED:- none}"
[ -z "$FAILED" ]
