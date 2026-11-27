#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds the host engine on the node (release and assertion build), writes the CSR dumps of the
# testbed instances below 1e7 nonzeros to the scratch, runs the engine on them, and runs the
# correctness check (check/run_check.py) with solution files on the scratch. One step after the
# other. A step that ended with exit code 0 leaves a marker with the identity of the source tree
# (tools/run_info.py) and is skipped at the next start while the source tree is the same; the
# runners skip what has a result of the same source tree and binary. A failed build ends the
# run: nothing runs on a binary that is not the one of this source tree. The two build steps
# run at every start: the build directories are shared by all tags and source trees, so a
# marker cannot say that the binaries in them are still those of this tree.
#
# Start it detached from the worktree root, inside the dev container:
#   setsid nohup bash gpu_presolve/tools/stage1_host_runs.sh <tag> > /dev/null 2>&1 < /dev/null &
# <tag> names the engine version: results go to gpu_presolve/results/stage1/host/<tag>, markers
# to $DEPS/results/stage1/host_runs/<tag>. Progress: tail $DEPS/results/stage1/host_runs.log.
# The exit code is 0 when no step failed; $STATE/ENDED lists the failed steps.
#
# Solution files: $GPRS_SOLUTIONS_DIR (default $DEPS/solutions) gets what HiGHS finds on the
# original models; $GPRS_EXTERNAL_SOLUTIONS_DIR (default $DEPS/solutions_external) is read as
# well, for solutions converted from other sources.

set -u -o pipefail

TAG=${1:?usage: stage1_host_runs.sh <tag>}
DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
STATE=$DEPS/results/stage1/host_runs/$TAG
LOG=$DEPS/results/stage1/host_runs.log
DUMPS_DEV=$DEPS/csr_dumps_dev
DUMPS_LARGE=$DEPS/csr_dumps
SOLUTIONS=${GPRS_SOLUTIONS_DIR:-$DEPS/solutions}
SOLUTIONS_EXTERNAL=${GPRS_EXTERNAL_SOLUTIONS_DIR:-$DEPS/solutions_external}
OUT=gpu_presolve/results/stage1/host/$TAG
STOCK=$DEPS/build/baseline_stock/baseline_tool
RELEASE=$DEPS/build/gprs_host
ASSERT=$DEPS/build/gprs_host_assert
CHECK=$DEPS/build/gprs_check
RUN=gpu_presolve/tools/run_engine.py
RUN_CHECK=gpu_presolve/check/run_check.py
# Where the objective sense of an instance is read from.
STATS=(--stats-dir "$DEPS/results/stage1/dumps_dev_stats"
  --stats-dir gpu_presolve/results/stage1/refs/stage1c
  --stats-dir gpu_presolve/results/stage1/refs/stage1c_large)

mkdir -p "$STATE"
FAILED=""
say() { echo "$(date +%F\ %T) $(hostname) [$TAG] $*" >> "$LOG"; }

LOCK=$DEPS/results/stage1/host_runs/lock
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
# From here on this run owns the tag: the list of an earlier run goes.
rm -f "$STATE/ENDED"

# The source tree the markers of this run stand for.
if ! IDENTITY=$(python3 gpu_presolve/tools/run_info.py); then
  say "cannot identify the source tree: not starting"
  echo " identity" > "$STATE/ENDED"
  exit 1
fi
say "start, pid $$, load $(cut -d' ' -f1-3 /proc/loadavg), source $IDENTITY"

# step <name> <command>...: returns the exit code of the command, 0 for a step that is done.
step() {
  local step_name=$1 code
  shift
  case "$step_name" in
    build | build_check) ;;
    *)
      if [ "$(cat "$STATE/$step_name.done" 2> /dev/null)" = "$IDENTITY" ]; then
        say "step $step_name: already done for this source tree"
        return 0
      fi
      ;;
  esac
  rm -f "$STATE/$step_name.done"
  say "step $step_name: start, load $(cut -d' ' -f1-3 /proc/loadavg)"
  "$@" >> "$LOG" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "$IDENTITY" > "$STATE/$step_name.done"
  else
    FAILED="$FAILED $step_name"
  fi
  say "step $step_name: end, exit $code"
  return "$code"
}

finish() {
  echo "$FAILED" > "$STATE/ENDED"
  say "end; failed steps:${FAILED:- none}"
  [ -z "$FAILED" ]
  exit $?
}

# logged <log file> <command>...: the output goes to the file; on failure its end is shown.
logged() {
  local log=$1
  shift
  if ! "$@" > "$log" 2>&1; then
    echo "failed: $*"
    tail -20 "$log"
    return 1
  fi
}

build() {
  logged "$RELEASE.configure.log" cmake -S gpu_presolve -B "$RELEASE" -G Ninja || return 1
  logged "$RELEASE.build.log" cmake --build "$RELEASE" -j 16 || return 1
  logged "$ASSERT.configure.log" cmake -S gpu_presolve -B "$ASSERT" -G Ninja -DGPRS_ASSERT=ON ||
    return 1
  logged "$ASSERT.build.log" cmake --build "$ASSERT" -j 16 || return 1
  logged "$RELEASE.tests.log" "$RELEASE/gprs_tests" || return 1
  tail -1 "$RELEASE.tests.log"
  logged "$ASSERT.tests.log" "$ASSERT/gprs_tests" || return 1
  tail -1 "$ASSERT.tests.log"
}

build_check() {
  logged "$CHECK.configure.log" cmake -S gpu_presolve/check -B "$CHECK" -G Ninja || return 1
  logged "$CHECK.build.log" cmake --build "$CHECK" -j 16 || return 1
}

# The dumps do not depend on the engine version: their marker is shared between tags.
dumps() {
  local set count
  if [ -f "$DEPS/results/stage1/host_runs/dumps_dev.done" ]; then
    return 0
  fi
  mkdir -p "$DUMPS_DEV"
  for set in dev status; do
    python3 gpu_presolve/baseline/run_baseline.py --set "$set" --variant dumps --tool "$STOCK" \
      --jobs 8 --stats-only --dump-csr-dir "$DUMPS_DEV" \
      --out "$DEPS/results/stage1/dumps_dev_stats" --skip-done | tail -2 ||
      echo "run_baseline.py --set $set: exit code ${PIPESTATUS[0]}"
  done
  count=$(find "$DUMPS_DEV" -name '*.csr' | wc -l)
  echo "$count dumps in $DUMPS_DEV"
  [ "$count" -ge 35 ] || return 1
  touch "$DEPS/results/stage1/host_runs/dumps_dev.done"
}

step build build || finish
step dumps dumps || finish
# The assertion build on every instance: a reduction that breaks an invariant stops here.
step assert_dev python3 "$RUN" --tool "$ASSERT/gprs_presolve" --dumps "$DUMPS_DEV" \
  --out "$OUT/assert_dev" --set dev --reps 1 --jobs 8 --kill-after 1800 --skip-done
step assert_status python3 "$RUN" --tool "$ASSERT/gprs_presolve" --dumps "$DUMPS_DEV" \
  --out "$OUT/assert_status" --set status --reps 1 --jobs 4 --kill-after 1800 --skip-done
# The timed runs: one core, three repetitions in turn, nothing else running in this script.
step release_dev python3 "$RUN" --tool "$RELEASE/gprs_presolve" --dumps "$DUMPS_DEV" \
  --out "$OUT/release_dev" --set dev --reps 3 --jobs 1 --kill-after 1800 --skip-done
step release_large python3 "$RUN" --tool "$RELEASE/gprs_presolve" --dumps "$DUMPS_LARGE" \
  --out "$OUT/release_large" --set large --reps 1 --jobs 1 --kill-after 3600 --skip-done

# The correctness check. Solution files first: HiGHS on the original models, every improving
# solution. Then per instance the crush check in default and primal-only mode and, below 1e7
# nonzeros, the relaxations and the MIP of the reduced model.
step build_check build_check || finish
mkdir -p "$SOLUTIONS" "$SOLUTIONS_EXTERNAL"
CHECK_ARGS=(--tool "$CHECK/gprs_check" --dumps "$DUMPS_DEV" --dumps "$DUMPS_LARGE" "${STATS[@]}"
  --solutions-dir "$SOLUTIONS" --solutions-dir "$SOLUTIONS_EXTERNAL" --skip-done)
step solutions_dev python3 "$RUN_CHECK" "${CHECK_ARGS[@]}" --set dev --make-solutions \
  --solution-time-limit 300 --jobs 4 --threads 4
step check_dev python3 "$RUN_CHECK" "${CHECK_ARGS[@]}" --out "$OUT/check_dev" --set dev \
  --lp-time-limit 120 --mip-time-limit 300 --jobs 4 --threads 4
# The instances without a solution: the reduced bnatt500 and fhnw-binpack4-4 must be infeasible.
step check_status python3 "$RUN_CHECK" "${CHECK_ARGS[@]}" --out "$OUT/check_status" --set status \
  --only bnatt500,fhnw-binpack4-4,neos-2075418-temuka \
  --lp-time-limit 120 --mip-time-limit 300 --jobs 3 --threads 4
step check_status_large python3 "$RUN_CHECK" "${CHECK_ARGS[@]}" --out "$OUT/check_status_large" \
  --set status --only neos-3230511-yuna --lp-time-limit 0 --mip-time-limit 0 --cost-variants 0 \
  --jobs 1
# Above 1e7 nonzeros: no solve of the reduced model, structure and crush check alone, and one
# cost variant (two more presolves). Few points per instance, and one pushed point for each: a
# point of dlr2 is 290 MB. An instance without a usable point fails the step: its crush check
# did not run.
step solutions_large python3 "$RUN_CHECK" "${CHECK_ARGS[@]}" --set large --make-solutions \
  --solution-time-limit 1800 --max-points 4 --jobs 2 --threads 8
step check_large python3 "$RUN_CHECK" "${CHECK_ARGS[@]}" --out "$OUT/check_large" --set large \
  --lp-time-limit 0 --mip-time-limit 0 --cost-variants 1 --push-points 1 --require-points \
  --jobs 2

finish
