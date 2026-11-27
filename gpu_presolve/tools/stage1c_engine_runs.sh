#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Node runs of one engine version of Stage 1c (the cheap column rules). The version is a commit
# on the fork. It is checked out as a git worktree of the clone $DEPS/src/v4 and built into its
# own directories, so the synced tree and the builds of other versions are not touched:
#   $DEPS/src/<tag>               the source tree at <commit>
#   $DEPS/build/<tag>/release     engine, tests, tools
#   $DEPS/build/<tag>/assert      the same with assertions
#   $DEPS/build/<tag>/check       the check tool gprs_check
# Runners and check scripts are those of the version's own source tree.
#
# Start it detached from the worktree root, inside the dev container:
#   setsid nohup bash gpu_presolve/tools/stage1c_engine_runs.sh <tag> <commit> [<base tag>] \
#     > /dev/null 2>&1 < /dev/null &
# <commit> is the full hash. Progress: tail $DEPS/results/stage1c/engine_runs.log. At the end
# $STATE/ENDED lists the failed steps (empty: none). A step that ended with exit code 0 leaves
# a marker with the commit and is skipped at the next start; the runners skip instances that
# have a result of the same source tree and binary.
#
# Environment:
#   GPRS_OFF   families or sub-rules (names of gprs_presolve --off, separated by spaces) that
#              the "<tag>_off" runs switch off. With the rules of a milestone off the engine
#              must give the results of the milestone before it.
#   GPRS_SKIP  step groups to leave out, separated by spaces: assert, check, check_solver,
#              timing, large.
#
# Steps and where their results go (below gpu_presolve/results in the synced tree):
#   source, build, build_check
#   assert_all, assert_dev, assert_status
#       the assertion build on the 240 benchmark instances and on the testbed below 1e7
#       nonzeros: bench240/<tag>/assert_all, stage1/host/<tag>/assert_dev and assert_status
#   release_all, release_dev, release_large
#       one process at a time, three repetitions (one on the large set):
#       bench240/<tag>/release_all, stage1/host/<tag>/release_dev and release_large
#   off_all, off_dev, off_large
#       with GPRS_OFF, several processes: bench240/<tag>_off/release_all,
#       stage1/host/<tag>_off/release_dev and release_large
#   check_all, check_dev, check_status, check_large
#       structure and crush check without a solver: bench240/<tag>/check_all,
#       stage1/host/<tag>/check_dev, check_status and check_large
#   check_solver_dev
#       the check with HiGHS on the testbed below 1e7 nonzeros (relaxations, MIP of the
#       reduced model): stage1/host/<tag>/check_solver_dev
#   time_<side>_<set>_<pass>
#       with a base tag: three passes over the 240 and the testbed below 1e7 nonzeros (before
#       the checks) and over the large set (after them); in each pass the base binary, this
#       version and (with GPRS_OFF) this version with the rules off, one process at a time:
#       stage1c/timing/<tag>/<side>_<set>_<pass> with side base, new or off
# analysis/stage1c_identity.py compares two versions result by result;
# analysis/stage1c_time.py reads the timing passes.

set -u -o pipefail

TAG=${1:?usage: stage1c_engine_runs.sh <tag> <commit> [<base tag>]}
COMMIT=${2:?usage: stage1c_engine_runs.sh <tag> <commit> [<base tag>]}
BASE=${3:-}
DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
REPO=$DEPS/src/v4
SRC=$DEPS/src/$TAG
BUILD=$DEPS/build/$TAG
STATE=$DEPS/results/stage1c/engine_runs/$TAG
LOG=$DEPS/results/stage1c/engine_runs.log
OUT_BENCH=gpu_presolve/results/bench240
OUT_HOST=gpu_presolve/results/stage1/host
OUT_TIME=gpu_presolve/results/stage1c/timing/$TAG
BENCH=(--dumps "$DEPS/csr_dumps_bench" --testbed gpu_presolve/testbed/benchmark240.json --set all)
DEV=(--dumps "$DEPS/csr_dumps_dev" --set dev)
STATUS=(--dumps "$DEPS/csr_dumps_dev" --set status)
LARGE=(--dumps "$DEPS/csr_dumps" --set large)
# Where the objective sense of an instance is read from.
STATS=(--stats-dir "$DEPS/results/stage1/dumps_dev_stats"
  --stats-dir gpu_presolve/results/stage1/refs/stage1c
  --stats-dir gpu_presolve/results/stage1/refs/stage1c_large
  --stats-dir "$DEPS/results/bench240/stats")
# Solution files: the testbed's (found by HiGHS, and converted published ones) and those of
# the 240 benchmark instances.
SOLUTIONS=(--solutions-dir "$DEPS/solutions" --solutions-dir "$DEPS/solutions_external"
  --solutions-dir "$DEPS/solutions_bench")
OFF_ARGS=()
for name in ${GPRS_OFF:-}; do
  OFF_ARGS+=("--tool-arg=--off" "--tool-arg=$name")
done

mkdir -p "$STATE" "$BUILD"
FAILED=""
say() { echo "$(date +%F\ %T) $(hostname) [$TAG] $*" >> "$LOG"; }
skip() {
  case " ${GPRS_SKIP:-} " in
    *" $1 "*) return 0 ;;
  esac
  return 1
}

LOCK=$DEPS/results/stage1c/engine_runs/lock
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

say "start, pid $$, load $(cut -d' ' -f1-3 /proc/loadavg), commit $COMMIT, base ${BASE:-none}, off: ${GPRS_OFF:-none}, skip: ${GPRS_SKIP:-none}"

# step <name> <command>...: returns the exit code of the command, 0 for a step that is done.
step() {
  local step_name=$1 code
  shift
  case "$step_name" in
    source | build | build_check) ;;
    *)
      if [ "$(cat "$STATE/$step_name.done" 2> /dev/null)" = "$COMMIT" ]; then
        say "step $step_name: already done for this commit"
        return 0
      fi
      ;;
  esac
  rm -f "$STATE/$step_name.done"
  say "step $step_name: start, load $(cut -d' ' -f1-3 /proc/loadavg)"
  "$@" >> "$LOG" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "$COMMIT" > "$STATE/$step_name.done"
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

source_tree() {
  # The clone follows one branch only: fetch the commit itself.
  git -C "$REPO" fetch -q origin "$COMMIT" || return 1
  if [ ! -e "$SRC/.git" ]; then
    git -C "$REPO" worktree add -q --detach "$SRC" "$COMMIT" || return 1
  fi
  if [ "$(git -C "$SRC" rev-parse HEAD)" != "$COMMIT" ]; then
    git -C "$SRC" checkout -q --detach "$COMMIT" || return 1
  fi
  if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
    echo "the source tree $SRC has changes"
    return 1
  fi
  git -C "$SRC" log -1 --format='%H %s'
}

build() {
  local kind
  for kind in release assert; do
    local flags=()
    [ "$kind" = assert ] && flags=(-DGPRS_ASSERT=ON)
    logged "$BUILD/$kind.configure.log" cmake -S "$SRC/gpu_presolve" -B "$BUILD/$kind" -G Ninja \
      ${flags[@]+"${flags[@]}"} || return 1
    logged "$BUILD/$kind.build.log" cmake --build "$BUILD/$kind" -j 16 || return 1
    logged "$BUILD/$kind.tests.log" "$BUILD/$kind/gprs_tests" || return 1
    tail -1 "$BUILD/$kind.tests.log"
  done
}

build_check() {
  logged "$BUILD/check.configure.log" cmake -S "$SRC/gpu_presolve/check" -B "$BUILD/check" \
    -G Ninja || return 1
  logged "$BUILD/check.build.log" cmake --build "$BUILD/check" -j 16 || return 1
}

# engine <step> <source tree> <binary> <result directory> <runner arguments>...
engine() {
  local step_name=$1 src=$2 tool=$3 out=$4
  shift 4
  step "$step_name" python3 "$src/gpu_presolve/tools/run_engine.py" --tool "$tool" --out "$out" \
    --kill-after 3600 --skip-done "$@"
}

# check <step> <result directory> <runner arguments>...
check() {
  local step_name=$1 out=$2
  shift 2
  step "$step_name" python3 "$SRC/gpu_presolve/check/run_check.py" \
    --tool "$BUILD/check/gprs_check" "${STATS[@]}" "${SOLUTIONS[@]}" --skip-done --out "$out" \
    "$@"
}

step source source_tree || finish
step build build || finish
RELEASE=$BUILD/release/gprs_presolve
ASSERT=$BUILD/assert/gprs_presolve

# The assertion build on every instance: a reduction that breaks an invariant stops here.
if ! skip assert; then
  engine assert_all "$SRC" "$ASSERT" "$OUT_BENCH/$TAG/assert_all" "${BENCH[@]}" --reps 1 --jobs 8
  engine assert_dev "$SRC" "$ASSERT" "$OUT_HOST/$TAG/assert_dev" "${DEV[@]}" --reps 1 --jobs 8
  engine assert_status "$SRC" "$ASSERT" "$OUT_HOST/$TAG/assert_status" "${STATUS[@]}" --reps 1 \
    --jobs 4
fi

# The results of the version: one process at a time, so that the seconds can be read.
engine release_all "$SRC" "$RELEASE" "$OUT_BENCH/$TAG/release_all" "${BENCH[@]}" --reps 3 --jobs 1
engine release_dev "$SRC" "$RELEASE" "$OUT_HOST/$TAG/release_dev" "${DEV[@]}" --reps 3 --jobs 1
skip large ||
  engine release_large "$SRC" "$RELEASE" "$OUT_HOST/$TAG/release_large" "${LARGE[@]}" --reps 1 \
    --jobs 1

# The same with the named rules off: sizes and hashes only, several processes.
if [ ${#OFF_ARGS[@]} -gt 0 ]; then
  engine off_all "$SRC" "$RELEASE" "$OUT_BENCH/${TAG}_off/release_all" "${BENCH[@]}" --reps 1 \
    --jobs 8 "${OFF_ARGS[@]}"
  engine off_dev "$SRC" "$RELEASE" "$OUT_HOST/${TAG}_off/release_dev" "${DEV[@]}" --reps 1 \
    --jobs 8 "${OFF_ARGS[@]}"
  skip large ||
    engine off_large "$SRC" "$RELEASE" "$OUT_HOST/${TAG}_off/release_large" "${LARGE[@]}" \
      --reps 1 --jobs 2 "${OFF_ARGS[@]}"
fi

# Time against the base version: three passes, the binaries in turn, one process at a time.
# timing_set <set name> <runner arguments>...
timing_set() {
  local set_name=$1 pass
  shift
  for pass in 1 2 3; do
    engine "time_base_${set_name}_$pass" "$DEPS/src/$BASE" \
      "$DEPS/build/$BASE/release/gprs_presolve" "$OUT_TIME/base_${set_name}_$pass" "$@" \
      --reps 1 --jobs 1
    engine "time_new_${set_name}_$pass" "$SRC" "$RELEASE" "$OUT_TIME/new_${set_name}_$pass" \
      "$@" --reps 1 --jobs 1
    if [ ${#OFF_ARGS[@]} -gt 0 ]; then
      engine "time_off_${set_name}_$pass" "$SRC" "$RELEASE" "$OUT_TIME/off_${set_name}_$pass" \
        "$@" --reps 1 --jobs 1 "${OFF_ARGS[@]}"
    fi
  done
}

TIMING=""
if [ -n "$BASE" ] && ! skip timing; then
  if [ -x "$DEPS/build/$BASE/release/gprs_presolve" ]; then
    TIMING=yes
  else
    say "no release build of the base tag $BASE: no timing"
    FAILED="$FAILED timing_no_base"
  fi
fi
if [ -n "$TIMING" ]; then
  timing_set all "${BENCH[@]}"
  timing_set dev "${DEV[@]}"
fi

# The correctness check without a solver: presolve status, structure check, crush check with
# the stored solution files, in default and primal-only mode.
if ! skip check; then
  if step build_check build_check; then
    NO_SOLVER=(--lp-time-limit 0 --mip-time-limit 0)
    check check_all "$OUT_BENCH/$TAG/check_all" "${BENCH[@]}" "${NO_SOLVER[@]}" \
      --cost-variants 1 --push-points 4 --jobs 8
    check check_dev "$OUT_HOST/$TAG/check_dev" "${DEV[@]}" "${NO_SOLVER[@]}" --jobs 8
    check check_status "$OUT_HOST/$TAG/check_status" "${STATUS[@]}" \
      --only bnatt500,fhnw-binpack4-4,neos-2075418-temuka "${NO_SOLVER[@]}" --jobs 3
    skip large ||
      check check_large "$OUT_HOST/$TAG/check_large" "${LARGE[@]}" "${NO_SOLVER[@]}" \
        --cost-variants 1 --push-points 1 --require-points --jobs 2
    skip check_solver ||
      check check_solver_dev "$OUT_HOST/$TAG/check_solver_dev" "${DEV[@]}" --lp-time-limit 120 \
        --mip-time-limit 300 --jobs 4 --threads 4
  fi
fi

if [ -n "$TIMING" ] && ! skip large; then
  timing_set large "${LARGE[@]}"
fi

finish
