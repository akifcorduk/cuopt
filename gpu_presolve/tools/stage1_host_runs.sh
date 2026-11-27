#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds the host engine on the node (release and assertion build), writes the CSR dumps of the
# testbed instances below 1e7 nonzeros to the scratch, and runs the engine on them. One step
# after the other; a step that ended with exit code 0 leaves a marker and is skipped at the next
# start, and the runner skips repetitions that have a result.
#
# Start it detached from the worktree root, inside the dev container:
#   setsid nohup bash gpu_presolve/tools/stage1_host_runs.sh <tag> > /dev/null 2>&1 < /dev/null &
# <tag> names the engine version: results go to gpu_presolve/results/stage1/host/<tag>, markers
# to $DEPS/results/stage1/host_runs/<tag>. Progress: tail $DEPS/results/stage1/host_runs.log.

set -u

TAG=${1:?usage: stage1_host_runs.sh <tag>}
DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
STATE=$DEPS/results/stage1/host_runs/$TAG
LOG=$DEPS/results/stage1/host_runs.log
DUMPS_DEV=$DEPS/csr_dumps_dev
DUMPS_LARGE=$DEPS/csr_dumps
OUT=gpu_presolve/results/stage1/host/$TAG
STOCK=$DEPS/build/baseline_stock/baseline_tool
RELEASE=$DEPS/build/gprs_host
ASSERT=$DEPS/build/gprs_host_assert
RUN=gpu_presolve/tools/run_engine.py

mkdir -p "$STATE"
rm -f "$STATE/ENDED"
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

build() {
  cmake -S gpu_presolve -B "$RELEASE" -G Ninja > "$RELEASE.configure.log" 2>&1 || return 1
  cmake --build "$RELEASE" -j 16 | tail -1 || return 1
  cmake -S gpu_presolve -B "$ASSERT" -G Ninja -DGPRS_ASSERT=ON > "$ASSERT.configure.log" 2>&1 ||
    return 1
  cmake --build "$ASSERT" -j 16 | tail -1 || return 1
  "$RELEASE/gprs_tests" | tail -1 && "$ASSERT/gprs_tests" | tail -1
}

# The dumps do not depend on the engine version: their marker is shared between tags.
dumps() {
  if [ -f "$DEPS/results/stage1/host_runs/dumps_dev.done" ]; then
    return 0
  fi
  mkdir -p "$DUMPS_DEV"
  for set in dev status; do
    python3 gpu_presolve/baseline/run_baseline.py --set "$set" --variant dumps --tool "$STOCK" \
      --jobs 8 --stats-only --dump-csr-dir "$DUMPS_DEV" \
      --out "$DEPS/results/stage1/dumps_dev_stats" --skip-done | tail -2
  done
  count=$(find "$DUMPS_DEV" -name '*.csr' | wc -l)
  echo "$count dumps in $DUMPS_DEV"
  [ "$count" -ge 35 ] || return 1
  touch "$DEPS/results/stage1/host_runs/dumps_dev.done"
}

step build build
step dumps dumps
# The assertion build on every instance: a reduction that breaks an invariant stops here.
step assert_dev python3 "$RUN" --tool "$ASSERT/gprs_presolve" --dumps "$DUMPS_DEV" \
  --out "$OUT/assert_dev" --set dev --reps 1 --jobs 8 --kill-after 1800 --skip-done
step assert_status python3 "$RUN" --tool "$ASSERT/gprs_presolve" --dumps "$DUMPS_DEV" \
  --out "$OUT/assert_status" --set status --reps 1 --jobs 4 --kill-after 1800 --skip-done
# The timed runs: one core, three repetitions in turn.
step release_dev python3 "$RUN" --tool "$RELEASE/gprs_presolve" --dumps "$DUMPS_DEV" \
  --out "$OUT/release_dev" --set dev --reps 3 --jobs 1 --kill-after 1800 --skip-done
step release_large python3 "$RUN" --tool "$RELEASE/gprs_presolve" --dumps "$DUMPS_LARGE" \
  --out "$OUT/release_large" --set large --reps 1 --jobs 1 --kill-after 3600 --skip-done

echo "$FAILED" > "$STATE/ENDED"
say "end; failed steps:${FAILED:- none}"
