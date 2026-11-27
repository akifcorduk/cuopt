#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The review run of milestone 1, one process at a time, restartable by step:
#   faults   the comparison of state_build_bench must fail on a wrong device result
#   memory   builds while the device memory is held (state_build_test --memory)
#   cold     first build of a process, default pages     <instance>.review.cold.jsonl
#   warm     the pass of run_state_build_m1.sh           <instance>.review.json
#   cold_hugetlb, warm_hugetlb: the same with GLIBC_TUNABLES=glibc.malloc.hugetlb=1
#   extras   run_state_build_m1_extras.sh                <instance>.review.<check>.json
#   models   state_build_test in the given build dirs (release, assertions, other CCCL)
#   big      state_build_test --big above 2^31 nonzeros: 140 GB of the device, 150 GB of RAM
#   run_state_build_review.sh <build dir> <result dir> <scratch dir> [steps...]
# REVIEW_OTHER_BUILDS: more build dirs for the step models.
# A finished step leaves <scratch dir>/<step>.done and is skipped by a later call. The log of a
# step is <scratch dir>/<step>.log.

set -u
build=$1
results=$2
scratch=$3
shift 3
steps=${*:-faults memory cold warm cold_hugetlb warm_hugetlb}
here=$(dirname "$0")
deps=/home/scratch.acoerduek_sw/gpu_presolve_deps
kenana=$deps/csr_dumps_dev/neos-5116085-kenana.csr

mkdir -p "$results" "$scratch"

# Every fault must end with exit code 3.
faults() {
  local wrong=0 kind status
  for kind in reverse_column drop_lock; do
    "$build/state_build_bench" "$kenana" "$results/neos-5116085-kenana.review.fault_$kind.json" \
      --fault $kind --transfers pageable --no-spmv --reps 2 --host-reps 1
    status=$?
    echo "fault $kind in every run: exit=$status"
    if [ $status -ne 3 ]; then wrong=1; fi
  done
  "$build/state_build_bench" "$kenana" "$results/neos-5116085-kenana.review.fault_one_run.json" \
    --fault drop_lock --fault-run 1 --transfers pageable --no-spmv --reps 3 --host-reps 1
  status=$?
  echo "fault drop_lock in the second timed run only: exit=$status"
  if [ $status -ne 3 ]; then wrong=1; fi
  return $wrong
}

step() {
  case $1 in
    faults) faults ;;
    memory) "$build/state_build_test" --memory ;;
    cold) bash "$here/run_state_build_cold.sh" "$build" "$results" .review ;;
    warm) bash "$here/run_state_build_m1.sh" "$build" "$results" .review ;;
    cold_hugetlb)
      GLIBC_TUNABLES=glibc.malloc.hugetlb=1 \
        bash "$here/run_state_build_cold.sh" "$build" "$results" .review.hugetlb 3 1 1
      ;;
    warm_hugetlb)
      GLIBC_TUNABLES=glibc.malloc.hugetlb=1 \
        bash "$here/run_state_build_m1.sh" "$build" "$results" .review.hugetlb
      ;;
    extras) bash "$here/run_state_build_m1_extras.sh" "$build" "$results" .review ;;
    models)
      local tree wrong=0
      for tree in "$build" ${REVIEW_OTHER_BUILDS:-}; do
        echo "== state_build_test of $tree"
        "$tree/state_build_test" || wrong=1
      done
      return $wrong
      ;;
    big) "$build/state_build_test" --big 2164260864 ;;
  esac
}

for name in $steps; do
  if [ -f "$scratch/$name.done" ]; then continue; fi
  echo "[review] $name begins $(date +%H:%M:%S), load $(cut -d' ' -f1-3 /proc/loadavg)"
  step "$name" > "$scratch/$name.log" 2>&1
  status=$?
  echo "[review] $name ends $(date +%H:%M:%S) exit=$status"
  if [ $status -eq 0 ]; then touch "$scratch/$name.done"; fi
done
echo "[review] finished $(date +%H:%M:%S)"
