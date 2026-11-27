#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The checks and the extra measurement of milestone 1, one process at a time:
#   <instance>.reverse_rows.json      rows in decreasing column order against the host loop
#   <instance>.wide.json              64-bit positions in the two sorts
#   <instance>.untouched.json         download into memory that was never touched
#   <instance>.untouched.hugetlb.json the same with GLIBC_TUNABLES=glibc.malloc.hugetlb=1
#   run_state_build_m1_extras.sh <build dir> <result dir> [tag]
# The tag goes between the instance and the suffix above, so that a second run of the checks
# leaves the files of the first. The script fails when a run fails.

set -u
build=$1
results=$2
tag=${3:-}
deps=/home/scratch.acoerduek_sw/gpu_presolve_deps
failed=0

dump() {
  if [ -f "$deps/csr_dumps_dev/$1.csr" ]; then
    echo "$deps/csr_dumps_dev/$1.csr"
  else
    echo "$deps/csr_dumps/$1.csr"
  fi
}

run() {
  local name=$1 suffix=$tag$2 status
  echo "== $name$suffix $(date +%H:%M:%S) GLIBC_TUNABLES=${GLIBC_TUNABLES:-}"
  "$build/state_build_bench" "$(dump "$name")" "$results/$name$suffix.json" "${@:3}"
  status=$?
  echo "exit=$status $name$suffix"
  if [ $status -ne 0 ]; then failed=$((failed + 1)); fi
}

mkdir -p "$results"
for name in istanbul-no-cutoff bab6 neos-5116085-kenana usafa t11nonreg; do
  run "$name" .reverse_rows --reverse-rows
done
for name in neos-5116085-kenana usafa dlr2; do
  run "$name" .wide --wide --transposes radix_sort,scatter_sort --transfers pageable --no-spmv \
    --host-reps 3
done
for name in neos-5116085-kenana usafa a2864-99blp neos-4332810-sesia neos-4535459-waipa; do
  run "$name" .untouched --untouched --no-spmv --host-reps 3
  GLIBC_TUNABLES=glibc.malloc.hugetlb=1 run "$name" .untouched.hugetlb --untouched --no-spmv \
    --host-reps 3
done
echo "== finished $(date +%H:%M:%S), $failed runs failed"
[ $failed -eq 0 ]
