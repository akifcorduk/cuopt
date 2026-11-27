#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The first build of a process, which is what a presolve call gets: every measurement is its
# own process (state_build_bench --cold and --cold-host), one at a time. Per instance: the host
# loop, the three transposes with the pageable transfer, and radix_sort with the two pinned
# transfers, each `processes` times.
#   run_state_build_cold.sh <build dir> <result dir> [suffix [processes [large [largest]]]]
# The lines of an instance go to <result dir>/<instance><suffix>.cold.jsonl. The instances are
# those of run_state_build_m1.sh and nine more between 5e5 and 8e6 nonzeros, where the device
# starts to pay. `processes` (default 3) is for the instances below 1e7 nonzeros, `large`
# (default 2) for those up to sesia, `largest` (default 1) for dlr2, hawaiiv10-130 and waipa,
# where a process takes half a minute and the first build differs little from the second.
# The script fails when a run fails.

set -u
build=$1
results=$2
suffix=${3:-}
processes=${4:-3}
processes_large=${5:-2}
processes_largest=${6:-1}

deps=/home/scratch.acoerduek_sw/gpu_presolve_deps
dev="istanbul-no-cutoff piperout-d27 rd-rplusc-21 neos-4306827-ravan bab6 splice1k1 savsched1
     supportcase7 neos-4647030-tutaki supportcase12 neos-5052403-cygnet neos-5116085-kenana
     supportcase19 neos-5114902-kasavu neos-2075418-temuka"
large="usafa a2864-99blp neos-3740487-motru square47 t11nonreg neos-4332810-sesia"
largest="dlr2 hawaiiv10-130 neos-4535459-waipa"

mkdir -p "$results"
failed=0
cold() {
  local dump=$1 lines=$2 status
  "$build/state_build_bench" "$dump" - "${@:3}" >> "$lines"
  status=$?
  if [ $status -ne 0 ]; then
    echo "exit=$status $dump ${*:3}"
    failed=$((failed + 1))
  fi
}
instance() {
  local dump=$1 count=$2 name lines k variant
  name=$(basename "$dump" .csr)
  lines="$results/$name$suffix.cold.jsonl"
  echo "== $name $(date +%H:%M:%S) GLIBC_TUNABLES=${GLIBC_TUNABLES:-}"
  : > "$lines"
  for k in $(seq "$count"); do
    cold "$dump" "$lines" --cold-host
    for variant in cusparse:pageable radix_sort:pageable scatter_sort:pageable \
      radix_sort:pinned radix_sort:pinned_chunked; do
      cold "$dump" "$lines" --cold --transposes "${variant%%:*}" --transfers "${variant##*:}"
    done
  done
}
for name in $dev; do
  instance "$deps/csr_dumps_dev/$name.csr" "$processes"
done
for name in $large; do
  instance "$deps/csr_dumps/$name.csr" "$processes_large"
done
for name in $largest; do
  instance "$deps/csr_dumps/$name.csr" "$processes_largest"
done
echo "== finished $(date +%H:%M:%S), $failed runs failed"
[ $failed -eq 0 ]
