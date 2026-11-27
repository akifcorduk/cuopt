#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs state_build_bench on the instances of milestone 1, one process at a time: six dev
# instances from 7e4 to 5e6 nonzeros and the nine instances above 1e7.
#   run_state_build_m1.sh <build dir> <result dir> [suffix [bench options...]]
# The result of an instance is <result dir>/<instance><suffix>.json; the suffix tells a run
# with another environment (GLIBC_TUNABLES) apart. The tables go to stdout. The script fails
# when a run fails; a run returns 3 when a device result differs from the host.

set -u
build=$1
results=$2
suffix=${3:-}
shift 2
if [ $# -gt 0 ]; then shift; fi

deps=/home/scratch.acoerduek_sw/gpu_presolve_deps
dev="istanbul-no-cutoff piperout-d27 neos-4306827-ravan bab6 neos-5116085-kenana neos-5052403-cygnet"
large="usafa a2864-99blp neos-3740487-motru square47 t11nonreg neos-4332810-sesia dlr2 hawaiiv10-130 neos-4535459-waipa"

mkdir -p "$results"
failed=0
run() {
  local name status
  name=$(basename "$1" .csr)
  echo "== $name $(date +%H:%M:%S) GLIBC_TUNABLES=${GLIBC_TUNABLES:-}"
  "$build/state_build_bench" "$1" "$results/$name$suffix.json" "${@:2}"
  status=$?
  echo "exit=$status $name"
  if [ $status -ne 0 ]; then failed=$((failed + 1)); fi
}
for name in $dev; do
  run "$deps/csr_dumps_dev/$name.csr" "$@"
done
for name in $large; do
  run "$deps/csr_dumps/$name.csr" "$@"
done
echo "== finished $(date +%H:%M:%S), $failed runs failed"
[ $failed -eq 0 ]
