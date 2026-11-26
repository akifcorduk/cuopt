#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Rough first look: stock HiGHS presolve only, 4 instances at a time. Not a measurement.
DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
D=/home/scratch.acoerduek_sw/cuopt_dev/datasets/mip/miplib2017
OUT=${OUT:-$DEPS/results/stage0/first_look}
mkdir -p "$OUT"
run_one() {
  n=$1
  [ -f "$D/$n.mps" ] || { echo "$n MISSING" > "$OUT/$n.status"; return; }
  tmp=/dev/shm/gprs_first_look_$n.mps
  printf 'write_presolved_model_file = %s\nthreads = 1\ntime_limit = %s\n' "$tmp" "${TL:-900}" > "$OUT/$n.opt"
  t0=$(date +%s.%N)
  "$DEPS/highs_stock_install/bin/highs" --options_file "$OUT/$n.opt" "$D/$n.mps" > "$OUT/$n.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f s wall incl. read and write\n", b - a }' > "$OUT/$n.time"
  echo "$n exit=$rc" > "$OUT/$n.status"
  rm -f "$tmp"
}
export -f run_one; export DEPS D OUT TL
printf '%s\n' "$@" | xargs -P 4 -I{} bash -c 'run_one {}'
echo FIRST_LOOK_DONE
