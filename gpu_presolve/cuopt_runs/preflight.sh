#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Short runs of cuOpt's MIP solver with the gprs presolve off, before PaPILO and instead of
# PaPILO, on a list of MPS files. For the dev node, before a cluster run: it shows that the
# adapter runs, what it removes, and whether status and objective agree between the modes.
#
# usage, from the worktree root inside the dev container:
#   bash gpu_presolve/cuopt_runs/preflight.sh <out dir> <seconds> <parallel runs> <mps> [<mps> ...]
# The mode is taken from GPRS_PREFLIGHT_MODE, which only a build with the pre-flight patch
# reads (the committed code takes mip_hyper_params_t::presolve_gprs):
#   git apply gpu_presolve/cuopt_runs/preflight_mode.patch, build, run, git apply -R the patch.
# The patch is never committed. One log per instance and mode in <out dir>; the table at the
# end is also in <out dir>/table.txt. reduce_to_mps.cpp, next to this file, writes the model the
# engine leaves as an MPS file, to run a solver on exactly that model without the adapter.

set -u -o pipefail

OUT=${1:?usage: preflight.sh <out dir> <seconds> <parallel runs> <mps> ...}
SECONDS_LIMIT=${2:?}
PARALLEL=${3:?}
shift 3
mkdir -p "$OUT"

run_one() {
  local mps=$1 mode=$2 name log
  name=$(basename "$mps" .mps)
  log=$OUT/$name.m$mode.log
  GPRS_PREFLIGHT_MODE=$mode timeout $((SECONDS_LIMIT * 4 + 300)) ./cpp/build/solve_MIP \
    --path "$mps" --time-limit "$SECONDS_LIMIT" --num-cpu-threads 8 > "$log" 2>&1
  echo "exit $?" >> "$log"
}

running=0
for mps in "$@"; do
  for mode in 0 1 2; do
    run_one "$mps" "$mode" &
    running=$((running + 1))
    if [ "$running" -ge "$PARALLEL" ]; then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait

{
  printf '%-30s %s %-5s %-44s %-12s %s\n' instance m exit "gprs (rows, cols, nnz after; status)" \
    objective end
  for mps in "$@"; do
    name=$(basename "$mps" .mps)
    for mode in 0 1 2; do
      log=$OUT/$name.m$mode.log
      code=$(grep -a '^exit ' "$log" | tail -1 | cut -d' ' -f2)
      gprs=$(grep -a 'GPRS presolve' "$log" | head -1 |
        sed -E 's/.*-> ([0-9]+) constraints, ([0-9]+) variables, ([0-9]+) nonzeros.*status ([a-z_]+).*/\1 \2 \3 \4/')
      objective=$(grep -a -E 'Best objective|Optimal solution found|[Ii]nfeasible|[Uu]nbounded|no solution found' "$log" |
        tail -1 | cut -c1-70)
      bad=$(grep -a -c -i -E 'assert|terminate called|Segmentation|cuda_error|what\(\)' "$log")
      printf '%-30s %s %-5s %-44s %s%s\n' "$name" "$mode" "${code:-?}" "${gprs:--}" \
        "${objective:-no final line}" "$([ "$bad" -gt 0 ] && echo "  CRASH-LINES:$bad")"
    done
  done
} | tee "$OUT/table.txt"
