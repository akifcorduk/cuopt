#!/bin/bash
# build_check.sh <out dir>: gprs_check of the worktree against the release library of the
# worktree's build and the stock HiGHS static library of the audit (64-bit HighsInt).
set -euo pipefail
SCRATCH=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad
S=$SCRATCH/s1c_m1_validity
G=/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-2/gpu_presolve
HS=$SCRATCH/src/highs
HB=$SCRATCH/audit/hb/stock_i64
OUT=$1
mkdir -p "$OUT"
flags=(-std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Werror "-I$G/include")
INC=(-isystem "$HB/inc" -isystem "$HS/extern" -isystem "$HS/highs")
for d in interfaces io io/filereader ipm ipm/ipx ipm/basiclu lp_data mip model parallel pdlp pdlp/cupdlp pdlp/hipdlp presolve qpsolver simplex util; do
  INC+=(-isystem "$HS/highs/$d")
done
c++ "${flags[@]}" "${INC[@]}" -c "$G/check/gprs_check.cpp" -o "$OUT/gprs_check.o"
c++ "$OUT/gprs_check.o" "${GPRS_LIB:-$S/build/release/libgprs_host.a}" "$HB/libhighs.a" -o "$OUT/gprs_check"
echo "built $OUT/gprs_check"
