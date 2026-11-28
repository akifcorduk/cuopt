#!/usr/bin/env bash
# ubsan_run.sh <build dir> <log>: both builds with UBSan and the libc++ debug hardening mode,
# the tests of both, then a campaign slice of each without a profile, with --singletons and
# with --columns.
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_validity
G=/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-2/gpu_presolve
OUT=$1
LOG=$2
cd "$G"
CXX=$S/tools/ubsan_cxx.sh JOBS=${JOBS:-3} GPRS_BUILD_DIR=$OUT ./build_host_local.sh > "$LOG" 2>&1
echo "build and tests exit $?" >> "$LOG"
for b in assert release; do
  "$OUT/$b/gprs_fuzz" 9700000000 25000 > "$OUT/$b.fuzz_plain.txt" 2>&1
  echo "$b fuzz plain exit $?" >> "$LOG"
  "$OUT/$b/gprs_fuzz" 9700000000 25000 --singletons > "$OUT/$b.fuzz_sing.txt" 2>&1
  echo "$b fuzz singletons exit $?" >> "$LOG"
  "$OUT/$b/gprs_fuzz" 9700000000 25000 --columns > "$OUT/$b.fuzz_cols.txt" 2>&1
  echo "$b fuzz columns exit $?" >> "$LOG"
done
grep -h "^models \|^crush \|^columns \|runtime error" "$OUT"/*.fuzz_*.txt | cut -c1-260 >> "$LOG"
