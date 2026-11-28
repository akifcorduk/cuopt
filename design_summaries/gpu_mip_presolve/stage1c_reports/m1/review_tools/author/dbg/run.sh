#!/usr/bin/env bash
# run.sh: builds dbg.cpp with case.inc against the assertion objects of quick_tests.sh and runs it.
set -euo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1
G=/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-1/gpu_presolve
c++ -std=c++17 -O1 -Wall -Wextra -ffp-contract=off "-I$G/include" "-I$G/tests" -DGPRS_ASSERT=1 \
  "$S/dbg/dbg.cpp" "$S"/qt/obj/src_*.o "$S/qt/obj/tests_oracle.o" -o "$S/dbg/dbg"
"$S/dbg/dbg"
