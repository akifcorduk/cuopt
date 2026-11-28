#!/usr/bin/env bash
# build.sh: the identity driver against the base engine (the groundwork commit, its library
# from build_base) and against the engine of the worktree (its library from build), each in the
# release and in the assertion build. The generator is the worktree's tests/oracle.cpp in all
# four, compiled against the headers of its engine.
set -euo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1
W=/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-1/gpu_presolve
B=$S/base_src/gpu_presolve
F=(-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off)
cd "$S/ident"
c++ "${F[@]}" "-I$B/include" "-I$W/tests" ident.cpp "$W/tests/oracle.cpp" "$S/build_base/release/libgprs_host.a" -o ident_base_rel &
c++ "${F[@]}" -DGPRS_ASSERT=1 "-I$B/include" "-I$W/tests" ident.cpp "$W/tests/oracle.cpp" "$S/build_base/assert/libgprs_host.a" -o ident_base_as &
c++ "${F[@]}" -DIDENT_NEW "-I$W/include" "-I$W/tests" ident.cpp "$W/tests/oracle.cpp" "${NEWBUILD:-$S/build}/release/libgprs_host.a" -o ident_new_rel &
c++ "${F[@]}" -DIDENT_NEW -DGPRS_ASSERT=1 "-I$W/include" "-I$W/tests" ident.cpp "$W/tests/oracle.cpp" "${NEWBUILD:-$S/build}/assert/libgprs_host.a" -o ident_new_as &
wait
ls -la ident_base_rel ident_base_as ident_new_rel ident_new_as
