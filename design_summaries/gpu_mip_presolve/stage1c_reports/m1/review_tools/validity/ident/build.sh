#!/usr/bin/env bash
set -euo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_validity
W=$S/commit_src/gpu_presolve
BS=$S/base_src/gpu_presolve
F=(-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off)
cd "$S/ident"
c++ "${F[@]}" "-I$BS/include" "-I$W/tests" ident.cpp "$W/tests/oracle.cpp" "$S/build_base/release/libgprs_host.a" -o ident_base_rel &
c++ "${F[@]}" -DGPRS_ASSERT=1 "-I$BS/include" "-I$W/tests" ident.cpp "$W/tests/oracle.cpp" "$S/build_base/assert/libgprs_host.a" -o ident_base_as &
c++ "${F[@]}" -DIDENT_NEW "-I$W/include" "-I$W/tests" ident.cpp "$W/tests/oracle.cpp" "${NEWBUILD:-$S/build_commit}/release/libgprs_host.a" -o ident_new_rel &
c++ "${F[@]}" -DIDENT_NEW -DGPRS_ASSERT=1 "-I$W/include" "-I$W/tests" ident.cpp "$W/tests/oracle.cpp" "${NEWBUILD:-$S/build_commit}/assert/libgprs_host.a" -o ident_new_as &
wait
ls -la ident_base_rel ident_base_as ident_new_rel ident_new_as
