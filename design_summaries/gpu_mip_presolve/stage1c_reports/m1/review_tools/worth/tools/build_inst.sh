#!/usr/bin/env bash
# build_inst.sh: the instrumented release engine tool, from the worktree sources with the scratch
# instrumentation applied, into $S/bin/gprs_presolve_inst.
set -euo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_worth
W=${SRC:-/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-3/gpu_presolve}
I=$S/inst
out=${OUT:-$S/bin/gprs_presolve_inst}
mkdir -p "$I/src" "$I/tools" "$I/obj"
keep=$(mktemp)
cp "$I/src/diag.hpp" "$keep"
rm -rf "$I/src" "$I/include"
cp -R "$W/src" "$W/include" "$I/"
cp "$keep" "$I/src/diag.hpp"
python3 "$S/tools/instrument.py" "$W/src/rules_singleton_cols.cpp" "$I/src/rules_singleton_cols.cpp"
python3 "$S/tools/instrument_presolve.py" "$W/src/presolve.cpp" "$I/src/presolve.cpp"
python3 "$S/tools/instrument_header.py" "$W/include/gprs/presolve.hpp" "$I/include/gprs/presolve.hpp"
python3 "$S/tools/instrument_tool.py" "$W/tools/gprs_presolve.cpp" "$I/tools/gprs_presolve.cpp"
rm -f "$I"/obj/*.o
flags="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -I$I/include"
pids=()
for f in "$I"/src/*.cpp "$I/tools/gprs_presolve.cpp"; do
  c++ $flags -c "$f" -o "$I/obj/$(basename "$f" .cpp).o" &
  pids+=("$!")
  if ((${#pids[@]} >= 4)); then
    for p in "${pids[@]}"; do wait "$p"; done
    pids=()
  fi
done
for p in ${pids[@]+"${pids[@]}"}; do wait "$p"; done
c++ $flags "$I"/obj/*.o -o "$out"
echo "built $out"
