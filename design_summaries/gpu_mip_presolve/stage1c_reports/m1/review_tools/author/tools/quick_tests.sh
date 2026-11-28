#!/usr/bin/env bash
# quick_tests.sh <gpu_presolve tree> <out dir> <filter> [test source basename ...]
# The assertion build of the library (every source) and of the named test files (default: all),
# linked with the harness, and gprs_tests <filter>. Objects that are newer than their source are
# kept; a header change needs CLEAN=1.
set -uo pipefail
G=$1
OUT=$2
FILTER=$3
shift 3
mkdir -p "$OUT/obj"
[[ "${CLEAN:-0}" == 1 ]] && rm -f "$OUT"/obj/*.o
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off "-I$G/include" -DGPRS_ASSERT=1)
sources=("$G"/src/*.cpp "$G/tests/main.cpp" "$G/tests/oracle.cpp")
if (($# > 0)); then
  for t in "$@"; do sources+=("$G/tests/$t.cpp"); done
else
  for t in "$G"/tests/test_*.cpp; do sources+=("$t"); done
fi
pids=()
objs=()
bad=0
for f in "${sources[@]}"; do
  obj="$OUT/obj/$(basename "$(dirname "$f")")_$(basename "$f" .cpp).o"
  objs+=("$obj")
  if [[ -f "$obj" && "$obj" -nt "$f" ]]; then continue; fi
  c++ "${flags[@]}" -c "$f" -o "$obj" &
  pids+=("$!")
  if ((${#pids[@]} >= ${JOBS:-6})); then
    for p in "${pids[@]}"; do wait "$p" || bad=1; done
    pids=()
  fi
done
for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || bad=1; done
((bad)) && { echo "compile failed"; exit 1; }
c++ "${flags[@]}" "${objs[@]}" -o "$OUT/gprs_tests" || exit 1
"$OUT/gprs_tests" "$FILTER"
