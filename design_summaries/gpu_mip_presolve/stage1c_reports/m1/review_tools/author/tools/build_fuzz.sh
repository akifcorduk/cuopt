#!/usr/bin/env bash
# build_fuzz.sh <gpu_presolve tree> <out dir> [assert]: the library, gprs_fuzz and gprs_presolve
# with the flags of build_host_local.sh, without the tests; only sources newer than their
# object are compiled (a header change needs a clean out dir).
set -euo pipefail
G=$1
OUT=$2
MODE=${3:-release}
mkdir -p "$OUT/obj"
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off "-I$G/include")
[[ "$MODE" == assert ]] && flags+=(-DGPRS_ASSERT=1)
pids=()
for f in "$G"/src/*.cpp "$G/tools/gprs_fuzz.cpp" "$G/tools/gprs_presolve.cpp" "$G/tests/oracle.cpp"; do
  obj="$OUT/obj/$(basename "$(dirname "$f")")_$(basename "$f" .cpp).o"
  c++ "${flags[@]}" -c "$f" -o "$obj" &
  pids+=("$!")
  if ((${#pids[@]} >= ${JOBS:-6})); then
    for p in "${pids[@]}"; do wait "$p"; done
    pids=()
  fi
done
for p in ${pids[@]+"${pids[@]}"}; do wait "$p"; done
rm -f "$OUT/libgprs_host.a"
ar rcs "$OUT/libgprs_host.a" "$OUT"/obj/src_*.o
c++ "${flags[@]}" "$OUT/obj/tools_gprs_fuzz.o" "$OUT/obj/tests_oracle.o" "$OUT/libgprs_host.a" -o "$OUT/gprs_fuzz"
c++ "${flags[@]}" "$OUT/obj/tools_gprs_presolve.o" "$OUT/libgprs_host.a" -o "$OUT/gprs_presolve"
echo "built $OUT/gprs_fuzz"
