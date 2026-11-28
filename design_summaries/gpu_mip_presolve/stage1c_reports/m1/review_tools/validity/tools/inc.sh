#!/usr/bin/env bash
# inc.sh [test name filter]   (JOBS, default 4; BUILDS, default "assert release")
# Incremental build of the worktree into $S/build (the directory of build_host_local.sh): a
# source is recompiled when it is newer than its object, every source when a header is newer
# than the library. Then the tests whose name contains the filter ("-" for no test run).
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_validity
G=/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-2/gpu_presolve
jobs="${JOBS:-4}"
filter="${1:-}"
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off "-I$G/include")
status=0
for name in ${BUILDS:-assert release}; do
  define=""
  [[ $name == assert ]] && define="-DGPRS_ASSERT=1"
  dir=$S/build/$name
  mkdir -p "$dir/obj"
  all=0
  for h in "$G"/include/gprs/*.hpp "$G"/src/*.hpp "$G"/tests/*.hpp "$G"/check/*.hpp; do
    [[ "$h" -nt "$dir/libgprs_host.a" ]] && all=1
  done
  pids=()
  failed=0
  for src in "$G"/src/*.cpp "$G"/tests/*.cpp "$G"/tools/*.cpp; do
    obj="$dir/obj/$(basename "$(dirname "$src")")_$(basename "$src" .cpp).o"
    if ((all)) || [[ ! -f "$obj" || "$src" -nt "$obj" ]]; then
      c++ "${flags[@]}" ${define:+"$define"} -c "$src" -o "$obj" &
      pids+=("$!")
      if ((${#pids[@]} >= jobs)); then
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        pids=()
      fi
    fi
  done
  for pid in ${pids[@]+"${pids[@]}"}; do wait "$pid" || failed=1; done
  if ((failed)); then
    echo "== $name: compile FAILED"
    status=1
    continue
  fi
  # Objects of sources that no longer exist.
  for obj in "$dir"/obj/*.o; do
    base=$(basename "$obj" .o)
    kind=${base%%_*}
    rest=${base#*_}
    [[ -f "$G/$kind/$rest.cpp" ]] || rm -f "$obj"
  done
  rm -f "$dir/libgprs_host.a"
  ar rcs "$dir/libgprs_host.a" "$dir"/obj/src_*.o
  c++ "${flags[@]}" "$dir"/obj/tests_*.o "$dir/libgprs_host.a" -o "$dir/gprs_tests" || status=1
  c++ "${flags[@]}" "$dir/obj/tools_gprs_presolve.o" "$dir/libgprs_host.a" -o "$dir/gprs_presolve" || status=1
  c++ "${flags[@]}" "$dir/obj/tools_gprs_fuzz.o" "$dir/obj/tests_oracle.o" "$dir/libgprs_host.a" -o "$dir/gprs_fuzz" || status=1
  if [[ "$filter" != "-" ]]; then
    echo "== $name"
    if [[ -n "$filter" ]]; then
      "$dir/gprs_tests" "$filter" 2>&1 | grep -v "^pass " || true
    else
      "$dir/gprs_tests" 2>&1 | grep -v "^pass " || true
    fi
  fi
done
exit $status
