#!/usr/bin/env bash
# mutant.sh <mutants dir> <name> <first seed> <seeds per profile> [tests]
# Builds gprs_fuzz (and with `tests` gprs_tests) of one wrong engine with the release flags of
# build_host_local.sh, -Werror included, and runs it without a profile option, with
# --singletons and with --columns, one process each after the other. Objects of sources that do
# not differ come from the release build of the right engine (GPRS_BASE_BUILD).
# Prints one line: name, build, models, oracle failures, structure failures, second presolves
# that changed, the structure failures by part and the oracle failures by the start of their
# message.
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_validity
G=${GPRS_TREE:-/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-2/gpu_presolve}
BASE=${GPRS_BASE_BUILD:-/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_validity/build_final/release}
DIR=$1
NAME=$2
FIRST=$3
COUNT=$4
TESTS=${5:-}
MUT=$DIR/$NAME/src
OUT=$DIR/$NAME/build
mkdir -p "$OUT/obj"
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off "-I$G/include")
changed_headers=()
for h in "$G"/src/*.hpp; do
  if ! cmp -s "$h" "$MUT/$(basename "$h")"; then changed_headers+=("$(basename "$h")"); fi
done
objs=()
changed=0
build=ok
for f in "$G"/src/*.cpp; do
  src=$(basename "$f" .cpp)
  same=1
  cmp -s "$f" "$MUT/$src.cpp" || same=0
  for h in ${changed_headers[@]+"${changed_headers[@]}"}; do
    grep -q "\"$h\"" "$MUT/$src.cpp" && same=0
  done
  if ((same)); then
    objs+=("$BASE/obj/src_$src.o")
    continue
  fi
  if ! c++ "${flags[@]}" -c "$MUT/$src.cpp" -o "$OUT/obj/$src.o" 2> "$OUT/obj/$src.err"; then
    build=werror
    if ! c++ "${flags[@]}" -Wno-error -c "$MUT/$src.cpp" -o "$OUT/obj/$src.o" 2>> "$OUT/obj/$src.err"; then
      echo "$NAME build FAILED ($src.cpp)"
      exit 1
    fi
  fi
  objs+=("$OUT/obj/$src.o")
  changed=$((changed + 1))
done
if ((changed == 0)); then
  echo "$NAME build NO-SOURCE-DIFFERS"
  exit 1
fi
if ! c++ "${flags[@]}" "${objs[@]}" "$BASE/obj/tools_gprs_fuzz.o" "$BASE/obj/tests_oracle.o" -o "$OUT/gprs_fuzz"; then
  echo "$NAME link FAILED"
  exit 1
fi
if [[ "$TESTS" == tests ]]; then
  if ! c++ "${flags[@]}" "$BASE"/obj/tests_*.o "${objs[@]}" -o "$OUT/gprs_tests"; then
    echo "$NAME tests link FAILED"
    exit 1
  fi
  "$OUT/gprs_tests" > "$OUT/tests.txt" 2>&1
fi
"$OUT/gprs_fuzz" "$FIRST" "$COUNT" > "$OUT/plain.txt" 2>&1
"$OUT/gprs_fuzz" "$FIRST" "$COUNT" --singletons > "$OUT/sing.txt" 2>&1
"$OUT/gprs_fuzz" "$FIRST" "$COUNT" --columns > "$OUT/cols.txt" 2>&1
outs=("$OUT/plain.txt" "$OUT/sing.txt" "$OUT/cols.txt")
models=$(grep -h "^models " "${outs[@]}" | awk '{m += $2} END {print m + 0}')
oracle=$(grep -h "^models " "${outs[@]}" | awk '{f += $8} END {print f + 0}')
structure=$(grep -h "^structure " "${outs[@]}" | awk '{f += $5} END {print f + 0}')
second=$(grep -h "^crush " "${outs[@]}" | awk '{c += $15} END {print c + 0}')
parts=$(grep -h "^  part " "${outs[@]}" | awk '{n = $2; $1 = ""; $2 = ""; c[$0] += n} END {for (k in c) printf "%s:%d;", k, c[k]}' | sed 's/ x //g')
messages=$(grep -h "^  [0-9]* x " "${outs[@]}" | awk '{n = $1; $1 = ""; $2 = ""; c[$0] += n} END {for (k in c) printf "%d x%s; ", c[k], k}')
tests=""
[[ "$TESTS" == tests ]] && tests=" tests: $(tail -1 "$OUT/tests.txt")"
printf "%-18s build %-6s models %8d oracle %7d structure %7d second %5d | %s | %s%s\n" "$NAME" "$build" "$models" "$oracle" "$structure" "$second" "${parts:--}" "${messages:--}" "$tests"
