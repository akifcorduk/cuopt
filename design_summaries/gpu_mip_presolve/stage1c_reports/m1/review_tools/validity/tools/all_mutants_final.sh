#!/usr/bin/env bash
# all_mutants.sh <mutants dir> <first seed> <seeds per profile> <parallel> <out table>
# mutant.sh for every wrong engine of the directory; the ev_ engines also build gprs_tests.
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_validity
DIR=$1
FIRST=$2
COUNT=$3
PAR=$4
TABLE=$5
mkdir -p "$DIR/_lines"
pids=()
for d in "$DIR"/*/; do
  name=$(basename "$d")
  [[ "$name" == _lines ]] && continue
  tests=""
  tests=tests
  "$S/tools/mutant_final.sh" "$DIR" "$name" "$FIRST" "$COUNT" $tests > "$DIR/_lines/$name.txt" 2>&1 &
  pids+=("$!")
  if ((${#pids[@]} >= PAR)); then
    for p in "${pids[@]}"; do wait "$p"; done
    pids=()
  fi
done
if ((${#pids[@]} > 0)); then
  for p in "${pids[@]}"; do wait "$p"; done
fi
cat "$DIR"/_lines/*.txt > "$TABLE"
wc -l "$TABLE"
