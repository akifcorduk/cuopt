#!/usr/bin/env bash
# run_random.sh <rel|as> <mode: off|subs|primal> <first seed> <seeds per profile>
# The base engine and the engine under review in the same mode on the random models of the
# campaign (no profile option, --singletons, --columns), compared chunk by chunk (1000 seeds
# of one profile). At most four processes.
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_validity
cd "$S/ident"
t=$1
mode=$2
first=$3
n=$4
mkdir -p out
run() {
  local s=$1 flag=""
  [[ $s == sing ]] && flag="--singletons"
  [[ $s == cols ]] && flag="--columns"
  ./ident_base_$t random "$first" "$n" "$mode" $flag > "out/base_${t}_${mode}_$s.txt" 2> "out/base_${t}_${mode}_$s.err" &
  ./ident_new_$t random "$first" "$n" "$mode" $flag > "out/new_${t}_${mode}_$s.txt" 2> "out/new_${t}_${mode}_$s.err" &
}
run plain
run sing
wait
run cols
wait
for s in plain sing cols; do
  echo "$t $mode $s: chunks $(wc -l < "out/base_${t}_${mode}_$s.txt") differing $(diff "out/base_${t}_${mode}_$s.txt" "out/new_${t}_${mode}_$s.txt" | grep -c '^<') base $(cat "out/base_${t}_${mode}_$s.err") new $(cat "out/new_${t}_${mode}_$s.err")"
done
