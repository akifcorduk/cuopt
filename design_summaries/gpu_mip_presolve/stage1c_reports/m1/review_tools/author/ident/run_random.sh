#!/usr/bin/env bash
# run_random.sh <rel|as> <first seed> <seeds per profile> [parallel]
# The base engine and the new engine, both with the family singleton_cols off, on the random
# models of the campaign: no profile option, --singletons and --columns. One line per kind with
# the chunks (of 1000 seeds of one profile) compared and those that differ.
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1
cd "$S/ident"
t=$1
first=$2
n=$3
mkdir -p out
for s in plain sing cols; do
  flag=""
  [[ $s == sing ]] && flag="--singletons"
  [[ $s == cols ]] && flag="--columns"
  ./ident_base_$t random "$first" "$n" off $flag > "out/base_${t}_$s.txt" 2> "out/base_${t}_$s.err" &
  ./ident_new_$t random "$first" "$n" off $flag > "out/new_${t}_$s.txt" 2> "out/new_${t}_$s.err" &
done
wait
for s in plain sing cols; do
  echo "$t $s: chunks $(wc -l < "out/base_${t}_$s.txt") differing $(diff "out/base_${t}_$s.txt" "out/new_${t}_$s.txt" | grep -c '^<') base $(cat "out/base_${t}_$s.err") new $(cat "out/new_${t}_$s.err")"
done
