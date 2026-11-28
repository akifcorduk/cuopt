#!/usr/bin/env bash
# run_dumps.sh <rel|as> <dump dir> <tag> [parallel]
# The base engine and the engine under review on every dump of the directory in eight option
# variants, in the modes off (family off), subs (its four reductions off) and primal
# (primal-only with the free singleton column off). Prints result lines and those that differ.
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_validity
cd "$S/ident"
t=$1
dir=$2
tag=$3
par=${4:-4}
mkdir -p "out/$tag"
run() {
  local bin=$1 mode=$2 out=$3
  ls "$dir"/*.csr | xargs -P "$par" -n 1 ./one.sh "$bin" "$mode" "$tag"
  cat out/$tag/*.csr.$bin.$mode.txt > "$out"
  rm -f out/$tag/*.csr.$bin.$mode.txt
}
for mode in off subs primal; do
  run ident_base_$t $mode "out/${tag}_base_${mode}_$t.txt"
  run ident_new_$t $mode "out/${tag}_new_${mode}_$t.txt"
  echo "$tag $t $mode: lines $(wc -l < "out/${tag}_base_${mode}_$t.txt") differ $(diff "out/${tag}_base_${mode}_$t.txt" "out/${tag}_new_${mode}_$t.txt" | grep -c '^<')"
done
