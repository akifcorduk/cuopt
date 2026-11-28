#!/usr/bin/env bash
# run_dumps.sh <rel|as> <dump dir> <tag> [parallel]
# The base engine with the family singleton_cols on (as it is at the base commit, where the
# family is empty) and off, and the new engine with it off, on every dump of the directory, in
# eight option variants each. Prints the number of result lines and of those that differ.
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1
cd "$S/ident"
t=$1
dir=$2
tag=$3
par=${4:-6}
mkdir -p "out/$tag"
run() {
  local bin=$1 mode=$2 out=$3
  ls "$dir"/*.csr | xargs -P "$par" -n 1 ./one.sh "$bin" "$mode" "$tag"
  cat out/$tag/*.csr.$bin.$mode.txt > "$out"
}
run ident_base_$t on "out/${tag}_base_on_$t.txt"
run ident_base_$t off "out/${tag}_base_off_$t.txt"
run ident_new_$t off "out/${tag}_new_off_$t.txt"
echo "$tag $t: lines $(wc -l < "out/${tag}_base_off_$t.txt") | new off against base off: $(diff "out/${tag}_base_off_$t.txt" "out/${tag}_new_off_$t.txt" | grep -c '^<') differ | new off against base on: $(diff "out/${tag}_base_on_$t.txt" "out/${tag}_new_off_$t.txt" | grep -c '^<') differ"
