#!/usr/bin/env bash
# camp.sh <release|assert> <seeds per profile> <first seed of the plain run>
# The campaign of one build: no profile option, --singletons and --columns, five processes each,
# one after the other; the seed ranges of the three are 10 million seeds apart. Output in
# $S/camp/<build>_<kind>/ and the totals in $S/camp/<build>_<kind>.txt.
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1
W=/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-1/gpu_presolve
build=$1
count=$2
first=$3
mkdir -p "$S/${CAMP:-camp}"
k=0
for kind in plain singletons columns; do
  flag=()
  [[ $kind != plain ]] && flag=("--$kind")
  start=$(date +%s)
  "$W/tools/run_campaign.sh" "$S/build/$build/gprs_fuzz" "$S/${CAMP:-camp}/${build}_$kind" \
    $((first + k * 10000000)) "$count" 5 ${flag[@]+"${flag[@]}"} > "$S/${CAMP:-camp}/${build}_$kind.txt" 2>&1
  echo "$build $kind: exit $? in $(($(date +%s) - start)) s, first seed $((first + k * 10000000))"
  k=$((k + 1))
done
