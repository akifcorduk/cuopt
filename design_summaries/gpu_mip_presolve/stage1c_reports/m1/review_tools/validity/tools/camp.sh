#!/usr/bin/env bash
# camp.sh <build dir name under $S> <release|assert> <seeds per profile> <first seed> <procs> [tag]
# plain, --singletons and --columns one after the other; seed ranges 50 million apart.
set -uo pipefail
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_validity
W=/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-2/gpu_presolve
bdir=$1
build=$2
count=$3
first=$4
procs=$5
tag=${6:-camp}
mkdir -p "$S/$tag"
k=0
for kind in plain singletons columns; do
  flag=()
  [[ $kind != plain ]] && flag=("--$kind")
  start=$(date +%s)
  "$W/tools/run_campaign.sh" "$S/$bdir/$build/gprs_fuzz" "$S/$tag/${build}_$kind" \
    $((first + k * 50000000)) "$count" "$procs" ${flag[@]+"${flag[@]}"} > "$S/$tag/${build}_$kind.txt" 2>&1
  echo "$build $kind: exit $? in $(($(date +%s) - start)) s, first seed $((first + k * 50000000))"
  k=$((k + 1))
done
