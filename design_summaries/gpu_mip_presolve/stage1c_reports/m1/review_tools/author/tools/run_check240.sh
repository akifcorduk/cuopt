#!/usr/bin/env bash
# run_check240.sh <tool> <tag> [parallel]
# gprs_check on the 240 local dumps without a solver and without points (check_one.sh).
set -u
SCRATCH=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad
S=$SCRATCH/s1c_m1
D=$SCRATCH/csr_dumps_bench
TOOL=$1
TAG=$2
PAR=${3:-6}
OUT=$S/check_out/${TAG}_240
mkdir -p "$OUT"
ls "$D"/*.csr | xargs -P "$PAR" -n 1 "$S/tools/check_one.sh" "$TOOL" "$OUT"
echo "$TAG: $(ls "$OUT"/*.json | wc -l) results, exit codes: $(cat "$OUT"/*.exit | sort | uniq -c | tr '\n' ' ')"
