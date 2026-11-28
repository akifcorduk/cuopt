#!/usr/bin/env bash
# run_check17.sh <tool> <tag> [extra run_check.py arguments]
# gprs_check on the 17 local dumps, default then primal-only, without a solver.
set -u
S=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1
G=/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-1/gpu_presolve
R=/Users/acoerduek/Documents/workspace/cuopt-wt/gpu-presolve/gpu_presolve
D=$R/local_dumps
DEV=istanbul-no-cutoff,wachplan,radiationm18-12-05,ns1830653,unitcal_7,brazil3,piperout-d27,uccase9,sorrell3,supportcase42,triptim1,neos-4306827-ravan,var-smallemery-m6j6,rd-rplusc-21,neos-5116085-kenana
STATUS=bnatt500,fhnw-binpack4-4
TOOL=$1
TAG=$2
shift 2
OUT=$S/check_out
mkdir -p "$OUT"
SOL=(--solutions-dir "$D/solutions" --solutions-dir "$D/solutions_long")
for mode in default primal; do
  extra=()
  if [ "$mode" = primal ]; then extra=(--primal-only); fi
  for set in dev status; do
    only=$DEV
    if [ "$set" = status ]; then only=$STATUS; fi
    python3 "$G/check/run_check.py" --tool "$TOOL" --dumps "$D" \
      --stats-dir "$R/results/stage1/refs/stage1c" --out "$OUT/${TAG}_${mode}_$set" --set "$set" \
      --only "$only" "${SOL[@]}" ${extra[@]+"${extra[@]}"} --lp-time-limit 0 --mip-time-limit 0 \
      --cost-variants 2 --push-points 4 --jobs "${JOBS:-5}" --threads 1 \
      "$@" > "$OUT/${TAG}_${mode}_$set.log" 2>&1
    echo "$TAG $mode $set: exit $?"
  done
done
