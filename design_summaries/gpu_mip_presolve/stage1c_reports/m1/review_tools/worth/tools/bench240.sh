#!/usr/bin/env bash
# bench240.sh <tool> <tag> [gprs_presolve option]...
# The tool on the 240 local dumps, one run each, into $S/bench/<tag>, and the analysis against
# the reference (REF, default m1_a) and the engine before into $S/logs/an_<tag>_<ref>.txt.
set -uo pipefail
SCRATCH=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad
S=$SCRATCH/s1c_m1_worth
W=/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-3/gpu_presolve
tool=$1
tag=$2
shift 2
args=()
for a in "$@"; do args+=("--tool-arg=$a"); done
mkdir -p "$S/bench" "$S/logs"
cd "$W"
python3 tools/run_engine.py --tool "$tool" --dumps "${DUMPS:-$SCRATCH/csr_dumps_bench}" \
  --testbed testbed/benchmark240.json --set all --reps 1 --jobs "${JOBS:-4}" --kill-after 900 \
  ${args[@]+"${args[@]}"} --out "$S/bench/$tag" > "$S/logs/bench_$tag.log" 2>&1
echo "$tag: run exit $?"
ref=${REF:-m1_a}
python3 analysis/bench240_stage1c.py --engine "$S/bench/$tag" \
  --ref results/bench240/stage1c_refs/$ref --before "${BEFORE:-$SCRATCH/bench_m0_local}" \
  --csv "$S/bench/${tag}_vs_${ref}.csv" > "$S/logs/an_${tag}_${ref}.txt" 2>&1
echo "$tag: analysis exit $?"
grep -A3 "^instances  " "$S/logs/an_${tag}_${ref}.txt" | head -4
