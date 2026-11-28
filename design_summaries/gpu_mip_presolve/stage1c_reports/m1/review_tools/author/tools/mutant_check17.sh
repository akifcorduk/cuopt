#!/usr/bin/env bash
# mutant_check17.sh <mutants dir> <name>...: gprs_check of each wrong engine on the 17 local
# dumps, without a solver, in both modes, with 2 cost variants and 4 pushed points per point.
# The tool is linked from gprs_check.o of the right engine, the objects that the wrong engine's
# campaign build compiled and the release objects of the right engine for the rest.
set -uo pipefail
SCRATCH=/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad
S=$SCRATCH/s1c_m1
HB=$SCRATCH/audit/hb/stock_i64
DIR=$1
shift
for name in "$@"; do
  out=$S/check_mut/$name
  mkdir -p "$out"
  objs=()
  for base in "$S"/build/release/obj/src_*.o; do
    src=$(basename "$base" .o)
    src=${src#src_}
    if [[ -f "$DIR/$name/build/obj/$src.o" ]]; then
      objs+=("$DIR/$name/build/obj/$src.o")
    else
      objs+=("$base")
    fi
  done
  if ! c++ "$S/check_build/gprs_check.o" "${objs[@]}" "$HB/libhighs.a" -o "$out/gprs_check" \
    > "$out/link.log" 2>&1; then
    echo "$name: link failed"
    continue
  fi
  JOBS=${JOBS:-5} bash "$S/tools/run_check17.sh" "$out/gprs_check" "mut_$name" --kill-after 300 \
    > /dev/null 2>&1
  echo "$name: $(python3 "$S/tools/check_sum.py" "$S/check_out/mut_${name}_default_dev" \
    "$S/check_out/mut_${name}_default_status" "$S/check_out/mut_${name}_primal_dev" \
    "$S/check_out/mut_${name}_primal_status" | grep "^results\|FAIL\|stack error" | tr '\n' ' ' | cut -c1-900)"
done
