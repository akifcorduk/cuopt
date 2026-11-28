#!/usr/bin/env bash
# check_one.sh <tool> <out dir> <dump>: gprs_check on one dump without a solver and without
# points: the presolve in both modes, the structure check of both results, two cost variants.
name=$(basename "$3" .csr)
"$1" "$3" "$2/$name.json" --lp-time-limit 0 --mip-time-limit 0 --threads 1 --cost-variants 2 \
  --push-points 0 > "$2/$name.log" 2>&1
echo $? > "$2/$name.exit"
