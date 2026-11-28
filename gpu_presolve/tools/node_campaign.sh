#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# A large random campaign of one engine version on a node: gprs_fuzz of the tag that
# stage1c_engine_runs.sh built ($DEPS/build/<tag>/release and assert), split over processes
# by the tag's own tools/run_campaign.sh. The three kinds (no profile option, --singletons,
# --columns) run one after the other, first in the release build, then in the assertion build.
#
# Start it detached inside the dev container (any directory):
#   setsid nohup bash node_campaign.sh <tag> <name> <first seed> <seeds per profile> \
#     <processes> [gprs_fuzz options] > /dev/null 2>&1 < /dev/null &
# The run k (0 to 5, in the order above) uses the seeds from <first seed> + k * <seeds per
# profile> on, so the six ranges do not overlap. The assertion build gets half the seeds.
#
# Environment:
#   CAMPAIGN_BUILDS  builds to run, separated by spaces (default: "release assert")
#   CAMPAIGN_KINDS   kinds to run (default: "plain singletons columns")
#
# Results in $DEPS/results/campaign/<tag>/<name>/:
#   log.txt                  one line per run: exit code, seconds, first seed, seeds
#   <build>_<kind>.txt       the totals of run_campaign.sh and the first failure lines
#   <build>_<kind>/part.*    the output of each process (every failure line)
#   ENDED                    written at the end: the runs that did not exit with 0

set -uo pipefail

if (($# < 5)); then
  echo "usage: node_campaign.sh <tag> <name> <first seed> <seeds per profile> <processes>" \
    "[gprs_fuzz options]" >&2
  exit 2
fi
TAG=$1
NAME=$2
FIRST=$3
COUNT=$4
PROCS=$5
shift 5

DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
RUNNER=$DEPS/src/$TAG/gpu_presolve/tools/run_campaign.sh
OUT=$DEPS/results/campaign/$TAG/$NAME
BUILDS=${CAMPAIGN_BUILDS:-release assert}
KINDS=${CAMPAIGN_KINDS:-plain singletons columns}

if [ ! -f "$RUNNER" ]; then
  echo "node_campaign.sh: no source tree of tag $TAG ($RUNNER)" >&2
  exit 2
fi
if ((PROCS > $(nproc))); then
  echo "node_campaign.sh: $PROCS processes on $(nproc) CPUs" >&2
  exit 2
fi
mkdir -p "$OUT"
rm -f "$OUT/ENDED"
commit=$(git -C "$DEPS/src/$TAG" rev-parse HEAD)
echo "$(date '+%F %T') $(hostname) start: tag $TAG commit $commit, first seed $FIRST," \
  "seeds per profile $COUNT, processes $PROCS, options: $*, load $(cut -d' ' -f1-3 /proc/loadavg)" \
  >> "$OUT/log.txt"

failed=()
k=0
for build in release assert; do
  for kind in plain singletons columns; do
    run=$k
    k=$((k + 1))
    case " $BUILDS " in *" $build "*) ;; *) continue ;; esac
    case " $KINDS " in *" $kind "*) ;; *) continue ;; esac
    bin=$DEPS/build/$TAG/$build/gprs_fuzz
    count=$COUNT
    [ "$build" = assert ] && count=$((COUNT / 2))
    flag=()
    [ "$kind" != plain ] && flag=("--$kind")
    first=$((FIRST + run * COUNT))
    start=$(date +%s)
    bash "$RUNNER" "$bin" "$OUT/${build}_$kind" "$first" "$count" "$PROCS" \
      ${flag[@]+"${flag[@]}"} "$@" > "$OUT/${build}_$kind.txt" 2>&1
    code=$?
    echo "$(date '+%F %T') ${build}_$kind: exit $code in $(($(date +%s) - start)) s," \
      "first seed $first, seeds per profile $count," \
      "$(grep -m1 '^models ' "$OUT/${build}_$kind.txt" | cut -d' ' -f1-2)" >> "$OUT/log.txt"
    [ "$code" -ne 0 ] && failed+=("${build}_$kind")
  done
done
echo "$(date '+%F %T') end; runs that did not exit with 0: ${failed[*]:-none}" >> "$OUT/log.txt"
echo "${failed[*]:-}" > "$OUT/ENDED"
