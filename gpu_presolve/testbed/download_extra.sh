#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Downloads the testbed instances that are not in the MIPLIB 2017 benchmark set, as .mps.gz, into
# the directory named in testbed.json under extra_dataset_dirs. Run on the node:
#   bash gpu_presolve/testbed/download_extra.sh [destination]
# A file that is present and passes the gzip check is not downloaded again.

set -euo pipefail

DEST=${1:-/home/scratch.acoerduek_sw/gpu_presolve_deps/miplib_extra}
URL=https://miplib.zib.de/WebData/instances

NAMES=(
  supportcase21i
  piperout-d27
  gfd-schedulen55f2d50m30k3i
  neos-4306827-ravan
  neos-5116085-kenana
  usafa
  a2864-99blp
  neos-3740487-motru
  t11nonreg
  neos-4332810-sesia
  dlr2
  hawaiiv10-130
  neos-4535459-waipa
  neos-3230511-yuna
)

mkdir -p "$DEST"
for name in "${NAMES[@]}"; do
  file=$DEST/$name.mps.gz
  if [ -f "$file" ] && gzip -t "$file" 2> /dev/null; then
    echo "$name present $(stat -c %s "$file")"
    continue
  fi
  curl -fsSL --retry 3 -o "$file.part" "$URL/$name.mps.gz"
  gzip -t "$file.part"
  mv "$file.part" "$file"
  echo "$name downloaded $(stat -c %s "$file")"
done
du -sh "$DEST"
