#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds HiGHS 283bfc859 with highs_diag.patch. Run on the node, inside the dev container:
#   bash gpu_presolve/baseline/build_highs_diag.sh
# JOBS sets the number of build jobs (default 8).

set -euo pipefail

DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
COMMIT=283bfc859
PATCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/highs_diag.patch
SRC=$DEPS/highs_diag_src
BUILD=$DEPS/build/highs_diag
INSTALL=$DEPS/highs_diag_install
STAMP=$SRC/.git/highs_diag_patch.sha256

if [ ! -d "$SRC/.git" ]; then
  git clone --quiet --no-checkout "$DEPS/highs" "$SRC"
fi

# The source copy is rewritten only when the patch changed, so that a re-run with the same patch
# rebuilds nothing.
patch_sum=$(sha256sum "$PATCH" | cut -d' ' -f1)
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$patch_sum" ]; then
  rm -f "$STAMP"
  git -C "$SRC" checkout --quiet --force --detach "$COMMIT"
  git -C "$SRC" clean --quiet -fd
  git -C "$SRC" apply "$PATCH"
  echo "$patch_sum" > "$STAMP"
fi

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIGHSINT64=ON \
  -DFAST_BUILD=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_INSTALL_PREFIX="$INSTALL"
cmake --build "$BUILD" -j "${JOBS:-8}"
cmake --install "$BUILD" > /dev/null

echo "installed: $INSTALL/bin/highs"
"$INSTALL/bin/highs" --version
