#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds HiGHS 283bfc859 with highs_diag.patch, and the baseline tool against that build. Run on
# the node, inside the dev container:
#   bash gpu_presolve/baseline/build_highs_diag.sh
# JOBS sets the number of build jobs (default 8).
# NAME names the build (default diag): source copy highs_<NAME>_src, build tree
# build/highs_<NAME>, install highs_<NAME>_install, tool build tree build/baseline_<NAME>. Nothing
# outside these four is written, so a new name builds the patch next to a build that is in use:
#   NAME=diag3 bash gpu_presolve/baseline/build_highs_diag.sh

set -euo pipefail

DEPS=/home/scratch.acoerduek_sw/gpu_presolve_deps
COMMIT=283bfc859
NAME=${NAME:-diag}
if [[ ! "$NAME" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "NAME must be letters, digits and underscores: \"$NAME\"" >&2
  exit 2
fi
if [ "$NAME" = stock ]; then
  echo "NAME=stock would overwrite the unpatched install highs_stock_install" >&2
  exit 2
fi
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH=$HERE/highs_diag.patch
SRC=$DEPS/highs_${NAME}_src
BUILD=$DEPS/build/highs_$NAME
INSTALL=$DEPS/highs_${NAME}_install
TOOL_BUILD=$DEPS/build/baseline_$NAME
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

# The tool is compiled against the installed headers and has to follow every rebuild of them.
cmake -S "$HERE" -B "$TOOL_BUILD" -G Ninja -DHIGHS_ROOT="$INSTALL"
cmake --build "$TOOL_BUILD" -j "${JOBS:-8}"
echo "tool: $TOOL_BUILD/baseline_tool"
