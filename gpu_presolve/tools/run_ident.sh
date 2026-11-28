#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Are two builds of the engine one? Builds tools/gprs_ident.cpp of this tree against the headers
# and the library of each build and compares what the two print, model by model.
#
#   run_ident.sh <out dir> <include dir a> <libgprs_host.a a> <include dir b> <libgprs_host.a b>
#                random <first seed> <seeds per profile> [mode]...
#   run_ident.sh <out dir> <include dir a> <libgprs_host.a a> <include dir b> <libgprs_host.a b>
#                dumps <dump dir> [mode]...
#
# mode: default, primal, off (see gprs_ident.cpp); without one, default and primal.
# random runs the models of the campaign without a profile option, with --singletons and with
# --columns, in chunks of 1000 seeds per profile; dumps runs every *.csr of the directory in
# eight option variants. Environment: CXX (default c++), CXXFLAGS_EXTRA (for example
# -DGPRS_ASSERT=1 when the libraries are assertion builds), JOBS (default 2: the two builds
# run side by side).
# A tree that is older than this file needs only the headers that gprs_ident.cpp names: the
# generator is this tree's tests/oracle.cpp for both.
# Prints, per mode and model set, the number of lines and of lines that differ, and exits with
# 1 when one differs.

set -uo pipefail

if (($# < 7)); then
  sed -n '5,22p' "$0" >&2
  exit 2
fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$1"
inc_a="$2"
lib_a="$3"
inc_b="$4"
lib_b="$5"
kind="$6"
shift 6
cxx="${CXX:-c++}"
jobs="${JOBS:-2}"
read -r -a extra <<< "${CXXFLAGS_EXTRA:-}"
flags=(-std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off ${extra[@]+"${extra[@]}"})
mkdir -p "$out"

build() {
  "$cxx" "${flags[@]}" "-I$1" "$here/gprs_ident.cpp" "$here/../tests/oracle.cpp" "$2" -o "$3"
}
build "$inc_a" "$lib_a" "$out/ident_a" &
pid=$!
if ((jobs < 2)); then wait "$pid" || exit 1; fi
build "$inc_b" "$lib_b" "$out/ident_b" || exit 1
if ((jobs >= 2)); then wait "$pid" || exit 1; fi

bad=0
# both <tag> <gprs_ident arguments>: the two tools side by side, then the comparison.
both() {
  local tag="$1"
  shift
  "$out/ident_a" "$@" > "$out/a_$tag.txt" 2> "$out/a_$tag.err" &
  local pid_a=$!
  if ((jobs < 2)); then wait "$pid_a"; fi
  "$out/ident_b" "$@" > "$out/b_$tag.txt" 2> "$out/b_$tag.err"
  if ((jobs >= 2)); then wait "$pid_a"; fi
  local lines differ
  lines=$(wc -l < "$out/a_$tag.txt")
  differ=$(diff "$out/a_$tag.txt" "$out/b_$tag.txt" | grep -c '^[<>]')
  echo "$tag: lines $lines, differing $differ, a: $(cat "$out/a_$tag.err"), b: $(cat "$out/b_$tag.err")"
  if ((differ != 0 || lines == 0)); then bad=1; fi
}

if [[ "$kind" == random ]]; then
  first="$1"
  count="$2"
  shift 2
  for mode in "${@:-default primal}"; do
    for m in $mode; do
      both "random_${m}_plain" random "$first" "$count" "$m"
      both "random_${m}_singletons" random "$first" "$count" "$m" --singletons
      both "random_${m}_columns" random "$first" "$count" "$m" --columns
    done
  done
elif [[ "$kind" == dumps ]]; then
  dir="$1"
  shift
  for mode in "${@:-default primal}"; do
    for m in $mode; do
      : > "$out/a_dumps_$m.txt"
      : > "$out/b_dumps_$m.txt"
      for dump in "$dir"/*.csr; do
        "$out/ident_a" dump "$dump" "$m" >> "$out/a_dumps_$m.txt" &
        pid_a=$!
        if ((jobs < 2)); then wait "$pid_a"; fi
        "$out/ident_b" dump "$dump" "$m" >> "$out/b_dumps_$m.txt"
        if ((jobs >= 2)); then wait "$pid_a"; fi
      done
      lines=$(wc -l < "$out/a_dumps_$m.txt")
      differ=$(diff "$out/a_dumps_$m.txt" "$out/b_dumps_$m.txt" | grep -c '^[<>]')
      echo "dumps_$m: lines $lines, differing $differ"
      if ((differ != 0 || lines == 0)); then bad=1; fi
    done
  done
else
  echo "run_ident.sh: random or dumps, not $kind" >&2
  exit 2
fi
exit $bad
