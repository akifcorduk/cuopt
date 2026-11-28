#!/usr/bin/env bash
# run.sh <binary> <dump dir> <out file> [parallel]: second.cpp on every dump of the directory.
set -uo pipefail
bin=$1
dir=$2
out=$3
par=${4:-2}
tmp=$(dirname "$out")/tmp_$(basename "$out")
rm -rf "$tmp"
mkdir -p "$tmp"
ls "$dir"/*.csr | xargs -P "$par" -n 1 sh -c '"$0" "$2" > "$1/$(basename "$2").txt" 2>&1' "$bin" "$tmp"
cat "$tmp"/*.txt > "$out"
rm -rf "$tmp"
wc -l "$out"
