#!/usr/bin/env bash
# one.sh <binary> <mode> <tag> <dump>: one dump through one identity driver.
cd "$(dirname "${BASH_SOURCE[0]}")"
"./$1" dump "$4" "$2" > "out/$3/$(basename "$4").$1.$2.txt" 2>&1
