#!/usr/bin/env bash
# each_test.sh <gprs_tests binary> <tests dir> [name prefix]: every test alone, so that a test
# that crashes does not hide the others. Prints the tests that fail or crash with their first
# lines.
set -uo pipefail
bin=$1
tests=$2
prefix=${3:-}
fail=0
for name in $(grep -ho "GPRS_TEST([a-z0-9_]*" "$tests"/*.cpp | sed 's/GPRS_TEST(//' | sort -u); do
  [[ -n "$prefix" && "$name" != "$prefix"* ]] && continue
  out=$("$bin" "$name" 2>&1)
  rc=$?
  # The filter is a substring: count only this test's own line.
  if ((rc != 0)) && ! grep -q "^pass $name\$" <<< "$out"; then
    fail=$((fail + 1))
    echo "== $name rc=$rc"
    grep -v "^pass\|^FAIL\|tests, " <<< "$out" | sed 's|.*/tests/||' | cut -c1-200 | head -${LINES_PER_TEST:-4}
  fi
done
echo "failing or crashing tests: $fail"
