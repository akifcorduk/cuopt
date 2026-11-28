#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One long random run of gprs_fuzz split over processes.
#
#   run_campaign.sh <gprs_fuzz binary> <out dir> <first seed> <seeds per profile> <processes>
#                   [gprs_fuzz options]
#
# Each process gets its own seed range and writes <out dir>/part.<k>.txt. Prints the totals,
# the first failure lines, and exits with 1 when a process failed or reported a failure.
# A rule change needs tens of millions of models in the release and in the assertion build,
# with and without --singletons: the one wrong reduction found so far had a rate of one in a
# few million models.

set -uo pipefail

if (($# < 5)); then
  echo "usage: run_campaign.sh <gprs_fuzz binary> <out dir> <first seed> <seeds per profile>" \
    "<processes> [gprs_fuzz options]" >&2
  exit 2
fi
bin="$1"
out="$2"
first="$3"
count="$4"
procs="$5"
shift 5
if [[ ! -x "$bin" ]]; then
  echo "run_campaign.sh: $bin is not an executable" >&2
  exit 2
fi
mkdir -p "$out"
rm -f "$out"/part.*.txt
part=$((count / procs))
pids=()
for ((k = 0; k < procs; ++k)); do
  "$bin" $((first + k * part)) "$part" "$@" > "$out/part.$k.txt" 2>&1 &
  pids+=("$!")
done
bad=0
for p in "${pids[@]}"; do
  wait "$p" || bad=1
done
finished=$(grep -l "^structure " "$out"/part.*.txt 2> /dev/null | wc -l)
if ((finished != procs)); then
  echo "run_campaign.sh: $((procs - finished)) of $procs processes did not finish"
  bad=1
fi
grep -h "^models " "$out"/part.*.txt |
  awk '{m += $2; r += $6; f += $8; o += $10; b += $12}
       END {print "models", m, "reduced", r, "oracle failures", f,
                  "reduced outside the oracle class", o, "base outside", b}'
grep -h "^structure " "$out"/part.*.txt |
  awk '{c += $3; f += $5; a += $7; s += $9; x += $11; if ($13 + 0 > w) w = $13 + 0;
        r += $19; k += $21; u += $25; d += $27; q += $29; t += $31; n += $33;
        if ($35 + 0 > m) m = $35 + 0; v += $37}
       END {print "structure check: results with a reduced model", c, "failed", f,
                  "failed alone", a, "missed with a reduced model", s,
                  "of them relaxed", v, "missed without one", x,
                  "largest passing violation", w;
            print "rows: left the model", r, "kept", k, "kept and not tested", u,
                  "| results with a subtraction", d, "rows made of equations", q,
                  "tested with an equation", t, "not tested", n,
                  "largest multiplier of an equation", m + 0}'
grep -h "^crush " "$out"/part.*.txt |
  awk '{r += $3; d += $5; p += $7; s += $9; f += $11; n += $13; c += $15}
       END {print "crush test: results", r, "of them with a dual finding", d,
                  "points crushed", p, "points left out", s, "dual findings", f,
                  "| second presolves of a fixed point", n, "changed", c}'
grep -h "^columns " "$out"/part.*.txt |
  awk '{a += $3; b += $4; c += $6; d += $7; e += $9; f += $10; g += $12; h += $13;
        i += $17; j += $19; k += $21; l += $23; m += $26; n += $28; o += $30; p += $32}
       END {print "singleton columns, results and findings: free_singleton", a, b,
                  "free_singleton_dual", c, d, "zero_cost_singleton", e, f,
                  "forcing_column", g, h;
            print "results with a record: defined", i, "defined by a dual rule", j,
                  "interval", k, "forcing", l, "| boxes tested", m, "snapshots tested", n,
                  "not tested", o, "results with limited", p}'
grep -h "^probing " "$out"/part.*.txt |
  awk '{a += $3; b += $5; c += $7; d += $10; e += $12; f += $14; g += $17; u += $19; h += $21;
        i += $23; j += $26; if ($28 + 0 > k) k = $28 + 0}
       END {print "simple probing: results with an alias", a, "aliases", b, "without a row", c,
                  "| rows in a batch", d, "deleted", e, "left with a cut edge", f;
            print "rows left: columns connected", g, "bounds one apart, not 0 and 1:", u,
                  "scale limit", h, "size of an offset", i,
                  "| calls with an alias depth above 1:", j, "largest depth", k + 0}'
grep -h "^FAIL\|^STRUCTURE-ONLY\|^MISSED\|^SECOND" "$out"/part.*.txt | cut -c1-200 | head -20
exit $bad
