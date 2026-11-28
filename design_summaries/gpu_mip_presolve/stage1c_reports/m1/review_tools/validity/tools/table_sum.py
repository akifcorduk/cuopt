"""table_sum.py <mutants dir>: one line per wrong engine from its _lines file, and the totals."""

import glob
import os
import re
import sys

d = sys.argv[1]
rows = []
for f in sorted(glob.glob(d + "/_lines/*.txt")):
    name = os.path.basename(f)[:-4]
    text = open(f).read()
    m = re.search(
        r"build (\S+)\s+models\s+(\d+) oracle\s+(\d+) structure\s+(\d+) second\s+(\d+)", text
    )
    t = re.search(r"tests: (\d+) tests, (\d+) failed", text)
    crashed = "Segmentation fault" in text or "Abort trap" in text
    if not m:
        rows.append((name, "no line", 0, 0, 0, 0, None, crashed))
        continue
    rows.append(
        (
            name,
            m.group(1),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
            int(m.group(5)),
            int(t.group(2)) if t else None,
            crashed,
        )
    )
caught = [r for r in rows if r[3] or r[4] or r[5]]
print("engines %d caught by a campaign check %d (oracle %d, structure %d, second presolve %d)" % (
    len(rows),
    len(caught),
    sum(1 for r in rows if r[3]),
    sum(1 for r in rows if r[4]),
    sum(1 for r in rows if r[5]),
))
print("not caught by a campaign check:")
for r in rows:
    if not (r[3] or r[4] or r[5]):
        print("  %-18s build %s models %d tests failed %s crashed %s" % (r[0], r[1], r[2], r[6], r[7]))
if len(sys.argv) > 2:
    for r in rows:
        if r[0] in sys.argv[2:] or sys.argv[2] == "all":
            print("%-18s oracle %6d structure %6d second %5d tests failed %s%s" % (
                r[0], r[3], r[4], r[5], r[6], " (crash)" if r[7] else ""))
