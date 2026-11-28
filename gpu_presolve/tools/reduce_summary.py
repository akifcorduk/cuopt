# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make the small summary.json of an engine run that the repository stores.

Usage: reduce_summary.py <summary.json of tools/run_engine.py> <out file>
       reduce_summary.py --all <dir of run dirs> <store dir>

The reduced file has the layout and the field names of the full one and
holds, per instance, the sizes of the model and of the reduced model, the
status, the hash, the fixed-point flag, the rounds, the objective nonzeros
and what the singleton-column rules report: their line of "rules", their
four reductions of "sub_rules", "columns", "col_from_rows", and the
counters "column_counts" and "left" with "second" (gprs_presolve --left).
A counter of "column_counts" or "left" that is zero is left out, and so is
a zero entry of "columns" and "col_from_rows". Times and the other
families are dropped; of the run, the identity of the tool, its arguments
and the number of failures stay.
analysis/bench240_stage1c.py and tools/switch_table.py read a reduced file
as they read a full one.
--all reduces <dir>/<label>/summary.json of every run directory into
<store dir>/<label>/summary.json.
"""

import json
import sys
from pathlib import Path

RUN_KEYS = ("identity", "host", "start_time", "tool", "tool_args", "reps")
INSTANCE_KEYS = (
    "name",
    "band",
    "status",
    "nnz",
    "rows",
    "cols",
    "reduced",
    "hash",
    "same_hash",
    "errors",
    "fixed_point",
    "rounds",
    "objective_nnz",
    "second",
)
FAMILY = "singleton_cols"
REDUCTIONS = (
    "free_singleton",
    "zero_cost_singleton",
    "forcing_column",
    "free_singleton_dual",
)
SPARSE = ("columns", "col_from_rows", "column_counts", "left")


def reduce_row(row):
    """Return the stored part of the row of one instance."""
    small = {key: row[key] for key in INSTANCE_KEYS if key in row}
    stats = (row.get("rules") or {}).get(FAMILY)
    if stats is not None:
        stats = {k: v for k, v in stats.items() if k != "seconds"}
        small["rules"] = {FAMILY: stats}
    if "sub_rules" in row:
        small["sub_rules"] = {
            k: v for k, v in row["sub_rules"].items() if k in REDUCTIONS
        }
    for key in SPARSE:
        if key in row:
            small[key] = {k: v for k, v in row[key].items() if v}
    return small


def reduce_summary(source, target):
    """Write the reduced form of the summary `source` to `target`."""
    with open(source) as f:
        full = json.load(f)
    small = {key: full[key] for key in RUN_KEYS if key in full}
    small["failures"] = full.get("failures", 0)
    small["instances"] = [reduce_row(row) for row in full["instances"]]
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(small, f, separators=(",", ":"))
        f.write("\n")


def main():
    """Reduce one summary, or with --all every run directory."""
    if len(sys.argv) == 4 and sys.argv[1] == "--all":
        runs = sorted(Path(sys.argv[2]).glob("*/summary.json"))
        for source in runs:
            reduce_summary(
                source, Path(sys.argv[3]) / source.parent.name / "summary.json"
            )
        print("%d summaries reduced" % len(runs))
    elif len(sys.argv) == 3:
        reduce_summary(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
