# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare an engine run with a HiGHS reference and the engine before Stage 1c.

Usage: bench240_stage1c.py [--engine DIR]... [--ref PATH]... [--before DIR]...
                           [--testbed] [--without NAME[,NAME]]... [--csv FILE]

--engine  result directory of tools/run_engine.py (summary.json; a field the
          summary does not carry is read from the instance reports next to
          it). Default: the engine before Stage 1c.
--ref     a HiGHS reference of baseline/run_baseline.py: a directory with
          summary.json or summary.csv, or such a file. The counters need the
          JSON. Default: results/bench240/highs_cheap_all (all cheap rules
          on); others are a ladder step results/bench240/stage1c_refs/<step>
          and results/bench240/cheap_ablation/<name>.csv.
--before  the engine before Stage 1c. Default:
          results/bench240/v4pre/release_all.
--testbed the defaults of the testbed: engine and before
          results/stage1/host/v4/release_dev and release_large, reference
          results/stage1c/refs/cheap_dev and cheap_large.
--without one more line of means: the first instance set less the named
          instances (the largest gains, for example).
--csv     write the table per instance.
An option given several times names several summaries that are read as one
side. A relative path is taken from the current directory, or else from
gpu_presolve/.

Presolve only, no scaling. An instance counts when both sides have a result:
one whose side hit the time limit, was not run, or ended without a model
(infeasible, unbounded) is listed and left out. A size of zero counts as one
in a ratio. Printed:
1. The measures remaining nonzeros, rows and columns, each as the geometric
   mean of reference / engine (below 1 the engine leaves more): without ex9,
   ex10 and enlight_hard, which a full presolve reduces to almost nothing;
   the same with every lead of the engine counted as zero (a ratio above 1
   as 1); per size class; over all instances. Per measure the number of
   instances below 0.95 and where the engine leaves less, the same or more.
   Matrix nonzeros plus objective nonzeros are a fourth measure when both
   sides report objective nonzeros (OBJECTIVE_ENGINE, OBJECTIVE_REFERENCE),
   and one line says which side does not. The Stage 1c gate is 0.99 in each
   measure, a lower bound only.
2. Per measure the instances below 0.95 and those where the engine leaves
   less, with both sizes.
3. Every instance where the engine leaves more nonzeros, rows or columns than
   the engine before Stage 1c.
4. The findings of the engine per rule family and sub-rule next to the HiGHS
   counters of the matching rule (the table RULES), the work per family in
   passes of the input model (nonzeros + 8 * (rows + columns)), the rounds,
   and the fixed-point flags: how many are set, which were set before and
   are not now.
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
BEFORE_BENCH = ["results/bench240/v4pre/release_all"]
BEFORE_TESTBED = [
    "results/stage1/host/v4/release_dev",
    "results/stage1/host/v4/release_large",
]
REF_BENCH = ["results/bench240/highs_cheap_all"]
REF_TESTBED = [
    "results/stage1c/refs/cheap_dev",
    "results/stage1c/refs/cheap_large",
]

MEASURES = (("nnz", "nonzeros"), ("rows", "rows"), ("cols", "columns"))
LEFT_OUT = ("ex9", "ex10", "enlight_hard")
BELOW = 0.95
MODEL_STATUSES = ("reduced", "unchanged", "reduced_to_empty")
ONE_SIDED = ("not in the engine result", "not in the reference")
# options_t::max_rounds and line_visit_work of the engine.
MAX_ROUNDS = 20
LINE_VISIT_WORK = 8
STAGE1C_FAMILIES = (
    "simple_probing",
    "singleton_cols",
    "dual_cols",
    "parallel_cols",
)
# Passes of the Stage 1c families together above which an instance is named.
NAMED_PASSES = 10.0

# Objective nonzeros of the reduced model in an engine result and of the
# presolved model in a reference summary. No stored result has such a field
# yet, so the names are this script's until the two tools write them.
OBJECTIVE_ENGINE = "reduced.obj_nnz"
OBJECTIVE_REFERENCE = "obj_nnz_after"

# One line per rule family or sub-rule: the finding counters of the engine
# (paths into the result of an instance, summed) and the HiGHS counters of
# the matching rule (DIAG lines of baseline/HIGHS_DIAG.md as <kind>.<field>,
# summed; a name of `fixing` or `bound_change` stands for its count). Each
# side counts what its own rule calls a finding, so a line shows where a rule
# fires and how often, not which side finds more. A milestone that adds a
# family or a sub-rule adds or corrects its line; the sub-rule fields of the
# Stage 1c families are the option names of the design until the engine
# writes them.
RULES = (
    (
        "row_rules",
        "",
        ("rules.row_rules.findings",),
        (
            "reductions.redundant_row",
            "reductions.singleton_row",
            "reductions.forcing_row",
        ),
    ),
    (
        "propagation",
        "",
        ("rules.propagation.findings",),
        ("bound_change.implied_bound", "bound_change.binary_row_test"),
    ),
    ("tightening", "", ("rules.tightening.findings",), ()),
    ("doubleton", "", ("rules.doubleton.findings",), ("doubleton.removed",)),
    (
        "parallel_rows",
        "",
        ("rules.parallel_rows.findings",),
        ("parallel.duplicate_rows", "parallel.row_additions"),
    ),
    (
        "parallel_rows",
        "merges",
        ("parallel_rows.merges",),
        ("parallel.duplicate_rows",),
    ),
    (
        "parallel_rows",
        "subtractions",
        ("parallel_rows.subtractions",),
        ("parallel.row_additions",),
    ),
    # HiGHS's dominated tests run before its lock rule and take its count.
    (
        "dual_fixing",
        "",
        ("rules.dual_fixing.findings",),
        (
            "fixing.dual_fixing_lock",
            "fixing.empty_col",
            "fixing.dominated_col",
            "fixing.weakly_dominated_col",
        ),
    ),
    (
        "simple_probing",
        "",
        ("rules.simple_probing.findings",),
        ("substitutions.simple_probing", "fixing.simple_probing"),
    ),
    (
        "singleton_cols",
        "",
        ("rules.singleton_cols.findings",),
        (
            "reductions.free_col_substitution",
            "reductions.zero_obj_singleton_col",
            "reductions.forcing_column",
        ),
    ),
    (
        "singleton_cols",
        "free_singleton, primal and dual",
        (
            "singleton_cols.free_singleton",
            "singleton_cols.free_singleton_dual",
        ),
        ("reductions.free_col_substitution",),
    ),
    (
        "singleton_cols",
        "free_singleton",
        ("singleton_cols.free_singleton",),
        (),
    ),
    (
        "singleton_cols",
        "free_singleton_dual",
        ("singleton_cols.free_singleton_dual",),
        (),
    ),
    (
        "singleton_cols",
        "zero_cost_singleton",
        ("singleton_cols.zero_cost_singleton",),
        ("reductions.zero_obj_singleton_col",),
    ),
    (
        "singleton_cols",
        "forcing_column",
        ("singleton_cols.forcing_column",),
        ("reductions.forcing_column",),
    ),
    (
        "dual_cols",
        "",
        ("rules.dual_cols.findings",),
        (
            "fixing.dual_fixing_bound",
            "bound_change.dual_fixing",
            "substitutions.dual_fixing",
        ),
    ),
    (
        "dual_cols",
        "dual_bound",
        ("dual_cols.dual_bound",),
        ("fixing.dual_fixing_bound", "bound_change.dual_fixing"),
    ),
    (
        "dual_cols",
        "lock_alias",
        ("dual_cols.lock_alias",),
        ("substitutions.dual_fixing",),
    ),
    (
        "parallel_cols",
        "",
        ("rules.parallel_cols.findings",),
        ("parallel.cols_dominated", "parallel.cols_merged"),
    ),
    (
        "parallel_cols",
        "parallel_dominance",
        ("parallel_cols.parallel_dominance",),
        ("parallel.cols_dominated",),
    ),
    (
        "parallel_cols",
        "parallel_merge",
        ("parallel_cols.parallel_merge",),
        ("parallel.cols_merged",),
    ),
    ("not built", "implied equation", (), ("reductions.implied_equation",)),
    (
        "not built",
        "single-equation dual fixing",
        (),
        ("fixing.dual_fixing_single_equation",),
    ),
    ("not built", "column stuffing", (), ("fixing.col_stuffing",)),
    (
        "not built",
        "implied-integer conversion",
        (),
        ("bound_change.implied_integer",),
    ),
)


def parse_args():
    """Return the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--engine", action="append")
    parser.add_argument("--ref", action="append")
    parser.add_argument("--before", action="append")
    parser.add_argument("--testbed", action="store_true")
    parser.add_argument("--without", action="append", default=[])
    parser.add_argument("--csv")
    return parser.parse_args()


def locate(text):
    """Return a path as given, or under gpu_presolve/ when it is not there."""
    path = Path(text)
    inside = PROJECT / path
    if not path.is_absolute() and not path.exists() and inside.exists():
        return inside
    return path


def field(row, path):
    """Return the value at a dotted path of a result, or None without it."""
    value = row
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def add_report_fields(directory, row, reps):
    """Add what the fastest instance report holds beyond the summary."""
    if "seconds" not in row:
        return
    for rep in range(reps):
        path = directory / ("%s.rep%d.json" % (row["name"], rep))
        if not path.is_file():
            continue
        with open(path) as f:
            report = json.load(f)
        if report.get("seconds") == row["seconds"]:
            for key, value in report.items():
                row.setdefault(key, value)
            return


def engine_rows(directories):
    """Return the instances of engine summaries by name, a line per summary."""
    rows = {}
    sources = []
    for directory in directories:
        path = directory / "summary.json"
        if not path.is_file():
            sys.exit("no engine summary %s" % path)
        with open(path) as f:
            summary = json.load(f)
        for row in summary["instances"]:
            if row["name"] in rows:
                sys.exit("%s is in two engine summaries" % row["name"])
            add_report_fields(directory, row, summary.get("reps", 0))
            rows[row["name"]] = row
        identity = summary.get("identity") or {}
        source = "source commit %s" % str(identity.get("source_commit"))[:12]
        if identity.get("source_commit") is None:
            checksum = str(identity.get("source_checksum"))[:12]
            source = "source checksum %s" % checksum
        sources.append(
            "%s: %d instances, %d with a failure, %s, started %s"
            % (
                directory,
                len(summary["instances"]),
                summary.get("failures", 0),
                source,
                summary.get("start_time"),
            )
        )
    return rows, sources


def from_csv(cells):
    """Return a line of a reference summary.csv typed as in summary.json."""
    row = dict(cells)
    for key in ["rows", "cols", "nnz", OBJECTIVE_REFERENCE] + [
        key + "_after" for key, _ in MEASURES
    ]:
        if key in cells:
            row[key] = int(float(cells[key])) if cells[key] != "" else None
    row["reached_time_limit"] = cells.get("reached_time_limit") in (
        "1",
        "True",
        "true",
    )
    return row


def reference_rows(paths):
    """Return the instances of references by name, a line per summary."""
    rows = {}
    sources = []
    for path in paths:
        if path.is_dir():
            both = [path / "summary.json", path / "summary.csv"]
            path = both[0] if both[0].is_file() else both[1]
        if not path.is_file():
            sys.exit("no reference summary %s (.json or .csv)" % path)
        if path.suffix == ".json":
            with open(path) as f:
                summary = json.load(f)
            found = summary["instances"]
            source = "variant %s, presolve_rule_off %s, switched off: %s" % (
                summary.get("variant"),
                summary.get("presolve_rule_off"),
                (summary.get("env") or {}).get("HIGHS_DIAG_OFF"),
            )
        else:
            with open(path, newline="") as f:
                found = [from_csv(cells) for cells in csv.DictReader(f)]
            source = "a table without counters"
        for row in found:
            if row["name"] in rows:
                sys.exit("%s is in two reference summaries" % row["name"])
            rows[row["name"]] = row
        sources.append("%s: %d instances, %s" % (path, len(found), source))
    return rows, sources


def engine_sizes(row):
    """Return what the engine leaves in each measure, or why it has none."""
    if row is None:
        return None, ONE_SIDED[0]
    if "reduced" not in row:
        return None, "engine: no result"
    if row["status"] == "limit":
        return None, "engine: time limit"
    if row["status"] not in MODEL_STATUSES:
        return None, "engine: no reduced model (%s)" % row["status"]
    return [row["reduced"][key] for key, _ in MEASURES], ""


def reference_sizes(row):
    """Return what the reference leaves in each measure, or why it has none."""
    if row is None:
        return None, ONE_SIDED[1]
    if row.get("reached_time_limit"):
        return None, "reference: time limit"
    sizes = [row.get(key + "_after") for key, _ in MEASURES]
    if None in sizes:
        return None, "reference: no result (%s)" % row.get("status")
    return sizes, ""


def compare(engine, reference):
    """Return a record per instance with both sides, the others by reason."""
    records = []
    left = {}
    for name in set(engine) | set(reference):
        ours, why_ours = engine_sizes(engine.get(name))
        theirs, why_theirs = reference_sizes(reference.get(name))
        if ours is None or theirs is None:
            reason = "; ".join(w for w in (why_ours, why_theirs) if w)
            left.setdefault(reason, []).append(name)
            continue
        records.append(
            {
                "name": name,
                "band": reference[name].get("band") or engine[name]["band"],
                "engine": engine[name],
                "reference": reference[name],
                "ours": ours,
                "theirs": theirs,
            }
        )
    records.sort(key=lambda r: (r["engine"]["nnz"], r["name"]))
    return records, left


def print_left_out(left):
    """Print the instances without a result on both sides, by reason."""
    for reason in sorted(left):
        names = sorted(left[reason])
        listed = names[:8] if reason in ONE_SIDED else names
        print(
            "left out, %s (%d): %s%s"
            % (
                reason,
                len(names),
                ", ".join(listed),
                ", ..." if len(listed) < len(names) else "",
            )
        )


def add_objective(records, before):
    """Add nonzeros plus objective nonzeros as a measure; False without them."""
    ours = [field(r["engine"], OBJECTIVE_ENGINE) for r in records]
    theirs = [field(r["reference"], OBJECTIVE_REFERENCE) for r in records]
    lacking = [
        "%s has no %s on %d of %d instances"
        % (side, path, values.count(None), len(values))
        for side, path, values in (
            ("the engine result", OBJECTIVE_ENGINE, ours),
            ("the reference", OBJECTIVE_REFERENCE, theirs),
        )
        if None in values
    ]
    then = [
        field(before.get(r["name"], {}), OBJECTIVE_ENGINE) for r in records
    ]
    known = [
        "%s %d" % (side, sum(values))
        for side, values in (
            ("the engine", ours),
            ("the reference", theirs),
            ("the engine before Stage 1c", then),
        )
        if None not in values
    ]
    if known:
        print(
            "\nObjective nonzeros left on %d instances: %s"
            % (len(records), ", ".join(known))
        )
    if lacking:
        print(
            "\nNonzeros plus objective nonzeros are not compared: %s"
            % "; ".join(lacking)
        )
        return False
    for r, mine, other in zip(records, ours, theirs):
        r["ours"].append(r["ours"][0] + mine)
        r["theirs"].append(r["theirs"][0] + other)
    return True


def geomean(values):
    """Return the geometric mean of positive values, or None when empty."""
    values = list(values)
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def show(value, form="%.4f"):
    """Return a number for a table cell, a dash for a missing one."""
    return "-" if value is None else form % value


def instance_sets(records, without):
    """Return the instance sets of the means: label, records, lead capped."""
    left_out = [n for n in LEFT_OUT if any(r["name"] == n for r in records)]
    first = [r for r in records if r["name"] not in LEFT_OUT]
    label = "without " + ", ".join(left_out) if left_out else "all"
    sets = [
        (label, first, False),
        ("  the engine's lead counted as zero", first, True),
    ]
    for text in without:
        names = text.split(",")
        unknown = [n for n in names if all(r["name"] != n for r in first)]
        if unknown:
            sys.exit("--without: not compared: %s" % ", ".join(unknown))
        chosen = [r for r in first if r["name"] not in names]
        sets.append(("  also without " + ", ".join(names), chosen, False))
    bands = []
    for r in first:
        if r["band"] not in bands:
            bands.append(r["band"])
    for band in bands:
        chosen = [r for r in first if r["band"] == band]
        sets.append(("  class %s" % band, chosen, False))
    if left_out:
        sets.append(("all", records, False))
    return sets


def print_measures(columns, sets):
    """Print the geometric means and the counts of every instance set."""
    width = max(len(label) for label, _, _ in sets)
    print(
        "\nGeometric mean of reference / engine; below 1 the engine leaves "
        "more"
    )
    print(
        "%-*s %4s %s"
        % (width, "instances", "n", " ".join("%16s" % c for c in columns))
    )
    for label, chosen, capped in sets:
        means = [
            geomean(
                min(r["ratio"][k], 1.0) if capped else r["ratio"][k]
                for r in chosen
            )
            for k in range(len(columns))
        ]
        print(
            "%-*s %4d %s"
            % (
                width,
                label,
                len(chosen),
                " ".join("%16s" % show(m) for m in means),
            )
        )
    print(
        "\nInstances below %.2f | the engine leaves less / the same / more"
        % BELOW
    )
    print(
        "%-*s %4s %s"
        % (width, "instances", "n", " ".join("%24s" % c for c in columns))
    )
    for label, chosen, capped in sets:
        if capped or label.startswith("  class"):
            continue
        cells = []
        for k in range(len(columns)):
            ratios = [r["ratio"][k] for r in chosen]
            cells.append(
                "%3d | %3d / %3d / %3d"
                % (
                    sum(1 for v in ratios if v < BELOW),
                    sum(1 for v in ratios if v > 1.0),
                    sum(1 for v in ratios if v == 1.0),
                    sum(1 for v in ratios if v < 1.0),
                )
            )
        print(
            "%-*s %4d %s"
            % (width, label, len(chosen), " ".join("%24s" % c for c in cells))
        )


def print_lists(columns, records):
    """Print per measure the instances below the mark and the engine's leads."""
    for k, column in enumerate(columns):
        below = sorted(
            (r for r in records if r["ratio"][k] < BELOW),
            key=lambda r: (r["ratio"][k], r["name"]),
        )
        ahead = sorted(
            (r for r in records if r["ratio"][k] > 1.0),
            key=lambda r: (-r["ratio"][k], r["name"]),
        )
        for title, chosen in (
            ("below %.2f" % BELOW, below),
            ("the engine leaves less", ahead),
        ):
            print(
                "\n%s, %s: %d of %d instances"
                % (column.capitalize(), title, len(chosen), len(records))
            )
            if chosen:
                print(
                    "  %9s %12s %12s  %-5s %s"
                    % ("ratio", "reference", "engine", "class", "instance")
                )
            for r in chosen:
                print(
                    "  %9.4f %12d %12d  %-5s %s%s"
                    % (
                        r["ratio"][k],
                        r["theirs"][k],
                        r["ours"][k],
                        r["band"],
                        r["name"],
                        "  (not in the first mean)"
                        if r["name"] in LEFT_OUT
                        else "",
                    )
                )


def print_before(engine, before):
    """Print every instance where the engine leaves more than before."""
    common = []
    lost = []
    for name in sorted(set(engine) & set(before)):
        now, why = engine_sizes(engine[name])
        then, _ = engine_sizes(before[name])
        if now is not None and then is not None:
            common.append((name, then, now))
        elif then is not None:
            lost.append("%s (%s)" % (name, why))
    more = [c for c in common if any(n > t for n, t in zip(c[2], c[1]))]
    print(
        "\nAgainst the engine before Stage 1c, %d instances with a result "
        "in both: the engine leaves less on %s; more in a measure on %d"
        % (
            len(common),
            ", ".join(
                "%d in %s"
                % (sum(1 for c in common if c[2][k] < c[1][k]), label)
                for k, (_, label) in enumerate(MEASURES)
            ),
            len(more),
        )
    )
    for name, then, now in more:
        print(
            "  %-28s %s"
            % (
                name,
                ", ".join(
                    "%s %d -> %d%s"
                    % (
                        label,
                        then[k],
                        now[k],
                        " MORE" if now[k] > then[k] else "",
                    )
                    for k, (_, label) in enumerate(MEASURES)
                ),
            )
        )
    if lost:
        print("  a result before and none now: %s" % ", ".join(lost))


def highs_counter(diag, path):
    """Return a counter of the DIAG lines of an instance, or None without it."""
    kind, key = path.split(".")
    lines = diag.get(kind)
    if lines is None:
        return None
    if isinstance(lines, dict):
        return lines.get(key, {}).get("count", 0)
    if kind == "reductions":
        lines = [line for line in lines if line.get("stage") == "all"]
    values = [line[key] for line in lines if key in line]
    return sum(values) if values else None


def line_counts(record, line):
    """Return the engine's and HiGHS's count of a line of RULES, or None."""
    _, _, fields, counters = line
    ours = [field(record["engine"], path) for path in fields]
    diag = record["reference"].get("diag") or {}
    theirs = [highs_counter(diag, path) for path in counters]
    return (
        None if not ours or None in ours else sum(ours),
        None if not theirs or None in theirs else sum(theirs),
    )


def count_cell(values):
    """Return the sum of the known values and how many are above zero."""
    known = [v for v in values if v is not None]
    if not known:
        return "-"
    return "%d (%d)" % (sum(known), sum(1 for v in known if v > 0))


def print_rules(records):
    """Print the engine's findings next to the HiGHS counters, per line."""
    print(
        "\nFindings of the engine and HiGHS's counters of the matching rule, "
        "summed over %d instances.\nEach side counts what its own rule calls "
        "a finding. In brackets: instances with a count above zero. A dash: "
        "the result has no such field." % len(records)
    )
    form = "%-15s %-37s %18s %18s  %s"
    print(form % ("family", "sub-rule", "engine", "HiGHS", "HiGHS counters"))
    for line in RULES:
        counts = [line_counts(r, line) for r in records]
        cells = (
            line[0],
            line[1],
            count_cell([c[0] for c in counts]) if line[2] else "",
            count_cell([c[1] for c in counts]) if line[3] else "",
            " + ".join(line[3]),
        )
        print((form % cells).rstrip())


def passes(row):
    """Return the work of each family in passes of the input model."""
    one_pass = row["nnz"] + LINE_VISIT_WORK * (row["rows"] + row["cols"])
    return {
        family: stats["work"] / one_pass
        for family, stats in (row.get("rules") or {}).items()
        if "work" in stats
    }


def total_passes(rows):
    """Return the work of all families by name, for rows that report work."""
    work = ((name, passes(row)) for name, row in rows.items())
    return [(name, sum(w.values())) for name, w in work if w]


def spread_line(label, values):
    """Return median, 90% quantile and largest of values by name as a line."""
    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    return "  %-20s %4d %8.2f %8.2f %8.2f  %s" % (
        label,
        len(ordered),
        statistics.median(v for _, v in ordered),
        ordered[int(0.9 * len(ordered))][1],
        ordered[-1][1],
        ordered[-1][0],
    )


def print_work(rows, before):
    """Print the work per family in passes of the input model."""
    work = {name: passes(row) for name, row in rows.items()}
    work = {name: w for name, w in work.items() if w}
    if not work:
        print("\nThe engine result reports no work units")
        return
    families = []
    for w in work.values():
        families += [family for family in w if family not in families]
    print(
        "\nWork in passes of the input model (nonzeros + %d * (rows + "
        "columns))" % LINE_VISIT_WORK
    )
    print(
        "  %-20s %4s %8s %8s %8s  %s"
        % ("family", "n", "median", "90%", "largest", "on")
    )
    for family in families:
        values = [(n, w[family]) for n, w in work.items() if family in w]
        print(spread_line(family, values))
    print(spread_line("all families", total_passes(rows)))
    then = total_passes(before)
    if then:
        print(spread_line("before Stage 1c, all", then))
    new = {
        n: {f: v for f, v in w.items() if f in STAGE1C_FAMILIES}
        for n, w in work.items()
    }
    new = {n: w for n, w in new.items() if w}
    if not new:
        return
    print(
        spread_line(
            "Stage 1c families", [(n, sum(w.values())) for n, w in new.items()]
        )
    )
    named = sorted(
        (
            (sum(w.values()), n, max(w, key=lambda f: w[f]))
            for n, w in new.items()
            if sum(w.values()) > NAMED_PASSES
        ),
        reverse=True,
    )
    print(
        "  above %g passes in the Stage 1c families: %s"
        % (
            NAMED_PASSES,
            ", ".join("%s %.1f (mostly %s)" % (n, v, f) for v, n, f in named)
            or "none",
        )
    )


def rounds_text(rows):
    """Return median and largest number of rounds and who is on the limit."""
    rounds = sorted((row["rounds"], name) for name, row in rows.items())
    on_limit = [name for count, name in rounds if count >= MAX_ROUNDS]
    return "median %g, largest %d (%s), on max_rounds = %d: %s" % (
        statistics.median(count for count, _ in rounds),
        rounds[-1][0],
        rounds[-1][1],
        MAX_ROUNDS,
        ", ".join(on_limit) or "none",
    )


def print_rounds_and_flags(rows, before):
    """Print the rounds and the fixed-point flags, now and before."""
    rows = {n: r for n, r in rows.items() if "rounds" in r}
    before = {n: r for n, r in before.items() if "rounds" in r}
    if rows:
        print("\nRounds, %d instances: %s" % (len(rows), rounds_text(rows)))
    if before:
        print(
            "  before Stage 1c, %d instances: %s"
            % (len(before), rounds_text(before))
        )
    flags = {
        n: r["fixed_point"] for n, r in rows.items() if "fixed_point" in r
    }
    if not flags:
        print("\nThe engine result has no fixed-point flag")
        return
    then = {
        n: r["fixed_point"] for n, r in before.items() if "fixed_point" in r
    }
    print(
        "\nFixed-point flag: set on %d of %d instances%s"
        % (
            sum(flags.values()),
            len(flags),
            "; before Stage 1c on %d of %d" % (sum(then.values()), len(then))
            if then
            else "",
        )
    )
    lists = [("not set", [n for n in flags if not flags[n]])]
    if then:
        lists += [
            (
                "set before and not now",
                [n for n in flags if not flags[n] and then.get(n)],
            ),
            (
                "set now and not before",
                [n for n in flags if flags[n] and then.get(n) is False],
            ),
        ]
    for title, names in lists:
        print("  %s: %s" % (title, ", ".join(sorted(names)) or "none"))


def write_csv(path, engine, reference, before, records, left):
    """Write the table per instance."""
    by_name = {r["name"]: r for r in records}
    why = {name: reason for reason, names in left.items() for name in names}
    keys = [key for key, _ in MEASURES]
    families = []
    for row in engine.values():
        families += [f for f in passes(row) if f not in families]
    lines = [
        (re.sub("[^a-z0-9]+", "_", " ".join(line[:2]).strip()), line)
        for line in RULES
    ]
    header = ["instance", "class"] + keys + ["left_out", "ref_status"]
    header += ["ref_" + key for key in keys] + ["engine_status"]
    header += ["engine_" + key for key in keys]
    header += ["ratio_" + key for key in keys]
    header += ["before_" + key for key in keys]
    header += ["ref_obj_nnz", "engine_obj_nnz", "before_obj_nnz"]
    header += ["rounds", "fixed_point", "before_rounds", "before_fixed_point"]
    header += ["passes"] + ["passes_" + family for family in families]
    header += ["engine_" + label for label, line in lines if line[2]]
    header += ["highs_" + label for label, line in lines if line[3]]
    names = sorted(
        set(engine) | set(reference),
        key=lambda n: ((engine.get(n) or reference[n])["nnz"], n),
    )
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        for name in names:
            ours = engine.get(name) or {}
            theirs = reference.get(name) or {}
            old = before.get(name) or {}
            record = by_name.get(name)
            work = passes(ours) if ours else {}
            cells = [name, theirs.get("band") or ours.get("band")]
            cells += [(ours or theirs).get(key) for key in keys]
            cells += [why.get(name, ""), theirs.get("status")]
            cells += [theirs.get(key + "_after") for key in keys]
            cells += [ours.get("status")]
            cells += [field(ours, "reduced." + key) for key in keys]
            cells += [
                "%.6f" % record["ratio"][k] if record else None
                for k in range(len(keys))
            ]
            cells += [field(old, "reduced." + key) for key in keys]
            cells += [
                field(theirs, OBJECTIVE_REFERENCE),
                field(ours, OBJECTIVE_ENGINE),
                field(old, OBJECTIVE_ENGINE),
                ours.get("rounds"),
                ours.get("fixed_point"),
                old.get("rounds"),
                old.get("fixed_point"),
                "%.4f" % sum(work.values()) if work else None,
            ]
            cells += [
                "%.4f" % work[family] if family in work else None
                for family in families
            ]
            both = {"engine": ours, "reference": theirs}
            counts = [line_counts(both, line) for _, line in lines]
            cells += [c[0] for c, (_, line) in zip(counts, lines) if line[2]]
            cells += [c[1] for c, (_, line) in zip(counts, lines) if line[3]]
            writer.writerow(["" if c is None else c for c in cells])
    print("\nwrote %s: %d instances" % (path, len(names)))


def main():
    """Print the comparison of one engine result with one reference."""
    args = parse_args()
    defaults = BEFORE_TESTBED if args.testbed else BEFORE_BENCH
    engine_dirs = [locate(p) for p in args.engine or defaults]
    before_dirs = [locate(p) for p in args.before or defaults]
    ref_paths = [
        locate(p)
        for p in args.ref or (REF_TESTBED if args.testbed else REF_BENCH)
    ]
    engine, sources = engine_rows(engine_dirs)
    print("\n".join("engine     " + s for s in sources))
    reference, sources = reference_rows(ref_paths)
    print("\n".join("reference  " + s for s in sources))
    if before_dirs == engine_dirs:
        before = {}
        print("before     the engine result itself, nothing to compare")
    else:
        before, sources = engine_rows(before_dirs)
        print("\n".join("before     " + s for s in sources))

    records, left = compare(engine, reference)
    if not records:
        sys.exit("no instance has a result on both sides")
    print_left_out(left)
    columns = [label for _, label in MEASURES]
    if add_objective(records, before):
        columns.append("nnz + objective")
    for r in records:
        r["ratio"] = [
            max(theirs, 1) / max(ours, 1)
            for theirs, ours in zip(r["theirs"], r["ours"])
        ]
    print_measures(columns, instance_sets(records, args.without))
    print_lists(columns, records)
    if before:
        print_before(engine, before)
    print_rules(records)
    with_result = {n: r for n, r in engine.items() if "reduced" in r}
    before_result = {n: r for n, r in before.items() if "reduced" in r}
    print_work(with_result, before_result)
    print_rounds_and_flags(with_result, before_result)
    if args.csv:
        write_csv(args.csv, engine, reference, before, records, left)
    return 0


if __name__ == "__main__":
    sys.exit(main())
