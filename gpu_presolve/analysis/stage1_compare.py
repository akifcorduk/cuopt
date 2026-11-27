# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare a host engine run with the Stage 1 reference and apply the gates.

Usage: stage1_compare.py [tag]   (default v1)
Reads results/stage1/host/<tag>/release_dev (and release_large when present)
and the reference runs results/stage1/refs/stage1c and stage1c_large: their
summaries for sizes and seconds, their instance files for the statistics of
the presolved model.

Reductions gate: geometric mean of nnz(reference) / nnz(engine) at least
0.99 over the instances that have the reference, every instance below 0.95
listed. An instance whose engine status carries no reduced model, or that
has no engine result, is left out of the mean and fails the gate.
Time gate: t <= 1.2 * t(reference) + 0.02 s per instance of bands A and B;
the other bands and the median per band are reported.
Runs: an instance with an error in a repetition or with different hashes
between repetitions is flagged and fails.

Writes results/stage1/host/<tag>/compare.csv and exits with code 1 when a
gate fails.
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
STAGE1 = RESULTS / "stage1"
SHIFT = 0.1
REFERENCES = ("stage1c", "stage1c_large")
MODEL_STATUSES = ("reduced", "unchanged", "reduced_to_empty")

HEADER = [
    "band",
    "instance",
    "nnz",
    "ref_rows",
    "ref_cols",
    "ref_nnz",
    "engine_rows",
    "engine_cols",
    "engine_nnz",
    "nnz_ratio",
    "ref_seconds",
    "engine_seconds",
    "limit_seconds",
    "status",
    "rounds",
    "flags",
    "ref_status",
    "ref_runs",
    "engine_runs",
    "engine_errors",
    "ref_scaling_seconds",
    "ref_hash_seconds",
    "engine_hash_seconds",
    "ref_dynamism_p50",
    "ref_dynamism_p99",
    "ref_finite_bounds",
]


def load(path):
    """Return the instances of a runner summary by name."""
    with open(path) as f:
        return {r["name"]: r for r in json.load(f)["instances"]}


def presolved_statistics(variant, name):
    """Return dynamism and finite bounds of the reference's presolved model."""
    path = STAGE1 / "refs" / variant / (name + ".json")
    if not path.is_file():
        return {}
    with open(path) as f:
        presolved = json.load(f).get("presolved")
    if not presolved:
        return {}
    bounds = presolved["bounds"]
    return {
        "dynamism_p50": presolved["row_dynamism"]["p50"],
        "dynamism_p99": presolved["row_dynamism"]["p99"],
        "finite_bounds": 2 * presolved["cols"]
        - bounds["lower_infinite"]
        - bounds["upper_infinite"],
    }


def reference():
    """Return the Stage 1 reference by instance name, and those without sizes."""
    ref = {}
    unusable = []
    for variant in REFERENCES:
        path = STAGE1 / "refs" / variant / "summary.json"
        if not path.is_file():
            continue
        for name, r in load(path).items():
            diag = r.get("diag") or {}
            if not diag.get("total"):
                unusable.append(name)
                continue
            total = diag["total"][0]
            ref[name] = {
                "band": r["band"],
                "nnz": r["nnz"],
                "ref": [
                    int(total[k].split(":")[1])
                    for k in ("rows", "cols", "nnz")
                ],
                "seconds": r["presolve_core_seconds"],
                "status": r.get("status"),
                "scaling_seconds": diag.get("stage", {})
                .get("final_scaling", {})
                .get("seconds"),
                "hash_seconds": (diag.get("parallel") or [{}])[0].get(
                    "hash_seconds"
                ),
            }
            ref[name].update(presolved_statistics(variant, name))
    return ref, unusable


def shifted_mean(values):
    """Return the shifted geometric mean of the values."""
    return (
        math.exp(sum(math.log(v + SHIFT) for v in values) / len(values))
        - SHIFT
    )


def cell(value, form="%s"):
    """Return a CSV cell, empty for a value that is not stored."""
    return "" if value is None else form % value


def compare(runs, ref):
    """Return one record per instance that has a reference and a result."""
    records = []
    for n in sorted(
        (n for n in runs if n in ref and "reduced" in runs[n]),
        key=lambda n: ref[n]["nnz"],
    ):
        r = runs[n]
        ours = [r["reduced"][k] for k in ("rows", "cols", "nnz")]
        theirs = ref[n]["ref"]
        has_model = r["status"] in MODEL_STATUSES
        limit = 1.2 * ref[n]["seconds"] + 0.02
        record = {
            "name": n,
            "band": ref[n]["band"],
            "ours": ours,
            "theirs": theirs,
            "has_model": has_model,
            "ratio": max(theirs[2], 1) / max(ours[2], 1)
            if has_model
            else None,
            "seconds": r["seconds"],
            "ref_seconds": ref[n]["seconds"],
            "limit": limit,
            "slow": r["seconds"] > limit,
            "errors": len(r.get("errors", [])),
            "runs": len(r.get("all_seconds", [])),
            "same_hash": r["same_hash"],
        }
        flags = []
        if not has_model:
            flags.append("no reduced model: " + r["status"])
        elif record["ratio"] < 0.95:
            flags.append("fewer reductions")
        if record["slow"]:
            flags.append("slow")
        if not r["same_hash"]:
            flags.append("hash differs between runs")
        if record["errors"]:
            flags.append("errors in %d repetitions" % record["errors"])
        record["flags"] = flags
        records.append(record)
    return records


def engine_hash_seconds(host, run):
    """Return the model_hash seconds of the fastest repetition, or None.

    From the summary, or for a summary without it from the run reports.
    """
    if "seconds_outside_rules" in run:
        return run["seconds_outside_rules"].get("hash")
    for directory in ("release_dev", "release_large"):
        for path in sorted(
            (host / directory).glob(run["name"] + ".rep*.json")
        ):
            if path.name.endswith(".run.json"):
                continue
            with open(path) as f:
                report = json.load(f)
            if report.get("seconds") == run["seconds"]:
                return report.get("seconds_outside_rules", {}).get("hash")
    return None


def csv_row(record, run, ref, hash_seconds):
    """Return the row of compare.csv of one instance."""
    return (
        [record["band"], record["name"], ref["nnz"]]
        + record["theirs"]
        + record["ours"]
        + [
            cell(record["ratio"], "%.4f"),
            "%.6f" % record["ref_seconds"],
            "%.6f" % record["seconds"],
            "%.3f" % record["limit"],
            run["status"],
            run["rounds"],
            "; ".join(record["flags"]),
            cell(ref["status"]),
            1,
            record["runs"],
            record["errors"],
            cell(ref["scaling_seconds"], "%.6f"),
            cell(ref["hash_seconds"], "%.6f"),
            cell(hash_seconds, "%.6f"),
            cell(ref.get("dynamism_p50"), "%.6g"),
            cell(ref.get("dynamism_p99"), "%.6g"),
            cell(ref.get("finite_bounds")),
        ]
    )


def print_table(records, runs, ref):
    """Print one line per instance."""
    print(
        "%-3s %-24s %10s | %10s %10s %6s | %8s %8s %7s  %s"
        % (
            "",
            "instance",
            "nnz",
            "reference",
            "engine",
            "ratio",
            "ref s",
            "engine s",
            "limit",
            "status",
        )
    )
    for record in records:
        n = record["name"]
        print(
            "%-3s %-24s %10d | %10d %10d %6s | %8.3f %8.3f %7.3f  %s"
            % (
                record["band"],
                n[:24],
                ref[n]["nnz"],
                record["theirs"][2],
                record["ours"][2],
                cell(record["ratio"], "%.3f"),
                record["ref_seconds"],
                record["seconds"],
                record["limit"],
                "; ".join(record["flags"]) or runs[n]["status"],
            )
        )


def reductions_gate(records, no_result):
    """Print the reductions gate and return whether it holds."""
    ratios = [r["ratio"] for r in records if r["has_model"]]
    no_model = [r["name"] for r in records if not r["has_model"]]
    mean = (
        math.exp(sum(math.log(v) for v in ratios) / len(ratios))
        if ratios
        else float("nan")
    )
    ok = mean >= 0.99 and not no_model and not no_result
    print(
        "reductions gate: %s. Geometric mean of nnz(reference) / nnz(engine) "
        "= %.4f over %d instances (needs 0.99); below 0.95: %d; above 1.05: "
        "%d"
        % (
            "pass" if ok else "FAIL",
            mean,
            len(ratios),
            sum(1 for v in ratios if v < 0.95),
            sum(1 for v in ratios if v > 1.05),
        )
    )
    if no_model:
        print("  no reduced model, not in the mean: %s" % ", ".join(no_model))
    if no_result:
        print(
            "  reference without an engine result: %s" % ", ".join(no_result)
        )
    return ok


def time_gate(records):
    """Print the time gate and the bands and return whether the gate holds."""
    slow = [r for r in records if r["slow"]]
    gated = [r["name"] for r in slow if r["band"][0] in "AB"]
    other = [r["name"] for r in slow if r["band"][0] not in "AB"]
    print(
        "time gate: %s. %d of %d instances above 1.2 * t + 0.02 s; in bands "
        "A and B: %d%s"
        % (
            "FAIL" if gated else "pass",
            len(slow),
            len(records),
            len(gated),
            ": " + ", ".join(gated) if gated else "",
        )
    )
    if other:
        print("  above the limit in the other bands: %s" % ", ".join(other))
    few = sum(1 for r in records if r["runs"] < 3)
    print(
        "  the reference is one run per instance; the engine has fewer than "
        "three runs on %d instances (the gate asks for the minimum of three "
        "alternating runs on both sides)" % few
    )
    print(
        "%-3s %3s %9s %9s %9s %9s"
        % ("", "n", "ref med", "eng med", "ref sgm", "eng sgm")
    )
    bands = {}
    for r in records:
        bands.setdefault(r["band"], []).append(r)
    for band in sorted(bands):
        theirs = [r["ref_seconds"] for r in bands[band]]
        ours = [r["seconds"] for r in bands[band]]
        print(
            "%-3s %3d %9.3f %9.3f %9.3f %9.3f%s"
            % (
                band,
                len(ours),
                statistics.median(theirs),
                statistics.median(ours),
                shifted_mean(theirs),
                shifted_mean(ours),
                "  median above the reference"
                if statistics.median(ours) > statistics.median(theirs)
                else "",
            )
        )
    print(
        "sum of seconds: reference %.1f, engine %.1f"
        % (
            sum(r["ref_seconds"] for r in records),
            sum(r["seconds"] for r in records),
        )
    )
    return not gated


def runs_gate(records):
    """Print the instances with errors or differing hashes; True without."""
    bad = [r["name"] for r in records if r["errors"] or not r["same_hash"]]
    print(
        "runs: %s. %d instances with an error in a repetition or a hash that "
        "differs between repetitions%s"
        % (
            "FAIL" if bad else "pass",
            len(bad),
            ": " + ", ".join(bad) if bad else "",
        )
    )
    return not bad


def main():
    """Print the comparison, write compare.csv and return the exit code."""
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    host = STAGE1 / "host" / tag
    runs = load(host / "release_dev" / "summary.json")
    if (host / "release_large" / "summary.json").is_file():
        runs.update(load(host / "release_large" / "summary.json"))
    ref, unusable = reference()
    records = compare(runs, ref)
    compared = {r["name"] for r in records}
    no_result = sorted(n for n in ref if n not in compared)

    print_table(records, runs, ref)
    for n in sorted(n for n in runs if n in ref and n not in compared):
        print("no result: %s %s" % (n, runs[n].get("errors")))
    for n in sorted(n for n in runs if n not in ref):
        print("no reference: %s" % n)
    for n in unusable:
        print("reference without sizes: %s" % n)
    print()
    ok = reductions_gate(records, no_result)
    ok = time_gate(records) and ok
    ok = runs_gate(records) and ok

    with open(host / "compare.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for record in records:
            n = record["name"]
            writer.writerow(
                csv_row(
                    record, runs[n], ref[n], engine_hash_seconds(host, runs[n])
                )
            )
    print("written: %s" % (host / "compare.csv"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
