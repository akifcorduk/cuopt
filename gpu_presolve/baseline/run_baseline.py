#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import concurrent.futures
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = Path("/home/scratch.acoerduek_sw/gpu_presolve_deps/build")
CSV_COLUMNS = [
    "name",
    "band",
    "where",
    "status",
    "rows",
    "cols",
    "nnz",
    "rows_after",
    "cols_after",
    "nnz_after",
    "rows_removed_pct",
    "cols_removed_pct",
    "nnz_removed_pct",
    "presolve_seconds",
    "presolve_core_seconds",
    "read_seconds",
    "reached_time_limit",
    "peak_rss_mb",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline_tool on the testbed and summarise the results."
    )
    parser.add_argument("--set", choices=["dev", "large", "status", "all"])
    parser.add_argument("--variant", default="stock")
    parser.add_argument("--tool")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--kill-after", type=float)
    parser.add_argument("--out")
    parser.add_argument("--only")
    parser.add_argument(
        "--env", action="append", default=[], metavar="KEY=VALUE"
    )
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--dump-csr-dir")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--testbed", default=str(PROJECT_DIR / "testbed" / "testbed.json")
    )
    parser.add_argument("--dataset-dir")
    args = parser.parse_args()

    args.only = args.only.split(",") if args.only else []
    if args.set is None:
        args.set = "all" if args.only else "dev"
    if args.tool is None:
        args.tool = str(
            BUILD_DIR / ("baseline_" + args.variant) / "baseline_tool"
        )
    if args.out is None:
        args.out = str(PROJECT_DIR / "results" / "stage0" / args.variant)
    if args.kill_after is None:
        args.kill_after = args.time_limit + 3600.0
    for item in args.env:
        if "=" not in item:
            parser.error("--env needs KEY=VALUE, got " + item)
    args.env = dict(item.split("=", 1) for item in args.env)
    if args.dump_csr_dir:
        dump_dir = Path(args.dump_csr_dir).resolve()
        if (
            PROJECT_DIR.parent in dump_dir.parents
            or dump_dir == PROJECT_DIR.parent
        ):
            parser.error("--dump-csr-dir must be outside the synced worktree")
    return args


def select_instances(testbed, set_name, only):
    main = testbed["main"]
    sets = {
        "dev": [e for e in main if e["where"] == "dev"],
        "large": [e for e in main if e["where"] == "dev-large"],
        "status": testbed["status"],
        "all": main + testbed["status"],
    }
    selected = sets[set_name]
    if only:
        heldout = {e["name"] for e in testbed["heldout"]}
        blocked = [n for n in only if n in heldout]
        if blocked:
            sys.exit("held-out instances are not run: " + ", ".join(blocked))
        names = {e["name"] for e in selected}
        unknown = [n for n in only if n not in names]
        if unknown:
            sys.exit("not in set '%s': %s" % (set_name, ", ".join(unknown)))
        selected = [e for e in selected if e["name"] in only]
    return selected


def find_mps(dataset_dir, name):
    for suffix in (".mps", ".mps.gz"):
        path = Path(dataset_dir) / (name + suffix)
        if path.exists():
            return path
    return None


def instance_files(out_dir, name):
    return {
        "json": out_dir / (name + ".json"),
        "tmp": out_dir / (name + ".json.tmp"),
        "log": out_dir / (name + ".json.log"),
        "out": out_dir / (name + ".out"),
        "run": out_dir / (name + ".run.json"),
    }


def run_one(entry, args, out_dir):
    name = entry["name"]
    mps = find_mps(args.dataset_dir, name)
    files = instance_files(out_dir, name)
    for path in files.values():
        if path.exists():
            path.unlink()
    if mps is None:
        return
    command = [args.tool, str(mps), str(files["json"])]
    command += ["--time-limit", repr(args.time_limit)]
    if args.stats_only:
        command.append("--stats-only")
    if args.dump_csr_dir:
        command += [
            "--dump-csr",
            str(Path(args.dump_csr_dir) / (name + ".csr")),
        ]
    env = dict(os.environ)
    env.update(args.env)
    start = time.monotonic()
    returncode = None
    with open(files["out"], "w") as out:
        try:
            returncode = subprocess.run(
                command,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=args.kill_after,
            ).returncode
        except subprocess.TimeoutExpired:
            pass
    run = {
        "command": command,
        "env": args.env,
        "returncode": returncode,
        "killed": returncode is None,
        "wall_seconds": time.monotonic() - start,
    }
    with open(files["run"], "w") as f:
        json.dump(run, f, indent=1)
        f.write("\n")


def parse_value(text):
    for convert in (int, float):
        try:
            value = convert(text)
        except ValueError:
            continue
        if math.isfinite(value):
            return value
    return text


# DIAG <kind> key=value ... ; a kind whose lines carry a name= field becomes a dictionary by
# name ({"stage": {<name>: {...}}}), where a repeated name replaces the earlier line and is
# counted in "_replaced". Any other kind becomes the list of its lines in order of appearance.
# A DIAG line without a kind is counted in "_malformed".
def parse_diag(paths):
    diag = {}
    replaced = 0
    malformed = 0
    for path in paths:
        if not path.exists():
            continue
        with open(path, errors="replace") as f:
            for line in f:
                if not line.startswith("DIAG "):
                    continue
                tokens = line.split()
                if len(tokens) < 2 or "=" in tokens[1]:
                    malformed += 1
                    continue
                kind = tokens[1]
                fields = {}
                for token in tokens[2:]:
                    key, _, value = token.partition("=")
                    fields[key] = parse_value(value)
                if kind not in diag:
                    diag[kind] = {} if "name" in fields else []
                if isinstance(diag[kind], list):
                    diag[kind].append(fields)
                else:
                    name = str(fields.pop("name", ""))
                    replaced += name in diag[kind]
                    diag[kind][name] = fields
    if replaced:
        diag["_replaced"] = replaced
    if malformed:
        diag["_malformed"] = malformed
    return diag


def percent_removed(before, after):
    return 100.0 * (before - after) / before if before else None


def summarize(entry, args, out_dir):
    name = entry["name"]
    files = instance_files(out_dir, name)
    row = {column: None for column in CSV_COLUMNS}
    row.update(
        name=name,
        band=entry["band"],
        where=entry["where"],
        rows=entry["rows"],
        cols=entry["cols"],
        nnz=entry["nnz"],
    )
    row["tool_wall_seconds"] = None
    row["diag"] = {}
    if find_mps(args.dataset_dir, name) is None:
        row["status"] = "missing"
        return row
    if not files["run"].exists():
        row["status"] = "not_run"
        return row
    with open(files["run"]) as f:
        run = json.load(f)
    row["tool_wall_seconds"] = run["wall_seconds"]
    row["diag"] = parse_diag([files["log"], files["out"]])
    if run["killed"]:
        row["status"] = "killed"
        return row
    if run["returncode"] != 0 or not files["json"].exists():
        row["status"] = "failed(%s)" % run["returncode"]
        return row
    with open(files["json"]) as f:
        result = json.load(f)
    original = result["original"]
    row.update(
        rows=original["rows"], cols=original["cols"], nnz=original["nnz"]
    )
    row["read_seconds"] = result["read_seconds"]
    row["peak_rss_mb"] = result["peak_rss_mb"]
    if "presolve" not in result:
        row["status"] = "stats_only"
        return row
    presolve = result["presolve"]
    row["status"] = presolve["status"]
    row["presolve_seconds"] = presolve["seconds"]
    row["presolve_core_seconds"] = presolve.get("core_seconds")
    row["reached_time_limit"] = presolve["reached_time_limit"]
    if presolve["has_model"]:
        for size in ("rows", "cols", "nnz"):
            after = result["presolved"][size]
            row[size + "_after"] = after
            row[size + "_removed_pct"] = percent_removed(original[size], after)
    return row


def diag_stages(row):
    stages = row["diag"].get("stage", {})
    return stages if isinstance(stages, dict) else {}


def csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return "%.6g" % value
    return value


def write_summary(rows, args, out_dir):
    summary = {
        "variant": args.variant,
        "tool": args.tool,
        "set": args.set,
        "time_limit": args.time_limit,
        "env": args.env,
        "stats_only": args.stats_only,
        "instances": rows,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")

    stages = []
    for row in rows:
        for stage in diag_stages(row):
            if stage not in stages:
                stages.append(stage)
    header = CSV_COLUMNS + ["stage_%s_seconds" % stage for stage in stages]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            cells = [csv_cell(row[column]) for column in CSV_COLUMNS]
            for stage in stages:
                fields = diag_stages(row).get(stage, {})
                cells.append(csv_cell(fields.get("seconds")))
            writer.writerow(cells)


def main():
    args = parse_args()
    with open(args.testbed) as f:
        testbed = json.load(f)
    if args.dataset_dir is None:
        args.dataset_dir = testbed["dataset_dir"]
    entries = select_instances(testbed, args.set, args.only)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        if not Path(args.tool).exists():
            sys.exit("tool not found: " + args.tool)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs
        ) as pool:
            futures = {
                pool.submit(run_one, e, args, out_dir): e for e in entries
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                done += 1
                row = summarize(futures[future], args, out_dir)
                seconds = row["presolve_seconds"]
                print(
                    "[%d/%d] %s: %s%s"
                    % (
                        done,
                        len(entries),
                        row["name"],
                        row["status"],
                        ""
                        if seconds is None
                        else ", presolve %.2f s" % seconds,
                    ),
                    flush=True,
                )

    rows = [summarize(entry, args, out_dir) for entry in entries]
    write_summary(rows, args, out_dir)
    missing = [row["name"] for row in rows if row["status"] == "missing"]
    if missing:
        print(
            "missing MPS files (%d): %s" % (len(missing), ", ".join(missing))
        )
    bad = [
        row["name"]
        for row in rows
        if row["status"] in ("killed", "not_run")
        or row["status"].startswith("failed")
    ]
    if bad:
        print("not completed (%d): %s" % (len(bad), ", ".join(bad)))
    print("summary: %s" % (out_dir / "summary.csv"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
