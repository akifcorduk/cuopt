# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn the published MIPLIB solutions of testbed instances into solution files.

MIPLIB publishes the solutions of an instance as
miplib.zib.de/downloads/solutions/<instance>/<k>/<instance>.sol.gz, with the
values by variable name. The correctness check (check/run_check.py) reads
<instance>.gprssol: one value per column of the CSR dump. convert_solution
(next to this file) does the conversion with the names of the MPS file.

--download fetches the files that are not there yet into
<download dir>/<instance>/<k>/<instance>.sol.gz; without it only files that
are already there are converted. The converted files go to --out, which
run_check.py takes as a --solutions-dir; --out/summary.json lists, per
instance, the points with their objective and violations on the model, and
the instances without a published solution.
"""

import argparse
import gzip
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEPS = Path("/home/scratch.acoerduek_sw/gpu_presolve_deps")
SITE = "https://miplib.zib.de/"
LINK = re.compile(r"downloads/solutions/([^\"/]+)/(\d+)/([^\"/]+)\.sol\.gz")


def parse_args():
    """Return the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool", default=str(DEPS / "build/baseline_stock/convert_solution")
    )
    parser.add_argument(
        "--download-dir", default=str(DEPS / "miplib_solutions")
    )
    parser.add_argument("--out", default=str(DEPS / "solutions_external"))
    parser.add_argument("--dumps", action="append")
    parser.add_argument("--dataset-dir", action="append")
    parser.add_argument("--only")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--testbed", default=str(PROJECT_DIR / "testbed" / "testbed.json")
    )
    args = parser.parse_args()
    if args.dumps is None:
        args.dumps = [str(DEPS / "csr_dumps_dev"), str(DEPS / "csr_dumps")]
    return args


def find(name, dirs, suffixes):
    """Return the first existing file name + suffix in the directories."""
    for d in dirs:
        for suffix in suffixes:
            path = Path(d) / (name + suffix)
            if path.is_file():
                return path
    return None


def fetch(url):
    """Return the content behind a URL."""
    request = urllib.request.Request(url, headers={"User-Agent": "curl"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def download(name, target):
    """Fetch the published solutions of an instance; return their count."""
    page = fetch(SITE + "instance_details_%s.html" % name).decode(
        "utf-8", "replace"
    )
    links = sorted({m.group(0) for m in LINK.finditer(page)})
    for link in links:
        instance, index, file_name = LINK.fullmatch(link).groups()
        if instance != name or file_name != name:
            continue
        path = target / name / index / (name + ".sol.gz")
        if path.is_file():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        part.write_bytes(fetch(SITE + link))
        part.rename(path)
    return len(links)


def convert(args, name, files, out):
    """Convert the solution files of one instance; return its record."""
    record = {"name": name, "files": [str(f) for f in files]}
    mps = find(name, args.dataset_dir, (".mps", ".mps.gz"))
    dump = find(name, args.dumps, (".csr",))
    if mps is None or dump is None:
        record["error"] = "no MPS file" if mps is None else "no dump"
        return record
    work = out / "unpacked" / name
    work.mkdir(parents=True, exist_ok=True)
    plain = []
    for f in files:
        target = work / ("%s.%s.sol" % (name, f.parent.name))
        with gzip.open(f, "rb") as source, open(target, "wb") as sink:
            shutil.copyfileobj(source, sink)
        plain.append(target)
    result = out / (name + ".gprssol")
    part = out / (name + ".gprssol.part")
    command = [args.tool, str(mps), str(dump), str(part)] + [
        str(p) for p in plain
    ]
    done = subprocess.run(command, capture_output=True, text=True)
    shutil.rmtree(work)
    if done.returncode != 0:
        part.unlink(missing_ok=True)
        record["error"] = done.stderr.strip()[-500:] or (
            "exit code %d" % done.returncode
        )
        return record
    part.rename(result)
    record.update(json.loads(done.stdout))
    record["out"] = str(result)
    return record


def main():
    """Convert every published solution of the selected instances."""
    args = parse_args()
    with open(args.testbed) as f:
        testbed = json.load(f)
    if args.dataset_dir is None:
        args.dataset_dir = [testbed["dataset_dir"]] + testbed.get(
            "extra_dataset_dirs", []
        )
    names = [e["name"] for e in testbed["main"] + testbed["status"]]
    if args.only:
        names = [n for n in names if n in set(args.only.split(","))]
    downloads = Path(args.download_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    failed = 0
    for name in names:
        if args.download:
            try:
                download(name, downloads)
            except OSError as error:
                records.append({"name": name, "error": "download: %s" % error})
                print("%s: download failed: %s" % (name, error), flush=True)
                failed += 1
                continue
        files = sorted((downloads / name).glob("*/%s.sol.gz" % name))
        if not files:
            records.append({"name": name, "files": []})
            print("%s: no published solution" % name, flush=True)
            continue
        record = convert(args, name, files, out)
        records.append(record)
        if "error" in record:
            failed += 1
            print("%s: failed: %s" % (name, record["error"]), flush=True)
            continue
        for p in record["points"]:
            print(
                "%s: objective %.10g (published %s), violations: row %.2g, "
                "bound %.2g, integrality %.2g"
                % (
                    name,
                    p["objective"],
                    p["reported_objective"],
                    p["row_violation"],
                    p["bound_violation"],
                    p["integrality_violation"],
                ),
                flush=True,
            )
    with open(out / "summary.json", "w") as f:
        json.dump({"instances": records, "failed": failed}, f, indent=1)
    with_points = sum(1 for r in records if r.get("points"))
    print(
        "%d instances, %d with converted points, %d failed"
        % (len(records), with_points, failed)
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
