# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Table of the results of state_build_bench.

usage: state_build_table.py <result dir> [suffix] [transfer]

Reads <instance><suffix>.json for every instance in the directory (suffix "" by
default, so "usafa.json" and not "usafa.hugetlb.json") and prints, per instance
and transpose, the best run of the given transfer (default pageable): the
phases, the total, the host times it stands against and the bit comparison.
"median" is the median total of the timed runs: a run in twenty or so stalls
for 30 ms to 0.7 s in cudaMalloc or cudaFree, which the best run does not
show. "call" is the best total plus what a call pays beside the build: the
pinned memory of the transfer and, for cusparse, the handle. These are builds
in a process that has built before; state_build_cold_table.py has the first
build. Times are seconds.
"""

import json
import statistics
import sys
from pathlib import Path

COLUMNS = (
    "alloc_seconds",
    "upload_seconds",
    "device_seconds",
    "download_seconds",
    "download_resize_seconds",
    "free_seconds",
    "total_seconds",
)


def load(directory, suffix):
    """Return the timing results with the given suffix, by nonzeros."""
    results = []
    for path in sorted(Path(directory).glob("*.json")):
        stem = path.name[: -len(".json")]
        data = json.loads(path.read_text())
        if data.get("mode") != "timing":
            continue
        if stem != data["instance"] + suffix:
            continue
        results.append(data)
    return sorted(results, key=lambda d: d["nnz"])


def main():
    directory = sys.argv[1]
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    transfer = sys.argv[3] if len(sys.argv) > 3 else "pageable"
    results = load(directory, suffix)

    print(
        f"{'instance':<22}{'nnz':>11} {'host -fold':>10} {'host derv':>9} "
        f"{'transpose':<13}{'upload':>8}{'device':>8}{'downld':>8}{'(resize)':>9}"
        f"{'mem':>7}{'total':>8}{'median':>8}{'call':>8} {'derv/t':>6} {'ns/nnz':>6} {'equal':>6}"
        f" {'dev GB':>6}"
    )
    staging_key = {
        "pinned": "pinned_seconds",
        "pinned_chunked": "pinned_chunks_seconds",
    }
    for data in results:
        host = data["host"]
        first = True
        for variant in data["variants"]:
            if variant["transfer"] != transfer:
                continue
            best = dict(zip(COLUMNS, variant["best"]))
            total = best["total_seconds"]
            median = statistics.median(run[-1] for run in variant["runs"])
            call = total + data["once"].get(staging_key.get(transfer), 0.0)
            if variant["transpose"] == "cusparse":
                call += data["once"]["cusparse_create_seconds"]
            memory = max(
                variant["device_bytes_allocated_peak"],
                variant["device_bytes_sampled_peak"],
            )
            head = (
                f"{data['instance'][:21]:<22}{data['nnz']:>11} "
                f"{host['build_without_fold_seconds']:>10.4f} "
                f"{host['derived_arrays_seconds']:>9.4f} "
                if first
                else " " * 54
            )
            print(
                head + f"{variant['transpose']:<13}"
                f"{best['upload_seconds']:>8.4f}{best['device_seconds']:>8.4f}"
                f"{best['download_seconds']:>8.4f}"
                f"{best['download_resize_seconds']:>9.4f}"
                f"{best['alloc_seconds'] + best['free_seconds']:>7.4f}"
                f"{total:>8.4f}{median:>8.4f}{call:>8.4f} "
                f"{host['derived_arrays_seconds'] / total:>6.2f} "
                f"{1e9 * total / data['nnz']:>6.2f} "
                f"{variant['runs_equal']:>3}/{variant['runs_checked']:<2} "
                f"{memory / 1e9:>6.2f}"
            )
            first = False


if __name__ == "__main__":
    main()
