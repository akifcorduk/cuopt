# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Table of the first build of a process (state_build_bench --cold, --cold-host).

usage: state_build_cold_table.py <result dir> [suffix]

Reads <instance><suffix>.cold.jsonl for every instance in the directory
(suffix "" by default) and prints per instance the medians over the
processes: the host loop as the first work of a process, and per device
variant the first call (the build, the pinned memory of its transfer and
the cuSPARSE handle when it is used) and the second build of the same
process. The last column is host / first call of radix_sort with the
pageable transfer. Times are seconds.
"""

import json
import statistics
import sys
from pathlib import Path

VARIANTS = (
    ("cusparse", "pageable"),
    ("radix_sort", "pageable"),
    ("scatter_sort", "pageable"),
    ("radix_sort", "pinned"),
    ("radix_sort", "pinned_chunked"),
)
TRANSFER_LABEL = {"pageable": "", "pinned": "/pin", "pinned_chunked": "/chunk"}


def load(directory, suffix):
    """Return {instance: lines} for the files with the given suffix."""
    results = {}
    ending = suffix + ".cold.jsonl"
    for path in sorted(Path(directory).glob("*" + ending)):
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        if lines and path.name == lines[0]["instance"] + ending:
            results[lines[0]["instance"]] = lines
    return results


def main():
    directory = sys.argv[1]
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    results = load(directory, suffix)

    head = f"{'instance':<22}{'nnz':>11}{'n':>3}{'busy':>6}{'host':>9}"
    for transpose, transfer in VARIANTS:
        label = transpose[:5] + TRANSFER_LABEL[transfer]
        head += f"{label + ' 1st':>17}{'2nd':>8}"
    print(head + f"{'host/radix':>11}{'equal':>7}")

    for lines in sorted(results.values(), key=lambda rows: rows[0]["nnz"]):
        host = [row for row in lines if row["mode"] == "cold_host"]
        device = [row for row in lines if row["mode"] == "cold"]
        host_first = statistics.median(
            row["derived_arrays_first_seconds"] for row in host
        )
        busy = statistics.mean(row["cpus_busy_mean"] for row in lines)
        equal = all(
            row["first_equal"] and row["second_equal"] for row in device
        ) and all(row["derived_arrays_equal_state"] for row in host)
        text = (
            f"{lines[0]['instance'][:21]:<22}{lines[0]['nnz']:>11}{len(host):>3}"
            f"{busy:>6.1f}{host_first:>9.4f}"
        )
        radix_first = None
        for transpose, transfer in VARIANTS:
            rows = [
                row
                for row in device
                if row["transpose"] == transpose
                and row["transfer"] == transfer
            ]
            if not rows:
                text += f"{'-':>17}{'-':>8}"
                continue
            first = statistics.median(
                row["first_call_seconds"] for row in rows
            )
            second = statistics.median(
                row["second_call_seconds"] for row in rows
            )
            text += f"{first:>17.4f}{second:>8.4f}"
            if (transpose, transfer) == ("radix_sort", "pageable"):
                radix_first = first
        ratio = (
            f"{host_first / radix_first:>11.2f}"
            if radix_first
            else f"{'-':>11}"
        )
        print(text + ratio + f"{'yes' if equal else 'NO':>7}")


if __name__ == "__main__":
    main()
