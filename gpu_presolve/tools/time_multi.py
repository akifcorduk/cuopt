# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time several gprs_presolve binaries against each other on one machine.

Usage: time_multi.py <dump dir> <passes> <table file>
                     <name>=<binary>[,option]... [<name>=<binary>[,option]...]

Alternating passes over the *.csr dumps of the directory, one process at a
time. In every pass each dump is run by every binary; the order of the
binaries turns by one with every pass, because the first run on a dump is
the slow one (the dump is read into the file cache). The engine reports its
own seconds, so reading the dump is not part of them.

Prints, per binary, the sum over the instances of the minimum over the
passes of the total seconds and of the seconds of every part (outside the
rules, and per rule family), each with its difference to the first binary
in percent, then the work units of the families from the first pass, and
the load average after every pass. Writes the minimum per instance to the
table file: instance, total seconds per binary, seconds of the singleton
columns per binary.

Two binaries that do the same work differed by 0.1% to 0.4% in the sum on a
machine with a load average of 4 to 6; take three passes or more and name
the load next to a result.
"""

import glob
import json
import os
import subprocess
import sys
import tempfile

PARTS = [
    "total",
    "state",
    "start",
    "storage",
    "finish",
    "hash",
    "row_rules",
    "propagation",
    "tightening",
    "doubleton",
    "parallel_rows",
    "dual_fixing",
    "simple_probing",
    "singleton_cols",
    "dual_cols",
    "parallel_cols",
]
FAMILIES = PARTS[6:]


def main():
    """Run the passes and print the sums."""
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    directory, passes, table = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    binaries = []
    for arg in sys.argv[4:]:
        name, command = arg.split("=", 1)
        binaries.append((name, command.split(",")))
    dumps = sorted(glob.glob(directory + "/*.csr"))
    best = {name: {} for name, _ in binaries}
    work = {name: {} for name, _ in binaries}
    with tempfile.TemporaryDirectory() as tmp:
        path = tmp + "/report.json"
        for p in range(passes):
            turn = p % len(binaries)
            for dump in dumps:
                instance = os.path.basename(dump)[:-4]
                for name, command in binaries[turn:] + binaries[:turn]:
                    subprocess.run(
                        [command[0], dump, *command[1:], "--json", path],
                        stdout=subprocess.DEVNULL,
                        check=True,
                    )
                    with open(path) as f:
                        report = json.load(f)
                    values = {"total": report["seconds"]}
                    values.update(report["seconds_outside_rules"])
                    for rule, stats in report["rules"].items():
                        values[rule] = stats["seconds"]
                        if p == 0:
                            work[name][rule] = (
                                work[name].get(rule, 0) + stats["work"]
                            )
                    slot = best[name].setdefault(instance, {})
                    for key, value in values.items():
                        slot[key] = min(slot.get(key, 1e30), value)
            print(
                "pass %d done, load average %.2f %.2f %.2f"
                % ((p + 1,) + os.getloadavg()),
                flush=True,
            )
    first = binaries[0][0]
    print(
        "%d instances, %d passes, minimum per instance, summed"
        % (len(dumps), passes)
    )
    print(
        "%-16s" % "part"
        + "".join("%12s %7s" % (name, "%") for name, _ in binaries)
    )
    for key in PARTS:
        line = "%-16s" % key
        base = sum(v.get(key, 0.0) for v in best[first].values())
        for name, _ in binaries:
            total = sum(v.get(key, 0.0) for v in best[name].values())
            change = 100 * (total / base - 1) if base > 0 else 0.0
            line += "%12.5f %+7.1f" % (total, change)
        print(line)
    print("work units")
    for key in FAMILIES:
        line = "%-16s" % key
        for name, _ in binaries:
            line += "%20d" % work[name].get(key, 0)
        print(line)
    line = "%-16s" % "all families"
    for name, _ in binaries:
        line += "%20d" % sum(work[name].values())
    print(line)
    with open(table, "w") as f:
        f.write("instance " + " ".join(name for name, _ in binaries) + "\n")
        for dump in dumps:
            instance = os.path.basename(dump)[:-4]
            totals = ["%.6f" % best[n][instance]["total"] for n, _ in binaries]
            family = [
                "%.6f" % best[n][instance].get("singleton_cols", 0.0)
                for n, _ in binaries
            ]
            f.write(" ".join([instance] + totals + family) + "\n")


if __name__ == "__main__":
    main()
