"""time_multi.py <dump dir> <passes> <table file> <name>=<binary>[,option...] ...: alternating
passes of several gprs_presolve binaries over the dumps of a directory, one process at a time.
Prints, per binary, the sum over the instances of the minimum of the total seconds and of the
seconds of each part, and the work units of the first pass; writes the minimum per instance
to the table file."""
import glob
import json
import os
import subprocess
import sys
import tempfile

D = sys.argv[1]
passes = int(sys.argv[2])
table = sys.argv[3]
binaries = []
for arg in sys.argv[4:]:
    name, command = arg.split("=", 1)
    binaries.append((name, command.split(",")))
dumps = sorted(glob.glob(D + "/*.csr"))
best = {name: {} for name, _ in binaries}
work = {name: {} for name, _ in binaries}
with tempfile.TemporaryDirectory() as tmp:
    path = tmp + "/r.json"
    for p in range(passes):
        for dump in dumps:
            inst = os.path.basename(dump)[:-4]
            # The order of the binaries turns by one with every pass: the first run on a dump
            # is the slow one.
            turn = p % len(binaries)
            for name, command in binaries[turn:] + binaries[:turn]:
                subprocess.run(
                    [command[0], dump, *command[1:], "--json", path],
                    stdout=subprocess.DEVNULL,
                    check=True,
                )
                r = json.load(open(path))
                values = {"total": r["seconds"]}
                values.update(r["seconds_outside_rules"])
                for rule, s in r["rules"].items():
                    values[rule] = s["seconds"]
                    if p == 0:
                        work[name][rule] = work[name].get(rule, 0) + s["work"]
                slot = best[name].setdefault(inst, {})
                for key, v in values.items():
                    slot[key] = min(slot.get(key, 1e30), v)
        print("pass %d done, load %s" % (p + 1, os.getloadavg()), flush=True)
keys = ["total", "state", "start", "storage", "finish", "hash", "row_rules", "propagation",
        "tightening", "doubleton", "parallel_rows", "dual_fixing", "simple_probing",
        "singleton_cols", "dual_cols", "parallel_cols"]
first = binaries[0][0]
print("%d instances, %d passes, minimum per instance, summed" % (len(dumps), passes))
print("%-16s" % "part" + "".join("%12s %7s" % (name, "%") for name, _ in binaries))
for key in keys:
    line = "%-16s" % key
    base = sum(v.get(key, 0.0) for v in best[first].values())
    for name, _ in binaries:
        s = sum(v.get(key, 0.0) for v in best[name].values())
        line += "%12.5f %+7.1f" % (s, 100 * (s / base - 1) if base > 0 else 0.0)
    print(line)
print("work units")
for key in keys[6:]:
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
        inst = os.path.basename(dump)[:-4]
        f.write(inst + " " + " ".join("%.6f" % best[name][inst]["total"] for name, _ in binaries)
                + " " + " ".join("%.6f" % best[name][inst].get("singleton_cols", 0.0)
                                 for name, _ in binaries) + "\n")
