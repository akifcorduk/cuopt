"""check_sum.py <dir>...: over the gprs_check results (*.json without run records and summary)
of the directories: verdict counts, failed verdicts by name, and the sums of the structure
check per mode (rows, kept rows, added rows, columns of records, boxes, snapshots), with the
instances that have a record of a dual rule or an interval or forcing record in primal-only."""
import collections
import glob
import json
import sys

verdicts = collections.Counter()
failed = collections.Counter()
failed_at = collections.defaultdict(list)
sums = {"default": collections.Counter(), "primal_only": collections.Counter()}
status = {"default": collections.Counter(), "primal_only": collections.Counter()}
stack_errors = []
files = 0
dual_in_primal = []
for root in sys.argv[1:]:
    for f in sorted(glob.glob(root + "/*.json")):
        if f.endswith(".run.json") or f.endswith("summary.json"):
            continue
        j = json.load(open(f))
        files += 1
        name = f.split("/")[-1][:-5]
        for check, verdict in j["checks"].items():
            verdicts[verdict] += 1
            if verdict == "FAIL":
                failed[check] += 1
                failed_at[check].append(name)
        for mode in ("default", "primal_only"):
            s = j["structure"].get(mode)
            if not s:
                continue
            if s.get("stack_error"):
                stack_errors.append((name, mode, s["stack_error"]))
            c = sums[mode]
            c["ran"] += 1 if s["ran"] else 0
            c["rows_tested"] += s["rows"]["tested"]
            c["rows_failed"] += s["rows"]["failed"]
            c["kept_tested"] += s["kept_rows"]["tested"]
            c["kept_parallel"] += s["kept_rows"]["parallel"]
            c["kept_not_tested"] += s["kept_rows"]["not_tested"]
            c["kept_failed"] += s["kept_rows"]["failed"]
            c["added_rows"] += s["added_rows"]["rows"]
            c["added_tested"] += s["added_rows"]["tested"]
            c["added_not_tested"] += s["added_rows"]["not_tested"]
            for k, v in s["columns"].items():
                if k in ("box_worst", "snapshot_worst", "box_column", "integral_column", "snapshot_row"):
                    continue
                c[k] += v
            for k in ("box_worst", "snapshot_worst"):
                v = s["columns"][k]
                if isinstance(v, (int, float)) and v > c.get(k + "_max", 0):
                    c[k + "_max"] = v
            c["objective_failed"] += s["objective"]["failed"]
            c["postsolve_failed"] += s["postsolve"]["failed"]
            c["bounds_failed"] += s["bounds"]["failed"]
            c["lost_integers"] += s["lost_integers"]
            if mode == "primal_only" and (s["columns"]["interval"] or s["columns"]["forcing"]):
                dual_in_primal.append(name)
print("results %d | verdicts %s" % (files, dict(verdicts)))
for check, n in sorted(failed.items()):
    print("  FAIL %s: %d (%s)" % (check, n, ", ".join(failed_at[check][:8])))
for e in stack_errors:
    print("  stack error: %s %s %s" % e)
for mode in ("default", "primal_only"):
    print(mode, dict(sorted(sums[mode].items())))
print("primal-only results with an interval or forcing record:", dual_in_primal)
