"""show.py <instance> <engine tag>...: sizes, findings and column statistics of an instance in
engine runs, one block per tag."""
import json
import os
import sys

S = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad/s1c_m1_worth"
name = sys.argv[1]
for tag in sys.argv[2:]:
    path = tag if os.path.isabs(tag) else S + "/bench/" + tag
    d = json.load(open(path + "/summary.json"))
    for i in d["instances"]:
        if i["name"] != name:
            continue
        r = i["reduced"]
        print("%s %s: input %d nnz %d rows %d cols -> %d nnz %d rows %d cols; rounds %d fixed_point %s status %s"
              % (name, tag, i["nnz"], i["rows"], i["cols"], r["nnz"], r["rows"], r["cols"],
                 i["rounds"], i["fixed_point"], i["status"]))
        for rule, s in i["rules"].items():
            if s["calls"] or s["findings"]:
                print("   %-15s calls %3d work %11d first %11d findings %8d rows %7d cols %7d nnz %8d  %.4fs"
                      % (rule, s["calls"], s["work"], s.get("first_work", 0), s["findings"],
                         s["rows_removed"], s["cols_removed"], s["nnz_removed"], s["seconds"]))
        for key in ("sub_rules", "columns", "col_from_rows", "objective_nnz", "parallel_rows",
                    "dual_findings"):
            if key in i:
                print("   %s: %s" % (key, i[key]))
