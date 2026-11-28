"""highs_run.py <tag> <rule bits off> <extra switches off or -> <jobs> <instance>...: the local
instrumented HiGHS (presolve only) with the switch set of reference step m1 plus the extra
switches, on the named instances ('@m1' = every instance where the stored m1_a reference counts
a free column substitution or a zero-cost singleton column). One line per instance with sizes
and the DIAG lines of the singleton rules; results in highs/<tag>/."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

SCRATCH = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad"
S = SCRATCH + "/s1c_m1_worth"
W = "/Users/acoerduek/Documents/workspace/cuopt/.claude/worktrees/wf_baf55528-756-3/gpu_presolve"
TOOL = SCRATCH + "/s1c_highs/bin/baseline_tool_new"
MPS = "/Users/acoerduek/Downloads/miplib2017"
HEAVY = ("probing,enumeration,clique_substitutions,clique_merging,implied_integers,aggregator,"
         "sparsify,fourier_motzkin,strengthen_inequalities,dominated_columns")
MINOR = "implied_equations,implied_integer_conversion"
M1_OFF = HEAVY + "," + MINOR + ",dual_fixing_substitution,parallel_columns,simple_probing"

tag, bits, extra, jobs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
names = sys.argv[5:]
if names == ["@m1"]:
    ref = json.load(open(W + "/results/bench240/stage1c_refs/m1_a/summary.json"))
    names = []
    for i in ref["instances"]:
        for r in i["diag"]["reductions"]:
            if r["stage"] == "all" and (r["free_col_substitution"] or r["zero_obj_singleton_col"]):
                names.append(i["name"])
out = S + "/highs/" + tag
os.makedirs(out, exist_ok=True)
off = M1_OFF if extra == "-" else M1_OFF + "," + extra


def run(name):
    js = out + "/" + name + ".json"
    if not os.path.exists(js):
        env = dict(os.environ, HIGHS_DIAG_OFF=off)
        subprocess.run([TOOL, MPS + "/" + name + ".mps", js, "--presolve-rule-off", bits,
                        "--time-limit", "600"], env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    d = json.load(open(js))
    p = d.get("presolved", {})
    lines = []
    log = js + ".log"
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            if line.startswith("DIAG free_col_substitution name=singleton_col") or \
               line.startswith("DIAG reductions stage=all"):
                lines.append(line.strip())
    return name, p.get("nnz"), p.get("rows"), p.get("cols"), lines


with ThreadPoolExecutor(jobs) as pool:
    for name, nnz, rows, cols, lines in pool.map(run, names):
        print("%s %s %s %s" % (name, nnz, rows, cols))
        for line in lines:
            if "free_col_substitution name" in line:
                print("   " + line[len("DIAG free_col_substitution "):])
            else:
                keep = [w for w in line.split() if w.split("=")[0] in
                        ("free_col_substitution", "zero_obj_singleton_col", "forcing_column",
                         "doubleton_equation", "fixed_col", "redundant_row", "duplicate_row")]
                print("   " + " ".join(keep))
        sys.stdout.flush()
