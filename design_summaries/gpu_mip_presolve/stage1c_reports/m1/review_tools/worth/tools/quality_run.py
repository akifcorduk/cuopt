"""quality_run.py <out file> <jobs> <tool> [tool option]...: the quality tool on the 240 local
dumps, one JSON line per instance."""
import glob
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

SCRATCH = "/private/tmp/claude-502/-Users-acoerduek--claude-spares-spare-bsnTJo/c660eb6b-7edc-4afc-bf62-0442cb32d870/scratchpad"
out, jobs, tool, args = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
dumps = sorted(glob.glob(SCRATCH + "/csr_dumps_bench/*.csr"))


def run(dump):
    p = subprocess.run([tool, dump] + args, capture_output=True, text=True)
    assert p.returncode == 0, (dump, p.stderr)
    return p.stdout.strip()


with ThreadPoolExecutor(jobs) as pool, open(out, "w") as f:
    for line in pool.map(run, dumps):
        f.write(line + "\n")
print("wrote %s: %d instances" % (out, len(dumps)))
