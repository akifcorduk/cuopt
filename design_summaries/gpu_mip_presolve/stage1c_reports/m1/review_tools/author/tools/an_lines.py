"""an_lines.py <analysis text>...: of each analysis of bench240_stage1c.py the geometric means
over 237 instances, the same with the engine's lead as zero, the counts below 0.95 and of
instances where the engine leaves less / the same / more, and the line against the engine
before."""
import re
import sys

for path in sys.argv[1:]:
    tag = path.split("/")[-1][3:-4]
    means = lead = counts = before = ""
    lines = open(path).read().split("\n")
    seen = 0
    for n, line in enumerate(lines):
        if line.startswith("without ex9, ex10, enlight_hard"):
            seen += 1
            if seen == 1:
                means = " ".join(line.split()[-3:])
                lead = " ".join(lines[n + 1].split()[-3:])
            elif seen == 2:
                counts = re.sub(r"\s+", " ", line.split("237", 1)[1]).strip()
        if line.startswith("Against the engine before"):
            before = line.split("both: ")[1]
    print("%-11s GM %s | lead as zero %s | below 0.95, less/same/more: %s | before: %s"
          % (tag, means, lead, counts, before))
