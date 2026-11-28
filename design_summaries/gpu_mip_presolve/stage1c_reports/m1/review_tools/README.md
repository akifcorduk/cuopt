# Tools of the M1 reviews (singleton-column rules), as they were

Copied on 2026-11-28 from the scratch directories of the three M1 agents (author, validity
reviewer, worth-and-cost reviewer), because the figures of 24 register entries rest on them
(ASSUMPTIONS.md, AS-338) and the scratch directories do not survive a reboot.

They are a record, not maintained code: paths inside point to the scratch directories and to the
builds of that night, and none of it is built by the project. The source trees, the builds and
the wrong engines the agents made are not copied; the scripts here made them.

- `worth/tools/instrument*.py`: turn a copy of the engine sources into the instrumented build
  (switches read from the environment, refusal counters by cause, the second presolve that
  classifies the singleton columns that are left, the counter of repeated scans).
- `worth/tools/gprs_quality.cpp`, `quality_*.py`: read the quality of a reduced model (cost
  range, coefficient ranges, sides, equations).
- `worth/tools/highs_*.py`, `href.py`: runs of the instrumented HiGHS that count pivot kinds.
- `validity/exact/`: the check in exact rational arithmetic. `validity/probe/`: the failing
  models (also in `../probes`). `validity/second/`, `*/ident/`: the identity tools.
- `*/tools/time_multi.py`: the timing script with the turning order.
