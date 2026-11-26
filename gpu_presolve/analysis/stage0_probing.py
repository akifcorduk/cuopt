# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 0 analysis: what HiGHS probing does on the dev-size testbed.

Reads the Stage 0 result files next to this directory
(``../results/stage0``) and prints markdown tables:

1. probing per instance (candidates, probes, rounds, stop reasons, what
   was found, seconds),
2. every probing round,
3. the work counters of the probing loop (per instance and per round),
4. coverage and the limit that ended each round,
5. yield (fixings and substitutions per second and per work unit),
6. the enumeration stage, clique merging and variable-bound aggregation,
7. sizes for a GPU design that probes many variables at once,
8. presolve result with probing switched off (only if the
   ``diag_noprobing`` run is present), and with clique merging switched
   off (only if the ``diag_nocliquemerge`` run is present),
9. sums per band.

Sources: ``diag/summary.json`` (HiGHS 1.16.0 with ``highs_diag.patch``),
``stock/summary.json`` (stock HiGHS, ``presolve_core_seconds``),
``stats/<name>.json`` (structure of the original model),
``diag/<name>.json`` and ``diag_noprobing/<name>.json`` (structure
statistics of the presolved model), ``fixed_costs.json`` (GPU launch and
read-back cost) and ``testbed/testbed.json`` (the ``main`` list,
instances with ``where == "dev"``). The fields are described in
``baseline/HIGHS_DIAG.md`` and ``baseline/README.md``.

One probe is one binary column probed at both values (HiGHS
``numProbed``; a value that is infeasible ends the probe).
``probing_work = prop_entries + activity_entries`` is counted in the
probing loop only: row entries scanned by propagation plus column
entries visited to update row activities, the latter once when a bound
change is applied and once when it is undone. ``prop_sweeps`` counts
passes over the queue of model rows only: what the clique table and the
stored implications fix is applied inside the bound change and adds no
sweep. The counts are those of HiGHS's scheme (activity deltas along the
column of every changed bound); they are not the entry count of a scheme
that recomputes every touched row.

Run without arguments: ``python3 stage0_probing.py``.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
STAGE0_DIR = PROJECT_DIR / "results" / "stage0"
TESTBED_FILE = PROJECT_DIR / "testbed" / "testbed.json"

# HiGHS constants read from the source at commit 283bfc859
# (HPresolve.cpp, HPresolveProbing.cpp, HighsCliqueTable.h).
PROBING_CONTINGENT = 1000
SPLAY_MIN = 100000
SPLAY_PER_NNZ = 10
SPLAY_FAIL_PENALTY = 100
DELETIONS_MIN = 1000
DELETIONS_DIVISOR = 20
TIGHTEN_AFTER_PROBES = 2500
NEW_ITEMS_MIN = 1000000
NEW_ITEMS_PER_NNZ = 2

FULL_ROUND_COVERAGE = 0.95


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def num(value):
    return f"{value:,}"


def dec(value, digits=3):
    return f"{value:.{digits}f}"


def ratio(a, b, digits=1, scale=1.0):
    if not b:
        return "-"
    return f"{scale * a / b:.{digits}f}"


def percent(a, b, digits=1):
    return ratio(a, b, digits, 100.0)


def pair(a, b):
    return f"{num(a)} / {num(b)}"


def table(title, headers, rows, notes=()):
    print(f"\n### {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(str(cell) for cell in row) + " |")
    for note in notes:
        print(f"\n{note}")


def streak_cap(nnz):
    """Probes until the neighbourhood contingent is used up by failures.

    HiGHS starts a round with max(100000, 10 nnz) clique-table queries
    and takes 100 * (failures in a row so far) off after every probe
    that neither deletes a column nor adds a clique. With no query
    made, k failures in a row use 50 k (k - 1); the round stops at the
    first k for which that exceeds the contingent.
    """
    budget = max(SPLAY_MIN, SPLAY_PER_NNZ * nnz)
    k = 1
    while SPLAY_FAIL_PENALTY * k * (k - 1) // 2 <= budget:
        k += 1
    return k


def streak_nnz_range(probes):
    """Nonzero counts [low, high) for which `streak_cap` gives `probes`.

    None when no nonzero count gives this number of probes (it is below
    the cap of the minimum contingent).
    """
    low = SPLAY_FAIL_PENALTY * (probes - 1) * (probes - 2) // 2
    high = SPLAY_FAIL_PENALTY * probes * (probes - 1) // 2
    if high <= SPLAY_MIN:
        return None
    first = -(-low // SPLAY_PER_NNZ) if low >= SPLAY_MIN else 0
    return first, -(-high // SPLAY_PER_NNZ)


def load_summary(variant):
    summary = load_json(STAGE0_DIR / variant / "summary.json")
    return {e["name"]: e for e in summary["instances"]}, summary


def load_variant(variant, switch):
    """Instances of a run made with HIGHS_DIAG_OFF=<switch>, or {}."""
    if not (STAGE0_DIR / variant / "summary.json").exists():
        return {}
    instances, summary = load_summary(variant)
    if summary.get("env", {}).get("HIGHS_DIAG_OFF") != switch:
        return {}
    return instances


def presolved_stats(variant, name):
    """Structure statistics of the presolved model, or None."""
    path = STAGE0_DIR / variant / f"{name}.json"
    if not path.exists():
        return None
    return load_json(path).get("presolved")


def load_instances():
    testbed = load_json(TESTBED_FILE)
    dev = [e for e in testbed["main"] if e["where"] == "dev"]
    diag, _ = load_summary("diag")
    stock, _ = load_summary("stock")
    off = load_variant("diag_noprobing", "probing")
    merge_off = load_variant("diag_nocliquemerge", "clique_merging")

    instances = []
    for entry in dev:
        name = entry["name"]
        d = diag[name]
        stats = load_json(STAGE0_DIR / "stats" / f"{name}.json")
        rounds = d["diag"]["probing_round"]

        def total(key):
            return sum(r[key] for r in rounds)

        inst = {
            "name": name,
            "band": entry["band"],
            "rows": d["rows"],
            "cols": d["cols"],
            "nnz": d["nnz"],
            "rows_after": d["rows_after"],
            "cols_after": d["cols_after"],
            "nnz_after": d["nnz_after"],
            "bin_model": stats["original"]["col_types"]["binary"],
            "rounds": rounds,
            "line": d["diag"]["probing"][0],
            "diag": d["diag"],
            "presolve_s": d["diag"]["total"][0]["presolve_seconds"],
            "diag_core_s": d["presolve_core_seconds"],
            "stock_core_s": stock[name]["presolve_core_seconds"],
            "stock": stock[name],
            "off": off.get(name),
            "merge_off": merge_off.get(name),
            "cand": rounds[0]["binaries"],
            "probed": total("probed"),
            "work": total("probing_work"),
            "prop_entries": total("prop_entries"),
            "act_entries": total("activity_entries"),
            "sweeps": total("prop_sweeps"),
            "prop_rows": total("prop_rows"),
            "seconds": total("seconds"),
            "prepare_s": total("prepare_seconds"),
            "loop_s": total("loop_seconds"),
            "finalise_s": total("finalise_seconds"),
            "fixed": total("fixed"),
            "subst": total("substitutions"),
            "bounds": total("bounds_tightened"),
            "lifted": total("lifted_nonzeros"),
            "new_cliques": sum(
                r["cliques_after"] - r["cliques_before"] for r in rounds
            ),
            "new_impl": sum(
                r["implications"] - r["implications_before"] for r in rounds
            ),
            "max_work": max(r["max_probe_work"] for r in rounds),
            "max_sweeps": max(r["max_probe_sweeps"] for r in rounds),
        }
        inst["found"] = inst["fixed"] + inst["subst"]
        instances.append(inst)
    return instances


def same_without_probing(inst):
    """Rows, columns and nonzeros of the presolved model are equal."""
    off = inst["off"]
    if off is None:
        return None
    return (off["rows_after"], off["cols_after"], off["nnz_after"]) == (
        inst["rows_after"],
        inst["cols_after"],
        inst["nnz_after"],
    )


def stats_differ_without_probing(inst):
    """Keys of the presolved structure statistics that differ, or None.

    The statistics (column types, row classes, bounds, coefficient
    range, length percentiles) are not the model itself: equal
    statistics do not prove equal models, different ones prove a
    difference.
    """
    on = presolved_stats("diag", inst["name"])
    off = presolved_stats("diag_noprobing", inst["name"])
    if on is None or off is None:
        return None
    return sorted(k for k in set(on) | set(off) if on.get(k) != off.get(k))


def consistency(instances):
    print("## Input checks\n")
    print(f"- instances: {len(instances)} (testbed main list, where = dev)")
    switches = defaultdict(int)
    blocks = defaultdict(int)
    mismatch = []
    attempted_differs = []
    stopped = []
    sizes_differ = []
    stage_gap = 0.0
    for inst in instances:
        sw = inst["diag"]["switches"][0]
        key = (sw["off"], sw["probing_contingent"], sw["probing_nostop"])
        switches[key] += 1
        blocks[len(inst["diag"]["total"])] += 1
        line = inst["line"]
        if (
            line["probed"] != inst["probed"]
            or line["probing_work"] != inst["work"]
            or line["rounds"] != len(inst["rounds"])
        ):
            mismatch.append(inst["name"])
        for r in inst["rounds"]:
            if r["attempted"] != r["probed"]:
                attempted_differs.append(f"{inst['name']}#{r['round']}")
        if inst["diag"]["total"][0]["stopped"]:
            stopped.append(inst["name"])
        st = inst["stock"]
        if (st["rows_after"], st["cols_after"], st["nnz_after"]) != (
            inst["rows_after"],
            inst["cols_after"],
            inst["nnz_after"],
        ):
            sizes_differ.append(inst["name"])
        stage_s = inst["diag"]["stage"]["probing"]["seconds"]
        stage_gap = max(stage_gap, abs(stage_s - inst["seconds"]))
    for key, count in switches.items():
        print(
            f"- DIAG switches off={key[0]} probing_contingent={key[1]}"
            f" probing_nostop={key[2]}: {count} instances"
        )
    print(f"- presolve blocks per instance (DIAG total lines): {dict(blocks)}")
    print(
        "- round lines add up to the DIAG probing line (probed, work,"
        f" rounds): mismatches {mismatch or 'none'}"
    )
    print(
        "- rounds where attempted differs from probed:"
        f" {attempted_differs or 'none'}"
    )
    print(
        "- largest difference between the probing stage seconds and the"
        f" sum of the round seconds: {stage_gap:.6f} s"
    )
    print(f"- presolve stopped on a limit: {stopped or 'none'}")
    print(
        "- presolved sizes differ between the stock and the patched run:"
        f" {sizes_differ or 'none'}"
    )


def stops(inst):
    return "/".join(r["stop"] for r in inst["rounds"])


def table_overview(instances):
    rows = []
    for inst in instances:
        r1 = inst["rounds"][0]
        line = inst["line"]
        rows.append(
            [
                inst["name"],
                inst["band"],
                num(inst["nnz"]),
                num(inst["bin_model"]),
                num(inst["cand"]),
                len(inst["rounds"]),
                num(inst["probed"]),
                percent(r1["probed"], r1["binaries"], 2),
                stops(inst),
                num(inst["fixed"]),
                num(inst["subst"]),
                num(inst["bounds"]),
                num(line["implications"]),
                f"{num(line['cliques_before'])} -> "
                f"{num(line['cliques_after'])}",
                num(inst["lifted"]),
            ]
        )
    table(
        "1a. Probing per instance: counts",
        [
            "instance",
            "band",
            "nnz",
            "binaries (model)",
            "candidates (round 1)",
            "rounds",
            "probes",
            "round-1 coverage %",
            "stop per round",
            "fixed",
            "substitutions",
            "bounds tightened",
            "implications (end)",
            "cliques (first loop start -> last loop end)",
            "lifted nnz",
        ],
        rows,
        [
            "`binaries (model)` is the binary count of the original model"
            " (`stats/<name>.json`); `candidates` is the list HiGHS"
            " builds at the start of round 1, on the model as presolve"
            " has left it (it can hold columns that became binary)."
            " `probes` is the sum over the rounds; a column can be"
            " probed again in a later round. `lifted nnz` are"
            " clique-extension nonzeros from clique merging.",
        ],
    )

    rows = []
    for inst in instances:
        rows.append(
            [
                inst["name"],
                dec(inst["seconds"]),
                dec(inst["prepare_s"]),
                dec(inst["loop_s"]),
                dec(inst["finalise_s"]),
                dec(inst["presolve_s"]),
                percent(inst["seconds"], inst["presolve_s"]),
                dec(inst["stock_core_s"]),
            ]
        )
    table(
        "1b. Probing per instance: seconds",
        [
            "instance",
            "probing s",
            "prepare s",
            "loop s",
            "finalise s",
            "presolve s (patched)",
            "probing % of presolve",
            "stock core s",
        ],
        rows,
        [
            "Seconds are from the patched run (`diag/summary.json`), sums"
            " over the rounds; `stock core s` is `presolve_core_seconds`"
            " of the stock run.",
        ],
    )
    total_probing = sum(i["seconds"] for i in instances)
    total_presolve = sum(i["presolve_s"] for i in instances)
    print(
        f"\nSum over {len(instances)} instances: probing"
        f" {total_probing:.2f} s of {total_presolve:.2f} s presolve"
        f" ({100 * total_probing / total_presolve:.1f}%); prepare"
        f" {sum(i['prepare_s'] for i in instances):.2f} s, loop"
        f" {sum(i['loop_s'] for i in instances):.2f} s, finalise"
        f" {sum(i['finalise_s'] for i in instances):.2f} s. Stock core"
        f" seconds, sum: {sum(i['stock_core_s'] for i in instances):.2f}."
    )
    rest = [i for i in instances if i["name"] != "rd-rplusc-21"]
    rp = sum(i["seconds"] for i in rest)
    rs = sum(i["presolve_s"] for i in rest)
    print(
        f"Without rd-rplusc-21: probing {rp:.2f} s of {rs:.2f} s"
        f" ({100 * rp / rs:.1f}%)."
    )
    shares = sorted(
        (100 * i["seconds"] / i["presolve_s"], i["name"]) for i in instances
    )
    print(
        "Probing share of presolve per instance: median"
        f" {statistics.median(s for s, _ in shares):.1f}%;"
        f" {sum(1 for s, _ in shares if s >= 50)} instances at 50% or"
        " more: "
        + ", ".join(f"{n} {s:.0f}%" for s, n in shares if s >= 50)
        + "."
    )
    with_cand = [i for i in instances if i["cand"]]
    cov = sorted(100 * i["rounds"][0]["probed"] / i["cand"] for i in with_cand)
    print(
        f"Round-1 coverage over the {len(with_cand)} instances with"
        f" candidates: median {statistics.median(cov):.1f}%; below 10%:"
        f" {sum(1 for c in cov if c < 10)}; 10% to 50%:"
        f" {sum(1 for c in cov if 10 <= c < 50)}; 50% to 95%:"
        f" {sum(1 for c in cov if 50 <= c < 95)}; 95% or more:"
        f" {sum(1 for c in cov if c >= 95)}."
    )
    cand_total = sum(i["cand"] for i in with_cand)
    r1_total = sum(i["rounds"][0]["probed"] for i in with_cand)
    print(
        f"All round-1 candidates: {cand_total:,}; probed in round 1:"
        f" {r1_total:,} ({100 * r1_total / cand_total:.2f}%); probes in"
        f" all rounds: {sum(i['probed'] for i in instances):,}."
    )


def table_rounds(instances):
    rows = []
    for inst in instances:
        for r in inst["rounds"]:
            rows.append(
                [
                    inst["name"],
                    r["round"],
                    num(r["binaries"]),
                    num(r["probed"]),
                    percent(r["probed"], r["binaries"], 2),
                    r["stop"],
                    num(r["fixed"]),
                    num(r["substitutions"]),
                    num(r["bounds_tightened"]),
                    f"{num(r['implications_before'])} -> "
                    f"{num(r['implications'])}",
                    f"{num(r['cliques_before'])} -> {num(r['cliques_after'])}",
                    num(r["lifted_nonzeros"]),
                    dec(r["seconds"]),
                    dec(r["prepare_seconds"]),
                    dec(r["loop_seconds"]),
                    dec(r["finalise_seconds"]),
                ]
            )
    table(
        "2. Probing rounds",
        [
            "instance",
            "round",
            "candidates",
            "probed",
            "coverage %",
            "stop",
            "fixed",
            "substitutions",
            "bounds tightened",
            "implications (loop start -> end)",
            "cliques (loop start -> end)",
            "lifted nnz",
            "s",
            "prepare s",
            "loop s",
            "finalise s",
        ],
        rows,
        [
            "`fixed`, `substitutions`, `bounds tightened` and `lifted nnz`"
            " are what `finaliseProbing` applies at the end of the round:"
            " they include fixings found by the propagation of"
            " `prepareProbing` and by clique merging, not only by the"
            " probes. `implications` are stored implied bounds on"
            " non-binary columns; an implied binary becomes a two-entry"
            " clique.",
        ],
    )


def table_work(instances):
    probing = [i for i in instances if i["probed"]]
    rows = []
    for inst in probing:
        p = inst["probed"]
        rows.append(
            [
                inst["name"],
                num(p),
                num(inst["work"]),
                num(inst["prop_entries"]),
                num(inst["act_entries"]),
                percent(inst["act_entries"], inst["work"]),
                num(inst["sweeps"]),
                num(inst["prop_rows"]),
                ratio(inst["work"], p, 0),
                num(inst["max_work"]),
                ratio(inst["max_work"], inst["nnz"], 4),
                ratio(inst["sweeps"], p, 2),
                inst["max_sweeps"],
                ratio(inst["loop_s"], p, 1, 1e6),
                ratio(inst["loop_s"], inst["work"], 4, 1e6),
            ]
        )
    table(
        "3a. Work of the probing loop per instance",
        [
            "instance",
            "probes",
            "probing_work",
            "prop_entries",
            "activity_entries",
            "activity % of work",
            "prop_sweeps",
            "prop_rows",
            "work / probe",
            "max_probe_work",
            "max_probe_work / nnz",
            "sweeps / probe",
            "max_probe_sweeps",
            "loop us / probe",
            "loop s / 1e6 work",
        ],
        rows,
        [
            "`nnz` is the nonzero count of the original model; the matrix"
            " at probing time is normally between the original and the"
            " presolved size. `max_probe_work` and `max_probe_sweeps`"
            " are the largest values of one probe, both values of the"
            " column together. The two instances without candidates are"
            " left out.",
        ],
    )

    rows = []
    rates = []
    for inst in probing:
        for r in inst["rounds"]:
            p = r["probed"]
            if p == 0 or r["loop_seconds"] <= 0:
                continue
            new_cliques = r["cliques_after"] - r["cliques_before"]
            rate = r["probing_work"] / r["loop_seconds"] / 1e6
            rates.append((rate, f"{inst['name']}#{r['round']}"))
            rows.append(
                [
                    inst["name"],
                    r["round"],
                    num(p),
                    num(r["probing_work"]),
                    ratio(r["probing_work"], p, 0),
                    num(r["max_probe_work"]),
                    ratio(r["prop_sweeps"], p, 2),
                    r["max_probe_sweeps"],
                    dec(r["loop_seconds"]),
                    ratio(r["loop_seconds"], p, 1, 1e6),
                    ratio(r["loop_seconds"], r["probing_work"], 4, 1e6),
                    dec(rate, 1),
                    ratio(new_cliques, p, 1),
                    ratio(r["implications"] - r["implications_before"], p, 1),
                ]
            )
    table(
        "3b. Work of the probing loop per round",
        [
            "instance",
            "round",
            "probed",
            "probing_work",
            "work / probe",
            "max_probe_work",
            "sweeps / probe",
            "max_probe_sweeps",
            "loop s",
            "loop us / probe",
            "loop s / 1e6 work",
            "1e6 work / loop s",
            "new cliques / probe",
            "new implications / probe",
        ],
        rows,
    )
    rates.sort()
    quart = statistics.quantiles([v for v, _ in rates], n=4)
    quart_incl = statistics.quantiles(
        [v for v, _ in rates], n=4, method="inclusive"
    )
    print(
        f"\nRate over {len(rates)} rounds, in 1e6 work units per loop"
        f" second: min {rates[0][0]:.1f} ({rates[0][1]}), quartiles"
        f" {quart[0]:.1f} / {quart[1]:.1f} / {quart[2]:.1f} (exclusive"
        f" method; inclusive {quart_incl[0]:.1f} / {quart_incl[1]:.1f} /"
        f" {quart_incl[2]:.1f}), max {rates[-1][0]:.1f} ({rates[-1][1]})."
    )
    print(
        "Lowest rates: "
        + ", ".join(f"{name} {rate:.1f}" for rate, name in rates[:8])
        + "."
    )
    total_work = sum(i["work"] for i in probing)
    total_loop = sum(i["loop_s"] for i in probing)
    print(
        f"All instances: {total_work:,} work units in {total_loop:.2f}"
        f" loop seconds = {total_work / total_loop / 1e6:.1f}e6 per"
        " second; activity updates are"
        f" {100 * sum(i['act_entries'] for i in probing) / total_work:.1f}%"
        " of the work."
    )
    share = sorted(
        (100 * i["act_entries"] / i["work"], i["name"]) for i in probing
    )
    print(
        "Activity updates as a share of the work, per instance: median"
        f" {statistics.median(v for v, _ in share):.1f}%;"
        f" {sum(1 for v, _ in share if v >= 80)} of {len(share)}"
        " instances at 80% or more; below 30%: "
        + ", ".join(f"{n} {v:.1f}%" for v, n in share if v < 30)
        + "."
    )
    wpp = sorted((i["work"] / i["probed"], i["name"]) for i in probing)
    print(
        "Work per probe (instance means): min"
        f" {wpp[0][0]:.0f} ({wpp[0][1]}), median"
        f" {statistics.median(v for v, _ in wpp):.0f}, max"
        f" {wpp[-1][0]:.0f} ({wpp[-1][1]}); above 100,000: "
        + ", ".join(f"{n} {v:,.0f}" for v, n in wpp if v > 1e5)
        + "."
    )
    upp = sorted((1e6 * i["loop_s"] / i["probed"], i["name"]) for i in probing)
    print(
        "Loop microseconds per probe (instance means): min"
        f" {upp[0][0]:.1f} ({upp[0][1]}), median"
        f" {statistics.median(v for v, _ in upp):.1f}, max"
        f" {upp[-1][0]:.1f} ({upp[-1][1]}); above 1000: "
        + ", ".join(f"{n} {v:,.0f}" for v, n in upp if v > 1000)
        + "."
    )
    spp = sorted((i["sweeps"] / i["probed"], i["name"]) for i in probing)
    print(
        "Sweeps per probe (instance means): median"
        f" {statistics.median(v for v, _ in spp):.2f};"
        f" {sum(1 for v, _ in spp if v <= 3.5)} of {len(spp)} instances"
        " at 3.5 or below; above 5: "
        + ", ".join(f"{n} {v:.1f}" for v, n in spp if v > 5)
        + "."
    )
    no_rows = [
        f"{i['name']} ({num(i['prop_rows'])} rows taken from the queue in"
        f" {num(i['sweeps'])} sweeps, {num(i['rounds'][0]['cliques_before'])}"
        " cliques at the start of the loop,"
        f" {num(i['new_cliques'])} new cliques)"
        for i in probing
        if i["prop_entries"] == 0
    ]
    print(
        "Instances whose probes scan no row entry (prop_entries = 0,"
        f" all work is activity updates): {'; '.join(no_rows) or 'none'}."
        " A row taken from the queue adds no entries when both of its"
        " sides are redundant at that moment. The counters do not say"
        " what changed the bounds (clique table, stored implications or"
        " dual-fix probing)."
    )
    spread = []
    for inst in probing:
        per_round = [
            r["probing_work"] / r["probed"]
            for r in inst["rounds"]
            if r["probed"]
        ]
        if len(per_round) > 1:
            spread.append(
                (max(per_round) / min(per_round), inst["name"], per_round)
            )
    spread.sort(reverse=True)
    print(
        "Work per probe by round on the multi-round instances (largest"
        " / smallest round mean): "
        + "; ".join(
            f"{name} {factor:.1f} ("
            + " / ".join(f"{v:,.0f}" for v in values)
            + ")"
            for factor, name, values in spread
        )
        + "."
    )
    over = [
        f"{i['name']} {i['max_work'] / i['nnz']:.2f}"
        for i in probing
        if i["max_work"] > i["nnz"]
    ]
    print(
        "Instances where one probe costs more than the original nonzero"
        f" count (max_probe_work / nnz): {', '.join(over) or 'none'}."
    )


def best_round_coverage(inst):
    values = [
        r["probed"] / r["binaries"] for r in inst["rounds"] if r["binaries"]
    ]
    return max(values) if values else None


def table_coverage(instances):
    rows = []
    ordered = sorted(
        instances,
        key=lambda i: (
            2.0 if best_round_coverage(i) is None else best_round_coverage(i),
            i["name"],
        ),
    )
    for inst in ordered:
        r1 = inst["rounds"][0]
        best = best_round_coverage(inst)
        per_round = " / ".join(
            percent(r["probed"], r["binaries"], 2) for r in inst["rounds"]
        )
        if best is None:
            bounds = "-"
        else:
            low = 100 * r1["probed"] / r1["binaries"]
            high = min(100.0, 100 * inst["probed"] / r1["binaries"])
            bounds = f"{low:.2f} to {high:.2f}"
        rows.append(
            [
                inst["name"],
                num(r1["binaries"]),
                num(r1["probed"]),
                num(r1["binaries"] - r1["probed"]),
                per_round,
                "-" if best is None else f"{100 * best:.2f}",
                bounds,
                dec(inst["seconds"]),
            ]
        )
    table(
        "4a. Coverage, sorted by the best coverage of a single round",
        [
            "instance",
            "candidates (round 1)",
            "probed (round 1)",
            "not probed (round 1)",
            "coverage % per round",
            "best round %",
            "distinct round-1 candidates probed % (bounds)",
            "probing s",
        ],
        rows,
        [
            "The DIAG lines do not say which columns were probed. Inside"
            " one round every column is probed at most once. Across"
            " rounds the share of distinct round-1 candidates is only"
            " bounded: at least round 1, at most the sum of the rounds.",
        ],
    )
    low = [
        i
        for i in ordered
        if best_round_coverage(i) is not None and best_round_coverage(i) < 0.10
    ]
    left = sum(i["cand"] - i["rounds"][0]["probed"] for i in low)
    print(
        f"\nBest single-round coverage below 10%: {len(low)} instances ("
        + ", ".join(i["name"] for i in low)
        + f"); probing on them takes {sum(i['seconds'] for i in low):.2f}"
        f" s ({sum(i['loop_s'] for i in low):.2f} s in the loops) and"
        f" leaves {left:,} round-1 candidates unprobed."
    )

    by_reason = defaultdict(list)
    last_reason = defaultdict(list)
    for inst in instances:
        for r in inst["rounds"]:
            by_reason[r["stop"]].append(f"{inst['name']}#{r['round']}")
        last_reason[inst["rounds"][-1]["stop"]].append(inst["name"])
    rows = []
    for reason in sorted(by_reason, key=lambda k: -len(by_reason[k])):
        rows.append(
            [
                reason,
                len(by_reason[reason]),
                len(last_reason.get(reason, [])),
                ", ".join(by_reason[reason]),
            ]
        )
    table(
        "4b. Stop reasons",
        [
            "stop",
            "rounds",
            "instances whose last round ended so",
            "instance#round",
        ],
        rows,
        [
            "Rules, read from HiGHS at commit 283bfc859"
            " (`HPresolve::runProbing` in HPresolveProbing.cpp, the main"
            " loop of `HPresolve::presolve` in HPresolve.cpp,"
            " `HighsCliqueTable::setMaxEntries`):",
            "- `contingent`: probes done in this presolve call exceed"
            " 1000 + credits (reductions found by each probe, and the"
            " running deletion count whenever a probe deletes a"
            " column).",
            "- `neighbourhood`: clique-table pair queries of the round"
            " exceed max(100000, 10 nnz) + 1000 per new clique + 100 x"
            " deletions - 100 x (failures in a row) per fruitless probe"
            " (250 x from the 2500th probe of the round).",
            "- `deletions`: columns fixed or substituted in the round"
            " exceed max(1000, (rows + columns) / 20); min(...) from the"
            " 2500th probe of the round. The round ends, the reductions"
            " are applied and the main loop starts again.",
            "- `clique_limit` / `implication_limit`: more than"
            " max(1,000,000, 2 nnz) new cliques / implications in the"
            " round. `table_full`: clique table entries reach 2,000,000"
            " + 10 nnz.",
            "- `finished`: every candidate visited. `time`: time limit.",
            "- Another round follows only if the contingent is not used"
            " up and the round removed more than 1% of the rows or"
            " columns or stopped on `deletions`. `prepareProbing` clears"
            " the probed flags (`shrinkProblem` rebuilds the"
            " implications), so a later round probes columns again.",
        ],
    )

    rows = []
    for inst in instances:
        cumulative = 0
        for r in inst["rounds"]:
            cumulative += r["probed"]
            if r["stop"] != "contingent":
                continue
            rows.append(
                [
                    inst["name"],
                    r["round"],
                    num(cumulative),
                    cumulative - (PROBING_CONTINGENT + 1),
                    num(inst["fixed"]),
                    num(inst["subst"]),
                    num(inst["bounds"]),
                    num(r["probing_work"]),
                    ratio(r["probing_work"], inst["nnz"], 2),
                    dec(r["loop_seconds"]),
                ]
            )
    table(
        "4c. Rounds stopped by the probing contingent",
        [
            "instance",
            "round",
            "probes so far",
            "probes beyond 1001",
            "fixed",
            "substitutions",
            "bounds tightened",
            "probing_work",
            "work / nnz",
            "loop s",
        ],
        rows,
        [
            "The contingent starts at 1000 probes for the whole presolve"
            " call and a round stops when the probes done exceed it, so"
            " a presolve that finds nothing stops after 1001 probes."
            " Every reduction found raises it.",
        ],
    )

    rows = []
    quiet_rounds = 0
    quiet_exact = 0
    quiet_other = []
    for inst in instances:
        for r in inst["rounds"]:
            if r["stop"] != "neighbourhood":
                continue
            new_cliques = r["cliques_after"] - r["cliques_before"]
            cap_after = streak_cap(inst["nnz_after"])
            cap_before = streak_cap(inst["nnz"])
            quiet = (
                new_cliques == 0
                and r["fixed"] == 0
                and r["substitutions"] == 0
            )
            implied = "-"
            if quiet:
                quiet_rounds += 1
                nnz_range = streak_nnz_range(r["probed"])
                if nnz_range is not None:
                    implied = f"{num(nnz_range[0])} to {num(nnz_range[1] - 1)}"
                if r["probed"] == cap_after:
                    quiet_exact += 1
                elif nnz_range is not None:
                    enum_nnz = (
                        inst["nnz"]
                        - (inst["diag"]["stage"]["enumeration"]["nnz_removed"])
                    )
                    inside = nnz_range[0] <= enum_nnz < nnz_range[1]
                    quiet_other.append(
                        f"{inst['name']} round {r['round']}:"
                        f" {num(r['probed'])} probes need {implied}"
                        f" nonzeros; presolved {num(inst['nnz_after'])},"
                        f" original {num(inst['nnz'])}, original minus the"
                        " nonzeros removed by the enumeration stage"
                        f" {num(enum_nnz)}"
                        f" ({'inside' if inside else 'outside'} the range)"
                    )
            rows.append(
                [
                    inst["name"],
                    r["round"],
                    num(r["probed"]),
                    cap_after,
                    cap_before,
                    r["probed"] - cap_after,
                    num(new_cliques),
                    num(r["implications"] - r["implications_before"]),
                    num(r["fixed"] + r["substitutions"]),
                    "yes" if quiet else "no",
                    implied,
                ]
            )
    table(
        "4d. Rounds stopped by the neighbourhood contingent",
        [
            "instance",
            "round",
            "probed",
            "failure cap at presolved nnz",
            "failure cap at original nnz",
            "probed minus cap at presolved nnz",
            "new cliques",
            "new implications",
            "fixed + substitutions",
            "nothing found in the round",
            "nnz for which the cap equals the probes",
        ],
        rows,
        [
            "Failure cap = number of fruitless probes in a row after"
            " which the contingent max(100000, 10 nnz) is used up by the"
            " penalty 100 * (failures in a row) alone, with no clique"
            " query made: the smallest k with 50 k (k - 1) above the"
            " contingent. The nonzero count at probing time is normally"
            " between the two sizes. A new clique or a deleted column"
            " resets the run of failures and adds to the contingent, so"
            " rounds that found something go beyond the cap. `nothing"
            " found` = no new clique, no fixing, no substitution; stored"
            " implications do not count as a success for HiGHS either.",
        ],
    )
    print(
        f"\nRounds that found nothing: {quiet_rounds}; probed equals the"
        f" cap at the presolved nonzero count in {quiet_exact} of them."
    )
    for line in quiet_other:
        print(f"- {line}.")

    rows = []
    for inst in instances:
        for r in inst["rounds"]:
            if r["stop"] != "deletions":
                continue
            tight = r["probed"] >= TIGHTEN_AFTER_PROBES
            pick = min if tight else max
            before = pick(
                DELETIONS_MIN,
                (inst["rows"] + inst["cols"]) // DELETIONS_DIVISOR,
            )
            after = pick(
                DELETIONS_MIN,
                (inst["rows_after"] + inst["cols_after"]) // DELETIONS_DIVISOR,
            )
            rows.append(
                [
                    inst["name"],
                    r["round"],
                    num(r["probed"]),
                    num(r["fixed"]),
                    num(r["substitutions"]),
                    "min" if tight else "max",
                    num(after),
                    num(before),
                ]
            )
    table(
        "4e. Rounds stopped by deletions",
        [
            "instance",
            "round",
            "probed",
            "fixed",
            "substitutions",
            "rule in force at the stop",
            "threshold at presolved size",
            "threshold at original size",
        ],
        rows,
        [
            "Threshold = max(1000, (rows + columns) / 20) columns fixed"
            " or substituted in the round; from 2500 probes of the round"
            " on it is min(1000, (rows + columns) / 20). The model size"
            " at probing time is normally between the two sizes.",
        ],
    )

    rows = []
    for inst in instances:
        for r in inst["rounds"]:
            if r["stop"] not in ("clique_limit", "implication_limit"):
                continue
            rows.append(
                [
                    inst["name"],
                    r["round"],
                    r["stop"],
                    num(r["probed"]),
                    num(r["cliques_after"] - r["cliques_before"]),
                    num(r["implications"] - r["implications_before"]),
                    num(
                        max(
                            NEW_ITEMS_MIN,
                            NEW_ITEMS_PER_NNZ * inst["nnz_after"],
                        )
                    ),
                    num(max(NEW_ITEMS_MIN, NEW_ITEMS_PER_NNZ * inst["nnz"])),
                ]
            )
    table(
        "4f. Rounds stopped by the clique or implication limit",
        [
            "instance",
            "round",
            "stop",
            "probed",
            "new cliques",
            "new implications",
            "limit at presolved nnz",
            "limit at original nnz",
        ],
        rows,
        ["Limit = max(1,000,000, 2 nnz) new cliques or implications."],
    )


def merge_calls(inst, stage):
    calls = [
        c for c in inst["diag"]["cliquemerge_call"] if c["stage"] == stage
    ]
    keys = [
        "seconds",
        "rows_changed",
        "rows_deleted",
        "rows_deleted_before_merge",
        "nnz_added",
        "nnz_removed",
        "fixings",
    ]
    out = {k: sum(c[k] for c in calls) for k in keys}
    out["calls"] = len(calls)
    return out


def yield_class(inst):
    if inst["cand"] == 0:
        return "no candidates"
    cols = inst["diag"]["stage"]["probing"]["cols_removed"]
    if inst["found"] == 0:
        label = "nothing"
    elif inst["found"] <= 20:
        label = "almost nothing"
    elif inst["found"] >= 500 or 100 * cols >= inst["cols"]:
        label = "a lot"
    else:
        label = "some"
    if inst["seconds"] > 1.0:
        label += ", > 1 s"
    return label


def table_yield(instances):
    rows = []
    for inst in sorted(instances, key=lambda i: -i["seconds"]):
        stage = inst["diag"]["stage"]["probing"]
        same = same_without_probing(inst)
        rows.append(
            [
                inst["name"],
                dec(inst["seconds"]),
                num(inst["fixed"]),
                num(merge_calls(inst, "probing")["fixings"]),
                num(inst["subst"]),
                num(inst["bounds"]),
                num(inst["new_impl"]),
                num(inst["new_cliques"]),
                num(stage["rows_removed"]),
                num(stage["cols_removed"]),
                num(stage["nnz_removed"]),
                ratio(inst["found"], inst["seconds"], 1),
                ratio(inst["found"], inst["work"], 2, 1e6),
                yield_class(inst),
                "-" if same is None else ("yes" if same else "no"),
            ]
        )
    table(
        "5. Yield, sorted by probing seconds",
        [
            "instance",
            "probing s",
            "fixed",
            "of which by clique merging",
            "substitutions",
            "bounds tightened",
            "new implications",
            "new cliques",
            "stage rows removed",
            "stage cols removed",
            "stage nnz removed",
            "(fixed + subst.) / s",
            "(fixed + subst.) / 1e6 work",
            "class",
            "presolved sizes the same without probing",
        ],
        rows,
        [
            "Class, by the count of fixings and substitutions: `nothing`"
            " = 0; `almost nothing` = 1 to 20; `a lot` = 500 or more, or"
            " the probing stage removes at least 1% of the original"
            " columns. New implications and new cliques are counted"
            " inside the probing loops (end minus start, summed over the"
            " rounds). A negative `stage nnz removed` means the stage"
            " added nonzeros. The last column compares rows, columns and"
            " nonzeros of the presolved model with the `diag_noprobing`"
            " run (table 8); equal sizes are not an identical model.",
        ],
    )
    slow = [i for i in instances if i["seconds"] > 1.0]
    nothing = [i for i in slow if i["found"] == 0]
    little = [i for i in slow if 0 < i["found"] <= 20]
    print(
        f"\nProbing above 1 s: {len(slow)} instances,"
        f" {sum(i['seconds'] for i in slow):.2f} s in total."
    )
    print(
        "Of these, no fixing and no substitution: "
        + ", ".join(f"{i['name']} ({i['seconds']:.2f} s)" for i in nothing)
        + f"; sum {sum(i['seconds'] for i in nothing):.2f} s."
    )
    print(
        "Of these, 1 to 20 fixings and substitutions: "
        + (
            ", ".join(
                f"{i['name']} ({i['seconds']:.2f} s, {i['found']})"
                for i in little
            )
            or "none"
        )
        + "."
    )
    zero = [i for i in instances if i["cand"] and i["found"] == 0]
    print(
        f"No fixing and no substitution at all: {len(zero)} of"
        f" {sum(1 for i in instances if i['cand'])} instances with"
        f" candidates, {sum(i['seconds'] for i in zero):.2f} s of probing."
    )
    found_all = sum(i["found"] for i in instances)
    top = sorted(instances, key=lambda i: -i["found"])[:5]
    print(
        f"Fixings and substitutions over all instances: {found_all:,}"
        f" ({sum(i['fixed'] for i in instances):,} fixed,"
        f" {sum(i['subst'] for i in instances):,} substituted); largest: "
        + ", ".join(f"{i['name']} {i['found']:,}" for i in top)
        + f" = {100 * sum(i['found'] for i in top) / found_all:.1f}%."
    )


def reductions_of(inst, stage):
    for entry in inst["diag"]["reductions"]:
        if entry["stage"] == stage:
            return entry
    return {}


def table_enumeration(instances):
    rows = []
    for inst in instances:
        stage = inst["diag"]["stage"]["enumeration"]
        red = reductions_of(inst, "enumeration")
        subs = inst["diag"]["substitutions"][0]
        merge = merge_calls(inst, "enumeration")
        vb = inst["diag"]["varbound_aggregation"]["enumeration"]
        rows.append(
            [
                inst["name"],
                stage["calls"],
                dec(stage["seconds"]),
                percent(stage["seconds"], inst["presolve_s"]),
                dec(merge["seconds"]),
                num(stage["rows_removed"]),
                num(stage["cols_removed"]),
                num(stage["nnz_removed"]),
                num(red.get("fixed_col", 0)),
                num(subs["in_enumeration"]),
                num(merge["rows_deleted"]),
                num(merge["rows_deleted_before_merge"]),
                num(vb["rows_removed"]),
            ]
        )
    table(
        "6a. Enumeration stage",
        [
            "instance",
            "calls",
            "s",
            "% of presolve",
            "of which clique merging s",
            "rows removed",
            "cols removed",
            "nnz removed",
            "columns fixed",
            "substitutions applied",
            "rows deleted by clique merging",
            "rows deleted by the clique table before merging",
            "rows removed by variable-bound aggregation",
        ],
        rows,
        [
            "The stage holds `prepareProbing` (in its first call the"
            " first clique extraction from the model), the enumeration"
            " of at most 400 all-binary rows with at most 8 unfixed"
            " columns, and `finaliseProbing` (clique merging, fixings,"
            " substitutions, variable-bound aggregation). `columns"
            " fixed` is the `fixed_col` count of `DIAG reductions"
            " stage=enumeration`. The three row columns on the right add"
            " up to `rows removed`.",
        ],
    )
    total = sum(
        i["diag"]["stage"]["enumeration"]["seconds"] for i in instances
    )
    presolve = sum(i["presolve_s"] for i in instances)
    top = sorted(
        instances,
        key=lambda i: -i["diag"]["stage"]["enumeration"]["seconds"],
    )[:5]
    print(
        f"\nEnumeration stage: {total:.2f} s of {presolve:.2f} s presolve"
        f" ({100 * total / presolve:.1f}%); largest: "
        + ", ".join(
            f"{i['name']} {i['diag']['stage']['enumeration']['seconds']:.2f} s"
            for i in top
        )
        + "."
    )
    changed = []
    unchanged_s = 0.0
    for inst in instances:
        stage = inst["diag"]["stage"]["enumeration"]
        if any(
            stage[k] for k in ("rows_removed", "cols_removed", "nnz_removed")
        ):
            changed.append(inst["name"])
        else:
            unchanged_s += stage["seconds"]
    print(
        "Instances where the enumeration stage changes the model size:"
        f" {len(changed)} ({', '.join(changed)}); stage seconds on the"
        f" other {len(instances) - len(changed)}: {unchanged_s:.2f}."
    )
    fixed = sum(
        reductions_of(i, "enumeration").get("fixed_col", 0) for i in instances
    )
    subst = sum(
        i["diag"]["substitutions"][0]["in_enumeration"] for i in instances
    )
    rows_total = sum(
        i["diag"]["stage"]["enumeration"]["rows_removed"] for i in instances
    )
    print(
        f"Enumeration stage, all instances: {fixed:,} columns fixed,"
        f" {subst:,} substitutions, {rows_total:,} rows removed."
    )

    rows = []
    for inst in instances:
        e = merge_calls(inst, "enumeration")
        p = merge_calls(inst, "probing")
        rows.append(
            [
                inst["name"],
                f"{e['calls']} / {p['calls']}",
                f"{e['seconds']:.4f} / {p['seconds']:.4f}",
                pair(e["rows_changed"], p["rows_changed"]),
                pair(e["rows_deleted"], p["rows_deleted"]),
                pair(
                    e["rows_deleted_before_merge"],
                    p["rows_deleted_before_merge"],
                ),
                pair(e["nnz_added"], p["nnz_added"]),
                pair(e["nnz_removed"], p["nnz_removed"]),
                pair(e["fixings"], p["fixings"]),
            ]
        )
    table(
        "6b. Clique merging (called from enumeration / from probing)",
        [
            "instance",
            "calls",
            "merge s",
            "rows changed",
            "rows deleted",
            "rows deleted before merge",
            "nnz added",
            "nnz removed",
            "fixings",
        ],
        rows,
        [
            "Every cell is `enumeration / probing`. `merge s` is the time"
            " inside `runCliqueMerging`; applying its result is in the"
            " stage time.",
        ],
    )
    merge_total = sum(
        i["diag"]["cliquemerge"][0]["seconds"] for i in instances
    )
    acting = [
        i["name"]
        for i in instances
        if i["diag"]["cliquemerge"][0]["rows_changed"]
        or i["diag"]["cliquemerge"][0]["rows_deleted"]
    ]
    print(
        f"\nClique merging: {merge_total:.2f} s in total; it changes or"
        f" deletes rows on {len(acting)} instances"
        f" ({', '.join(acting)})."
    )
    slowest = sorted(
        instances, key=lambda i: -i["diag"]["cliquemerge"][0]["seconds"]
    )[:5]
    print(
        "Largest merge times: "
        + ", ".join(
            f"{i['name']} {i['diag']['cliquemerge'][0]['seconds']:.2f} s"
            for i in slowest
        )
        + "."
    )
    for key in ("rows_changed", "rows_deleted", "nnz_added", "nnz_removed"):
        ranked = sorted(
            instances, key=lambda i: -i["diag"]["cliquemerge"][0][key]
        )
        value = sum(i["diag"]["cliquemerge"][0][key] for i in instances)
        print(
            f"Clique merging, sum of {key}: {value:,}; largest: "
            + ", ".join(
                f"{i['name']} {i['diag']['cliquemerge'][0][key]:,}"
                for i in ranked[:4]
            )
            + "."
        )
    by_stage = {
        s: sum(merge_calls(i, s)["rows_deleted"] for i in instances)
        for s in ("enumeration", "probing")
    }
    print(
        "Rows deleted by clique merging called from enumeration:"
        f" {by_stage['enumeration']:,}; from probing:"
        f" {by_stage['probing']:,}."
    )
    before_probing = 0
    idle = []
    for inst in instances:
        seen_probing = False
        idle_s = 0.0
        idle_calls = 0
        for call in inst["diag"]["cliquemerge_call"]:
            if call["stage"] == "probing":
                seen_probing = True
            elif not seen_probing:
                before_probing += call["rows_deleted"]
            if not any(
                call[k]
                for k in (
                    "rows_changed",
                    "rows_deleted",
                    "nnz_added",
                    "nnz_removed",
                    "fixings",
                )
            ):
                idle_s += call["seconds"]
                idle_calls += 1
        idle.append((idle_s, idle_calls, inst["name"]))
    idle.sort(reverse=True)
    print(
        "Rows deleted by clique merging in enumeration calls that come"
        f" before the first probing round: {before_probing:,}."
    )
    print(
        "Time in clique merging calls that change nothing (no row"
        " changed or deleted, no fixing):"
        f" {sum(s for s, _, _ in idle):.2f} s of {merge_total:.2f} s;"
        " largest: "
        + ", ".join(
            f"{name} {s:.2f} s in {calls} calls" for s, calls, name in idle[:5]
        )
        + "."
    )
    for inst in instances:
        if inst["name"] not in ("sorrell3", "savsched1"):
            continue
        print(
            f"Clique merging calls of {inst['name']}, in order: "
            + "; ".join(
                f"{c['stage']} {c['seconds']:.2f} s,"
                f" {c['rows_deleted']:,} rows deleted, cliques"
                f" {c['cliques_before']:,} -> {c['cliques_after']:,}"
                for c in inst["diag"]["cliquemerge_call"]
            )
            + "."
        )
    with_off = [i for i in instances if i["off"]]
    if with_off:
        deleted_off = sum(
            i["off"]["diag"]["cliquemerge"][0]["rows_deleted"]
            for i in with_off
        )
        seconds_off = sum(
            i["off"]["diag"]["cliquemerge"][0]["seconds"] for i in with_off
        )
        enum_off = sum(
            i["off"]["diag"]["stage"]["enumeration"]["seconds"]
            for i in with_off
        )
        print(
            "With probing switched off (`diag_noprobing`): clique merging"
            f" deletes {deleted_off:,} rows in {seconds_off:.2f} s and"
            f" the enumeration stage takes {enum_off:.2f} s."
        )

    rows = []
    for inst in instances:
        for stage in ("enumeration", "probing"):
            vb = inst["diag"]["varbound_aggregation"][stage]
            if not (
                vb["rows_removed"]
                or vb["rows_modified"]
                or vb["seconds"] > 0.01
            ):
                continue
            rows.append(
                [
                    inst["name"],
                    stage,
                    vb["calls"],
                    dec(vb["seconds"], 4),
                    num(vb["rows_removed"]),
                    num(vb["rows_modified"]),
                    num(vb["vars_lifted"]),
                    num(vb["nnz_removed"]),
                ]
            )
    table(
        "6c. Variable-bound aggregation (calls with an effect or above"
        " 0.01 s; nothing on the other instances)",
        [
            "instance",
            "called from",
            "calls",
            "s",
            "rows removed",
            "rows modified",
            "binaries lifted",
            "nnz removed",
        ],
        rows,
    )
    vb_total = sum(
        i["diag"]["varbound_aggregation"][s]["seconds"]
        for i in instances
        for s in ("enumeration", "probing")
    )
    print(f"\nVariable-bound aggregation: {vb_total:.2f} s in total.")


def fixed_cost_us():
    data = load_json(STAGE0_DIR / "fixed_costs.json")
    launch = None
    readback = None
    for entry in data["results"]:
        if entry["name"] == "empty_kernel_launch_sync":
            launch = entry["median_us"]
        if (
            entry["name"] == "kernel_write_8_bytes_copy_to_host"
            and entry.get("variant") == "pinned"
        ):
            readback = entry["median_us"]
    return launch, readback


def table_gpu(instances):
    launch_us, readback_us = fixed_cost_us()
    probing = [i for i in instances if i["probed"]]
    rows = []
    for inst in instances:
        p = inst["probed"]
        cand = inst["cand"]
        if p == 0:
            rows.append([inst["name"], num(cand)] + ["-"] * 10)
            continue
        gpu_per_probe = (inst["prop_entries"] + inst["act_entries"] / 2) / p
        inst["pass_highs"] = cand * inst["work"] / p
        inst["pass_gpu"] = cand * gpu_per_probe
        inst["pass_s"] = cand * inst["loop_s"] / p
        overlay = inst["act_entries"] / (4 * p)
        measured = "-"
        for r in inst["rounds"]:
            if r["probed"] >= FULL_ROUND_COVERAGE * r["binaries"]:
                measured = (
                    f"{r['probing_work'] / 1e6:.2f} (round {r['round']},"
                    f" {num(r['probed'])} probes)"
                )
                break
        rows.append(
            [
                inst["name"],
                num(cand),
                dec(gpu_per_probe, 0),
                dec(overlay, 0),
                f"{overlay / inst['rows']:.2f} to"
                f" {overlay / inst['rows_after']:.2f}",
                dec(inst["pass_highs"] / 1e6, 2),
                dec(inst["pass_gpu"] / 1e6, 2),
                dec(inst["pass_gpu"] / inst["nnz"], 1),
                measured,
                dec(inst["pass_s"], 2),
                dec(inst["pass_gpu"] / inst["work"], 1),
                f"{inst['max_sweeps']}"
                f" ({inst['max_sweeps'] * readback_us / 1000:.2f} ms)",
            ]
        )
    table(
        "7. Sizes for probing every candidate once",
        [
            "instance",
            "probes for one pass (candidates)",
            "entries / probe without undo",
            "activity entries / probe / value",
            "the same / rows (original to presolved)",
            "one pass, 1e6 entries (HiGHS count)",
            "one pass, 1e6 entries without undo",
            "one pass without undo / nnz",
            "full round measured, 1e6 work",
            "one pass at the HiGHS rate, s",
            "one pass without undo / HiGHS work done",
            "max_probe_sweeps (sweep fixed cost)",
        ],
        rows,
        [
            "One pass = every round-1 candidate probed once at both"
            " values, on the model as it is (no reduction applied in"
            " between). `one pass` = candidates x mean work per probe of"
            " the probes HiGHS made; it assumes that the unprobed"
            " candidates cost as much as the probed ones, which the data"
            " cannot check (HiGHS probes the columns with the most"
            " clique implications first). `without undo` = prop_entries"
            " + activity_entries / 2: a probe that works on its own"
            " overlay does not undo its bound changes. `activity entries"
            " / probe / value` = activity_entries / (4 probes): the mean"
            " number of row-activity updates one value of one probe"
            " makes; a value above the row count means every row is"
            " updated more than once. `full round measured` is the work"
            " of the first round that probed at least 95% of its"
            " candidates. `one pass at the HiGHS rate` = candidates x"
            " loop seconds per probe. `one pass without undo / HiGHS"
            " work done` is the factor by which a GPU must beat the"
            " HiGHS entry rate to finish one pass in the time of the"
            " HiGHS probing loops. Sweep fixed cost = max_probe_sweeps x"
            f" {readback_us:.2f} us (one kernel with an 8-byte read-back"
            f" per sweep on the H200; a launch alone is {launch_us:.2f}"
            " us).",
        ],
    )
    cand_total = sum(i["cand"] for i in probing)
    highs_total = sum(i["pass_highs"] for i in probing)
    gpu_total = sum(i["pass_gpu"] for i in probing)
    print(
        f"\nAll {len(probing)} instances with probes: {cand_total:,}"
        f" candidates = {2 * cand_total:,} probe values; one pass is"
        f" {highs_total / 1e9:.2f}e9 entries in the HiGHS count and"
        f" {gpu_total / 1e9:.2f}e9 without undo; at the HiGHS rate"
        f" {sum(i['pass_s'] for i in probing):.0f} s. HiGHS itself did"
        f" {sum(i['work'] for i in probing) / 1e9:.2f}e9 in"
        f" {sum(i['probed'] for i in probing):,} probes and"
        f" {sum(i['loop_s'] for i in probing):.2f} loop seconds."
    )
    rest = [i for i in probing if i["name"] != "supportcase6"]
    print(
        "Without supportcase6: one pass is"
        f" {sum(i['pass_highs'] for i in rest) / 1e9:.2f}e9 entries in"
        " the HiGHS count and"
        f" {sum(i['pass_gpu'] for i in rest) / 1e9:.2f}e9 without undo;"
        f" at the HiGHS rate {sum(i['pass_s'] for i in rest):.0f} s."
    )
    by_pass = sorted(probing, key=lambda i: -i["pass_gpu"])
    print(
        "Largest one-pass estimates (1e9 entries without undo): "
        + ", ".join(
            f"{i['name']} {i['pass_gpu'] / 1e9:.2f}" for i in by_pass[:6]
        )
        + "."
    )
    in_nnz = sorted((i["pass_gpu"] / i["nnz"], i["name"]) for i in probing)
    print(
        "One pass without undo in units of the original nonzero count:"
        f" min {in_nnz[0][0]:.2f} ({in_nnz[0][1]}), median"
        f" {statistics.median(v for v, _ in in_nnz):.1f}, max"
        f" {in_nnz[-1][0]:.0f} ({in_nnz[-1][1]}); at or below 100:"
        f" {sum(1 for v, _ in in_nnz if v <= 100)} of {len(in_nnz)};"
        " above 1000: "
        + ", ".join(f"{n} {v:,.0f}" for v, n in in_nnz if v > 1000)
        + "."
    )
    factor = sorted((i["pass_gpu"] / i["work"], i["name"]) for i in probing)
    print(
        "One pass without undo / HiGHS work done: median"
        f" {statistics.median(v for v, _ in factor):.1f}; below 1: "
        + ", ".join(f"{n} {v:.2f}" for v, n in factor if v < 1)
        + "; above 100: "
        + ", ".join(f"{n} {v:,.0f}" for v, n in factor if v > 100)
        + "."
    )
    dense = [
        f"{i['name']} {i['act_entries'] / (4 * i['probed']) / i['rows']:.1f}"
        for i in probing
        if i["act_entries"] / (4 * i["probed"]) > i["rows"]
    ]
    print(
        "Mean row-activity updates per probe value above the original"
        f" row count (ratio): {', '.join(dense) or 'none'}."
    )
    dense_after = [
        f"{i['name']}"
        f" {i['act_entries'] / (4 * i['probed']) / i['rows_after']:.1f}"
        for i in probing
        if i["act_entries"] / (4 * i["probed"]) > i["rows_after"]
    ]
    print(
        "The same above the presolved row count (ratio):"
        f" {', '.join(dense_after) or 'none'}. The model at probing time"
        " has between the two row counts."
    )
    for inst in probing:
        overlay = inst["act_entries"] / (4 * inst["probed"])
        if not inst["rows_after"] < overlay <= inst["rows"]:
            continue
        enum_rows = inst["diag"]["stage"]["enumeration"]["rows_removed"]
        print(
            f"- {inst['name']}: {overlay:,.0f} updates per value;"
            f" {inst['rows']:,} rows originally, {inst['rows_after']:,}"
            f" presolved; the enumeration stage removes {enum_rows:,}"
            " rows."
        )
    small_before = 0
    small_after = 0
    for inst in probing:
        overlay = inst["act_entries"] / (4 * inst["probed"])
        small_before += overlay <= 0.02 * inst["rows"]
        small_after += overlay <= 0.02 * inst["rows_after"]
    print(
        "Mean row-activity updates per probe value at most 2% of the row"
        f" count: {small_before} instances with the original row count,"
        f" {small_after} with the presolved row count."
    )
    print(
        "Limits of these sizes: (1) `without undo` takes half of"
        " activity_entries as undo; bound changes that are not undone"
        " (fixings and tightenings kept after a probe, their"
        " propagation) make the true share smaller, so the figure is a"
        " lower bound. (2) activity_entries is the column length per"
        " bound change, the count of a scheme that adds deltas along"
        " columns. A scheme that recomputes every touched row pays the"
        " row length per touched row and sweep instead; the patch does"
        " not count distinct rows touched, so that cost is not in the"
        " data. (3) prop_sweeps and max_probe_sweeps count passes over"
        " the queue of model rows; fixings by the clique table and by"
        " stored implications are applied inside the bound change and"
        " add no sweep, so a design that propagates cliques in sweeps"
        " can need more sweeps than max_probe_sweeps."
    )
    sweeps = sorted((i["max_sweeps"], i["name"]) for i in probing)
    print(
        "max_probe_sweeps: median"
        f" {statistics.median(v for v, _ in sweeps)};"
        f" {sum(1 for v, _ in sweeps if v <= 10)} of {len(sweeps)}"
        " instances at 10 or below; above 50: "
        + ", ".join(f"{n} {v}" for v, n in sweeps if v > 50)
        + "."
    )


def table_bands(instances):
    bands = defaultdict(list)
    for inst in instances:
        bands[inst["band"]].append(inst)
        bands[f"{inst['band'][0]} (all)"].append(inst)
    bands["all"] = list(instances)
    rows = []
    for band in sorted(bands, key=lambda b: (len(b) > 2, b)):
        group = bands[band]
        probing = [i for i in group if i["probed"]]
        presolve = sum(i["presolve_s"] for i in group)
        seconds = sum(i["seconds"] for i in group)
        enum_s = sum(
            i["diag"]["stage"]["enumeration"]["seconds"] for i in group
        )
        rows.append(
            [
                band,
                len(group),
                num(sum(i["cand"] for i in group)),
                num(sum(i["probed"] for i in group)),
                dec(seconds, 2),
                dec(sum(i["loop_s"] for i in group), 2),
                dec(enum_s, 2),
                dec(presolve, 2),
                percent(seconds, presolve),
                percent(enum_s, presolve),
                num(sum(i["found"] for i in group)),
                dec(sum(i["work"] for i in probing) / 1e6, 1),
                dec(sum(i["pass_gpu"] for i in probing) / 1e6, 1),
                dec(sum(i["pass_s"] for i in probing), 1),
            ]
        )
    table(
        "9. Sums per band",
        [
            "band",
            "instances",
            "candidates (round 1)",
            "probes",
            "probing s",
            "loop s",
            "enumeration s",
            "presolve s",
            "probing % of presolve",
            "enumeration % of presolve",
            "fixed + substitutions",
            "HiGHS work, 1e6",
            "one pass without undo, 1e6 entries",
            "one pass at the HiGHS rate, s",
        ],
        rows,
        [
            "Bands by original nonzeros (testbed.json). The one-pass"
            " columns are the estimates of table 7.",
        ],
    )


def largest_other_stage(inst):
    on = inst["diag"]["stage"]
    off = inst["off"]["diag"]["stage"]
    delta, name = max(
        (on[k]["seconds"] - off[k]["seconds"], k) for k in on if k != "probing"
    )
    return f"{name} {delta:+.3f}"


def table_probing_off(instances):
    with_off = [i for i in instances if i["off"]]
    if not with_off:
        print(
            "\n### 8. Presolve with probing switched off\n\n"
            "`diag_noprobing/summary.json` is not present; skipped."
        )
        return
    rows = []
    for inst in with_off:
        off = inst["off"]
        d_nnz = off["nnz_after"] - inst["nnz_after"]
        delta_s = inst["diag_core_s"] - off["presolve_core_seconds"]
        stage_s = inst["diag"]["stage"]["probing"]["seconds"]
        rows.append(
            [
                inst["name"],
                num(inst["nnz_after"]),
                num(off["rows_after"] - inst["rows_after"]),
                num(off["cols_after"] - inst["cols_after"]),
                num(d_nnz),
                percent(d_nnz, off["nnz_after"]),
                dec(inst["diag_core_s"]),
                dec(off["presolve_core_seconds"]),
                dec(stage_s),
                dec(delta_s - stage_s),
                largest_other_stage(inst),
            ]
        )
    table(
        "8. Presolve with probing switched off (HIGHS_DIAG_OFF=probing)",
        [
            "instance",
            "nnz, probing on",
            "extra rows when off",
            "extra cols when off",
            "extra nnz when off",
            "nnz saved by probing %",
            "core s on",
            "core s off",
            "probing stage s",
            "on - off - stage s",
            "other stage with the largest on - off s",
        ],
        rows,
        [
            "Supplementary: `diag_noprobing/` is a run of the patched"
            " HiGHS with `HIGHS_DIAG_OFF=probing` (`runProbing` is not"
            " called). The sizes are exact. The two runs were not timed"
            " together, so the time columns are indicative only. `on -"
            " off - stage` is the time the other stages spend more (or"
            " less) because probing changed the model.",
        ],
    )
    same = [i for i in with_off if same_without_probing(i)]
    print(
        "\nPresolved rows, columns and nonzeros equal with and without"
        f" probing: {len(same)} of {len(with_off)} instances; probing"
        f" time on these: {sum(i['seconds'] for i in same):.2f} s ("
        + ", ".join(f"{i['name']} {i['seconds']:.2f}" for i in same)
        + ")."
    )
    differ = []
    unknown = []
    equal_stats = []
    for inst in same:
        keys = stats_differ_without_probing(inst)
        if keys is None:
            unknown.append(inst["name"])
        elif keys:
            differ.append(f"{inst['name']} ({', '.join(keys)})")
        else:
            equal_stats.append(inst)
    print(
        "Of these, the structure statistics of the presolved model"
        " (column types, row classes, bounds, coefficient range, length"
        f" percentiles) are also equal on {len(equal_stats)}"
        f" ({sum(i['seconds'] for i in equal_stats):.2f} s of probing);"
        f" they differ on: {'; '.join(differ) or 'none'}; not"
        f" comparable: {', '.join(unknown) or 'none'}. Equal statistics"
        " are not a comparison of the models themselves. A presolve"
        " that returns `Not reduced` hands back the original model"
        " without tightened bounds: "
        + (
            ", ".join(
                i["name"] for i in same if i["stock"]["status"] != "Reduced"
            )
            or "none"
        )
        + "."
    )
    low_gain = [
        i
        for i in with_off
        if i["seconds"] > 1.0
        and 100 * (i["off"]["nnz_after"] - i["nnz_after"])
        < i["off"]["nnz_after"]
    ]
    print(
        "Probing above 1 s and less than 1% of the presolved nonzeros"
        " saved: "
        + ", ".join(f"{i['name']} {i['seconds']:.2f} s" for i in low_gain)
        + f"; sum {sum(i['seconds'] for i in low_gain):.2f} s of"
        f" {sum(i['seconds'] for i in with_off):.2f} s probing."
    )
    on_total = sum(i["diag_core_s"] for i in with_off)
    off_total = sum(i["off"]["presolve_core_seconds"] for i in with_off)
    stage_total = sum(
        i["diag"]["stage"]["probing"]["seconds"] for i in with_off
    )
    print(
        f"Core seconds, sum: probing on {on_total:.2f}, off"
        f" {off_total:.2f}, difference {on_total - off_total:.2f};"
        f" probing stage {stage_total:.2f}."
    )
    nnz_on = sum(i["nnz_after"] for i in with_off)
    nnz_off = sum(i["off"]["nnz_after"] for i in with_off)
    print(
        f"Presolved nonzeros, sum: probing on {nnz_on:,}, off"
        f" {nnz_off:,} ({100 * (nnz_off - nnz_on) / nnz_off:.2f}% saved"
        " by probing)."
    )
    gains = sorted(
        (
            (
                100
                * (i["off"]["nnz_after"] - i["nnz_after"])
                / i["off"]["nnz_after"],
                i["name"],
            )
            for i in with_off
        ),
        reverse=True,
    )
    print(
        "Nonzero savings of 1% or more: "
        + ", ".join(f"{n} {v:.1f}%" for v, n in gains if v >= 1)
        + "."
    )
    print(
        "Instances with more nonzeros when probing is on: "
        + (", ".join(f"{n} {v:.2f}%" for v, n in gains if v < 0) or "none")
        + "."
    )


def table_merging_off(instances):
    with_off = [i for i in instances if i["merge_off"]]
    if not with_off:
        print(
            "\n### 8b. Presolve with clique merging switched off\n\n"
            "`diag_nocliquemerge/summary.json` is not present; skipped."
        )
        return
    rows = []
    for inst in with_off:
        off = inst["merge_off"]
        merge = inst["diag"]["cliquemerge"][0]
        d_rows = off["rows_after"] - inst["rows_after"]
        d_cols = off["cols_after"] - inst["cols_after"]
        d_nnz = off["nnz_after"] - inst["nnz_after"]
        if not (
            d_rows
            or d_cols
            or d_nnz
            or merge["rows_changed"]
            or merge["rows_deleted"]
        ):
            continue
        rows.append(
            [
                inst["name"],
                num(merge["rows_changed"]),
                num(merge["rows_deleted"]),
                dec(merge["seconds"]),
                num(d_rows),
                num(d_cols),
                num(d_nnz),
                off["status"],
                dec(inst["diag_core_s"]),
                dec(off["presolve_core_seconds"]),
            ]
        )
    table(
        "8b. Presolve with clique merging switched off"
        " (HIGHS_DIAG_OFF=clique_merging; instances where merging acts or"
        " the result differs)",
        [
            "instance",
            "rows changed by merging",
            "rows deleted by merging",
            "merge s",
            "extra rows when off",
            "extra cols when off",
            "extra nnz when off",
            "presolve status when off",
            "core s on",
            "core s off",
        ],
        rows,
        [
            "Supplementary: `diag_nocliquemerge/` is a run of the patched"
            " HiGHS in which `runCliqueMerging` is not called. The sizes"
            " are exact; the two runs were not timed together. A"
            " negative `extra nnz` means the presolved model has fewer"
            " nonzeros without merging (merging adds clique-extension"
            " nonzeros).",
        ],
    )
    extra_rows = sum(
        i["merge_off"]["rows_after"] - i["rows_after"] for i in with_off
    )
    extra_nnz = sum(
        i["merge_off"]["nnz_after"] - i["nnz_after"] for i in with_off
    )
    core_off = sum(i["merge_off"]["presolve_core_seconds"] for i in with_off)
    print(
        f"\nSums over the {len(with_off)} instances: extra rows when off"
        f" {extra_rows:,}, extra nonzeros {extra_nnz:,}; core seconds on"
        f" {sum(i['diag_core_s'] for i in with_off):.2f}, off"
        f" {core_off:.2f}."
    )


def main():
    instances = load_instances()
    print("# Stage 0: HiGHS probing on the dev-size testbed\n")
    consistency(instances)
    table_overview(instances)
    table_rounds(instances)
    table_work(instances)
    table_coverage(instances)
    table_yield(instances)
    table_enumeration(instances)
    table_gpu(instances)
    table_probing_off(instances)
    table_merging_off(instances)
    table_bands(instances)


if __name__ == "__main__":
    main()
