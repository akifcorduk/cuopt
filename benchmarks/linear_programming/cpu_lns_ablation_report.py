#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare CPU-LNS ablation result directories against the full-method control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def load_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def validate_campaigns(
    campaigns: dict[str, dict[str, Any]], expected_instances: int
) -> None:
    control = campaigns.get("none")
    if control is None:
        raise ValueError("missing control directory with ablation.variant=none")
    expected_names = control["instances"]
    if len(expected_names) != expected_instances:
        raise ValueError(
            f"control contains {len(expected_names)} instances; expected {expected_instances}"
        )
    invariant_command_keys = (
        "cuopt_cli",
        "determinism_mode",
        "feasibility_tolerance",
        "instances_file",
        "minimum_instances",
        "num_cpu_threads",
        "presolve",
        "seed",
        "time_limit",
    )
    for variant, campaign in campaigns.items():
        if campaign["instances"] != expected_names:
            raise ValueError(f"{variant}: corpus or order differs from control")
        if campaign["artifacts"]["cuopt_cli_sha256"] != control["artifacts"][
            "cuopt_cli_sha256"
        ]:
            raise ValueError(f"{variant}: cuopt_cli hash differs from control")
        for key in invariant_command_keys:
            if campaign["command"].get(key) != control["command"].get(key):
                raise ValueError(f"{variant}: command field {key} differs from control")


def paired_ratio(
    variant: dict[str, Any], control: dict[str, Any], field: str
) -> float | None:
    numerator = variant.get(field)
    denominator = control.get(field)
    if not finite(numerator) or not finite(denominator) or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def compare_variant(
    variant_name: str,
    campaign: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    control_by_name = {result["instance"]: result for result in control["results"]}
    variant_by_name = {result["instance"]: result for result in campaign["results"]}
    gained: list[str] = []
    lost: list[str] = []
    shared_feasible: list[str] = []
    ttff_ratios: list[float] = []
    ips_ratios: list[float] = []
    unsat_deltas: list[float] = []
    excess_deltas: list[float] = []
    objective_deltas: list[float] = []

    for instance, control_result in control_by_name.items():
        variant_result = variant_by_name[instance]
        control_feasible = bool(control_result["feasible"])
        variant_feasible = bool(variant_result["feasible"])
        if variant_feasible and not control_feasible:
            gained.append(instance)
        elif control_feasible and not variant_feasible:
            lost.append(instance)
        elif control_feasible:
            shared_feasible.append(instance)

        ratio = paired_ratio(
            variant_result, control_result, "first_feasible_solve_s"
        )
        if control_feasible and variant_feasible and ratio is not None:
            ttff_ratios.append(ratio)
        ratio = paired_ratio(variant_result, control_result, "iterations_per_second")
        if ratio is not None:
            ips_ratios.append(ratio)
        if finite(variant_result.get("final_unsat")) and finite(
            control_result.get("final_unsat")
        ):
            unsat_deltas.append(
                float(variant_result["final_unsat"])
                - float(control_result["final_unsat"])
            )
        if finite(variant_result.get("final_excess")) and finite(
            control_result.get("final_excess")
        ):
            excess_deltas.append(
                float(variant_result["final_excess"])
                - float(control_result["final_excess"])
            )
        if (
            control_feasible
            and variant_feasible
            and finite(variant_result.get("final_objective_solver"))
            and finite(control_result.get("final_objective_solver"))
        ):
            objective_deltas.append(
                float(variant_result["final_objective_solver"])
                - float(control_result["final_objective_solver"])
            )

    summary = campaign["summary"]
    control_summary = control["summary"]
    return {
        "variant": variant_name,
        "description": campaign["ablation"]["description"],
        "feasible_count": summary["feasible_count"],
        "feasible_delta": summary["feasible_count"]
        - control_summary["feasible_count"],
        "attachment_feasible_count": summary["attachment_feasible_count"],
        "attachment_feasible_delta": summary["attachment_feasible_count"]
        - control_summary["attachment_feasible_count"],
        "gained_feasible": gained,
        "lost_feasible": lost,
        "shared_feasible_count": len(shared_feasible),
        "median_ttff_ratio": median(ttff_ratios),
        "median_ips_ratio": median(ips_ratios),
        "median_final_unsat_delta": median(unsat_deltas),
        "median_final_excess_delta": median(excess_deltas),
        "median_final_objective_delta": median(objective_deltas),
        "objective_improved_count": summary["objective_improved_count"],
        "objective_best_updates": summary.get("objective_best_updates", 0),
        "sa_uphill_accepts": summary.get("sa_uphill_accepts", 0),
        "sa_escape_best_updates": summary.get("sa_escape_best_updates", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-instances", type=int, default=30)
    args = parser.parse_args()

    campaigns: dict[str, dict[str, Any]] = {}
    for result_path in sorted(args.results_root.glob("*/results.json")):
        campaign = load_result(result_path)
        variant = campaign["ablation"]["variant"]
        if variant in campaigns:
            raise ValueError(f"duplicate campaign for ablation variant {variant}")
        campaigns[variant] = campaign
    if not campaigns:
        parser.error(f"no */results.json files below {args.results_root}")
    validate_campaigns(campaigns, args.expected_instances)

    control = campaigns["none"]
    comparisons = [
        compare_variant(variant, campaign, control)
        for variant, campaign in sorted(campaigns.items())
    ]
    report = {
        "control_summary": control["summary"],
        "cuopt_cli_sha256": control["artifacts"]["cuopt_cli_sha256"],
        "command": control["command"],
        "comparisons": comparisons,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "ablation_report.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    with (args.output_dir / "ablation_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        rows = []
        for comparison in comparisons:
            row = comparison.copy()
            row["gained_feasible"] = ",".join(row["gained_feasible"])
            row["lost_feasible"] = ",".join(row["lost_feasible"])
            rows.append(row)
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(comparisons, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
