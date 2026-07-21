#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate repeated CPU-LNS objective trajectories and paired controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


HORIZONS_S = (2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def load_campaigns(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    campaigns: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(root.glob("*/seed_*/results.json")):
        campaign = load_json(path)
        variant = str(campaign["ablation"]["variant"])
        seed = int(campaign["command"]["seed"])
        key = (variant, seed)
        if key in campaigns:
            raise ValueError(f"duplicate campaign for {variant}, seed {seed}")
        campaigns[key] = campaign
    if not campaigns:
        raise ValueError(f"no variant/seed/results.json campaigns under {root}")
    return campaigns


def validate_campaigns(campaigns: dict[tuple[str, int], dict[str, Any]]) -> None:
    variants = sorted({variant for variant, _ in campaigns})
    seeds = sorted({seed for _, seed in campaigns})
    if "none" not in variants:
        raise ValueError("missing none control variant")
    expected_keys = {(variant, seed) for variant in variants for seed in seeds}
    if set(campaigns) != expected_keys:
        missing = sorted(expected_keys - set(campaigns))
        raise ValueError(f"incomplete variant/seed matrix: missing {missing}")

    control = campaigns[("none", seeds[0])]
    expected_instances = control["instances"]
    expected_hash = control["artifacts"]["cuopt_cli_sha256"]
    invariant_keys = (
        "cuopt_cli",
        "determinism_mode",
        "feasibility_tolerance",
        "instances_file",
        "minimum_instances",
        "num_cpu_threads",
        "presolve",
        "time_limit",
    )
    for key, campaign in campaigns.items():
        if campaign["instances"] != expected_instances:
            raise ValueError(f"{key}: instance corpus differs from control")
        if campaign["artifacts"]["cuopt_cli_sha256"] != expected_hash:
            raise ValueError(f"{key}: cuopt_cli hash differs from control")
        for command_key in invariant_keys:
            if campaign["command"].get(command_key) != control["command"].get(command_key):
                raise ValueError(f"{key}: command field {command_key} differs from control")


def index_results(
    campaigns: dict[tuple[str, int], dict[str, Any]],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for (variant, seed), campaign in campaigns.items():
        for result in campaign["results"]:
            indexed[(variant, seed, result["instance"])] = result
    return indexed


def best_at_horizon(result: dict[str, Any], horizon_s: float) -> float | None:
    selected: float | None = None
    for sample in result.get("objective_trajectory", []):
        elapsed = sample.get("solve_elapsed_s")
        best = sample.get("best_solver")
        if finite(elapsed) and float(elapsed) <= horizon_s and finite(best):
            selected = float(best)
    return selected


def relative_improvement(first: float, best: float) -> float:
    return (first - best) / (1.0 + abs(first))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    campaigns: dict[tuple[str, int], dict[str, Any]], root: Path
) -> dict[str, Any]:
    indexed = index_results(campaigns)
    variants = sorted({variant for variant, _ in campaigns})
    seeds = sorted({seed for _, seed in campaigns})
    instances = campaigns[("none", seeds[0])]["instances"]

    paired_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        improvements: list[float] = []
        relative_improvements: list[float] = []
        rates: list[float] = []
        uphill = 0
        escapes = 0
        improved_count = 0
        paired_deltas: list[float] = []
        wins = ties = losses = 0
        for seed in seeds:
            for instance in instances:
                result = indexed[(variant, seed, instance)]
                first = result.get("first_feasible_objective_solver")
                final = result.get("final_objective_solver")
                if finite(result.get("iterations_per_second")):
                    rates.append(float(result["iterations_per_second"]))
                diagnostics = result.get("diagnostics") or {}
                uphill += int(diagnostics.get("sa_uphill_accepts", 0))
                escapes += int(diagnostics.get("sa_escape_best_updates", 0))
                if finite(first) and finite(final):
                    first_f = float(first)
                    final_f = float(final)
                    improvement = first_f - final_f
                    improvements.append(improvement)
                    relative_improvements.append(relative_improvement(first_f, final_f))
                    improved_count += improvement > 1e-9
                    for horizon in HORIZONS_S:
                        best = best_at_horizon(result, horizon)
                        if best is None:
                            continue
                        trajectory_rows.append(
                            {
                                "variant": variant,
                                "seed": seed,
                                "instance": instance,
                                "horizon_s": horizon,
                                "first_objective_solver": first_f,
                                "best_objective_solver": best,
                                "improvement_solver": first_f - best,
                                "relative_improvement": relative_improvement(first_f, best),
                            }
                        )

                if variant == "none":
                    continue
                control = indexed[("none", seed, instance)]
                control_final = control.get("final_objective_solver")
                if not finite(final) or not finite(control_final):
                    continue
                delta = float(final) - float(control_final)
                tolerance = 1e-9 * (1.0 + abs(float(control_final)))
                paired_deltas.append(delta)
                if delta < -tolerance:
                    outcome = "win"
                    wins += 1
                elif delta > tolerance:
                    outcome = "loss"
                    losses += 1
                else:
                    outcome = "tie"
                    ties += 1
                paired_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "instance": instance,
                        "control_final_objective_solver": float(control_final),
                        "variant_final_objective_solver": float(final),
                        "variant_minus_control": delta,
                        "outcome_lower_is_better": outcome,
                    }
                )

        summaries.append(
            {
                "variant": variant,
                "campaign_count": len(seeds),
                "objective_trackable_count": len(improvements),
                "objective_improved_count": improved_count,
                "median_objective_improvement_solver": median(improvements),
                "sum_objective_improvement_solver": sum(improvements),
                "median_relative_improvement": median(relative_improvements),
                "median_iterations_per_second": median(rates),
                "sa_uphill_accepts": uphill,
                "sa_escape_best_updates": escapes,
                "paired_wins": wins,
                "paired_ties": ties,
                "paired_losses": losses,
                "median_final_objective_delta_vs_control": median(paired_deltas),
                "sum_final_objective_delta_vs_control": sum(paired_deltas),
            }
        )

    return {
        "results_root": str(root.resolve()),
        "variants": variants,
        "seeds": seeds,
        "instances": instances,
        "cuopt_cli_sha256": campaigns[("none", seeds[0])]["artifacts"][
            "cuopt_cli_sha256"
        ],
        "variant_summaries": summaries,
        "paired_objectives": paired_rows,
        "trajectory_samples": trajectory_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    campaigns = load_campaigns(args.results_root)
    validate_campaigns(campaigns)
    report = build_report(campaigns, args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trajectory_report.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    write_csv(args.output_dir / "variant_summary.csv", report["variant_summaries"])
    write_csv(args.output_dir / "paired_objectives.csv", report["paired_objectives"])
    write_csv(args.output_dir / "trajectory_samples.csv", report["trajectory_samples"])
    print(json.dumps(report["variant_summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
