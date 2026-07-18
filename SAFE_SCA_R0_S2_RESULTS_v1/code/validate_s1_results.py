#!/usr/bin/env python3
"""Fail closed when an S1 held-out result directory is incomplete or contaminated.

The validator prevents three easy-to-miss problems before statistical analysis:
missing cells, a Safe-SCA policy that used an oracle label, and a result folder
whose frozen Safe-SCA configuration drifted across cells.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GAMES = ["chicken", "deadlock", "hawk_dove", "stag_hunt", "battle_of_the_sexes", "public_goods"]
CELLS = ["het_notom", "het_gated_atom_talk", "het_gsaca", "het_point_sca", "het_safe_sca", "het_oracle_sca"]
S1_CELLS = {"het_point_sca", "het_safe_sca", "het_oracle_sca"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def read_json(path: Path, errors: list[str]) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Unreadable JSON {path}: {exc}")
        return None


def validate(
    root: Path,
    frozen_config: dict,
    *,
    games: list[str],
    seeds: list[int],
    cells: list[str],
    expected_episodes: int,
) -> dict:
    missing: list[str] = []
    errors: list[str] = []
    checked = 0
    expected_config = canonical(frozen_config)
    manifest = read_json(root / "ENVIRONMENT_MANIFEST_S1.json", errors)
    if manifest is None:
        missing.append(str(root / "ENVIRONMENT_MANIFEST_S1.json"))
    else:
        if not manifest.get("preflight_passed"):
            errors.append("S1 environment preflight was not successful")
        if manifest.get("allow_version_mismatch"):
            errors.append("S1 environment preflight allowed a version mismatch")
    for game in games:
        for seed in seeds:
            seed_dir = root / game / f"seed_{seed}"
            arm_order = read_json(seed_dir / "arm_order.json", errors)
            if arm_order is None:
                missing.append(str(seed_dir / "arm_order.json"))
            elif sorted(arm_order.get("arm_order", [])) != sorted(cells):
                errors.append(f"arm order is not a permutation of expected cells: {seed_dir}")
            for cell in cells:
                metrics_path = seed_dir / cell / "metrics.json"
                if not metrics_path.exists():
                    missing.append(str(metrics_path))
                    continue
                metrics = read_json(metrics_path, errors)
                if metrics is None:
                    continue
                checked += 1
                if metrics.get("n_episodes") != expected_episodes:
                    errors.append(f"{metrics_path}: n_episodes={metrics.get('n_episodes')} not {expected_episodes}")
                if cell not in S1_CELLS:
                    continue
                if canonical(metrics.get("s1_safe_config")) != expected_config:
                    errors.append(f"{metrics_path}: Safe-SCA config drift")
                total_payoffs = metrics.get("s1_total_episode_team_payoffs", [])
                if len(total_payoffs) != expected_episodes:
                    errors.append(f"{metrics_path}: total-horizon payoff list has {len(total_payoffs)} entries")
                if cell in {"het_safe_sca", "het_point_sca"} and metrics.get("s1_oracle_label_used"):
                    errors.append(f"{metrics_path}: deployable policy used oracle label")
                if cell == "het_oracle_sca" and not metrics.get("s1_oracle_label_used"):
                    errors.append(f"{metrics_path}: oracle diagnostic arm was not marked")
                decision_path = seed_dir / cell / "decision.json"
                if not decision_path.exists():
                    missing.append(str(decision_path))

    expected_cells = len(games) * len(seeds) * len(cells)
    return {
        "schema_version": 1,
        "root": str(root),
        "expected_cells": expected_cells,
        "checked_metrics": checked,
        "missing_count": len(missing),
        "missing": missing,
        "error_count": len(errors),
        "errors": errors,
        "ready_for_analysis": not missing and not errors and checked == expected_cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate completeness and provenance of S1 results")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(62, 82)))
    parser.add_argument("--expected-episodes", type=int, default=30)
    parser.add_argument("--out", type=Path, help="default: results directory")
    args = parser.parse_args()
    frozen = read_json(args.frozen_config, [])
    if frozen is None:
        parser.error(f"Cannot read frozen config: {args.frozen_config}")
    report = validate(args.results, frozen, games=GAMES, seeds=args.seeds,
                      cells=CELLS, expected_episodes=args.expected_episodes)
    out_dir = args.out or args.results
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "S1_INTEGRITY_REPORT.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[wrote] {output}")
    if not report["ready_for_analysis"]:
        raise SystemExit("S1 integrity validation failed; do not run final analysis yet")


if __name__ == "__main__":
    main()
