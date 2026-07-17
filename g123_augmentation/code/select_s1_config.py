#!/usr/bin/env python3
"""Freeze one Safe-SCA configuration from development warm-up trajectories.

This script is the only S1 component permitted to use known game classes, and
only on development seeds.  The resulting JSON contains no labels and is the
sole configuration accepted by ``run_s1_safe_sca.py --phase test``.

Selection is lexicographic and preregistered:
  1. minimize anti-coordination false-align count;
  2. among ties, maximize coordination align count;
  3. among ties, choose the shorter warm-up; then the more conservative
     (larger) tau; then the larger coverage requirement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safe_sca import CoverageCertifiedEstimator, SafeSCAConfig, select_safe_arm


ORACLE = {
    "chicken": "anti_coord", "deadlock": "anti_coord", "hawk_dove": "anti_coord",
    "stag_hunt": "coord", "battle_of_the_sexes": "coord", "public_goods": "coord",
}


def load_episodes(root: Path) -> list[tuple[str, int, list[list[dict]]]]:
    rows = []
    for trajectory_path in sorted(root.glob("*/seed_*/het_notom/trajectories.jsonl")):
        game = trajectory_path.parents[2].name
        seed = int(trajectory_path.parents[1].name.removeprefix("seed_"))
        episodes = []
        for line in trajectory_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            episodes.append(record["steps"])
        rows.append((game, seed, episodes))
    if not rows:
        raise ValueError(f"No development trajectories found below {root}")
    return rows


def summarize_prefix(episodes, warmup: int, config: SafeSCAConfig, seed: int):
    estimator = CoverageCertifiedEstimator()
    for episode in episodes[:warmup]:
        for step in episode:
            estimator.observe(step["actions"], step["rewards"])
    n_agents = len(episodes[0][0]["actions"])
    # All frozen S1 environments are binary-action games.  This is an
    # environment interface fact, not information inferred from later actions.
    n_actions = 2
    return estimator.summarize(
        n_agents=n_agents, n_actions=n_actions, config=config, seed=seed,
    )


def candidate_configs(args: argparse.Namespace):
    for warmup in args.warmup_grid:
        for tau in args.tau_grid:
            for coverage in args.coverage_grid:
                for min_stratum in args.min_stratum_grid:
                    yield SafeSCAConfig(
                        warmup_episodes=warmup,
                        tau=tau,
                        confidence=args.confidence,
                        bootstrap_samples=args.bootstrap_samples,
                        min_profile_coverage=coverage,
                        min_stratum_observations=min_stratum,
                    )


def evaluate(config: SafeSCAConfig, data) -> dict:
    rows = []
    for game, seed, episodes in data:
        if len(episodes) < config.warmup_episodes:
            raise ValueError(f"{game}/seed_{seed} has fewer than {config.warmup_episodes} episodes")
        summary = summarize_prefix(episodes, config.warmup_episodes, config, seed)
        arm, reasons = select_safe_arm(summary, config)
        label = ORACLE[game]
        rows.append({
            "game": game,
            "seed": seed,
            "oracle_structure": label,
            "selected_arm": arm,
            "reasons": reasons,
            "coverage": summary.to_dict(),
        })
    anti_false_align = sum(r["oracle_structure"] == "anti_coord" and r["selected_arm"] == "Gated"
                           for r in rows)
    coord_align = sum(r["oracle_structure"] == "coord" and r["selected_arm"] == "Gated"
                      for r in rows)
    anti_total = sum(r["oracle_structure"] == "anti_coord" for r in rows)
    coord_total = sum(r["oracle_structure"] == "coord" for r in rows)
    return {
        "config": config.to_dict(),
        "anti_false_align_count": anti_false_align,
        "anti_false_align_rate": anti_false_align / anti_total if anti_total else 0.0,
        "coord_align_count": coord_align,
        "coord_align_rate": coord_align / coord_total if coord_total else 0.0,
        "n_cells": len(rows),
        "per_cell": rows,
    }


def ranking(result: dict) -> tuple:
    cfg = result["config"]
    return (
        result["anti_false_align_count"],
        -result["coord_align_count"],
        cfg["warmup_episodes"],
        -cfg["tau"],
        -cfg["min_profile_coverage"],
        -cfg["min_stratum_observations"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and freeze a Safe-SCA development config")
    parser.add_argument("--dev-results", type=Path, required=True,
                        help="exp_s1_dev_warmup output directory")
    parser.add_argument("--frozen-config", type=Path, required=True,
                        help="new JSON passed unchanged to the held-out S1 test")
    parser.add_argument("--report", type=Path,
                        help="selection report JSON (default: next to frozen config)")
    parser.add_argument("--warmup-grid", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--tau-grid", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--coverage-grid", type=float, nargs="+", default=[0.125, 0.25, 0.50])
    parser.add_argument("--min-stratum-grid", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data = load_episodes(args.dev_results)
    results = [evaluate(config, data) for config in candidate_configs(args)]
    best = min(results, key=ranking)
    report = {
        "schema_version": 1,
        "selection_rule": (
            "minimize anti false-align count; maximize coordination align count; "
            "shorter warm-up; larger tau; larger coverage; larger stratum minimum"
        ),
        "development_oracle_used_only_here": True,
        "n_candidates": len(results),
        "selected": best,
        "ranked_candidates": sorted(results, key=ranking),
    }
    frozen_path = args.frozen_config
    report_path = args.report or frozen_path.with_name(f"{frozen_path.stem}_selection_report.json")
    if (frozen_path.exists() or report_path.exists()) and not args.force:
        raise FileExistsError("Refusing to overwrite output; use --force only before held-out test starts")
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(best["config"], indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "frozen_config": str(frozen_path),
        "report": str(report_path),
        "selected": best["config"],
        "anti_false_align": best["anti_false_align_count"],
        "coord_align": best["coord_align_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
