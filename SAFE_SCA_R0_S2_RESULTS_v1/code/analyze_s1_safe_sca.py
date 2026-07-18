#!/usr/bin/env python3
"""Analyze the held-out S1 Safe-SCA experiment.

Primary endpoint: total-horizon mean team payoff (warm-up included), paired
against NoAlign within each (game, seed).  Game classes are used only here for
reporting false-align risk and coordination-gain recovery; they are never
available to the Safe-SCA policy at inference time.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np


ORDER = ["chicken", "deadlock", "hawk_dove", "stag_hunt", "battle_of_the_sexes", "public_goods"]
ANTI = {"chicken", "deadlock", "hawk_dove"}
COORD = {"stag_hunt", "battle_of_the_sexes", "public_goods"}
POLICIES = ["het_notom", "het_gated_atom_talk", "het_gsaca", "het_point_sca", "het_safe_sca", "het_oracle_sca"]
DISPLAY = {
    "het_notom": "NoAlign",
    "het_gated_atom_talk": "Always Gated",
    "het_gsaca": "Legacy GSACA",
    "het_point_sca": "Point-SCA",
    "het_safe_sca": "Safe-SCA",
    "het_oracle_sca": "Label-oracle SCA (diagnostic)",
}


def team_payoff(metrics: dict[str, Any]) -> float:
    """Read the S1 total-horizon endpoint, with a legacy-cell fallback."""
    if "s1_total_team_payoff" in metrics:
        return float(metrics["s1_total_team_payoff"])
    if "team_mean_payoff" in metrics:
        return float(metrics["team_mean_payoff"])
    raise KeyError("metrics contains neither s1_total_team_payoff nor team_mean_payoff")


def load(root: Path) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    values: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for path in root.glob("*/seed_*/*/metrics.json"):
        game = path.parents[2].name
        seed = int(path.parents[1].name.removeprefix("seed_"))
        policy = path.parent.name
        if policy not in POLICIES:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        values[game][seed][policy] = metrics
    if not values:
        raise ValueError(f"No S1 metrics found below {root}")
    return values


def paired_bootstrap_ci(deltas: list[float], *, seed: int, n_boot: int = 20_000) -> tuple[float, float]:
    if not deltas:
        return float("nan"), float("nan")
    arr = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def rows_for_policy(data, game: str, policy: str) -> tuple[list[int], list[float], list[float]]:
    seeds, baseline, candidate = [], [], []
    for seed, cell_map in sorted(data.get(game, {}).items()):
        if "het_notom" not in cell_map or policy not in cell_map:
            continue
        seeds.append(seed)
        baseline.append(team_payoff(cell_map["het_notom"]))
        candidate.append(team_payoff(cell_map[policy]))
    return seeds, baseline, candidate


def aggregate(
    data,
    safety_margin: float,
    min_coordination_recovery: float = 0.30,
    min_coordination_games: int = 2,
) -> dict[str, Any]:
    per_game: list[dict[str, Any]] = []
    for game in ORDER:
        for policy in POLICIES:
            seeds, baseline, candidate = rows_for_policy(data, game, policy)
            if not seeds:
                continue
            delta = (np.asarray(candidate) - np.asarray(baseline)).astype(float)
            bootstrap_seed = zlib.crc32(f"{game}:{policy}".encode("utf-8"))
            ci_low, ci_high = paired_bootstrap_ci(delta.tolist(), seed=bootstrap_seed)
            per_game.append({
                "game": game,
                "group": "anti" if game in ANTI else "coord_or_boundary",
                "policy": policy,
                "policy_display": DISPLAY[policy],
                "n_paired": len(seeds),
                "seeds": seeds,
                "baseline_mean": float(np.mean(baseline)),
                "policy_mean": float(np.mean(candidate)),
                "delta_mean": float(np.mean(delta)),
                "delta_ci95": [ci_low, ci_high],
                "paired_win_rate": float(np.mean(delta > 0)),
            })

    safe_rows = [r for r in per_game if r["policy"] == "het_safe_sca"]
    safety_by_game = [
        {**r, "noninferior_to_noalign": r["delta_ci95"][0] >= -safety_margin}
        for r in safe_rows if r["game"] in ANTI
    ]
    safety_pass = bool(safety_by_game) and all(r["noninferior_to_noalign"] for r in safety_by_game)

    route_rows = []
    false_align = false_abstain = 0
    anti_total = coord_total = 0
    for game, seed_map in data.items():
        for seed, cells in seed_map.items():
            safe = cells.get("het_safe_sca")
            if not safe:
                continue
            selected = safe.get("s1_selected_post_warmup_arm")
            group = "anti" if game in ANTI else "coord_or_boundary"
            false_align_flag = game in ANTI and selected == "Gated"
            false_abstain_flag = game in COORD and selected != "Gated"
            if game in ANTI:
                anti_total += 1
                false_align += int(false_align_flag)
            if game in COORD:
                coord_total += 1
                false_abstain += int(false_abstain_flag)
            route_rows.append({
                "game": game,
                "seed": seed,
                "group": group,
                "selected_arm": selected,
                "false_align": false_align_flag,
                "false_abstain": false_abstain_flag,
                "decision_reasons": safe.get("s1_decision_reasons", []),
                "coverage": safe.get("s1_coverage", {}),
            })

    recovery = []
    for game in ORDER:
        _, baseline, safe = rows_for_policy(data, game, "het_safe_sca")
        _, _, gated = rows_for_policy(data, game, "het_gated_atom_talk")
        if not baseline or not gated:
            continue
        baseline_mean = float(np.mean(baseline))
        gated_gain = float(np.mean(gated) - baseline_mean)
        safe_gain = float(np.mean(safe) - baseline_mean)
        recovery.append({
            "game": game,
            "is_coordination_or_boundary": game in COORD,
            "gated_gain_over_noalign": gated_gain,
            "safe_sca_gain_over_noalign": safe_gain,
            "gain_recovery_fraction": safe_gain / gated_gain if abs(gated_gain) > 1e-12 else None,
        })

    utility_qualified = [
        row for row in recovery
        if row["is_coordination_or_boundary"]
        and row["gated_gain_over_noalign"] > 0
        and row["gain_recovery_fraction"] is not None
        and row["gain_recovery_fraction"] >= min_coordination_recovery
    ]
    utility_pass = len(utility_qualified) >= min_coordination_games

    return {
        "primary_endpoint": "total_horizon_team_payoff_including_warmup",
        "safety_margin": safety_margin,
        "utility_gate": {
            "min_coordination_recovery": min_coordination_recovery,
            "min_coordination_games": min_coordination_games,
            "qualified_games": [row["game"] for row in utility_qualified],
            "passed": utility_pass,
        },
        "per_game_policy": per_game,
        "safety_by_anti_game": safety_by_game,
        "safety_pass_all_anti_games": safety_pass,
        "method_paper_pass": safety_pass and utility_pass,
        "safe_sca_routing": {
            "n_rows": len(route_rows),
            "anti_false_align": false_align,
            "anti_total": anti_total,
            "anti_false_align_rate": false_align / anti_total if anti_total else None,
            "coord_false_abstain": false_abstain,
            "coord_total": coord_total,
            "coord_false_abstain_rate": false_abstain / coord_total if coord_total else None,
            "per_cell": route_rows,
        },
        "coordination_gain_recovery": recovery,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# S1 — Coverage-Certified Safe-SCA held-out analysis", ""]
    lines.append(f"Primary endpoint: `{report['primary_endpoint']}`.")
    lines.append(f"Safety margin: {report['safety_margin']:.3f}; "
                 f"all anti-coordination games pass: **{report['safety_pass_all_anti_games']}**.")
    utility = report["utility_gate"]
    lines.append(
        "Utility gate: recover at least "
        f"{utility['min_coordination_recovery']:.0%} of the positive Always-Gated gain in "
        f"{utility['min_coordination_games']} coordination/boundary games; "
        f"qualified={utility['qualified_games']}; pass: **{utility['passed']}**."
    )
    lines.append(f"Method-paper gate (safety AND utility): **{report['method_paper_pass']}**.")
    lines.append("")
    lines.append("## Paired payoff against NoAlign")
    lines.append("")
    lines.append("| game | policy | n | NoAlign | policy | delta | 95% paired bootstrap CI | win rate |")
    lines.append("|---|---|---:|---:|---:|---:|---|---:|")
    for row in report["per_game_policy"]:
        low, high = row["delta_ci95"]
        lines.append(
            f"| {row['game']} | {row['policy_display']} | {row['n_paired']} | "
            f"{row['baseline_mean']:.3f} | {row['policy_mean']:.3f} | {row['delta_mean']:+.3f} | "
            f"[{low:+.3f}, {high:+.3f}] | {row['paired_win_rate']:.0%} |"
        )
    lines.append("")
    lines.append("## Safe-SCA routing risk")
    lines.append("")
    routing = report["safe_sca_routing"]
    anti_rate = "n/a" if routing["anti_false_align_rate"] is None else f"{routing['anti_false_align_rate']:.1%}"
    coord_rate = "n/a" if routing["coord_false_abstain_rate"] is None else f"{routing['coord_false_abstain_rate']:.1%}"
    lines.append(f"- Anti-coordination false-align: **{routing['anti_false_align']}/{routing['anti_total']}** "
                 f"({anti_rate}).")
    lines.append(f"- Coordination/boundary false-abstain: **{routing['coord_false_abstain']}/{routing['coord_total']}** "
                 f"({coord_rate}).")
    lines.append("")
    lines.append("## Coordination gain recovery")
    lines.append("")
    lines.append("| game | Gated-NoAlign | Safe-SCA-NoAlign | recovery |")
    lines.append("|---|---:|---:|---:|")
    for row in report["coordination_gain_recovery"]:
        recovery = "n/a" if row["gain_recovery_fraction"] is None else f"{row['gain_recovery_fraction']:.1%}"
        lines.append(f"| {row['game']} | {row['gated_gain_over_noalign']:+.3f} | "
                     f"{row['safe_sca_gain_over_noalign']:+.3f} | {recovery} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze S1 Safe-SCA results")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="default: results directory")
    parser.add_argument("--safety-margin", type=float, default=0.10)
    parser.add_argument("--min-coordination-recovery", type=float, default=0.30)
    parser.add_argument("--min-coordination-games", type=int, default=2)
    args = parser.parse_args()
    if args.safety_margin < 0:
        parser.error("safety-margin must be non-negative")
    if not 0 <= args.min_coordination_recovery <= 1:
        parser.error("min-coordination-recovery must be in [0, 1]")
    if args.min_coordination_games <= 0:
        parser.error("min-coordination-games must be positive")
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)
    report = aggregate(
        load(args.results), args.safety_margin,
        min_coordination_recovery=args.min_coordination_recovery,
        min_coordination_games=args.min_coordination_games,
    )
    (out / "s1_safe_sca_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    (out / "s1_safe_sca_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"[wrote] {out / 's1_safe_sca_summary.json'}")
    print(f"[wrote] {out / 's1_safe_sca_summary.md'}")


if __name__ == "__main__":
    main()
