#!/usr/bin/env python3
"""Compare a same-seed R0 replay against the archived S1 reference output.

R0 is an execution reproducibility audit, not a parameter-search step.  This
script compares every requested `(game, seed, policy)` cell, checks Safe-SCA's
post-warm-up routing decision exactly, and enforces a preregistered payoff
tolerance.  It writes a machine-readable report and returns non-zero when the
replay is incomplete or outside the declared tolerance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GAMES = [
    "chicken", "deadlock", "hawk_dove",
    "stag_hunt", "battle_of_the_sexes", "public_goods",
]
CELLS = [
    "het_notom", "het_gated_atom_talk", "het_gsaca",
    "het_point_sca", "het_safe_sca", "het_oracle_sca",
]


def read_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Unreadable JSON {path}: {exc}")
        return None


def total_payoff(metrics: dict[str, Any]) -> float:
    if "s1_total_team_payoff" in metrics:
        return float(metrics["s1_total_team_payoff"])
    if "team_mean_payoff" in metrics:
        return float(metrics["team_mean_payoff"])
    raise KeyError("missing s1_total_team_payoff/team_mean_payoff")


def compare(
    reference: Path,
    replay: Path,
    *,
    games: list[str],
    seeds: list[int],
    cells: list[str],
    payoff_tolerance: float,
    route_mismatch_budget: int,
) -> dict[str, Any]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    route_mismatches = 0
    max_abs_difference = 0.0
    for game in games:
        for seed in seeds:
            for cell in cells:
                ref_path = reference / game / f"seed_{seed}" / cell / "metrics.json"
                replay_path = replay / game / f"seed_{seed}" / cell / "metrics.json"
                ref = read_json(ref_path, errors)
                candidate = read_json(replay_path, errors)
                if ref is None or candidate is None:
                    continue
                try:
                    ref_payoff = total_payoff(ref)
                    replay_payoff = total_payoff(candidate)
                except KeyError as exc:
                    errors.append(f"{game}/seed_{seed}/{cell}: {exc}")
                    continue
                difference = replay_payoff - ref_payoff
                max_abs_difference = max(max_abs_difference, abs(difference))
                row = {
                    "game": game,
                    "seed": seed,
                    "cell": cell,
                    "reference_total_payoff": ref_payoff,
                    "replay_total_payoff": replay_payoff,
                    "difference": difference,
                    "within_payoff_tolerance": abs(difference) <= payoff_tolerance,
                }
                if cell == "het_safe_sca":
                    ref_arm = ref.get("s1_selected_post_warmup_arm")
                    replay_arm = candidate.get("s1_selected_post_warmup_arm")
                    route_match = ref_arm == replay_arm
                    route_mismatches += int(not route_match)
                    row.update({
                        "reference_selected_arm": ref_arm,
                        "replay_selected_arm": replay_arm,
                        "route_match": route_match,
                    })
                rows.append(row)

    expected_rows = len(games) * len(seeds) * len(cells)
    payoff_violations = sum(not row["within_payoff_tolerance"] for row in rows)
    complete = len(rows) == expected_rows and not errors
    passed = complete and payoff_violations == 0 and route_mismatches <= route_mismatch_budget
    return {
        "schema_version": 1,
        "reference": str(reference),
        "replay": str(replay),
        "games": games,
        "seeds": seeds,
        "cells": cells,
        "expected_rows": expected_rows,
        "compared_rows": len(rows),
        "payoff_tolerance": payoff_tolerance,
        "route_mismatch_budget": route_mismatch_budget,
        "max_abs_payoff_difference": max_abs_difference,
        "payoff_violations": payoff_violations,
        "route_mismatches": route_mismatches,
        "errors": errors,
        "complete": complete,
        "passed": passed,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare an R0 Safe-SCA replay with S1")
    parser.add_argument("--reference", type=Path, required=True,
                        help="S1 exp_s1_safe_sca_test result directory")
    parser.add_argument("--replay", type=Path, required=True,
                        help="R0 exp_r0_safe_sca_test result directory")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--games", nargs="+", default=GAMES, choices=GAMES)
    parser.add_argument("--cells", nargs="+", default=CELLS, choices=CELLS)
    parser.add_argument("--payoff-tolerance", type=float, default=0.0,
                        help="preregister before R0; default requires exact equality")
    parser.add_argument("--route-mismatch-budget", type=int, default=0,
                        help="preregister before R0; default requires exact Safe-SCA routing")
    parser.add_argument("--out", type=Path, help="default: replay directory")
    args = parser.parse_args()
    if args.payoff_tolerance < 0:
        parser.error("payoff-tolerance must be non-negative")
    if args.route_mismatch_budget < 0:
        parser.error("route-mismatch-budget must be non-negative")
    report = compare(
        args.reference, args.replay, games=args.games, seeds=args.seeds,
        cells=args.cells, payoff_tolerance=args.payoff_tolerance,
        route_mismatch_budget=args.route_mismatch_budget,
    )
    out = args.out or args.replay
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "R0_REPLAY_COMPARISON.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"[wrote] {destination}")
    if not report["passed"]:
        raise SystemExit("R0 replay comparison failed; investigate before S2.")


if __name__ == "__main__":
    main()
