#!/usr/bin/env python3
"""Separate-matrix P3 analysis with predeclared transfer gates."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np

from p3_matrices import get_spec
from p3_protocol import load_protocol


DISPLAY = {
    "het_notom": "NoAlign",
    "het_gated_atom_talk": "Always Gated",
    "het_point_sca": "Point-SCA",
    "het_safe_sca": "Safe-SCA",
}


def team_payoff(metrics: dict[str, Any]) -> float:
    if "s1_total_team_payoff" in metrics:
        return float(metrics["s1_total_team_payoff"])
    if "team_mean_payoff" in metrics:
        return float(metrics["team_mean_payoff"])
    raise KeyError("metrics has neither s1_total_team_payoff nor team_mean_payoff")


def load(root: Path, protocol: dict) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    data: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for matrix in protocol["matrix_ids"]:
        for seed in protocol["seeds"]:
            for policy in protocol["policies"]:
                path = root / matrix / f"seed_{seed}" / policy / "metrics.json"
                if not path.exists():
                    raise ValueError(f"Missing metric required for P3 analysis: {path}")
                data[matrix][seed][policy] = json.loads(path.read_text(encoding="utf-8"))
    return data


def paired_ci(deltas: list[float], *, seed: int, n_boot: int) -> tuple[float, float]:
    arr = np.asarray(deltas, dtype=float)
    if not len(arr):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def policy_rows(data, matrix: str, policy: str) -> tuple[list[int], list[float], list[float]]:
    seeds, baseline, candidate = [], [], []
    for seed, rows in sorted(data[matrix].items()):
        seeds.append(seed)
        baseline.append(team_payoff(rows["het_notom"]))
        candidate.append(team_payoff(rows[policy]))
    return seeds, baseline, candidate


def aggregate(data, protocol: dict) -> dict[str, Any]:
    analysis = protocol["analysis"]
    n_boot = analysis["bootstrap_samples"]
    per_matrix_policy = []
    for matrix in protocol["matrix_ids"]:
        spec = get_spec(matrix)
        for policy in protocol["policies"]:
            seeds, baseline, candidate = policy_rows(data, matrix, policy)
            delta = (np.asarray(candidate) - np.asarray(baseline)).astype(float)
            low, high = paired_ci(delta.tolist(), seed=zlib.crc32(f"{matrix}:{policy}".encode()), n_boot=n_boot)
            per_matrix_policy.append({
                "matrix": matrix, "analysis_category": spec.analysis_category,
                "policy": policy, "policy_display": DISPLAY[policy], "n_paired": len(seeds),
                "seeds": seeds, "baseline_mean": float(np.mean(baseline)),
                "policy_mean": float(np.mean(candidate)), "delta_mean": float(np.mean(delta)),
                "delta_ci95": [low, high], "paired_win_rate": float(np.mean(delta > 0)),
            })
    safe_rows = [row for row in per_matrix_policy if row["policy"] == "het_safe_sca"]
    safety_rows = [{**row, "noninferior_to_noalign": row["delta_ci95"][0] >= -analysis["anti_noninferiority_margin"]}
                   for row in safe_rows if row["analysis_category"] == "anti"]

    route_rows, false_align, anti_total, false_abstain, coord_total = [], 0, 0, 0, 0
    for matrix, seed_map in data.items():
        category = get_spec(matrix).analysis_category
        for seed, cells in seed_map.items():
            safe = cells["het_safe_sca"]
            selected = safe.get("s1_selected_post_warmup_arm")
            is_false_align = category == "anti" and selected == "Gated"
            is_false_abstain = category == "coord_or_boundary" and selected != "Gated"
            if category == "anti":
                anti_total += 1; false_align += int(is_false_align)
            else:
                coord_total += 1; false_abstain += int(is_false_abstain)
            route_rows.append({
                "matrix": matrix, "seed": seed, "analysis_category": category,
                "selected_arm": selected, "false_align": is_false_align,
                "false_abstain": is_false_abstain,
                "decision_reasons": safe.get("s1_decision_reasons", []),
                "coverage": safe.get("s1_coverage", {}),
            })

    recovery = []
    for matrix in protocol["matrix_ids"]:
        spec = get_spec(matrix)
        _, baseline, safe = policy_rows(data, matrix, "het_safe_sca")
        _, _, gated = policy_rows(data, matrix, "het_gated_atom_talk")
        baseline_mean = float(np.mean(baseline))
        gated_gain = float(np.mean(gated) - baseline_mean)
        safe_gain = float(np.mean(safe) - baseline_mean)
        recovery.append({
            "matrix": matrix, "analysis_category": spec.analysis_category,
            "gated_gain_over_noalign": gated_gain, "safe_sca_gain_over_noalign": safe_gain,
            "gain_recovery_fraction": safe_gain / gated_gain if abs(gated_gain) > 1e-12 else None,
        })
    qualified = [row for row in recovery
                 if row["analysis_category"] == "coord_or_boundary"
                 and row["gated_gain_over_noalign"] > 0
                 and row["gain_recovery_fraction"] is not None
                 and row["gain_recovery_fraction"] >= analysis["min_coordination_recovery"]]
    safety_pass = bool(safety_rows) and all(row["noninferior_to_noalign"] for row in safety_rows)
    routing_pass = false_align <= analysis["max_anti_false_align"]
    utility_pass = len(qualified) >= analysis["min_coordination_matrices"]
    return {
        "primary_endpoint": "total_horizon_team_payoff_including_warmup",
        "analysis_config": analysis, "per_matrix_policy": per_matrix_policy,
        "safety_by_anti_matrix": safety_rows, "safety_pass_all_anti_matrices": safety_pass,
        "safe_sca_routing": {
            "anti_false_align": false_align, "anti_total": anti_total,
            "anti_false_align_rate": false_align / anti_total if anti_total else None,
            "coord_false_abstain": false_abstain, "coord_total": coord_total,
            "coord_false_abstain_rate": false_abstain / coord_total if coord_total else None,
            "per_cell": route_rows,
        },
        "routing_pass": routing_pass, "coordination_gain_recovery": recovery,
        "utility_gate": {"qualified_matrices": [row["matrix"] for row in qualified], "passed": utility_pass},
        "method_p3_pass": safety_pass and routing_pass and utility_pass,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# P3 transfer analysis", ""]
    lines.append(f"Primary endpoint: `{report['primary_endpoint']}`.")
    lines.append(f"Anti-matrix safety pass: **{report['safety_pass_all_anti_matrices']}**.")
    lines.append(f"Anti false-align routing pass: **{report['routing_pass']}**.")
    lines.append(f"Utility qualified matrices: {report['utility_gate']['qualified_matrices']}; pass: **{report['utility_gate']['passed']}**.")
    lines.append(f"Overall P3 gate: **{report['method_p3_pass']}**.")
    lines.extend(["", "## Paired payoff against NoAlign", "", "| matrix | category | policy | n | delta | 95% paired CI |", "|---|---|---|---:|---:|---|"])
    for row in report["per_matrix_policy"]:
        low, high = row["delta_ci95"]
        lines.append(f"| {row['matrix']} | {row['analysis_category']} | {row['policy_display']} | {row['n_paired']} | {row['delta_mean']:+.3f} | [{low:+.3f}, {high:+.3f}] |")
    lines.extend(["", "## Routing", ""])
    routing = report["safe_sca_routing"]
    lines.append(f"- Anti false-align: **{routing['anti_false_align']}/{routing['anti_total']}**.")
    lines.append(f"- Coordination/boundary false-abstain: **{routing['coord_false_abstain']}/{routing['coord_total']}**.")
    lines.extend(["", "## Gain recovery", "", "| matrix | Gated-NoAlign | Safe-SCA-NoAlign | recovery |", "|---|---:|---:|---:|"])
    for row in report["coordination_gain_recovery"]:
        recovery = "n/a" if row["gain_recovery_fraction"] is None else f"{row['gain_recovery_fraction']:.1%}"
        lines.append(f"| {row['matrix']} | {row['gated_gain_over_noalign']:+.3f} | {row['safe_sca_gain_over_noalign']:+.3f} | {recovery} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen P3 transfer results")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol.resolve())
    root = args.results.resolve()
    integrity = root / "P3_INTEGRITY_REPORT.json"
    if not integrity.exists() or not json.loads(integrity.read_text(encoding="utf-8")).get("ready_for_analysis"):
        raise SystemExit("Run validate_p3_results.py successfully before P3 analysis.")
    report = aggregate(load(root, protocol), protocol)
    (root / "p3_transfer_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    (root / "p3_transfer_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    if not report["method_p3_pass"]:
        raise SystemExit("P3 gates did not all pass; report every matrix and do not claim transfer.")


if __name__ == "__main__":
    main()
