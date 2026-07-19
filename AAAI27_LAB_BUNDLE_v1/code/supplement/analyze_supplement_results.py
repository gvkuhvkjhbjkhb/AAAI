#!/usr/bin/env python3
"""Frozen paired analysis for P0 payoff prompt, P1 labels, and P2 bandit."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from p3_matrices import get_spec
from supplement_protocol import load_protocol, read_json


NOALIGN = "het_notom"
GATED = "het_gated_atom_talk"
SAFE = "het_safe_sca"


def _team_mean_payoff(data: dict) -> float:
    v = data.get("team_mean_payoff")
    if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        v = data.get("s1_total_team_payoff")
    return float(v)


def metric(root: Path, context: str, seed: int, policy: str) -> dict:
    path = root / context / f"seed_{seed}" / policy / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def decision(root: Path, context: str, seed: int, policy: str = SAFE) -> dict:
    path = root / context / f"seed_{seed}" / policy / "decision.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def payoff_range(matrix_id: str) -> float:
    spec = get_spec(matrix_id)
    teams = [float(np.mean(pair)) for row in spec.payoff_matrix for pair in row]
    span = max(teams) - min(teams)
    if span <= 0:
        raise ValueError(f"Degenerate team-payoff range for {matrix_id}")
    return span


def bootstrap_mean_ci(values: Iterable[float], draws: int, seed: int,
                      confidence: float = 0.95) -> tuple[float, float, float, float]:
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, size=(draws, x.size), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
    return float(x.mean()), float(lo), float(hi), float(p)


def hierarchical_ci(rows: list[dict], value_key: str, draws: int, seed: int,
                    confidence: float = 0.95) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["context"], []).append(float(row[value_key]))
    names = sorted(grouped)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for b in range(draws):
        selected = rng.choice(names, size=len(names), replace=True)
        values = []
        for name in selected:
            cell = np.asarray(grouped[str(name)], dtype=float)
            values.extend(rng.choice(cell, size=cell.size, replace=True).tolist())
        samples[b] = float(np.mean(values))
    alpha = 1.0 - confidence
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    observed = float(np.mean([r[value_key] for r in rows]))
    return observed, float(lo), float(hi)


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [1.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (n - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [centre - half, centre + half]


def analyze_p0(protocol: dict, supplement_root: Path, reference: Path,
               draws: int) -> tuple[dict, list[dict]]:
    root = supplement_root / "p0_payoff_prompt"
    rows = []
    for context in protocol["p3"]["matrix_ids"]:
        span = payoff_range(context)
        category = get_spec(context).analysis_category
        for seed in protocol["p3"]["seeds"]:
            pp = _team_mean_payoff(metric(root, context, seed, "het_payoff_prompt"))
            no = _team_mean_payoff(metric(root, context, seed, NOALIGN))
            gated = _team_mean_payoff(metric(root, context, seed, GATED))
            safe = _team_mean_payoff(metric(root, context, seed, SAFE))
            old_no = _team_mean_payoff(metric(reference, context, seed, NOALIGN))
            old_gated = _team_mean_payoff(metric(reference, context, seed, GATED))
            rows.append({
                "experiment": "p0", "context": context, "seed": seed,
                "category": category, "payoff_prompt": pp, "noalign": no,
                "gated": gated, "effect_vs_noalign": pp - no,
                "effect_vs_gated": pp - gated,
                "effect_vs_safe": pp - safe,
                "regret_to_better_fixed_arm": max(no, gated) - pp,
                "normalized_effect_vs_noalign": (pp - no) / span,
                "temporal_replay_drift_noalign": no - old_no,
                "temporal_replay_drift_gated": gated - old_gated,
            })
    matrices = []
    for i, context in enumerate(protocol["p3"]["matrix_ids"]):
        subset = [r for r in rows if r["context"] == context]
        mean, lo, hi, p = bootstrap_mean_ci(
            [r["effect_vs_noalign"] for r in subset], draws, 1000 + i)
        regret, rlo, rhi, _ = bootstrap_mean_ci(
            [r["regret_to_better_fixed_arm"] for r in subset], draws, 2000 + i)
        matrices.append({
            "matrix": context, "category": subset[0]["category"],
            "n": len(subset), "effect_vs_noalign": mean,
            "effect_vs_noalign_ci95": [lo, hi], "p_unadjusted": p,
            "regret_to_better_fixed_arm": regret,
            "regret_ci95": [rlo, rhi],
        })
    adjusted = holm_adjust([m["p_unadjusted"] for m in matrices])
    for item, p_adj in zip(matrices, adjusted):
        item["p_holm"] = p_adj
    overall_no = hierarchical_ci(rows, "effect_vs_noalign", draws, 3101)
    overall_gated = hierarchical_ci(rows, "effect_vs_gated", draws, 3102)
    overall_safe = hierarchical_ci(rows, "effect_vs_safe", draws, 3106)
    overall_regret = hierarchical_ci(rows, "regret_to_better_fixed_arm", draws, 3103)
    drift_no = hierarchical_ci(rows, "temporal_replay_drift_noalign", draws, 3104)
    drift_gated = hierarchical_ci(rows, "temporal_replay_drift_gated", draws, 3105)
    return {
        "question": "Does full payoff information close the utility gap on the exact P3 grid?",
        "primary_endpoint": "team_mean_payoff",
        "n_cells": len(rows), "matrices": matrices,
        "hierarchical_effect_vs_noalign": {"mean": overall_no[0], "ci95": list(overall_no[1:])},
        "hierarchical_effect_vs_gated": {"mean": overall_gated[0], "ci95": list(overall_gated[1:])},
        "hierarchical_effect_vs_safe_sca": {"mean": overall_safe[0], "ci95": list(overall_safe[1:])},
        "hierarchical_regret_to_better_fixed_arm": {"mean": overall_regret[0], "ci95": list(overall_regret[1:])},
        "temporal_sensitivity_vs_original_p3": {
            "noalign_drift": {"mean": drift_no[0], "ci95": list(drift_no[1:])},
            "gated_drift": {"mean": drift_gated[0], "ci95": list(drift_gated[1:])},
            "primary_contrasts_use_concurrent_reruns": True
        },
        "interpretation_rule": (
            "Report as an external baseline regardless of sign. A positive CI versus NoAlign "
            "supports payoff visibility; nonpositive results rule out missing payoff text as a "
            "sufficient explanation for P3 utility failure."
        ),
    }, rows


def analyze_p1(protocol: dict, supplement_root: Path, reference: Path,
               draws: int) -> tuple[dict, list[dict]]:
    root = supplement_root / "p1_label_swap"
    original_root = supplement_root / "p0_payoff_prompt"
    rows = []
    route_matches = 0
    for context in protocol["p3"]["matrix_ids"]:
        span = payoff_range(context)
        category = get_spec(context).analysis_category
        for seed in protocol["p3"]["seeds"]:
            original = {p: _team_mean_payoff(metric(original_root, context, seed, p))
                        for p in (NOALIGN, GATED, SAFE)}
            swapped = {p: _team_mean_payoff(metric(root, context, seed, p))
                       for p in (NOALIGN, GATED, SAFE)}
            route_o = decision(original_root, context, seed).get("selected_post_warmup_arm")
            route_s = decision(root, context, seed).get("selected_post_warmup_arm")
            route_match = route_o == route_s
            route_matches += int(route_match)
            rows.append({
                "experiment": "p1", "context": context, "seed": seed,
                "category": category, "route_original": route_o,
                "route_swapped": route_s, "route_match": route_match,
                "gated_interaction": ((swapped[GATED] - swapped[NOALIGN]) -
                                      (original[GATED] - original[NOALIGN])),
                "safe_interaction": ((swapped[SAFE] - swapped[NOALIGN]) -
                                     (original[SAFE] - original[NOALIGN])),
                "gated_interaction_normalized": (((swapped[GATED] - swapped[NOALIGN]) -
                                                   (original[GATED] - original[NOALIGN])) / span),
                "safe_interaction_normalized": (((swapped[SAFE] - swapped[NOALIGN]) -
                                                  (original[SAFE] - original[NOALIGN])) / span),
            })
    matrices = []
    margin = float(protocol["analysis"]["p1_normalized_equivalence_margin"])
    for i, context in enumerate(protocol["p3"]["matrix_ids"]):
        subset = [r for r in rows if r["context"] == context]
        item = {"matrix": context, "category": subset[0]["category"], "n": len(subset)}
        for j, policy in enumerate(("gated", "safe")):
            key = f"{policy}_interaction_normalized"
            mean, lo, hi, _ = bootstrap_mean_ci([r[key] for r in subset], draws,
                                                 4000 + 100 * i + j)
            item[f"{policy}_normalized_interaction"] = mean
            item[f"{policy}_normalized_interaction_ci95"] = [lo, hi]
            item[f"{policy}_equivalent_within_margin"] = lo > -margin and hi < margin
        item["route_agreement"] = sum(r["route_match"] for r in subset) / len(subset)
        matrices.append(item)
    route_rate = route_matches / len(rows)
    all_effects_equivalent = all(
        m["gated_equivalent_within_margin"] and m["safe_equivalent_within_margin"]
        for m in matrices
    )
    gate_pass = (route_rate >= protocol["analysis"]["p1_min_route_agreement"] and
                 all_effects_equivalent)
    return {
        "question": "Are policy effects and Safe-SCA routes robust to an action-label permutation?",
        "n_context_seed_pairs": len(rows), "route_matches": route_matches,
        "route_agreement": route_rate,
        "route_agreement_ci95_wilson": wilson(route_matches, len(rows)),
        "normalized_equivalence_margin": margin,
        "matrices": matrices,
        "label_robustness_gate_pass": gate_pass,
        "interpretation_rule": (
            "Only if the preregistered route and normalized effect-interaction gates pass may "
            "the paper describe P3 arm effects as robust to the tested label surface. Failure "
            "must retain action labels as an unresolved moderator."
        ),
    }, rows


def analyze_p2_domain(protocol: dict, bandit_root: Path, reference: Path,
                      contexts: list[str], seeds: list[int], draws: int,
                      domain: str) -> tuple[dict, list[dict]]:
    rows = []
    for context in contexts:
        for seed in seeds:
            b = metric(bandit_root, context, seed, "het_bandit_teammean")
            no = _team_mean_payoff(metric(reference, context, seed, NOALIGN))
            gated = _team_mean_payoff(metric(reference, context, seed, GATED))
            best = "Gated" if gated > no else "NoAlign"
            chosen = b["bandit_chosen_arm"]
            chosen_reference = gated if chosen == "Gated" else no
            online = float(b["bandit_online_total_team_mean_payoff"])
            rows.append({
                "experiment": "p2", "domain": domain, "context": context,
                "seed": seed, "best_reference_arm": best, "chosen_arm": chosen,
                "selection_correct": chosen == best,
                "reference_selection_regret": max(no, gated) - chosen_reference,
                "online_total_team_mean_payoff": online,
                "online_gain_vs_reference_noalign": online - no,
                "online_regret_to_better_reference_arm": max(no, gated) - online,
            })
    correct = sum(r["selection_correct"] for r in rows)
    regret = bootstrap_mean_ci([r["reference_selection_regret"] for r in rows], draws, 6001)
    online_gain = bootstrap_mean_ci([r["online_gain_vs_reference_noalign"] for r in rows], draws, 6002)
    per_context = []
    for i, context in enumerate(contexts):
        subset = [r for r in rows if r["context"] == context]
        gain = bootstrap_mean_ci([r["online_gain_vs_reference_noalign"] for r in subset],
                                 draws, 6100 + i)
        c = sum(r["selection_correct"] for r in subset)
        per_context.append({
            "context": context, "n": len(subset), "selection_accuracy": c / len(subset),
            "selection_accuracy_ci95_wilson": wilson(c, len(subset)),
            "online_gain_vs_reference_noalign": gain[0],
            "online_gain_vs_reference_noalign_ci95": [gain[1], gain[2]],
        })
    return {
        "domain": domain, "n_cells": len(rows), "correct_selections": correct,
        "selection_accuracy": correct / len(rows),
        "selection_accuracy_ci95_wilson": wilson(correct, len(rows)),
        "mean_reference_selection_regret": regret[0],
        "reference_selection_regret_ci95": [regret[1], regret[2]],
        "online_gain_vs_reference_noalign": online_gain[0],
        "online_gain_vs_reference_noalign_ci95": [online_gain[1], online_gain[2]],
        "per_context": per_context,
        "interpretation_rule": (
            "Use online total payoff including all 10 probe episodes as the deployment result; "
            "commit-only payoff is diagnostic and may not replace it."
        ),
    }, rows


def markdown_report(result: dict) -> str:
    p0, p1 = result["p0"], result["p1"]
    lines = [
        "# AAAI-27 supplemental controls: frozen analysis", "",
        "Primary endpoint throughout: `team_mean_payoff`.", "",
        "## P0: same-grid payoff-in-prompt", "",
        f"- Cells: {p0['n_cells']}",
        f"- Hierarchical effect vs NoAlign: {p0['hierarchical_effect_vs_noalign']['mean']:.4f} "
        f"{p0['hierarchical_effect_vs_noalign']['ci95']}",
        f"- Regret to better fixed arm: {p0['hierarchical_regret_to_better_fixed_arm']['mean']:.4f} "
        f"{p0['hierarchical_regret_to_better_fixed_arm']['ci95']}", "",
        "| Matrix | Type | PayoffPrompt-NoAlign (95% CI) | Holm p | Regret |",
        "|---|---|---:|---:|---:|",
    ]
    for m in p0["matrices"]:
        lines.append(f"| {m['matrix']} | {m['category']} | {m['effect_vs_noalign']:.4f} "
                     f"[{m['effect_vs_noalign_ci95'][0]:.4f}, {m['effect_vs_noalign_ci95'][1]:.4f}] "
                     f"| {m['p_holm']:.4f} | {m['regret_to_better_fixed_arm']:.4f} |")
    lines += ["", "## P1: action-label permutation", "",
              f"- Route agreement: {p1['route_matches']}/{p1['n_context_seed_pairs']} "
              f"({p1['route_agreement']:.3f})",
              f"- Full label-robustness gate: **{p1['label_robustness_gate_pass']}**", "",
              "| Matrix | Route agreement | Gated normalized interaction CI | Safe normalized interaction CI |",
              "|---|---:|---:|---:|"]
    for m in p1["matrices"]:
        lines.append(f"| {m['matrix']} | {m['route_agreement']:.2f} | "
                     f"{m['gated_normalized_interaction_ci95']} | "
                     f"{m['safe_normalized_interaction_ci95']} |")
    for key, value in result.get("p2", {}).items():
        lines += ["", f"## P2: online probe ({key})", "",
                  f"- Selection accuracy: {value['selection_accuracy']:.3f} "
                  f"{value['selection_accuracy_ci95_wilson']}",
                  f"- Online gain vs NoAlign (includes probes): "
                  f"{value['online_gain_vs_reference_noalign']:.4f} "
                  f"{value['online_gain_vs_reference_noalign_ci95']}"]
    lines += ["", "All matrices/contexts are retained; no gate failure deletes an experiment.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen supplemental controls")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--p3-reference-root", type=Path, required=True,
                        help="extracted exp_p3_transfer_test root")
    parser.add_argument("--source-reference-root", type=Path,
                        help="extracted S2 root; required only with --include-source-p2")
    parser.add_argument("--include-source-p2", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol.resolve())
    root = args.results_root.resolve()
    integrity = root / "SUPPLEMENT_INTEGRITY_REPORT.json"
    if not integrity.exists() or not read_json(integrity).get("ready_for_analysis"):
        raise SystemExit("Run validate_supplement_results.py successfully before analysis")
    draws = int(protocol["analysis"]["bootstrap_samples"])
    p3_reference = args.p3_reference_root.resolve()
    p0, rows0 = analyze_p0(protocol, root, p3_reference, draws)
    p1, rows1 = analyze_p1(protocol, root, p3_reference, draws)
    p2 = {}
    rows2: list[dict] = []
    p2_p3_root = root / "p2_teammean_bandit_p3"
    if p2_p3_root.exists():
        p2["p3"] , block_rows = analyze_p2_domain(
            protocol, p2_p3_root, p3_reference,
            protocol["p3"]["matrix_ids"], protocol["p3"]["seeds"], draws, "p3")
        rows2.extend(block_rows)
    if args.include_source_p2:
        if args.source_reference_root is None:
            parser.error("--source-reference-root is required with --include-source-p2")
        p2["source"], block_rows = analyze_p2_domain(
            protocol, root / "p2_teammean_bandit_source", args.source_reference_root.resolve(),
            protocol["source"]["games"], protocol["source"]["seeds"], draws, "source")
        rows2.extend(block_rows)
    result = {
        "schema_version": 1, "primary_endpoint": "team_mean_payoff",
        "bootstrap_samples": draws, "p0": p0, "p1": p1, "p2": p2,
        "claim_guardrail": (
            "P3 estimates policy-arm effects within matrix-seed contexts. It does not identify "
            "the payoff tensor as the sole cause of differences between source and P3 grids."
        ),
    }
    (root / "SUPPLEMENT_ANALYSIS.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (root / "SUPPLEMENT_ANALYSIS.md").write_text(markdown_report(result), encoding="utf-8")
    csv_rows = rows0 + rows1 + rows2
    fieldnames = sorted({key for row in csv_rows for key in row})
    with (root / "SUPPLEMENT_CELL_LEVEL.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
