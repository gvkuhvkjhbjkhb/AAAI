#!/usr/bin/env python3
"""Integrity checks for P0/P1/P2 results; no statistical interpretation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_supplement_campaign import OUTPUT_NAMES, contexts_for, metric_path, policies_for
from supplement_protocol import load_protocol, read_json


def check_metric(path: Path, protocol: dict, experiment: str, domain: str,
                 context: str, seed: int, policy: str) -> list[str]:
    errors: list[str] = []
    try:
        metric = read_json(path)
    except Exception as exc:  # report every corrupt file in one pass
        return [f"{path}: invalid JSON: {exc}"]
    value = metric.get("team_mean_payoff")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        value = metric.get("s1_total_team_payoff")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        errors.append(f"{path}: missing/non-finite team_mean_payoff")
    if metric.get("cell") != policy:
        errors.append(f"{path}: cell={metric.get('cell')!r}, expected {policy!r}")
    config = metric.get("config", {})
    if config.get("game") != context or int(config.get("seed", -1)) != seed:
        errors.append(f"{path}: config game/seed mismatch")
    if float(config.get("top_p", -1)) != protocol["sampling"]["top_p"]:
        errors.append(f"{path}: top_p mismatch")
    if int(config.get("gen_seed_base", -1)) != protocol["sampling"]["gen_seed_base"]:
        errors.append(f"{path}: gen_seed_base mismatch")
    if experiment == "p0":
        if policy == "het_payoff_prompt":
            if config.get("payoff_in_prompt") is not True:
                errors.append(f"{path}: payoff-in-prompt arm must expose the matrix")
            if any(config.get(key) for key in ("use_tom", "use_talk", "adaptive_tom")):
                errors.append(f"{path}: payoff-in-prompt arm accidentally enabled ToM/talk")
        elif config.get("payoff_in_prompt", False):
            errors.append(f"{path}: fixed-arm control must not expose the payoff table")
        if policy == "het_safe_sca" and not path.with_name("decision.json").exists():
            errors.append(f"{path.with_name('decision.json')}: missing concurrent Safe-SCA decision")
    elif experiment == "p1":
        if config.get("payoff_in_prompt", False):
            errors.append(f"{path}: P1 must not expose payoff table")
        if policy == "het_safe_sca":
            decision = path.with_name("decision.json")
            if not decision.exists():
                errors.append(f"{decision}: missing Safe-SCA decision")
            else:
                d = read_json(decision)
                if d.get("oracle_label_used"):
                    errors.append(f"{decision}: deployable controller used oracle label")
    else:
        if metric.get("bandit_selection_endpoint") != "team_mean_payoff":
            errors.append(f"{path}: bandit selection endpoint mismatch")
        if int(metric.get("bandit_k", -1)) != protocol["p2"]["bandit_k"]:
            errors.append(f"{path}: bandit K mismatch")
        if metric.get("bandit_chosen_arm") not in protocol["p2"]["arms"]:
            errors.append(f"{path}: invalid bandit arm")
        k = protocol["p2"]["bandit_k"]
        p0 = metric.get("bandit_probe_payoffs_NoAlign", [])
        p1 = metric.get("bandit_probe_payoffs_Gated", [])
        if len(p0) != k or len(p1) != k:
            errors.append(f"{path}: expected {k} probe episodes per arm")
        else:
            n_commit = int(metric.get("bandit_n_commit", -1))
            n_total = int(metric.get("bandit_total_episode_count", -1))
            expected = (sum(p0) + sum(p1) + n_commit * float(value)) / n_total
            observed = float(metric.get("bandit_online_total_team_mean_payoff", math.nan))
            if not math.isclose(expected, observed, abs_tol=1e-12):
                errors.append(f"{path}: online total does not include probes correctly")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate P0/P1/P2 result integrity")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", choices=["p0", "p1", "p2"],
                        default=["p0", "p1", "p2"])
    parser.add_argument("--include-source-p2", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol.resolve())
    root = args.results_root.resolve()
    blocks: list[tuple[str, str]] = []
    for experiment in dict.fromkeys(args.experiments):
        blocks.append((experiment, "p3"))
        if experiment == "p2" and args.include_source_p2:
            blocks.append(("p2", "source"))
    missing: list[str] = []
    errors: list[str] = []
    checked = 0
    expected = 0
    per_block = {}
    for experiment, domain in blocks:
        contexts, seeds = contexts_for(protocol, domain)
        block_expected = len(contexts) * len(seeds) * len(policies_for(protocol, experiment))
        block_checked = 0
        expected += block_expected
        for context in contexts:
            visibility = root / OUTPUT_NAMES[(experiment, domain)] / context / f"{experiment.upper()}_VISIBILITY.json"
            if not visibility.exists():
                missing.append(str(visibility))
            for seed in seeds:
                for policy in policies_for(protocol, experiment):
                    path = metric_path(root, protocol, experiment, domain, context, seed, policy)
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    checked += 1
                    block_checked += 1
                    errors.extend(check_metric(path, protocol, experiment, domain,
                                               context, seed, policy))
        per_block[f"{experiment}/{domain}"] = {
            "expected_metrics": block_expected, "checked_metrics": block_checked,
        }
    manifest = root / "ENVIRONMENT_MANIFEST_S1.json"
    if not manifest.exists():
        missing.append(str(manifest))
    else:
        m = read_json(manifest)
        if not m.get("preflight_passed") or m.get("allow_version_mismatch"):
            errors.append("Environment manifest is not a strict passing preflight")
    report = {
        "schema_version": 1, "expected_metrics": expected,
        "checked_metrics": checked, "missing": missing, "errors": errors,
        "per_block": per_block,
        "ready_for_analysis": checked == expected and not missing and not errors,
    }
    output = root / "SUPPLEMENT_INTEGRITY_REPORT.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready_for_analysis"]:
        raise SystemExit("Integrity validation failed; preserve all files and fix execution gaps")


if __name__ == "__main__":
    main()
