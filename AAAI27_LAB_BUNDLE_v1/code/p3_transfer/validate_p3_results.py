#!/usr/bin/env python3
"""Fail-closed completeness and frozen-configuration validator for P3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from p3_protocol import load_protocol, sha256_json


def expected_metric_paths(root: Path, protocol: dict) -> list[Path]:
    return [root / matrix / f"seed_{seed}" / policy / "metrics.json"
            for matrix in protocol["matrix_ids"]
            for seed in protocol["seeds"]
            for policy in protocol["policies"]]


def validate(root: Path, protocol: dict) -> dict[str, Any]:
    errors: list[str] = []
    missing: list[str] = []
    checked = 0
    safe_config_errors: list[str] = []
    for path in expected_metric_paths(root, protocol):
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Unreadable metrics {path}: {exc}")
            continue
        checked += 1
        if path.parent.name == "het_safe_sca":
            actual = metrics.get("s1_safe_config")
            if actual != protocol["safe_sca"]:
                safe_config_errors.append(f"{path}: frozen Safe-SCA config differs")
            if metrics.get("s1_policy") != "het_safe_sca":
                safe_config_errors.append(f"{path}: s1_policy is not het_safe_sca")
            if metrics.get("s1_selected_post_warmup_arm") not in {"NoAlign", "Gated"}:
                safe_config_errors.append(f"{path}: invalid/missing Safe-SCA selected arm")
    snapshot_path = root / "P3_CAMPAIGN_SNAPSHOT.json"
    snapshot_errors: list[str] = []
    if not snapshot_path.exists():
        snapshot_errors.append(f"Missing {snapshot_path}")
    else:
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if snapshot.get("protocol_sha256") != sha256_json(protocol):
                snapshot_errors.append("campaign snapshot protocol hash differs from supplied protocol")
            if snapshot.get("matrix_registry_sha256") != protocol["matrix_registry_sha256"]:
                snapshot_errors.append("campaign snapshot matrix registry hash differs from protocol")
        except (OSError, json.JSONDecodeError) as exc:
            snapshot_errors.append(f"Unreadable campaign snapshot: {exc}")
    expected = len(expected_metric_paths(root, protocol))
    report = {
        "schema_version": 1,
        "root": str(root),
        "expected_cells": expected,
        "checked_metrics": checked,
        "missing_count": len(missing),
        "missing": missing,
        "error_count": len(errors) + len(safe_config_errors) + len(snapshot_errors),
        "errors": errors + safe_config_errors + snapshot_errors,
        "ready_for_analysis": not missing and not errors and not safe_config_errors and not snapshot_errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate P3 transfer outputs")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol.resolve())
    report = validate(args.results.resolve(), protocol)
    target = args.results.resolve() / "P3_INTEGRITY_REPORT.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[wrote] {target}")
    if not report["ready_for_analysis"]:
        raise SystemExit("P3 integrity validation failed; do not analyze partial results.")


if __name__ == "__main__":
    main()
