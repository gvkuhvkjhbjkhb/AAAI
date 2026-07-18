#!/usr/bin/env python3
"""Launch the preregistered S1 Coverage-Certified Safe-SCA experiment.

The experiment is deliberately two-stage:

1. ``dev`` records label-free NoAlign warm-up trajectories on development
   seeds 42--51.  ``select_s1_config.py`` may use those *development* labels
   to freeze exactly one Safe-SCA configuration.
2. ``test`` evaluates all policies on fresh seeds 62--81.  Test execution
   refuses to start without the frozen JSON configuration and never passes an
   oracle label into ``het_safe_sca`` or ``het_point_sca``.

Each test cell has 30 episodes.  Safe/point/oracle policies first run the
same NoAlign warm-up and retain the same agent objects for their commit phase;
all warm-up payoffs are part of the primary total-horizon metric.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from safe_sca import load_config


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("GSACA_ROOT", HERE.parent)).resolve()
RUNNER = ROOT / "code" / "run_experiment_local.py"
DEFAULT_RESULTS = ROOT / "v2_results"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
GLM = "THUDM/GLM-4-9B-0414"
GAMES = ["chicken", "deadlock", "hawk_dove", "stag_hunt", "battle_of_the_sexes", "public_goods"]
DEV_SEEDS = list(range(42, 52))
TEST_SEEDS = list(range(62, 82))
TEST_CELLS = [
    "het_notom",             # deployment safety baseline
    "het_gated_atom_talk",   # always-align risk baseline
    "het_gsaca",             # existing integrated conditional mechanism
    "het_point_sca",         # unconstrained point-estimate two-arm rule
    "het_safe_sca",          # proposed label-free method
    "het_oracle_sca",        # marked diagnostic upper bound only
]
LAB_STACK_B_TARGET = {
    "hardware": "2x NVIDIA RTX 5090 (32GB, Blackwell sm_120)",
    "vllm": "0.25.1",
    "pytorch": "2.11.0+cu128",
    "transformers": "5.14.1",
    "precision": "bf16",
    "qwen_endpoint": "GPU0:8000",
    "glm_endpoint": "GPU1:8001",
}


def shard(items: list[int], n_shards: int) -> list[list[int]]:
    return [items[index::n_shards] for index in range(n_shards) if items[index::n_shards]]


def safe_flags(config_path: Path) -> list[str]:
    cfg = load_config(config_path)
    return [
        "--safe_warmup", str(cfg.warmup_episodes),
        "--safe_tau", str(cfg.tau),
        "--safe_confidence", str(cfg.confidence),
        "--safe_bootstrap_samples", str(cfg.bootstrap_samples),
        "--safe_min_profile_coverage", str(cfg.min_profile_coverage),
        "--safe_min_stratum_observations", str(cfg.min_stratum_observations),
    ]


def immutable_json(path: Path, payload: dict) -> None:
    """Write a configuration snapshot once; reject accidental drift on resume."""
    canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != canonical:
            raise RuntimeError(
                f"Existing immutable snapshot differs: {path}. Use a new output directory."
            )
        return
    path.write_text(canonical, encoding="utf-8")


def require_successful_preflight(output: Path) -> None:
    """Refuse a GPU launch without a strict, recorded environment check."""
    manifest_path = output / "ENVIRONMENT_MANIFEST_S1.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"S1 preflight manifest missing: {manifest_path}. "
            "Run preflight_s1.py against this output directory first."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable S1 preflight manifest: {manifest_path}: {exc}") from exc
    if not manifest.get("preflight_passed"):
        raise RuntimeError(f"S1 preflight did not pass: {manifest_path}")
    if manifest.get("allow_version_mismatch"):
        raise RuntimeError(
            "S1 preflight used --allow-version-mismatch. "
            "Do not use this result directory for the preregistered run."
        )


def base_command() -> list[str]:
    return [
        sys.executable, str(RUNNER), "--use_vllm", "--gen_seed_base", "1000",
        "--episodes", "30", "--horizon", "5", "--memory", "2", "--log_every", "100",
        "--models_het", QWEN, GLM, "--gate_trust_threshold", "0.6",
        "--gate_ema_alpha", "0.3", "--atom_warmup", "3", "--top_p", "0.9",
        "--latin_square",
    ]


def launch(args: argparse.Namespace, *, phase: str, output: Path, cells: list[str], seeds: list[int], extra: list[str]) -> None:
    snapshot = {
        "schema_version": 1,
        "phase": phase,
        "games": GAMES,
        "seeds": seeds,
        "cells": cells,
        "episodes": 30,
        "horizon": 5,
        "memory": 2,
        "top_p": 0.9,
        "latin_square": True,
        "models_het": [QWEN, GLM],
        "lab_stack_b_target": LAB_STACK_B_TARGET,
        "safe_config_flags": extra,
    }
    commands: list[tuple[int, list[str], Path]] = []
    for index, seed_shard in enumerate(shard(seeds, args.workers)):
        command = base_command() + extra + ["--games", *GAMES, "--seeds", *map(str, seed_shard),
                                             "--cells", *cells, "--out_dir", str(output)]
        log_path = output / "logs" / f"s1_{phase}_shard{index}.log"
        commands.append((index, command, log_path))

    print(f"[S1/{phase}] {len(GAMES)} games x {len(seeds)} seeds x {len(cells)} cells = "
          f"{len(GAMES) * len(seeds) * len(cells)} cells; {len(commands)} worker shards")
    if args.dry_run:
        print(json.dumps({"would_write_config_snapshot": snapshot}, indent=2, sort_keys=True))
        for _, command, _ in commands:
            print(" ".join(command))
        return

    require_successful_preflight(output)
    immutable_json(output / "CONFIG_SNAPSHOT_S1.json", snapshot)

    processes: list[tuple[int, subprocess.Popen, object, Path]] = []
    for index, command, log_path in commands:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(log_path, "w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((index, process, handle, log_path))
    started = time.time()
    for completed, (index, process, handle, log_path) in enumerate(processes, start=1):
        return_code = process.wait()
        handle.close()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {completed}/{len(processes)} "
              f"shard={index} rc={return_code} elapsed={time.time() - started:.0f}s log={log_path}")
        if return_code != 0:
            raise RuntimeError(f"S1 worker {index} failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S1 Coverage-Certified Safe-SCA launcher")
    parser.add_argument("--phase", choices=["dev", "test"], required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--safe-config", type=Path,
                        help="Frozen JSON from select_s1_config.py; required for test")
    parser.add_argument("--dev-seeds", type=int, nargs="+", default=DEV_SEEDS)
    parser.add_argument("--test-seeds", type=int, nargs="+", default=TEST_SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    if not RUNNER.exists():
        parser.error(f"runner not found: {RUNNER}")

    if args.phase == "dev":
        # A NoAlign-only observer run prevents labels or intervention state from
        # leaking into the threshold-selection input.
        launch(args, phase="dev", output=args.results_root / "exp_s1_dev_warmup",
               cells=["het_notom"], seeds=args.dev_seeds, extra=[])
        return

    if args.safe_config is None:
        parser.error("--safe-config is required for held-out test execution")
    config_path = args.safe_config.resolve()
    if not config_path.exists():
        parser.error(f"safe config not found: {config_path}")
    flags = safe_flags(config_path)
    launch(args, phase="test", output=args.results_root / "exp_s1_safe_sca_test",
           cells=TEST_CELLS, seeds=args.test_seeds, extra=flags)


if __name__ == "__main__":
    main()
