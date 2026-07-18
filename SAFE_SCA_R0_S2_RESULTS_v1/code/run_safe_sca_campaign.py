#!/usr/bin/env python3
"""Run an immutable Safe-SCA replay or independent-replication campaign.

This launcher is intentionally generic: it runs either the 72-cell R0 replay
or a fresh S2 seed block without editing the frozen S1 decision rule.  Before
it starts work, the exact output directory must already contain a successful
``ENVIRONMENT_MANIFEST_S1.json`` written by ``preflight_s1.py``.

Each task owns one ``(game, seed)`` pair and executes all six policies in the
seed-keyed Latin-square order.  This preserves the policy-order protocol while
allowing the two vLLM servers to batch requests from many independent tasks.
Existing ``metrics.json`` files are respected by ``run_experiment_local.py``,
so interrupted campaigns can resume safely without a force overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


GAMES = [
    "chicken", "deadlock", "hawk_dove",
    "stag_hunt", "battle_of_the_sexes", "public_goods",
]
CELLS = [
    "het_notom",             # NoAlign
    "het_gated_atom_talk",   # Always-Gated
    "het_gsaca",             # legacy integrated comparator
    "het_point_sca",         # label-free point-estimate comparator
    "het_safe_sca",          # proposed deployable controller
    "het_oracle_sca",        # diagnostic only; never a deployable result
]
QWEN = "Qwen/Qwen2.5-7B-Instruct"
GLM = "THUDM/GLM-4-9B-0414"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read JSON: {path}: {exc}") from exc


def require_strict_preflight(output: Path) -> None:
    """Fail closed unless preflight passed with no version override."""
    manifest_path = output / "ENVIRONMENT_MANIFEST_S1.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Missing preflight manifest: {manifest_path}. Run preflight_s1.py "
            "with --out-dir set to this exact campaign output directory first."
        )
    manifest = read_json(manifest_path)
    if not manifest.get("preflight_passed"):
        raise RuntimeError(f"Preflight did not pass: {manifest_path}: {manifest.get('failures')}")
    if manifest.get("allow_version_mismatch"):
        raise RuntimeError(
            "Preflight used a version override. This is not allowed for R0/S2 confirmation."
        )


def immutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Write once; a resume must be byte-for-byte protocol identical."""
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old != rendered:
            raise RuntimeError(
                f"Campaign snapshot differs: {path}. Refuse to mix protocols; use a new results root."
            )
        return
    path.write_text(rendered, encoding="utf-8")


def safe_flags(frozen: dict[str, Any]) -> list[str]:
    required = [
        "warmup_episodes", "tau", "confidence", "bootstrap_samples",
        "min_profile_coverage", "min_stratum_observations",
    ]
    missing = [key for key in required if key not in frozen]
    if missing:
        raise RuntimeError(f"Frozen configuration missing keys: {missing}")
    return [
        "--safe_warmup", str(frozen["warmup_episodes"]),
        "--safe_tau", str(frozen["tau"]),
        "--safe_confidence", str(frozen["confidence"]),
        "--safe_bootstrap_samples", str(frozen["bootstrap_samples"]),
        "--safe_min_profile_coverage", str(frozen["min_profile_coverage"]),
        "--safe_min_stratum_observations", str(frozen["min_stratum_observations"]),
    ]


def build_base_command(runner: Path, frozen: dict[str, Any]) -> list[str]:
    return [
        sys.executable, str(runner),
        "--use_vllm", "--gen_seed_base", "1000",
        "--episodes", "30", "--horizon", "5", "--memory", "2", "--log_every", "100",
        "--models_het", QWEN, GLM,
        "--gate_trust_threshold", "0.6", "--gate_ema_alpha", "0.3",
        "--atom_warmup", "3", "--top_p", "0.9", "--latin_square",
        *safe_flags(frozen),
    ]


def cell_complete(output: Path, game: str, seed: int, cell: str) -> bool:
    return (output / game / f"seed_{seed}" / cell / "metrics.json").exists()


def task_complete(output: Path, game: str, seed: int, cells: list[str]) -> bool:
    return all(cell_complete(output, game, seed, cell) for cell in cells)


def count_completed_cells(output: Path, games: list[str], seeds: list[int], cells: list[str]) -> int:
    return sum(
        cell_complete(output, game, seed, cell)
        for game in games for seed in seeds for cell in cells
    )


def open_task_log(log_dir: Path, game: str, seed: int, command: list[str]):
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{game}_seed_{seed}.log"
    handle = path.open("a", encoding="utf-8")
    handle.write("\n" + "=" * 88 + "\n")
    handle.write("COMMAND: " + " ".join(command) + "\n")
    handle.flush()
    return handle, path


def terminate_process(process: subprocess.Popen, *, grace_seconds: float = 20.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace_seconds)


def run_campaign(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    runner = root / "code" / "run_experiment_local.py"
    if not runner.exists():
        raise RuntimeError(f"Runner not found: {runner}")
    frozen_path = args.frozen_config.resolve()
    frozen = read_json(frozen_path)
    output = args.results_root.resolve() / f"exp_{args.campaign}_safe_sca_test"
    logs = output / "logs_campaign"
    games = args.games
    seeds = args.seeds
    cells = args.cells
    if len(set(games)) != len(games) or len(set(seeds)) != len(seeds) or len(set(cells)) != len(cells):
        raise RuntimeError("games, seeds, and cells must not contain duplicates")
    unknown_cells = sorted(set(cells) - set(CELLS))
    if unknown_cells:
        raise RuntimeError(f"Unknown S1 cells: {unknown_cells}")

    frozen_sha256 = hashlib.sha256(canonical(frozen).encode("utf-8")).hexdigest()
    snapshot = {
        "schema_version": 1,
        "campaign": args.campaign,
        "root": str(root),
        "runner": str(runner),
        "frozen_config_path": str(frozen_path),
        "frozen_config_sha256": frozen_sha256,
        "frozen_config": frozen,
        "games": games,
        "seeds": seeds,
        "cells": cells,
        "episodes": 30,
        "horizon": 5,
        "memory": 2,
        "top_p": 0.9,
        "latin_square": True,
        "models_het": [QWEN, GLM],
        "workers": args.workers,
        "task_timeout_seconds": args.task_timeout,
        "max_retries": args.max_retries,
    }
    total_cells = len(games) * len(seeds) * len(cells)
    base = build_base_command(runner, frozen)
    tasks = [(game, seed) for game in games for seed in seeds]
    pending = [(game, seed) for game, seed in tasks if not task_complete(output, game, seed, cells)]

    print(json.dumps({
        "campaign": args.campaign,
        "output": str(output),
        "total_cells": total_cells,
        "completed_cells": count_completed_cells(output, games, seeds, cells),
        "pending_game_seed_tasks": len(pending),
        "workers": args.workers,
    }, indent=2), flush=True)
    if args.dry_run:
        example_game, example_seed = tasks[0]
        example = base + ["--games", example_game, "--seeds", str(example_seed),
                          "--cells", *cells, "--out_dir", str(output)]
        print("DRY RUN example command:\n" + " ".join(example), flush=True)
        return

    require_strict_preflight(output)
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "CAMPAIGN_SNAPSHOT.json", snapshot)
    if not pending:
        print("All requested cells already exist; no generation launched.", flush=True)
        return

    attempts: dict[tuple[str, int], int] = {task: 0 for task in pending}
    queue = list(pending)
    running: dict[subprocess.Popen, tuple[str, int, float, object, Path]] = {}
    failures: list[tuple[str, int, str]] = []
    started = time.time()
    completed_tasks = 0

    while queue or running:
        while queue and len(running) < args.workers:
            game, seed = queue.pop(0)
            if task_complete(output, game, seed, cells):
                completed_tasks += 1
                continue
            attempts[(game, seed)] += 1
            command = base + ["--games", game, "--seeds", str(seed),
                              "--cells", *cells, "--out_dir", str(output)]
            handle, log_path = open_task_log(logs, game, seed, command)
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            running[process] = (game, seed, time.time(), handle, log_path)
            print(
                f"[{time.strftime('%H:%M:%S')}] START {game}/seed_{seed} "
                f"attempt={attempts[(game, seed)]} pid={process.pid} "
                f"running={len(running)} queued={len(queue)}",
                flush=True,
            )

        finished: list[subprocess.Popen] = []
        for process, (game, seed, started_task, handle, log_path) in list(running.items()):
            elapsed = time.time() - started_task
            rc = process.poll()
            timed_out = rc is None and elapsed > args.task_timeout
            if rc is None and not timed_out:
                continue
            if timed_out:
                handle.write(f"\n[TIMEOUT after {elapsed:.1f}s]\n")
                handle.flush()
                terminate_process(process)
                rc = process.returncode
            handle.close()
            finished.append(process)
            complete = rc == 0 and task_complete(output, game, seed, cells)
            if complete:
                completed_tasks += 1
                status = "OK"
            elif attempts[(game, seed)] <= args.max_retries:
                queue.append((game, seed))
                status = f"RETRY(rc={rc})"
            else:
                message = f"rc={rc}; timeout={timed_out}; log={log_path}"
                failures.append((game, seed, message))
                status = f"FAILED({message})"
            done_cells = count_completed_cells(output, games, seeds, cells)
            print(
                f"[{time.strftime('%H:%M:%S')}] DONE {game}/seed_{seed} {status} "
                f"elapsed={elapsed:.0f}s cells={done_cells}/{total_cells} "
                f"tasks_ok={completed_tasks}/{len(tasks)}",
                flush=True,
            )
        for process in finished:
            del running[process]
        if running:
            time.sleep(1.0)

    final_cells = count_completed_cells(output, games, seeds, cells)
    campaign_report = {
        "campaign": args.campaign,
        "output": str(output),
        "total_cells": total_cells,
        "completed_cells": final_cells,
        "complete": final_cells == total_cells and not failures,
        "wall_seconds": time.time() - started,
        "failures": [
            {"game": game, "seed": seed, "message": message}
            for game, seed, message in failures
        ],
        "attempts": {f"{game}/seed_{seed}": count for (game, seed), count in attempts.items()},
    }
    report_path = output / "CAMPAIGN_EXECUTION_REPORT.json"
    report_path.write_text(json.dumps(campaign_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(campaign_report, indent=2), flush=True)
    if not campaign_report["complete"]:
        raise SystemExit("Campaign incomplete; inspect logs and rerun the same command to resume.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run immutable Safe-SCA R0/S2 campaigns")
    parser.add_argument("--campaign", required=True, choices=["r0", "s2"],
                        help="r0 = same-seed execution replay; s2 = fresh independent seed block")
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("GSACA_ROOT", Path(__file__).resolve().parents[1])))
    parser.add_argument("--results-root", type=Path, required=True,
                        help="new root; output becomes <results-root>/exp_<campaign>_safe_sca_test")
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--games", nargs="+", default=GAMES, choices=GAMES)
    parser.add_argument("--cells", nargs="+", default=CELLS, choices=CELLS)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--task-timeout", type=int, default=7200)
    parser.add_argument("--max-retries", type=int, default=2,
                        help="additional attempts after the initial task attempt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    if args.task_timeout <= 0:
        parser.error("task-timeout must be positive")
    if args.max_retries < 0:
        parser.error("max-retries must be non-negative")
    run_campaign(args)


if __name__ == "__main__":
    main()
