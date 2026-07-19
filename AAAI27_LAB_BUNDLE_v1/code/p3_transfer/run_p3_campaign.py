#!/usr/bin/env python3
"""Resumable immutable launcher for the 320-cell P3 transfer suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from p3_matrices import registry_audit, registry_sha256
from p3_protocol import canonical, immutable_json, load_protocol, read_json, sha256_json


def cell_complete(output: Path, matrix_id: str, seed: int, policy: str) -> bool:
    return (output / matrix_id / f"seed_{seed}" / policy / "metrics.json").exists()


def task_complete(output: Path, matrix_id: str, seed: int, policies: list[str]) -> bool:
    return all(cell_complete(output, matrix_id, seed, policy) for policy in policies)


def completed_cells(output: Path, matrices: list[str], seeds: list[int], policies: list[str]) -> int:
    return sum(cell_complete(output, matrix, seed, policy)
               for matrix in matrices for seed in seeds for policy in policies)


def require_preflight(output: Path) -> None:
    manifest_path = output / "ENVIRONMENT_MANIFEST_S1.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"Missing {manifest_path}. Run base code/preflight_s1.py against this exact output first."
        )
    manifest = read_json(manifest_path)
    if not manifest.get("preflight_passed"):
        raise RuntimeError(f"P3 preflight did not pass: {manifest.get('failures')}")
    if manifest.get("allow_version_mismatch"):
        raise RuntimeError("P3 preflight used a version override; this is prohibited")


def terminate(process: subprocess.Popen, grace_seconds: float = 20.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace_seconds)


def build_snapshot(protocol: dict, protocol_path: Path, base_root: Path, output: Path) -> dict:
    return {
        "schema_version": 1,
        "campaign": "p3_transfer",
        "output": str(output),
        "base_root": str(base_root),
        "runner": str(Path(__file__).with_name("run_p3_matrix.py").resolve()),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_json(protocol),
        "protocol": protocol,
        "matrix_registry_sha256": registry_sha256(),
        "matrix_registry": registry_audit(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run immutable P3 transfer campaign")
    parser.add_argument("--base-root", type=Path, required=True,
                        help="g123_augmentation directory containing S1/S2 base code")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True,
                        help="output becomes <results-root>/exp_p3_transfer_test")
    parser.add_argument("--workers", type=int,
                        help="must equal the frozen protocol worker count (32)")
    parser.add_argument("--task-timeout", type=int,
                        help="must equal the frozen protocol task timeout")
    parser.add_argument("--max-retries", type=int,
                        help="must equal the frozen protocol additional-retry count")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    if not (args.base_root.resolve() / "code" / "run_experiment_local.py").exists():
        parser.error("base-root must contain code/run_experiment_local.py")
    workers = protocol["workers"] if args.workers is None else args.workers
    timeout = protocol["task_timeout_seconds"] if args.task_timeout is None else args.task_timeout
    retries = protocol["max_retries"] if args.max_retries is None else args.max_retries
    if workers != protocol["workers"] or timeout != protocol["task_timeout_seconds"] or retries != protocol["max_retries"]:
        parser.error("workers, timeout, and retries are frozen in the P3 protocol")
    if workers <= 0 or timeout <= 0 or retries < 0:
        parser.error("invalid worker/timeout/retry value")

    output = args.results_root.resolve() / "exp_p3_transfer_test"
    matrices, seeds, policies = protocol["matrix_ids"], protocol["seeds"], protocol["policies"]
    total_cells = len(matrices) * len(seeds) * len(policies)
    tasks = [(matrix, seed) for matrix in matrices for seed in seeds]
    pending = [(matrix, seed) for matrix, seed in tasks if not task_complete(output, matrix, seed, policies)]
    runner = Path(__file__).with_name("run_p3_matrix.py").resolve()
    snapshot = build_snapshot(protocol, protocol_path, args.base_root.resolve(), output)
    print(json.dumps({
        "output": str(output), "total_cells": total_cells,
        "completed_cells": completed_cells(output, matrices, seeds, policies),
        "pending_matrix_seed_tasks": len(pending), "workers": workers,
        "protocol_sha256": snapshot["protocol_sha256"],
        "matrix_registry_sha256": snapshot["matrix_registry_sha256"],
    }, indent=2), flush=True)
    example = [sys.executable, str(runner), "--base-root", str(args.base_root.resolve()),
               "--protocol", str(protocol_path), "--out-dir", str(output),
               "--matrix-id", matrices[0], "--seeds", str(seeds[0])]
    if args.dry_run:
        print("DRY RUN example command:\n" + " ".join(example), flush=True)
        return

    require_preflight(output)
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "P3_CAMPAIGN_SNAPSHOT.json", snapshot)
    immutable_json(output / "P3_MATRIX_REGISTRY.json", {
        "registry_sha256": registry_sha256(), "matrices": registry_audit(),
    })
    if not pending:
        print("All P3 cells already exist; nothing to launch.", flush=True)
        return

    logs = output / "logs_campaign"
    logs.mkdir(parents=True, exist_ok=True)
    attempts = {task: 0 for task in pending}
    queue = list(pending)
    running: dict[subprocess.Popen, tuple[str, int, float, object, Path]] = {}
    failures: list[dict] = []
    started = time.time()
    completed_tasks = 0

    while queue or running:
        while queue and len(running) < workers:
            matrix, seed = queue.pop(0)
            if task_complete(output, matrix, seed, policies):
                completed_tasks += 1
                continue
            attempts[(matrix, seed)] += 1
            command = [sys.executable, str(runner), "--base-root", str(args.base_root.resolve()),
                       "--protocol", str(protocol_path), "--out-dir", str(output),
                       "--matrix-id", matrix, "--seeds", str(seed)]
            log_path = logs / f"{matrix}_seed_{seed}.log"
            handle = log_path.open("a", encoding="utf-8")
            handle.write("\n" + "=" * 88 + "\nCOMMAND: " + " ".join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            running[process] = (matrix, seed, time.time(), handle, log_path)
            print(f"[{time.strftime('%H:%M:%S')}] START {matrix}/seed_{seed} "
                  f"attempt={attempts[(matrix, seed)]} pid={process.pid} "
                  f"running={len(running)} queued={len(queue)}", flush=True)

        finished = []
        for process, (matrix, seed, task_started, handle, log_path) in list(running.items()):
            elapsed = time.time() - task_started
            rc = process.poll()
            timed_out = rc is None and elapsed > timeout
            if rc is None and not timed_out:
                continue
            if timed_out:
                handle.write(f"\n[TIMEOUT after {elapsed:.1f}s]\n")
                handle.flush()
                terminate(process)
                rc = process.returncode
            handle.close()
            finished.append(process)
            if rc == 0 and task_complete(output, matrix, seed, policies):
                completed_tasks += 1
                status = "OK"
            elif attempts[(matrix, seed)] <= retries:
                queue.append((matrix, seed))
                status = f"RETRY(rc={rc})"
            else:
                message = f"rc={rc}; timeout={timed_out}; log={log_path}"
                failures.append({"matrix": matrix, "seed": seed, "message": message})
                status = f"FAILED({message})"
            print(f"[{time.strftime('%H:%M:%S')}] DONE {matrix}/seed_{seed} {status} "
                  f"elapsed={elapsed:.0f}s cells={completed_cells(output, matrices, seeds, policies)}/{total_cells} "
                  f"tasks_ok={completed_tasks}/{len(tasks)}", flush=True)
        for process in finished:
            del running[process]
        if running:
            time.sleep(1.0)

    report = {
        "campaign": "p3_transfer", "output": str(output), "total_cells": total_cells,
        "completed_cells": completed_cells(output, matrices, seeds, policies),
        "complete": completed_cells(output, matrices, seeds, policies) == total_cells and not failures,
        "wall_seconds": time.time() - started, "failures": failures,
        "attempts": {f"{matrix}/seed_{seed}": count for (matrix, seed), count in attempts.items()},
    }
    (output / "P3_CAMPAIGN_EXECUTION_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["complete"]:
        raise SystemExit("P3 campaign incomplete; preserve logs and rerun the exact command to resume.")


if __name__ == "__main__":
    main()
