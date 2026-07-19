#!/usr/bin/env python3
"""Resumable launcher for P0/P1/P2 supplemental controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from p3_label_variants import registry_audit as label_audit
from p3_matrices import registry_audit as p3_audit
from supplement_protocol import immutable_json, load_protocol, read_json, sha256_json


OUTPUT_NAMES = {
    ("p0", "p3"): "p0_payoff_prompt",
    ("p1", "p3"): "p1_label_swap",
    ("p2", "p3"): "p2_teammean_bandit_p3",
    ("p2", "source"): "p2_teammean_bandit_source",
}


def policies_for(protocol: dict, experiment: str) -> list[str]:
    if experiment == "p0":
        return list(protocol["p0"]["policies"])
    if experiment == "p1":
        return list(protocol["p1"]["policies"])
    return [protocol["p2"]["cell"]]


def contexts_for(protocol: dict, domain: str) -> tuple[list[str], list[int]]:
    if domain == "p3":
        return list(protocol["p3"]["matrix_ids"]), list(protocol["p3"]["seeds"])
    return list(protocol["source"]["games"]), list(protocol["source"]["seeds"])


def metric_path(root: Path, protocol: dict, experiment: str, domain: str,
                context: str, seed: int, policy: str) -> Path:
    return (root / OUTPUT_NAMES[(experiment, domain)] / context / f"seed_{seed}" /
            policy / "metrics.json")


def task_complete(root: Path, protocol: dict, task: tuple[str, str, str, int]) -> bool:
    experiment, domain, context, seed = task
    return all(metric_path(root, protocol, experiment, domain, context, seed, policy).exists()
               for policy in policies_for(protocol, experiment))


def count_cells(root: Path, protocol: dict, tasks: list[tuple[str, str, str, int]]) -> int:
    return sum(metric_path(root, protocol, exp, dom, ctx, seed, policy).exists()
               for exp, dom, ctx, seed in tasks
               for policy in policies_for(protocol, exp))


def require_preflight(root: Path) -> None:
    path = root / "ENVIRONMENT_MANIFEST_S1.json"
    if not path.exists():
        raise RuntimeError(f"Missing strict preflight manifest: {path}")
    manifest = read_json(path)
    if not manifest.get("preflight_passed") or manifest.get("allow_version_mismatch"):
        raise RuntimeError(f"Strict preflight failed or used override: {manifest.get('failures')}")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminate(process: subprocess.Popen, grace: float = 20.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch frozen P0/P1/P2 controls")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", choices=["p0", "p1", "p2"],
                        default=["p0", "p1", "p2"])
    parser.add_argument("--include-source-p2", action="store_true",
                        help="add the secondary 120-cell S2-source P2 block")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--task-timeout", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    execution = protocol["execution"]
    workers = execution["workers"] if args.workers is None else args.workers
    timeout = execution["task_timeout_seconds"] if args.task_timeout is None else args.task_timeout
    retries = execution["max_retries"] if args.max_retries is None else args.max_retries
    if (workers, timeout, retries) != (
        execution["workers"], execution["task_timeout_seconds"], execution["max_retries"]
    ):
        parser.error("workers/timeout/retries are frozen; do not override them")

    root = args.results_root.resolve()
    blocks: list[tuple[str, str]] = []
    for experiment in dict.fromkeys(args.experiments):
        blocks.append((experiment, "p3"))
        if experiment == "p2" and args.include_source_p2:
            blocks.append(("p2", "source"))
    tasks: list[tuple[str, str, str, int]] = []
    for experiment, domain in blocks:
        contexts, seeds = contexts_for(protocol, domain)
        tasks.extend((experiment, domain, context, seed)
                     for context in contexts for seed in seeds)
    total_cells = sum(len(policies_for(protocol, exp)) for exp, _, _, _ in tasks)
    pending = [task for task in tasks if not task_complete(root, protocol, task)]
    runner = Path(__file__).with_name("run_supplement_task.py").resolve()
    snapshot = {
        "schema_version": 1,
        "campaign": protocol["campaign"],
        "protocol": protocol,
        "protocol_sha256": sha256_json(protocol),
        "runner_sha256": file_sha256(runner),
        "blocks": [{"experiment": e, "domain": d} for e, d in blocks],
        "p3_registry": p3_audit(),
        "p1_label_registry": label_audit(),
        "total_cells": total_cells,
    }
    block_tag = "__".join(f"{experiment}_{domain}" for experiment, domain in blocks)
    status = {
        "results_root": str(root), "blocks": blocks, "total_cells": total_cells,
        "completed_cells": count_cells(root, protocol, tasks),
        "pending_tasks": len(pending), "workers": workers,
        "protocol_sha256": snapshot["protocol_sha256"],
    }
    print(json.dumps(status, indent=2), flush=True)
    if args.dry_run:
        if pending:
            e, d, c, s = pending[0]
            print("DRY RUN example:\n" + " ".join([
                sys.executable, str(runner), "--protocol", str(protocol_path),
                "--out-dir", str(root), "--experiment", e, "--domain", d,
                "--context", c, "--seed", str(s),
            ]), flush=True)
        return

    require_preflight(root)
    root.mkdir(parents=True, exist_ok=True)
    immutable_json(root / f"SUPPLEMENT_CAMPAIGN_SNAPSHOT__{block_tag}.json", snapshot)
    if not pending:
        print("All requested cells exist; nothing to launch.", flush=True)
        return

    logs = root / "logs_campaign"
    logs.mkdir(parents=True, exist_ok=True)
    attempts = {task: 0 for task in pending}
    queue = list(pending)
    running: dict[subprocess.Popen, tuple[tuple[str, str, str, int], float, object, Path]] = {}
    failures: list[dict] = []
    started = time.time()
    while queue or running:
        while queue and len(running) < workers:
            task = queue.pop(0)
            if task_complete(root, protocol, task):
                continue
            experiment, domain, context, seed = task
            attempts[task] += 1
            command = [
                sys.executable, str(runner), "--protocol", str(protocol_path),
                "--out-dir", str(root), "--experiment", experiment,
                "--domain", domain, "--context", context, "--seed", str(seed),
            ]
            log_path = logs / f"{experiment}_{domain}_{context}_seed_{seed}.log"
            handle = log_path.open("a", encoding="utf-8")
            handle.write("\n" + "=" * 80 + "\nCOMMAND: " + " ".join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            running[process] = (task, time.time(), handle, log_path)
            print(f"START {experiment}/{domain}/{context}/seed_{seed} pid={process.pid}", flush=True)

        finished = []
        for process, (task, task_started, handle, log_path) in list(running.items()):
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
            if rc == 0 and task_complete(root, protocol, task):
                outcome = "OK"
            elif attempts[task] <= retries:
                queue.append(task)
                outcome = f"RETRY(rc={rc})"
            else:
                experiment, domain, context, seed = task
                failures.append({"experiment": experiment, "domain": domain,
                                 "context": context, "seed": seed,
                                 "returncode": rc, "timeout": timed_out,
                                 "log": str(log_path)})
                outcome = "FAILED"
            print(f"DONE {'/'.join(map(str, task))} {outcome}; "
                  f"cells={count_cells(root, protocol, tasks)}/{total_cells}", flush=True)
        for process in finished:
            del running[process]
        if running:
            time.sleep(1.0)

    complete_cells = count_cells(root, protocol, tasks)
    report = {
        "campaign": protocol["campaign"], "total_cells": total_cells,
        "completed_cells": complete_cells,
        "complete": complete_cells == total_cells and not failures,
        "wall_seconds": time.time() - started, "failures": failures,
        "attempts": {"/".join(map(str, task)): count for task, count in attempts.items()},
    }
    (root / f"SUPPLEMENT_CAMPAIGN_EXECUTION_REPORT__{block_tag}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["complete"]:
        raise SystemExit("Campaign incomplete; preserve logs and rerun the same command")


if __name__ == "__main__":
    main()
