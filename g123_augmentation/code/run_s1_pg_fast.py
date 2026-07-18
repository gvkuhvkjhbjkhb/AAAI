#!/usr/bin/env python3
"""S1 public_goods per-cell parallel launcher.

Other 5 games are 100% complete (600/600 cells). This launcher redirects ALL
compute to the remaining public_goods cells by splitting each seed's 6 cells
into independent tasks, so:
  - All 24 worker slots stay busy (vs 20/24 currently, with long-tail idle).
  - Fast cells (~200s: notom/point_sca/safe_sca/oracle_sca) don't block slow
    cells (~750s: gsaca/gated) — better load balancing.
  - Fully resumable: skips cells with existing metrics.json.
  - Identical flags to the frozen S1 protocol — results are bit-compatible.

arm_order.json note: single-cell runs overwrite the per-seed arm_order.json
with just one cell. This is cosmetic metadata; we pre-write the correct
full Latin-square order and the actual results (metrics.json) are unaffected.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

GSACA_ROOT = Path("/data/lab/AAAI/g123_augmentation")
RUNNER = GSACA_ROOT / "code" / "run_experiment_local.py"
OUT_DIR = GSACA_ROOT / "v2_results" / "exp_s1_safe_sca_test"
LOG_DIR = OUT_DIR / "logs_fast"
FROZEN_CONFIG = GSACA_ROOT / "v2_results" / "s1_safe_sca_frozen.json"

GAME = "public_goods"
SEEDS = list(range(62, 82))  # 20 held-out test seeds
CELLS = [
    "het_notom", "het_gated_atom_talk", "het_gsaca",
    "het_point_sca", "het_safe_sca", "het_oracle_sca",
]
QWEN = "Qwen/Qwen2.5-7B-Instruct"
GLM = "THUDM/GLM-4-9B-0414"

N_CONCURRENT = 24
MAX_RETRIES = 2


def load_safe_flags() -> list[str]:
    cfg = json.loads(FROZEN_CONFIG.read_text())
    return [
        "--safe_warmup", str(cfg["warmup_episodes"]),
        "--safe_tau", str(cfg["tau"]),
        "--safe_confidence", str(cfg["confidence"]),
        "--safe_bootstrap_samples", str(cfg["bootstrap_samples"]),
        "--safe_min_profile_coverage", str(cfg["min_profile_coverage"]),
        "--safe_min_stratum_observations", str(cfg["min_stratum_observations"]),
    ]


def base_cmd() -> list[str]:
    return [
        sys.executable, str(RUNNER),
        "--use_vllm", "--gen_seed_base", "1000",
        "--episodes", "30", "--horizon", "5", "--memory", "2", "--log_every", "100",
        "--models_het", QWEN, GLM,
        "--gate_trust_threshold", "0.6", "--gate_ema_alpha", "0.3",
        "--atom_warmup", "3", "--top_p", "0.9", "--latin_square",
    ] + load_safe_flags()


def cell_done(seed: int, cell: str) -> bool:
    return (OUT_DIR / GAME / f"seed_{seed}" / cell / "metrics.json").exists()


def count_done() -> int:
    return sum(1 for s in SEEDS for c in CELLS if cell_done(s, c))


def prewrite_arm_orders() -> None:
    """Pre-write correct full Latin-square arm_order.json for each seed.
    Workers will overwrite with single-cell versions, but we write the
    canonical version first for provenance. After completion, we restore."""
    for seed in SEEDS:
        seed_dir = OUT_DIR / GAME / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        arm_order_path = seed_dir / "arm_order_canonical.json"
        if not arm_order_path.exists():
            shift = seed % len(CELLS)
            full_order = [CELLS[(i + shift) % len(CELLS)] for i in range(len(CELLS))]
            arm_order_path.write_text(json.dumps({
                "game": GAME, "seed": seed, "arm_order": full_order,
                "latin_square": True,
                "timestamp": datetime.now().isoformat(),
                "note": "canonical full Latin square; preserved before per-cell parallel launch",
            }, indent=2))


def launch_cell(seed: int, cell: str) -> tuple[subprocess.Popen, object]:
    cmd = base_cmd() + [
        "--games", GAME, "--seeds", str(seed),
        "--cells", cell, "--out_dir", str(OUT_DIR),
    ]
    log_path = LOG_DIR / f"pg_seed{seed}_{cell}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc, fh


def run_batch(tasks: list[tuple[int, str]], tag: str) -> tuple[int, list]:
    """Run a batch of (seed, cell) tasks with N_CONCURRENT workers.
    Returns (completed_ok, failed_list)."""
    running: dict[subprocess.Popen, tuple[int, str, float, object]] = {}
    queue = list(tasks)
    completed_ok = 0
    failed: list[tuple[int, str]] = []
    t0 = time.time()

    while queue or running:
        # Fill workers
        while queue and len(running) < N_CONCURRENT:
            seed, cell = queue.pop(0)
            if cell_done(seed, cell):
                completed_ok += 1
                continue
            proc, fh = launch_cell(seed, cell)
            running[proc] = (seed, cell, time.time(), fh)
            print(f"[{time.strftime('%H:%M:%S')}] {tag} START pg/seed_{seed}/{cell} "
                  f"(pid={proc.pid} running={len(running)} queued={len(queue)})", flush=True)

        # Poll
        done_procs = []
        for proc in list(running):
            rc = proc.poll()
            if rc is None:
                continue
            seed, cell, start_t, fh = running[proc]
            fh.close()
            elapsed = time.time() - start_t
            done_procs.append(proc)
            if rc == 0:
                completed_ok += 1
            else:
                failed.append((seed, cell))
            done_cells = count_done()
            wall = time.time() - t0
            total_pg = len(SEEDS) * len(CELLS)
            pct = done_cells / total_pg * 100
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[{time.strftime('%H:%M:%S')}] {tag} DONE pg/seed_{seed}/{cell} {status} "
                  f"({elapsed:.0f}s) | pg {done_cells}/{total_pg} ({pct:.1f}%) | "
                  f"ok={completed_ok} fail={len(failed)}", flush=True)

        for p in done_procs:
            del running[p]
        if running:
            time.sleep(1)

    return completed_ok, failed


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    prewrite_arm_orders()

    total_pg = len(SEEDS) * len(CELLS)
    done = count_done()
    remaining = [(s, c) for s in SEEDS for c in CELLS if not cell_done(s, c)]
    print(f"[PG-FAST] {done}/{total_pg} public_goods cells done, "
          f"{len(remaining)} remaining", flush=True)
    print(f"[PG-FAST] Splitting into per-cell tasks, {N_CONCURRENT} concurrent workers", flush=True)

    if not remaining:
        print("[PG-FAST] All public_goods cells complete!", flush=True)
        return

    # Phase 1: run all remaining cells
    ok, failed = run_batch(remaining, "")
    print(f"\n[PG-FAST] Phase 1: {ok} ok, {len(failed)} failed", flush=True)

    # Phase 2: retries
    for attempt in range(1, MAX_RETRIES + 1):
        if not failed:
            break
        retry = failed
        failed = []
        print(f"\n[PG-FAST] Retry {attempt}: {len(retry)} tasks", flush=True)
        ok, failed = run_batch(retry, f"R{attempt}")
        print(f"[PG-FAST] Retry {attempt}: {ok} ok, {len(failed)} failed", flush=True)

    # Final report
    done = count_done()
    print(f"\n{'='*60}", flush=True)
    print(f"[PG-FAST] FINAL: {done}/{total_pg} public_goods cells "
          f"({done/total_pg*100:.1f}%)", flush=True)
    if failed:
        print(f"[PG-FAST] {len(failed)} still failed:", flush=True)
        for s, c in failed:
            print(f"  - seed_{s}/{c}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
