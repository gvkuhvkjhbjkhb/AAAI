#!/usr/bin/env python3
"""S1 Fast Launcher — parallel (game, seed) sharding for maximum GPU utilization.

Bottleneck analysis:
  - Previous run used --workers 2 (only 2 shards by seed).
  - vLLM logs showed Running: 1 req, GPU KV cache 0.1% — GPUs heavily idle.
  - 720 cells total, only 17 completed before workers died.

Optimization:
  - Shard by (game, seed) → 120 independent tasks, each producing 6 cells.
  - Run up to N_CONCURRENT tasks simultaneously against the two vLLM servers.
  - Fast games (hawk_dove, stag_hunt, BoS ~5 min) free slots for slow games
    (chicken, deadlock, public_goods ~10-13 min) — better load balancing than
    seed-based sharding where every worker must run all 6 games.
  - Fully resumable: run_experiment_local.py skips cells with existing metrics.json.
  - Failed tasks are retried automatically.

The command-line flags are identical to what run_s1_safe_sca.py produces from
the frozen config, so results are bit-identical to the preregistered protocol.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
GSACA_ROOT = Path("/data/lab/AAAI/g123_augmentation")
RUNNER = GSACA_ROOT / "code" / "run_experiment_local.py"
OUT_DIR = GSACA_ROOT / "v2_results" / "exp_s1_safe_sca_test"
LOG_DIR = OUT_DIR / "logs_fast"
FROZEN_CONFIG = GSACA_ROOT / "v2_results" / "s1_safe_sca_frozen.json"

# ── Experiment parameters (must match run_s1_safe_sca.py / frozen config) ──
GAMES = [
    "chicken", "deadlock", "hawk_dove",
    "stag_hunt", "battle_of_the_sexes", "public_goods",
]
SEEDS = list(range(62, 82))  # 20 held-out test seeds
CELLS = [
    "het_notom", "het_gated_atom_talk", "het_gsaca",
    "het_point_sca", "het_safe_sca", "het_oracle_sca",
]
QWEN = "Qwen/Qwen2.5-7B-Instruct"
GLM = "THUDM/GLM-4-9B-0414"

# ── Concurrency ────────────────────────────────────────────────────────
# 24 workers ≈ 12 concurrent requests per vLLM server. The G1 run used 20
# total workers (G1+G2+G3) on the same 2 GPUs; 24 for S1-only is safe.
N_CONCURRENT = 24
TASK_TIMEOUT = 7200  # 2 hours per (game, seed) task
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


def task_complete(game: str, seed: int) -> bool:
    seed_dir = OUT_DIR / game / f"seed_{seed}"
    return all((seed_dir / cell / "metrics.json").exists() for cell in CELLS)


def count_done_cells() -> int:
    n = 0
    for game in GAMES:
        for seed in SEEDS:
            for cell in CELLS:
                if (OUT_DIR / game / f"seed_{seed}" / cell / "metrics.json").exists():
                    n += 1
    return n


def launch_task(game: str, seed: int) -> tuple[subprocess.Popen, object]:
    cmd = base_cmd() + [
        "--games", game, "--seeds", str(seed),
        "--cells", *CELLS, "--out_dir", str(OUT_DIR),
    ]
    log_path = LOG_DIR / f"{game}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc, fh


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    total_cells = len(GAMES) * len(SEEDS) * len(CELLS)

    # Build task list, skipping fully-complete (game, seed) pairs
    all_tasks = [(g, s) for g in GAMES for s in SEEDS]
    pending = [(g, s) for g, s in all_tasks if not task_complete(g, s)]
    done = count_done_cells()
    print(f"[S1-FAST] {done}/{total_cells} cells done | "
          f"{len(all_tasks) - len(pending)}/{len(all_tasks)} tasks complete | "
          f"{len(pending)} tasks pending", flush=True)

    if not pending:
        print("[S1-FAST] All tasks complete!", flush=True)
        return

    print(f"[S1-FAST] Launching {N_CONCURRENT} concurrent workers for {len(pending)} tasks", flush=True)
    print(f"[S1-FAST] Frozen config: {FROZEN_CONFIG}", flush=True)
    print(f"[S1-FAST] Output: {OUT_DIR}", flush=True)

    # ── Phase 1: run all pending tasks with N_CONCURRENT workers ───────
    running: dict[subprocess.Popen, tuple[str, int, float, object]] = {}
    queue = list(pending)
    completed_ok = 0
    failed: list[tuple[str, int]] = []
    t0 = time.time()
    tasks_started = 0

    while queue or running:
        # Fill up to N_CONCURRENT
        while queue and len(running) < N_CONCURRENT:
            game, seed = queue.pop(0)
            proc, fh = launch_task(game, seed)
            running[proc] = (game, seed, time.time(), fh)
            tasks_started += 1
            print(f"[{time.strftime('%H:%M:%S')}] START {game}/seed_{seed} "
                  f"(pid={proc.pid} running={len(running)} queued={len(queue)} "
                  f"started={tasks_started}/{len(pending)})", flush=True)

        # Poll for completions
        done_procs = []
        for proc in list(running):
            rc = proc.poll()
            if rc is None:
                continue
            game, seed, start_t, fh = running[proc]
            fh.close()
            elapsed = time.time() - start_t
            done_procs.append(proc)

            if rc == 0:
                completed_ok += 1
            else:
                failed.append((game, seed))

            done_cells = count_done_cells()
            wall = time.time() - t0
            pct = done_cells / total_cells * 100
            remaining_tasks = len(pending) - completed_ok - len(failed) - len(queue)
            active = len(running) - 1  # this one just finished
            throughput = completed_ok / (wall / 60) if wall > 60 else 0
            eta_min = (len(queue) + active) / throughput if throughput > 0 else 0

            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[{time.strftime('%H:%M:%S')}] DONE  {game}/seed_{seed} {status} "
                  f"({elapsed:.0f}s) | cells {done_cells}/{total_cells} ({pct:.1f}%) | "
                  f"ok={completed_ok} fail={len(failed)} | "
                  f"ETA ~{eta_min:.0f}min", flush=True)

        for p in done_procs:
            del running[p]

        if running:
            time.sleep(1)

    wall_phase1 = time.time() - t0
    print(f"\n[S1-FAST] Phase 1 done: {completed_ok} ok, {len(failed)} failed "
          f"in {wall_phase1/3600:.1f}h", flush=True)

    # ── Phase 2: retry failed tasks ────────────────────────────────────
    for attempt in range(1, MAX_RETRIES + 1):
        if not failed:
            break
        retry_list = failed
        failed = []
        print(f"\n[S1-FAST] Retry pass {attempt}: {len(retry_list)} tasks", flush=True)

        running = {}
        queue = list(retry_list)
        while queue or running:
            while queue and len(running) < N_CONCURRENT:
                game, seed = queue.pop(0)
                # Skip if somehow complete
                if task_complete(game, seed):
                    print(f"  [skip] {game}/seed_{seed} already complete", flush=True)
                    completed_ok += 1
                    continue
                proc, fh = launch_task(game, seed)
                running[proc] = (game, seed, time.time(), fh)
                print(f"  [{time.strftime('%H:%M:%S')}] RETRY {game}/seed_{seed} "
                      f"(pid={proc.pid})", flush=True)

            done_procs = []
            for proc in list(running):
                rc = proc.poll()
                if rc is None:
                    continue
                game, seed, start_t, fh = running[proc]
                fh.close()
                elapsed = time.time() - start_t
                done_procs.append(proc)
                if rc == 0:
                    completed_ok += 1
                    print(f"  [{time.strftime('%H:%M:%S')}] RETRY OK  {game}/seed_{seed} "
                          f"({elapsed:.0f}s)", flush=True)
                else:
                    failed.append((game, seed))
                    print(f"  [{time.strftime('%H:%M:%S')}] RETRY FAIL {game}/seed_{seed} "
                          f"(rc={rc} {elapsed:.0f}s)", flush=True)

            for p in done_procs:
                del running[p]
            if running:
                time.sleep(1)

    # ── Final report ───────────────────────────────────────────────────
    done_cells = count_done_cells()
    wall_total = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"[S1-FAST] FINAL: {done_cells}/{total_cells} cells "
          f"({done_cells/total_cells*100:.1f}%) in {wall_total/3600:.1f}h", flush=True)
    if failed:
        print(f"[S1-FAST] {len(failed)} tasks still failed after {MAX_RETRIES} retries:", flush=True)
        for g, s in failed:
            print(f"  - {g}/seed_{s}", flush=True)
    print(f"[S1-FAST] Output: {OUT_DIR}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
