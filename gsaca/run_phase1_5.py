#!/usr/bin/env python3
"""Phase 1-5 experiment orchestrator for Route A (DP-Gating).
Uses LOCAL vLLM servers (Qwen-7B on :8000, GLM-9B on :8001) for 10x speedup.
30 episodes, seeds 42-49.

Jobs:
  job1: Phase 1 - deadlock + stag_hunt (4 cells x 8 seeds x 2 games = 64 runs)
  job2: Phase 1 - hawk_dove + battle_of_the_sexes (64 runs)
  job3: Chicken reproduction (32) + Phase 4 public_goods (16) = 48 runs
  job4: Phase 3 threshold ablation (4 thr x 2 games x 8 seeds = 64 runs)
  job5: Phase 4 model coverage via SiliconFlow API (Qwen-14B + GLM-9B) = 32 runs
"""
import argparse, os, sys, time, json, random, traceback
from multiprocessing import Pool
from datetime import datetime
import numpy as np

sys.path.insert(0, '/data/lab/gsaca')
import hettom_baseline as hb

API_BASE = "https://api.siliconflow.cn/v1"
API_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-hhpwsdsxkbcjxvgnpcdrfjdvwbqpmcltpvuowvxlkewczhwl")

OUT = "/data/lab/results/phase1_5_unified"
EPISODES = 30
SEEDS = list(range(42, 50))

CELLS_4 = ["hom_notom", "het_notom", "het_gated_atom_talk", "het_dp_gated_atom_talk"]
CELLS_2 = ["het_gated_atom_talk", "het_dp_gated_atom_talk"]

CELL_CONFIGS = {
    "hom_notom": dict(homogeneous=True, use_tom=False, use_talk=False, adaptive_tom=False),
    "het_notom": dict(homogeneous=False, use_tom=False, use_talk=False, adaptive_tom=False),
    "het_gated_atom_talk": dict(homogeneous=False, use_tom=True, use_talk=True,
                                 adaptive_tom=True, gated_talk_tom=True),
    "het_dp_gated_atom_talk": dict(homogeneous=False, use_tom=True, use_talk=True,
                                    adaptive_tom=True, gated_talk_tom=True,
                                    diversity_preserving_gate=True),
}


def make_config(game, seed, cell, threshold=0.6, n_agents=None,
                models_het=None, model_homo=None):
    cfg = {
        "game": game, "n_agents": n_agents, "horizon": 5, "memory": 2,
        "models_homo": model_homo or "Qwen/Qwen2.5-7B-Instruct",
        "models_het": models_het or ["Qwen/Qwen2.5-7B-Instruct", "THUDM/GLM-4-9B-0414"],
        "temp_homo": 0.7, "temps_het": [0.7, 0.7],
        "roles": [f"player{i+1}" for i in range(n_agents or 2)],
        "tom_order": 1, "seed": seed,
        "api_base": API_BASE, "api_key": API_KEY,
        "gate_trust_threshold": threshold,
    }
    cfg.update(CELL_CONFIGS[cell])
    cfg["cell_name"] = cell
    return cfg


def run_one(task):
    game, seed, cell, threshold, out_dir, n_agents, models_het, model_homo = task
    seed_dir = os.path.join(out_dir, game, f"seed_{seed}")
    mpath = os.path.join(seed_dir, cell, "metrics.json")
    if os.path.exists(mpath):
        return f"[skip] {game} s{seed} {cell} thr={threshold}"

    random.seed(seed)
    np.random.seed(seed)
    cfg = make_config(game, seed, cell, threshold, n_agents, models_het, model_homo)

    t0 = time.time()
    for attempt in range(3):
        try:
            hb.run_config(cfg, EPISODES, seed_dir, log_every=10)
            return f"[done] {game} s{seed} {cell} thr={threshold} ({time.time()-t0:.0f}s)"
        except Exception as e:
            if attempt < 2:
                time.sleep(15 * (attempt + 1))
                random.seed(seed + attempt)
                np.random.seed(seed + attempt)
                continue
            return f"[FAIL] {game} s{seed} {cell} thr={threshold} ({time.time()-t0:.0f}s): {e}"


def gen_tasks(job):
    tasks = []
    if job == "job1":
        for game in ["deadlock", "stag_hunt"]:
            for seed in SEEDS:
                for cell in CELLS_4:
                    tasks.append((game, seed, cell, 0.6, f"{OUT}/phase1", None, None, None))

    elif job == "job2":
        for game in ["hawk_dove", "battle_of_the_sexes"]:
            for seed in SEEDS:
                for cell in CELLS_4:
                    tasks.append((game, seed, cell, 0.6, f"{OUT}/phase1", None, None, None))

    elif job == "job3":
        # Chicken reproduction (separate dir)
        for seed in SEEDS:
            for cell in CELLS_4:
                tasks.append(("chicken", seed, cell, 0.6, f"{OUT}/chicken_repro", None, None, None))
        # Phase 4: public_goods 4-agent
        for seed in SEEDS:
            for cell in CELLS_2:
                tasks.append(("public_goods", seed, cell, 0.6,
                              f"{OUT}/phase4_public_goods", 4, None, None))

    elif job == "job4":
        for thr in [0.4, 0.5, 0.7, 0.8]:
            for game in ["chicken", "hawk_dove"]:
                for seed in SEEDS:
                    tasks.append((game, seed, "het_dp_gated_atom_talk", thr,
                                  f"{OUT}/phase3_threshold/thr_{thr}", None, None, None))

    elif job == "job5":
        # Phase 4: model coverage via SiliconFlow API (Qwen-14B not local)
        large_qwen = ["Qwen/Qwen2.5-14B-Instruct", "THUDM/GLM-4-9B-0414"]
        for game in ["chicken", "hawk_dove"]:
            for seed in SEEDS:
                for cell in CELLS_2:
                    tasks.append((game, seed, cell, 0.6,
                                  f"{OUT}/phase4_model_coverage",
                                  None, large_qwen, "Qwen/Qwen2.5-14B-Instruct"))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    tasks = gen_tasks(args.job)
    eta_min = len(tasks) * 120 / args.workers / 60
    print(f"[{datetime.now()}] {args.job}: {len(tasks)} tasks, {args.workers} workers, ETA {eta_min:.0f} min", flush=True)

    with Pool(args.workers) as pool:
        done = skip = fail = 0
        for i, result in enumerate(pool.imap_unordered(run_one, tasks)):
            if "[done]" in result: done += 1
            elif "[skip]" in result: skip += 1
            else: fail += 1
            print(f"[{datetime.now()}] ({i+1}/{len(tasks)}) {result} | D:{done} S:{skip} F:{fail}", flush=True)

    print(f"\n[{datetime.now()}] {args.job} DONE: {done} done, {skip} skipped, {fail} failed", flush=True)


if __name__ == "__main__":
    main()
