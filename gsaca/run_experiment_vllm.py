#!/usr/bin/env python3
"""GSACA + Public Goods experiment runner — vLLM local GPU edition.

Routes each agent to the correct vLLM server based on model name:
  Qwen2.5-7B-Instruct -> localhost:8000 (GPU 0)
  GLM-4-9B-0414        -> localhost:8001 (GPU 1)

vLLM auto-batches concurrent requests from all workers, giving near-individual
latency with 6x parallelism.
"""
import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hettom_baseline as hb

# Local vLLM endpoints
VLLM_QWEN = "http://localhost:8000/v1"
VLLM_GLM = "http://localhost:8001/v1"
API_KEY = "dummy"

MODELS_HET = ["Qwen/Qwen2.5-7B-Instruct", "THUDM/GLM-4-9B-0414"]
MODEL_HOMO = "Qwen/Qwen2.5-7B-Instruct"

API_BASE_MAP = {
    "Qwen/Qwen2.5-7B-Instruct": VLLM_QWEN,
    "THUDM/GLM-4-9B-0414": VLLM_GLM,
}

ALL_GAMES = ["chicken", "hawk_dove", "deadlock",
             "stag_hunt", "battle_of_the_sexes", "public_goods"]

ORACLE_STRUCTURE = {
    "chicken": "anti_coord",
    "hawk_dove": "anti_coord",
    "deadlock": "anti_coord",
    "stag_hunt": "coord",
    "battle_of_the_sexes": "coord",
    "public_goods": "coord",
}


# =============================================================================
# Monkey-patch build_agents for per-model vLLM routing
# =============================================================================

_original_build_agents = hb.build_agents

def _patched_build_agents(config):
    game, agents = _original_build_agents(config)
    api_base_map = config.get("api_base_map")
    if api_base_map:
        for ag in agents:
            endpoint = api_base_map.get(ag.model_name)
            if endpoint:
                ag.api_base = endpoint
                ag.api_key = config.get("api_key", "dummy")
                ag._api_client = None
                ag._model = None
    return game, agents

hb.build_agents = _patched_build_agents


# =============================================================================
# Game Structure Estimator
# =============================================================================

class GameStructureEstimator:
    def __init__(self):
        self.observations = []

    def observe(self, actions, rewards):
        team_payoff = float(np.mean(rewards))
        self.observations.append((tuple(actions), team_payoff))

    def estimate(self):
        if len(self.observations) < 3:
            return "anti_coord", 0.0, len(self.observations)
        same_p, diff_p = [], []
        for actions, payoff in self.observations:
            if len(set(actions)) == 1:
                same_p.append(payoff)
            else:
                diff_p.append(payoff)
        if not same_p or not diff_p:
            return "anti_coord", 0.0, len(self.observations)
        mean_same = float(np.mean(same_p))
        mean_diff = float(np.mean(diff_p))
        split_score = mean_diff - mean_same
        structure = "anti_coord" if split_score > 0 else "coord"
        return structure, split_score, len(self.observations)


# =============================================================================
# Config builder
# =============================================================================

def make_config(game_name, cell_name, seed, n_agents=None,
                horizon=5, memory=2):
    if n_agents is None:
        n_agents = 4 if game_name == "public_goods" else 2
    base = {
        "game": game_name,
        "n_agents": n_agents,
        "horizon": horizon,
        "memory": memory,
        "models_homo": MODEL_HOMO,
        "models_het": MODELS_HET,
        "temp_homo": 0.7,
        "temps_het": [0.5, 0.8],
        "roles": [f"player{i+1}" for i in range(n_agents)],
        "tom_order": 1,
        "seed": seed,
        "api_base": VLLM_QWEN,
        "api_base_map": API_BASE_MAP,
        "api_key": API_KEY,
        "cell_name": cell_name,
    }
    if cell_name == "het_notom":
        base.update(homogeneous=False, use_tom=False, use_talk=False,
                    adaptive_tom=False)
    elif cell_name == "het_gated_atom_talk":
        base.update(homogeneous=False, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True,
                    diversity_preserving_gate=False)
    elif cell_name == "het_dp_gated_atom_talk":
        base.update(homogeneous=False, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True,
                    diversity_preserving_gate=True)
    elif cell_name == "het_gsaca":
        base.update(homogeneous=False, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True,
                    diversity_preserving_gate=True)
    else:
        raise ValueError(f"Unknown cell: {cell_name}")
    return base


# =============================================================================
# GSACA cell runner
# =============================================================================

def run_gsaca_cell(cfg, n_episodes, out_dir, log_every=5, warmup=5):
    cell = cfg["cell_name"]
    t0 = time.time()
    game, agents = hb.build_agents(cfg)
    horizon = cfg["horizon"]
    memory = cfg["memory"]

    estimator = GameStructureEstimator()
    episodes = []

    for ep_idx in range(warmup):
        ep = hb.run_episode(game, agents, horizon, memory)
        episodes.append(ep)
        for step in ep:
            estimator.observe(step["actions"], step["rewards"])

    structure, split_score, n_obs = estimator.estimate()
    oracle = ORACLE_STRUCTURE.get(cfg["game"], "unknown")
    detection_correct = (structure == oracle)

    if structure == "coord":
        for ag in agents:
            ag.gated_talk_tom = False
            ag.diversity_preserving_gate = False

    for ep_idx in range(warmup, n_episodes):
        ep = hb.run_episode(game, agents, horizon, memory)
        episodes.append(ep)
        if (ep_idx + 1) % log_every == 0:
            print(f"  [{cell}] ep {ep_idx+1}/{n_episodes} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    metrics = hb.compute_metrics(episodes, game, n_boot=2000,
                                  seed=cfg.get("seed", 0))
    metrics["cell"] = cell
    metrics["gsaca_detected_structure"] = structure
    metrics["gsaca_oracle_structure"] = oracle
    metrics["gsaca_detection_correct"] = detection_correct
    metrics["gsaca_split_score"] = split_score
    metrics["gsaca_warmup_episodes"] = warmup
    metrics["gsaca_n_observations"] = n_obs
    metrics["wall_time_s"] = time.time() - t0
    metrics["config"] = {k: v for k, v in cfg.items()
                         if k not in ("models_het", "temps_het", "roles",
                                       "api_base_map")}

    team_payoffs = [float(np.mean([np.mean(s["rewards"]) for s in ep]))
                    for ep in episodes]
    metrics["team_mean_payoff"] = float(np.mean(team_payoffs))

    cell_dir = os.path.join(out_dir, cell)
    os.makedirs(cell_dir, exist_ok=True)
    with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    traj_path = os.path.join(cell_dir, "trajectories.jsonl")
    with open(traj_path, "w") as ftraj:
        for ep_idx, ep in enumerate(episodes):
            ftraj.write(json.dumps({"episode": ep_idx, "steps": ep}) + "\n")

    print(f"  [{cell}] detect={structure}(oracle={oracle},"
          f"{'OK' if detection_correct else 'MISS'}) "
          f"split={split_score:.3f} payoff={metrics['cooperation_payoff']:.4f} "
          f"team={metrics['team_mean_payoff']:.4f} "
          f"div={metrics['perspective_diversity']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    del agents
    gc.collect()
    return metrics


# =============================================================================
# Standard cell runner
# =============================================================================

def run_standard_cell(cfg, n_episodes, out_dir, log_every=5):
    cell = cfg["cell_name"]
    t0 = time.time()
    game, agents = hb.build_agents(cfg)
    horizon = cfg["horizon"]
    memory = cfg["memory"]

    episodes = []
    cell_dir = os.path.join(out_dir, cell)
    os.makedirs(cell_dir, exist_ok=True)
    traj_path = os.path.join(cell_dir, "trajectories.jsonl")
    with open(traj_path, "w") as ftraj:
        for ep_idx in range(n_episodes):
            ep = hb.run_episode(game, agents, horizon, memory)
            episodes.append(ep)
            ftraj.write(json.dumps({"episode": ep_idx, "steps": ep}) + "\n")
            if (ep_idx + 1) % log_every == 0:
                print(f"  [{cell}] ep {ep_idx+1}/{n_episodes} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    metrics = hb.compute_metrics(episodes, game, n_boot=2000,
                                  seed=cfg.get("seed", 0))
    metrics["cell"] = cell
    metrics["wall_time_s"] = time.time() - t0
    metrics["config"] = {k: v for k, v in cfg.items()
                         if k not in ("models_het", "temps_het", "roles",
                                       "api_base_map")}

    team_payoffs = [float(np.mean([np.mean(s["rewards"]) for s in ep]))
                    for ep in episodes]
    metrics["team_mean_payoff"] = float(np.mean(team_payoffs))

    with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  [{cell}] payoff={metrics['cooperation_payoff']:.4f} "
          f"team={metrics['team_mean_payoff']:.4f} "
          f"div={metrics['perspective_diversity']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    del agents
    gc.collect()
    return metrics


# =============================================================================
# Main
# =============================================================================

def run_game_seed(args, game, seed):
    n_agents = 4 if game == "public_goods" else None
    seed_dir = os.path.join(args.out_dir, game, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    total_t0 = time.time()
    for cell in args.cells:
        mpath = os.path.join(seed_dir, cell, "metrics.json")
        if (not args.force) and os.path.exists(mpath):
            print(f"  [skip] {cell} (exists)", flush=True)
            continue
        cfg = make_config(game, cell, seed, n_agents=n_agents,
                          horizon=args.horizon, memory=args.memory)
        print(f"  [run]  {cell} @ {args.episodes}ep ...", flush=True)
        try:
            if cell == "het_gsaca":
                run_gsaca_cell(cfg, args.episodes, seed_dir,
                               log_every=args.log_every,
                               warmup=args.gsaca_warmup)
            else:
                run_standard_cell(cfg, args.episodes, seed_dir,
                                  log_every=args.log_every)
        except Exception as e:
            print(f"  [ERROR] {cell}: {e}")
            import traceback
            traceback.print_exc()
    print(f"  --- {game} seed {seed} done in {time.time()-total_t0:.0f}s ---",
          flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="GSACA experiment runner (vLLM local GPU)")
    ap.add_argument("--games", type=str, nargs="+", default=ALL_GAMES)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--memory", type=int, default=2)
    ap.add_argument("--cells", type=str, nargs="+",
                    default=["het_notom", "het_gated_atom_talk",
                             "het_dp_gated_atom_talk", "het_gsaca"])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--gsaca_warmup", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total_t0 = time.time()

    for game in args.games:
        print(f"\n{'='*60}")
        print(f"GAME: {game}")
        print(f"{'='*60}", flush=True)
        for seed in args.seeds:
            print(f"\n--- {game} seed {seed} @ "
                  f"{datetime.now().strftime('%H:%M:%S')} ---", flush=True)
            run_game_seed(args, game, seed)

    print(f"\n{'='*60}")
    print(f"Complete: {time.time()-total_t0:.0f}s total")
    print(f"Output: {args.out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
