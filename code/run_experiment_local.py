#!/usr/bin/env python3
"""GSACA + Public Goods experiment runner — local HF GPU edition.

Loads Qwen2.5-7B-Instruct + GLM-4-9B-0414 in 4-bit on the assigned GPU.
No API calls, no network latency. Each worker process loads both models once
(~9GB VRAM in 4-bit) and reuses them across all cells.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 run_experiment_local.py --games chicken ... &
  CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py --games stag_hunt ... &
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

# No API — triggers local HF model loading in LLMAgent._ensure_model()
MODELS_HET = ["Qwen/Qwen2.5-7B-Instruct", "THUDM/GLM-4-9B-0414"]
MODEL_HOMO = "Qwen/Qwen2.5-7B-Instruct"

ALL_GAMES = ["chicken", "hawk_dove", "deadlock",
             "stag_hunt", "battle_of_the_sexes", "public_goods",
             "matching_pennies"]

ORACLE_STRUCTURE = {
    "chicken": "anti_coord", "hawk_dove": "anti_coord",
    "deadlock": "anti_coord", "stag_hunt": "coord",
    "battle_of_the_sexes": "coord", "public_goods": "coord",
    "matching_pennies": "anti_coord",
}


class GameStructureEstimator:
    def __init__(self):
        self.observations = []

    def observe(self, actions, rewards):
        self.observations.append((tuple(actions), float(np.mean(rewards))))

    def estimate(self):
        if len(self.observations) < 3:
            return "anti_coord", 0.0, len(self.observations)
        same_p, diff_p = [], []
        for actions, payoff in self.observations:
            (same_p if len(set(actions)) == 1 else diff_p).append(payoff)
        if not same_p or not diff_p:
            return "anti_coord", 0.0, len(self.observations)
        split_score = float(np.mean(diff_p)) - float(np.mean(same_p))
        return ("anti_coord" if split_score > 0 else "coord"), split_score, len(self.observations)


def wrap_payoff_noise(game, noise_std):
    """Wrap a game's payoff callable to add i.i.d. Gaussian noise (stress test
    for the GSACA structure estimator). noise_std=0 leaves the game unchanged."""
    if not noise_std or noise_std <= 0:
        return game
    base_payoff = game.base.payoff
    rng = np.random.RandomState(12345)
    def noisy_payoff(actions):
        raw = list(base_payoff(actions))
        return [r + rng.normal(0, noise_std) for r in raw]
    game.base.payoff = noisy_payoff
    return game


def make_config(game_name, cell_name, seed, n_agents=None,
                horizon=5, memory=2, args=None):
    if n_agents is None:
        n_agents = 4 if game_name == "public_goods" else 2
    models_het = getattr(args, "models_het", MODELS_HET) if args else MODELS_HET
    model_homo = getattr(args, "model_homo", MODEL_HOMO) if args else MODEL_HOMO
    base = {
        "game": game_name, "n_agents": n_agents,
        "horizon": horizon, "memory": memory,
        "models_homo": model_homo, "models_het": models_het,
        "temp_homo": 0.7, "temps_het": [0.5, 0.8],
        "roles": [f"player{i+1}" for i in range(n_agents)],
        "tom_order": 1, "seed": seed,
        # NO api_base → triggers local HF model loading
        "cell_name": cell_name,
        # hyperparameters (defaults preserve original behavior)
        "gate_trust_threshold": getattr(args, "gate_trust_threshold", 0.6) if args else 0.6,
        "gate_ema_alpha": getattr(args, "gate_ema_alpha", 0.3) if args else 0.3,
        "atom_warmup": getattr(args, "atom_warmup", 3) if args else 3,
    }
    # ---- heterogeneous cells (original) ----
    if cell_name == "het_notom":
        base.update(homogeneous=False, use_tom=False, use_talk=False, adaptive_tom=False)
    elif cell_name == "het_gated_atom_talk":
        base.update(homogeneous=False, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True, diversity_preserving_gate=False)
    elif cell_name == "het_dp_gated_atom_talk":
        base.update(homogeneous=False, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True, diversity_preserving_gate=True)
    elif cell_name == "het_gsaca":
        base.update(homogeneous=False, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True, diversity_preserving_gate=True)
    # ---- V2 anti-coordination enhancement schemes (build on het_gsaca) ----
    elif cell_name == "het_role_asym":
        base.update(homogeneous=False, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True, diversity_preserving_gate=True,
                    role_asymmetric_hint=True)
    elif cell_name == "het_hist_split":
        base.update(homogeneous=False, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True, diversity_preserving_gate=True,
                    history_split_hint=True)
    elif cell_name == "het_adapt_interv":
        base.update(homogeneous=False, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True, diversity_preserving_gate=True,
                    adaptive_intervention=True, adaptive_interv_threshold=0.3)
    elif cell_name == "het_combo_anti":
        base.update(homogeneous=False, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True, diversity_preserving_gate=True,
                    role_asymmetric_hint=True, history_split_hint=True,
                     adaptive_intervention=True, adaptive_interv_threshold=0.3)
    # ---- 3-arm GSACA with abstention (split>τ→CGA, split<-τ→Gated, |split|≤τ→NoToM) ----
    elif cell_name == "het_3arm":
        base.update(homogeneous=False, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True, diversity_preserving_gate=True)
    # ---- payoff-in-prompt baseline (no alignment, LLM sees full matrix) ----
    elif cell_name == "het_payoff_prompt":
        base.update(homogeneous=False, use_tom=False, use_talk=False,
                    adaptive_tom=False, payoff_in_prompt=True)
    # ---- homogeneous control cells (same model both agents) ----
    elif cell_name == "hom_notom":
        base.update(homogeneous=True, use_tom=False, use_talk=False, adaptive_tom=False)
    elif cell_name == "hom_gated_atom_talk":
        base.update(homogeneous=True, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True, diversity_preserving_gate=False)
    elif cell_name == "hom_dp_gated_atom_talk":
        base.update(homogeneous=True, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True, diversity_preserving_gate=True)
    elif cell_name == "hom_gsaca":
        base.update(homogeneous=True, use_tom=True, use_talk=True,
                    adaptive_tom=True, gated_talk_tom=True, diversity_preserving_gate=True)
    else:
        raise ValueError(f"Unknown cell: {cell_name}")
    return base


def run_3arm_cell(cfg, n_episodes, out_dir, log_every=5, warmup=5, noise_std=0.0, abstain_tau=0.4):
    """3-arm GSACA: split>τ→CGA, split<-τ→Gated, |split|≤τ→NoToM (abstain)."""
    cell = cfg["cell_name"]
    t0 = time.time()
    game, agents = hb.build_agents(cfg)
    wrap_payoff_noise(game, noise_std)
    horizon, memory = cfg["horizon"], cfg["memory"]
    estimator = GameStructureEstimator()
    episodes = []

    for ep_idx in range(warmup):
        ep = hb.run_episode(game, agents, horizon, memory)
        episodes.append(ep)
        for step in ep:
            estimator.observe(step["actions"], step["rewards"])

    structure, split_score, n_obs = estimator.estimate()
    oracle = ORACLE_STRUCTURE.get(cfg["game"], "unknown")

    if split_score > abstain_tau:
        arm = "CGA"
    elif split_score < -abstain_tau:
        arm = "Gated"
        for ag in agents:
            ag.gated_talk_tom = False
            ag.diversity_preserving_gate = False
    else:
        arm = "NoToM"
        for ag in agents:
            ag.gated_talk_tom = False
            ag.diversity_preserving_gate = False
            ag.use_tom = False
            ag.use_talk = False

    detection_correct = (structure == oracle)
    print(f"  [{cell}] split={split_score:.3f} τ={abstain_tau} → arm={arm} (oracle={oracle})", flush=True)

    for ep_idx in range(warmup, n_episodes):
        ep = hb.run_episode(game, agents, horizon, memory)
        episodes.append(ep)
        if (ep_idx + 1) % log_every == 0:
            print(f"  [{cell}] ep {ep_idx+1}/{n_episodes} ({time.time()-t0:.0f}s)", flush=True)

    metrics = hb.compute_metrics(episodes, game, n_boot=2000, seed=cfg.get("seed", 0))
    metrics["cell"] = cell
    metrics["gsaca_detected_structure"] = structure
    metrics["gsaca_oracle_structure"] = oracle
    metrics["gsaca_detection_correct"] = detection_correct
    metrics["gsaca_split_score"] = split_score
    metrics["gsaca_warmup_episodes"] = warmup
    metrics["gsaca_n_observations"] = n_obs
    metrics["gsaca_3arm_selected"] = arm
    metrics["gsaca_3arm_tau"] = abstain_tau
    metrics["wall_time_s"] = time.time() - t0
    metrics["config"] = {k: v for k, v in cfg.items() if k not in ("models_het", "temps_het", "roles")}
    team_payoffs = [float(np.mean([np.mean(s["rewards"]) for s in ep])) for ep in episodes]
    metrics["team_mean_payoff"] = float(np.mean(team_payoffs))

    cell_dir = os.path.join(out_dir, cell)
    os.makedirs(cell_dir, exist_ok=True)
    with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(cell_dir, "trajectories.jsonl"), "w") as ftraj:
        for ep_idx, ep in enumerate(episodes):
            ftraj.write(json.dumps({"episode": ep_idx, "steps": ep}) + "\n")

    print(f"  [{cell}] arm={arm} split={split_score:.3f} payoff={metrics['cooperation_payoff']:.4f} "
          f"team={metrics['team_mean_payoff']:.4f} div={metrics['perspective_diversity']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del agents; gc.collect()
    return metrics


def run_gsaca_cell(cfg, n_episodes, out_dir, log_every=5, warmup=5, noise_std=0.0):
    cell = cfg["cell_name"]
    t0 = time.time()
    game, agents = hb.build_agents(cfg)
    wrap_payoff_noise(game, noise_std)
    horizon, memory = cfg["horizon"], cfg["memory"]
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
            print(f"  [{cell}] ep {ep_idx+1}/{n_episodes} ({time.time()-t0:.0f}s)", flush=True)

    metrics = hb.compute_metrics(episodes, game, n_boot=2000, seed=cfg.get("seed", 0))
    metrics["cell"] = cell
    metrics["gsaca_detected_structure"] = structure
    metrics["gsaca_oracle_structure"] = oracle
    metrics["gsaca_detection_correct"] = detection_correct
    metrics["gsaca_split_score"] = split_score
    metrics["gsaca_warmup_episodes"] = warmup
    metrics["gsaca_n_observations"] = n_obs
    metrics["wall_time_s"] = time.time() - t0
    metrics["config"] = {k: v for k, v in cfg.items() if k not in ("models_het", "temps_het", "roles")}
    team_payoffs = [float(np.mean([np.mean(s["rewards"]) for s in ep])) for ep in episodes]
    metrics["team_mean_payoff"] = float(np.mean(team_payoffs))

    cell_dir = os.path.join(out_dir, cell)
    os.makedirs(cell_dir, exist_ok=True)
    with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    traj_path = os.path.join(cell_dir, "trajectories.jsonl")
    with open(traj_path, "w") as ftraj:
        for ep_idx, ep in enumerate(episodes):
            ftraj.write(json.dumps({"episode": ep_idx, "steps": ep}) + "\n")

    print(f"  [{cell}] detect={structure}(oracle={oracle},{'OK' if detection_correct else 'MISS'}) "
          f"split={split_score:.3f} payoff={metrics['cooperation_payoff']:.4f} "
          f"team={metrics['team_mean_payoff']:.4f} div={metrics['perspective_diversity']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del agents; gc.collect()
    return metrics


def run_standard_cell(cfg, n_episodes, out_dir, log_every=5, noise_std=0.0):
    cell = cfg["cell_name"]
    t0 = time.time()
    game, agents = hb.build_agents(cfg)
    wrap_payoff_noise(game, noise_std)
    horizon, memory = cfg["horizon"], cfg["memory"]
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
                print(f"  [{cell}] ep {ep_idx+1}/{n_episodes} ({time.time()-t0:.0f}s)", flush=True)

    metrics = hb.compute_metrics(episodes, game, n_boot=2000, seed=cfg.get("seed", 0))
    metrics["cell"] = cell
    metrics["wall_time_s"] = time.time() - t0
    metrics["config"] = {k: v for k, v in cfg.items() if k not in ("models_het", "temps_het", "roles")}
    team_payoffs = [float(np.mean([np.mean(s["rewards"]) for s in ep])) for ep in episodes]
    metrics["team_mean_payoff"] = float(np.mean(team_payoffs))

    with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [{cell}] payoff={metrics['cooperation_payoff']:.4f} "
          f"team={metrics['team_mean_payoff']:.4f} div={metrics['perspective_diversity']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del agents; gc.collect()
    return metrics


def run_game_seed(args, game, seed):
    n_agents = 4 if game == "public_goods" else None
    seed_dir = os.path.join(args.out_dir, game, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    random.seed(seed); np.random.seed(seed)
    total_t0 = time.time()
    for cell in args.cells:
        mpath = os.path.join(seed_dir, cell, "metrics.json")
        if (not args.force) and os.path.exists(mpath):
            print(f"  [skip] {cell} (exists)", flush=True)
            continue
        cfg = make_config(game, cell, seed, n_agents=n_agents,
                          horizon=args.horizon, memory=args.memory, args=args)
        print(f"  [run]  {cell} @ {args.episodes}ep ...", flush=True)
        try:
            if cell == "het_3arm":
                run_3arm_cell(cfg, args.episodes, seed_dir,
                              log_every=args.log_every, warmup=args.gsaca_warmup,
                              noise_std=args.payoff_noise_std, abstain_tau=args.abstain_tau)
            elif cell.endswith("gsaca"):
                run_gsaca_cell(cfg, args.episodes, seed_dir,
                               log_every=args.log_every, warmup=args.gsaca_warmup,
                               noise_std=args.payoff_noise_std)
            else:
                run_standard_cell(cfg, args.episodes, seed_dir, log_every=args.log_every,
                                  noise_std=args.payoff_noise_std)
        except Exception as e:
            print(f"  [ERROR] {cell}: {e}")
            import traceback; traceback.print_exc()
    print(f"  --- {game} seed {seed} done in {time.time()-total_t0:.0f}s ---", flush=True)


def main():
    ap = argparse.ArgumentParser(description="GSACA experiment (local HF GPU)")
    ap.add_argument("--games", type=str, nargs="+", default=ALL_GAMES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--memory", type=int, default=2)
    ap.add_argument("--cells", type=str, nargs="+",
                    default=["het_notom", "het_gated_atom_talk",
                             "het_dp_gated_atom_talk", "het_gsaca"])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--gsaca_warmup", type=int, default=5)
    ap.add_argument("--gate_trust_threshold", type=float, default=0.6)
    ap.add_argument("--gate_ema_alpha", type=float, default=0.3)
    ap.add_argument("--atom_warmup", type=int, default=3)
    ap.add_argument("--payoff_noise_std", type=float, default=0.0)
    ap.add_argument("--role_asymmetric_hint", action="store_true")
    ap.add_argument("--history_split_hint", action="store_true")
    ap.add_argument("--adaptive_intervention", action="store_true")
    ap.add_argument("--adaptive_interv_threshold", type=float, default=0.3)
    ap.add_argument("--abstain_tau", type=float, default=0.4)
    ap.add_argument("--models_het", type=str, nargs=2, default=MODELS_HET)
    ap.add_argument("--model_homo", type=str, default=MODEL_HOMO)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total_t0 = time.time()

    for game in args.games:
        print(f"\n{'='*60}\nGAME: {game}\n{'='*60}", flush=True)
        for seed in args.seeds:
            print(f"\n--- {game} seed {seed} @ {datetime.now().strftime('%H:%M:%S')} ---", flush=True)
            run_game_seed(args, game, seed)

    print(f"\n{'='*60}\nComplete: {time.time()-total_t0:.0f}s total\nOutput: {args.out_dir}\n{'='*60}")


if __name__ == "__main__":
    main()
