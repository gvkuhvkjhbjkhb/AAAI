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
from safe_sca import (
    CoverageCertifiedEstimator,
    SafeSCAConfig,
    configure_agents_for_arm,
    select_point_estimate_arm,
    select_safe_arm,
)

# No API — triggers local HF model loading in LLMAgent._ensure_model()
MODELS_HET = ["Qwen/Qwen2.5-7B-Instruct", "THUDM/GLM-4-9B-0414"]
MODEL_HOMO = "Qwen/Qwen2.5-7B-Instruct"

# vLLM endpoints (used when --use_vllm is set)
VLLM_QWEN = "http://localhost:8000/v1"
VLLM_GLM = "http://localhost:8001/v1"
VLLM_API_KEY = "dummy"
VLLM_API_BASE_MAP = {
    "Qwen/Qwen2.5-7B-Instruct": VLLM_QWEN,
    "THUDM/GLM-4-9B-0414": VLLM_GLM,
}
# Allow extra / override endpoints via env (e.g. Llama for the QL pair):
#   VLLM_ENDPOINTS='{"NousResearch/Meta-Llama-3.1-8B-Instruct":"http://localhost:8002/v1"}'
_env_eps = os.environ.get("VLLM_ENDPOINTS")
if _env_eps:
    try:
        VLLM_API_BASE_MAP.update(json.loads(_env_eps))
    except Exception as _e:
        print(f"[vLLM] bad VLLM_ENDPOINTS env: {_e}", flush=True)

_original_build_agents = hb.build_agents

def _patched_build_agents_vllm(config):
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
    base_payoff = game.payoff
    rng = np.random.RandomState(12345)
    def noisy_payoff(actions):
        raw = list(base_payoff(actions))
        return [r + rng.normal(0, noise_std) for r in raw]
    game.payoff = noisy_payoff
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
        # vLLM: when use_vllm is set, inject api_base for API-mode inference
        "cell_name": cell_name,
        # hyperparameters (defaults preserve original behavior)
        "gate_trust_threshold": getattr(args, "gate_trust_threshold", 0.6) if args else 0.6,
        "gate_ema_alpha": getattr(args, "gate_ema_alpha", 0.3) if args else 0.3,
        "atom_warmup": getattr(args, "atom_warmup", 3) if args else 3,
        # reproducible generation seed base passed through to vLLM requests
        "gen_seed_base": getattr(args, "gen_seed_base", 1000) if args else 1000,
        # nucleus-sampling top_p (0.9 frozen main table; 1.0 = G2 ablation)
        "top_p": getattr(args, "top_p", 0.9) if args else 0.9,
    }
    # inject vLLM API routing if requested
    use_vllm = getattr(args, "use_vllm", False) if args else False
    if use_vllm:
        base["api_base"] = VLLM_QWEN
        base["api_base_map"] = VLLM_API_BASE_MAP
        base["api_key"] = VLLM_API_KEY
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
    # ---- end-to-end 2-arm attainment bandit (G1) ----
    # Probes NoAlign(=NoToM) and Gated arms for K episodes each, then commits
    # the remaining episodes to the probe-mean winner. Removes the offline
    # arm-labeling of the v6 §6.5 reconstruction.
    elif cell_name == "het_bandit":
        base.update(homogeneous=False, use_tom=False, use_talk=False,
                    adaptive_tom=False)
    # ---- S1: all decision policies begin with the same NoAlign warm-up. ----
    # The runner switches the agent pool post-warm-up; labels are used only by
    # het_oracle_sca, a clearly marked non-deployable diagnostic upper bound.
    elif cell_name in {"het_point_sca", "het_safe_sca", "het_oracle_sca"}:
        base.update(homogeneous=False, use_tom=False, use_talk=False,
                    adaptive_tom=False, gated_talk_tom=False,
                    diversity_preserving_gate=False)
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


def _arm_config(cfg, arm):
    """Derive an arm-specific config from the bandit base cfg.
    NoAlign = abstain (independent NoToM reasoning); Gated = forced alignment."""
    c = dict(cfg)
    if arm == "NoAlign":
        c.update(use_tom=False, use_talk=False, adaptive_tom=False,
                 gated_talk_tom=False, diversity_preserving_gate=False)
    else:  # Gated
        c.update(use_tom=True, use_talk=True, adaptive_tom=True,
                 gated_talk_tom=True, diversity_preserving_gate=False)
    return c


def run_bandit_cell(cfg, n_episodes, out_dir, log_every=5, bandit_k=5, noise_std=0.0):
    """End-to-end 2-arm attainment bandit (G1, replaces v6 §6.5 offline recon).

    For each (game, seed): probe NoAlign and Gated for K episodes each (order
    alternated by seed parity to remove run-order effects), select the
    probe-mean winner, then commit the remaining (n_episodes - 2K) episodes to
    that arm with fresh agents. Records probe means, the selection, and the
    delivered commit payoff. No game-label / oracle arm assignment is used.
    """
    cell = cfg["cell_name"]
    t0 = time.time()
    horizon, memory = cfg["horizon"], cfg["memory"]
    seed = cfg.get("seed", 0)
    n_commit = n_episodes - 2 * bandit_k
    if n_commit <= 0:
        raise ValueError(f"n_episodes={n_episodes} must exceed 2*bandit_k={2*bandit_k}")

    arms = ["NoAlign", "Gated"]
    if seed % 2 == 1:            # odd seed -> Gated first (Latin-style balance)
        arms = ["Gated", "NoAlign"]

    probe_means, probe_episodes = {}, {}
    for arm in arms:
        ac = _arm_config(cfg, arm)
        game, agents = hb.build_agents(ac)
        wrap_payoff_noise(game, noise_std)
        eps = [hb.run_episode(game, agents, horizon, memory) for _ in range(bandit_k)]
        probe_episodes[arm] = eps
        probe_means[arm] = float(np.mean(
            [np.mean([np.mean(s["rewards"]) for s in ep]) for ep in eps]))
        del agents; gc.collect()
        print(f"  [{cell}] probe {arm}: mean={probe_means[arm]:.4f} (K={bandit_k})", flush=True)

    # select winner by probe mean; ties -> NoAlign (conservative abstain)
    chosen = "Gated" if probe_means["Gated"] > probe_means["NoAlign"] else "NoAlign"

    cc = _arm_config(cfg, chosen)
    game, agents = hb.build_agents(cc)
    wrap_payoff_noise(game, noise_std)
    commit_episodes = []
    for ep_idx in range(n_commit):
        ep = hb.run_episode(game, agents, horizon, memory)
        commit_episodes.append(ep)
        if (ep_idx + 1) % log_every == 0:
            print(f"  [{cell}] commit {chosen} ep {ep_idx+1}/{n_commit} ({time.time()-t0:.0f}s)", flush=True)

    metrics = hb.compute_metrics(commit_episodes, game, n_boot=2000, seed=seed)
    metrics["cell"] = cell
    metrics["bandit_k"] = bandit_k
    metrics["bandit_probe_order"] = arms
    metrics["bandit_probe_mean_NoAlign"] = probe_means["NoAlign"]
    metrics["bandit_probe_mean_Gated"] = probe_means["Gated"]
    metrics["bandit_chosen_arm"] = chosen
    metrics["bandit_n_commit"] = n_commit
    for arm in arms:
        metrics[f"bandit_probe_payoffs_{arm}"] = [
            float(np.mean([np.mean(s["rewards"]) for s in ep])) for ep in probe_episodes[arm]]
    metrics["wall_time_s"] = time.time() - t0
    metrics["config"] = {k: v for k, v in cfg.items() if k not in ("models_het", "temps_het", "roles")}
    team_payoffs = [float(np.mean([np.mean(s["rewards"]) for s in ep])) for ep in commit_episodes]
    metrics["team_mean_payoff"] = float(np.mean(team_payoffs))

    cell_dir = os.path.join(out_dir, cell)
    os.makedirs(cell_dir, exist_ok=True)
    with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(cell_dir, "trajectories.jsonl"), "w") as ftraj:
        for ep_idx, ep in enumerate(commit_episodes):
            ftraj.write(json.dumps({"episode": ep_idx, "phase": "commit",
                                    "arm": chosen, "steps": ep}) + "\n")
        for arm in arms:
            for ep_idx, ep in enumerate(probe_episodes[arm]):
                ftraj.write(json.dumps({"episode": ep_idx, "phase": "probe",
                                        "arm": arm, "steps": ep}) + "\n")

    print(f"  [{cell}] chosen={chosen} probe(N={probe_means['NoAlign']:.3f},"
          f"G={probe_means['Gated']:.3f}) commit_payoff={metrics['cooperation_payoff']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del agents; gc.collect()
    return metrics


def _episode_team_payoff(episode):
    """Mean per-agent, per-round payoff for one episode."""
    return float(np.mean([np.mean(step["rewards"]) for step in episode]))


def _s1_config_from_args(args):
    return SafeSCAConfig(
        warmup_episodes=args.safe_warmup,
        tau=args.safe_tau,
        confidence=args.safe_confidence,
        bootstrap_samples=args.safe_bootstrap_samples,
        min_profile_coverage=args.safe_min_profile_coverage,
        min_stratum_observations=args.safe_min_stratum_observations,
    )


def run_s1_policy_cell(cfg, n_episodes, out_dir, args, log_every=5, noise_std=0.0):
    """Run one fully end-to-end S1 policy with total-horizon accounting.

    ``het_point_sca`` is the unconstrained point-estimate baseline.
    ``het_safe_sca`` is label-free Coverage-Certified Safe-SCA.
    ``het_oracle_sca`` uses a game-class label only as a diagnostic ceiling;
    its metrics explicitly mark it non-deployable.

    All three policies collect exactly the same NoAlign warm-up observations,
    preserve the same agent objects into the commit phase, and compute primary
    metrics over all episodes.  This makes trust-EMA transition cost visible
    instead of replacing the selected arm with fresh agents.
    """
    cell = cfg["cell_name"]
    if cell not in {"het_point_sca", "het_safe_sca", "het_oracle_sca"}:
        raise ValueError(f"S1 policy runner received unsupported cell {cell}")
    safe_cfg = _s1_config_from_args(args)
    if safe_cfg.warmup_episodes >= n_episodes:
        raise ValueError("safe_warmup must be smaller than episodes")

    t0 = time.time()
    seed = int(cfg.get("seed", 0))
    game, agents = hb.build_agents(cfg)
    wrap_payoff_noise(game, noise_std)
    horizon, memory = cfg["horizon"], cfg["memory"]
    estimator = CoverageCertifiedEstimator()
    warmup_episodes, commit_episodes = [], []

    # Frozen S1 protocol: NoAlign is the warm-up policy for all deployable
    # policy cells.  It is both safer and does not pre-warm Gated's EMA.
    configure_agents_for_arm(agents, "NoAlign")
    for ep_idx in range(safe_cfg.warmup_episodes):
        episode = hb.run_episode(game, agents, horizon, memory)
        warmup_episodes.append(episode)
        for step in episode:
            estimator.observe(step["actions"], step["rewards"])

    summary = estimator.summarize(
        n_agents=game.n_agents,
        n_actions=game.n_actions,
        config=safe_cfg,
        seed=seed,
    )
    if cell == "het_safe_sca":
        chosen_arm, reasons = select_safe_arm(summary, safe_cfg)
        decision_source = "coverage_certified_label_free"
        oracle_label_used = False
    elif cell == "het_point_sca":
        chosen_arm, reasons = select_point_estimate_arm(summary, tau=args.point_sca_tau)
        decision_source = "unconstrained_point_estimate"
        oracle_label_used = False
    else:
        oracle = ORACLE_STRUCTURE.get(cfg["game"])
        if oracle is None:
            raise ValueError(f"No oracle structure registered for {cfg['game']}")
        chosen_arm = "Gated" if oracle == "coord" else "NoAlign"
        reasons = ["oracle_game_class_for_diagnostic_upper_bound"]
        decision_source = "oracle_label_non_deployable"
        oracle_label_used = True

    configure_agents_for_arm(agents, chosen_arm)
    print(f"  [{cell}] warmup={safe_cfg.warmup_episodes} split="
          f"{summary.split_score if summary.split_score is not None else float('nan'):.3f} "
          f"ucb={summary.split_upper_bound if summary.split_upper_bound is not None else float('nan'):.3f} "
          f"coverage={summary.profile_coverage:.3f} -> {chosen_arm} "
          f"({','.join(reasons)})", flush=True)

    for ep_idx in range(safe_cfg.warmup_episodes, n_episodes):
        episode = hb.run_episode(game, agents, horizon, memory)
        commit_episodes.append(episode)
        if (ep_idx + 1) % log_every == 0:
            print(f"  [{cell}] ep {ep_idx + 1}/{n_episodes} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    all_episodes = warmup_episodes + commit_episodes
    metrics = hb.compute_metrics(all_episodes, game, n_boot=2000, seed=seed)
    post_metrics = hb.compute_metrics(commit_episodes, game, n_boot=2000, seed=seed + 1)
    warmup_payoffs = [_episode_team_payoff(ep) for ep in warmup_episodes]
    commit_payoffs = [_episode_team_payoff(ep) for ep in commit_episodes]
    all_payoffs = warmup_payoffs + commit_payoffs
    # Do not read a game-class label for deployable S1 cells. The label exists
    # only inside the explicitly named, non-deployable oracle comparator.
    oracle_for_record = (
        ORACLE_STRUCTURE.get(cfg["game"], "unknown") if oracle_label_used else None
    )
    detected_structure = (
        "anti_coord" if summary.split_score is not None and summary.split_score > 0 else "coord"
    )

    metrics.update({
        "cell": cell,
        "wall_time_s": time.time() - t0,
        "config": {k: v for k, v in cfg.items() if k not in ("models_het", "temps_het", "roles")},
        "s1_schema_version": 1,
        "s1_policy": cell,
        "s1_decision_source": decision_source,
        "s1_oracle_label_used": oracle_label_used,
        "s1_warmup_arm": "NoAlign",
        "s1_selected_post_warmup_arm": chosen_arm,
        "s1_decision_reasons": reasons,
        "s1_oracle_structure_for_reporting_only": oracle_for_record,
        "s1_point_detected_structure_for_reporting_only": detected_structure,
        "s1_safe_config": safe_cfg.to_dict(),
        "s1_coverage": summary.to_dict(),
        "s1_total_team_payoff": float(np.mean(all_payoffs)),
        "s1_warmup_team_payoff": float(np.mean(warmup_payoffs)),
        "s1_commit_team_payoff": float(np.mean(commit_payoffs)),
        "s1_total_episode_team_payoffs": all_payoffs,
        "s1_warmup_episode_team_payoffs": warmup_payoffs,
        "s1_commit_episode_team_payoffs": commit_payoffs,
        "s1_post_commit_metrics": post_metrics,
    })

    cell_dir = os.path.join(out_dir, cell)
    os.makedirs(cell_dir, exist_ok=True)
    decision_record = {
        "schema_version": 1,
        "cell": cell,
        "game": cfg["game"],
        "seed": seed,
        "decision_source": decision_source,
        "oracle_label_used": oracle_label_used,
        "warmup_arm": "NoAlign",
        "selected_post_warmup_arm": chosen_arm,
        "reasons": reasons,
        "safe_config": safe_cfg.to_dict(),
        "coverage": summary.to_dict(),
        "warmup_observations": [
            {"actions": list(actions), "team_payoff": payoff}
            for actions, payoff in estimator.observations
        ],
    }
    with open(os.path.join(cell_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(cell_dir, "decision.json"), "w") as handle:
        json.dump(decision_record, handle, indent=2)
    with open(os.path.join(cell_dir, "trajectories.jsonl"), "w") as handle:
        for ep_idx, episode in enumerate(warmup_episodes):
            handle.write(json.dumps({"episode": ep_idx, "phase": "warmup",
                                     "arm": "NoAlign", "steps": episode}) + "\n")
        for ep_idx, episode in enumerate(commit_episodes, start=safe_cfg.warmup_episodes):
            handle.write(json.dumps({"episode": ep_idx, "phase": "commit",
                                     "arm": chosen_arm, "steps": episode}) + "\n")

    print(f"  [{cell}] total_team={metrics['s1_total_team_payoff']:.4f} "
          f"commit_team={metrics['s1_commit_team_payoff']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del agents; gc.collect()
    return metrics


def latin_square_order(cells, seed):
    """Balance arm order within each (game, seed) using a fixed Latin square
    keyed on the seed. For the canonical n=3 arm set the rotation matches the
    pre-registered schedule:
      seeds 42,45,48,... -> cells as given (NoToM, Gated, CGA)
      seeds 43,46,49,... -> rotate by 1 (Gated, CGA, NoToM)
      seeds 44,47,50,... -> rotate by 2 (CGA, NoToM, Gated)
    Generalizes to any k arms via rotation by (seed % k)."""
    k = len(cells)
    shift = seed % k
    return [cells[(i + shift) % k] for i in range(k)]


def run_game_seed(args, game, seed):
    n_agents = 4 if game == "public_goods" else None
    seed_dir = os.path.join(args.out_dir, game, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    random.seed(seed); np.random.seed(seed)
    total_t0 = time.time()
    if getattr(args, "latin_square", False):
        run_cells = latin_square_order(args.cells, seed)
    else:
        run_cells = list(args.cells)
    # record the arm order actually used for this (game, seed) in a manifest
    with open(os.path.join(seed_dir, "arm_order.json"), "w") as _f:
        json.dump({"game": game, "seed": seed, "arm_order": run_cells,
                   "latin_square": bool(getattr(args, "latin_square", False)),
                   "timestamp": datetime.now().isoformat()}, _f, indent=2)
    for cell in run_cells:
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
            elif cell == "het_bandit":
                run_bandit_cell(cfg, args.episodes, seed_dir,
                                log_every=args.log_every, bandit_k=args.bandit_k,
                                noise_std=args.payoff_noise_std)
            elif cell in {"het_point_sca", "het_safe_sca", "het_oracle_sca"}:
                run_s1_policy_cell(cfg, args.episodes, seed_dir, args,
                                   log_every=args.log_every,
                                   noise_std=args.payoff_noise_std)
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
    ap.add_argument("--bandit_k", type=int, default=5,
                    help="G1 bandit: probe episodes per arm (NoAlign & Gated)")
    # S1 Coverage-Certified Safe-SCA.  These values are written into every
    # decision record and must be frozen before any held-out test execution.
    ap.add_argument("--safe_warmup", type=int, default=10,
                    help="S1: NoAlign warm-up episodes before a policy decision")
    ap.add_argument("--safe_tau", type=float, default=0.10,
                    help="S1: require split-score UCB < -tau before alignment")
    ap.add_argument("--safe_confidence", type=float, default=0.95,
                    help="S1: one-sided bootstrap confidence for split-score UCB")
    ap.add_argument("--safe_bootstrap_samples", type=int, default=2000,
                    help="S1: bootstrap draws used for the coverage certificate")
    ap.add_argument("--safe_min_profile_coverage", type=float, default=0.25,
                    help="S1: minimum unique joint-profile coverage required to align")
    ap.add_argument("--safe_min_stratum_observations", type=int, default=3,
                    help="S1: minimum same and differentiated observations required")
    ap.add_argument("--point_sca_tau", type=float, default=0.0,
                    help="S1 point-estimate comparator: align if split <= -tau")
    ap.add_argument("--top_p", type=float, default=0.9,
                    help="nucleus sampling top_p (0.9 frozen; 1.0 = G2 ablation)")
    ap.add_argument("--models_het", type=str, nargs=2, default=MODELS_HET)
    ap.add_argument("--model_homo", type=str, default=MODEL_HOMO)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--use_vllm", action="store_true",
                    help="Use vLLM API servers instead of local transformers")
    ap.add_argument("--latin_square", action="store_true",
                    help="Balance arm run-order within each (game,seed) via a "
                         "seed-keyed Latin square (removes server-warmup order bias)")
    ap.add_argument("--gen_seed_base", type=int, default=1000,
                    help="Base for the reproducible per-request vLLM generation "
                         "seed; final seed = gen_seed_base + experiment_seed*100 + agent_id")
    ap.add_argument("--auto_episodes", action="store_true",
                    help="Override --episodes per game: public_goods=20, others=30 "
                         "(matches the frozen main-table protocol; lets one worker "
                         "span mixed 2-player + public_goods games)")
    args = ap.parse_args()

    if any(cell in {"het_point_sca", "het_safe_sca", "het_oracle_sca"}
           for cell in args.cells):
        try:
            _s1_config_from_args(args).validate()
        except ValueError as exc:
            ap.error(f"invalid S1 Safe-SCA configuration: {exc}")

    # apply vLLM monkey-patch if requested
    if args.use_vllm:
        hb.build_agents = _patched_build_agents_vllm
        print("[vLLM] API mode enabled: Qwen→localhost:8000, GLM→localhost:8001", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    total_t0 = time.time()

    for game in args.games:
        if args.auto_episodes:
            args.episodes = 20 if game == "public_goods" else 30
        print(f"\n{'='*60}\nGAME: {game} (episodes={args.episodes})\n{'='*60}", flush=True)
        for seed in args.seeds:
            print(f"\n--- {game} seed {seed} @ {datetime.now().strftime('%H:%M:%S')} ---", flush=True)
            run_game_seed(args, game, seed)

    print(f"\n{'='*60}\nComplete: {time.time()-total_t0:.0f}s total\nOutput: {args.out_dir}\n{'='*60}")


if __name__ == "__main__":
    main()
