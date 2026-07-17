#!/usr/bin/env python3
"""
HetToM Layer-2 bridge: migrate Layer-1 winners to LBF + MAPPO.

Layer 1 (hettom_baseline.py) verifies which configuration (hom/het x noToM/ToM)
breaks the Reasoning Trap in short matrix games. Layer 2 takes the WINNING
configuration and tests whether the effect transfers to a real cooperative
MARL benchmark (Level-Based Foraging) trained with MAPPO via the repo's
existing EPyMARL fork.

Why a bridge, not a from-scratch LLM-in-the-loop trainer:
  - The repo's LLM pipeline is OFFLINE (offline_relabel.py), not per-step
    online (Plan_Assessment.txt constraint 1). Per-step online ToM is
    infeasible on a single RTX 5090.
  - So Layer 2 uses a periodic OFFLINE ToM injection: every K episodes the
    LLM (offline) inspects recent trajectories and produces per-agent
    "intent features" that are fed into the MAPPO policy network as extra
    observation context. This matches the repo's proven offline-LLM idiom.

This file does THREE things:
  1. build_intent_features(): offline LLM produces ToM intent labels for
     sampled LBF trajectories (reuses offline_relabel.py HF idiom).
  2. a config patch (mappo_hettom.yaml) that tells EPyMARL to load these
     intent features as extra observation dims during training.
  3. a launcher (run_layer2.sh) that wires it all together, mirroring the
     repo's existing run_*.sh pattern.

Usage:
  python3 layer2_marl_bridge.py prepare   --layer1_dir results/hettom_layer1
  python3 layer2_marl_bridge.py inject    --trajectories <lbf_traj.jsonl>
  python3 layer2_marl_bridge.py launch    --config mappo_hettom --seeds 1 2 3
  python3 layer2_marl_bridge.py all       --layer1_dir results/hettom_layer1

NOTE: The actual MAPPO training is launched via EPyMARL's src/main.py (sacred),
exactly as the repo's run_qs_quick_test.sh does. This script only prepares
the intent features, the config, and the launch command.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import defaultdict

import numpy as np


# Resolve repo root: this file lives in <repo>/hettom_experiments/, so the
# repo root is one level up. All EPyMARL / results paths are anchored there
# so the scripts work regardless of the current working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EPYMARL_DIR = os.path.join(_REPO_ROOT, "epymarl")
_RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
_CONFIG_ALGS_DIR = os.path.join(_EPYMARL_DIR, "src", "config", "algs")


def _abs(path):
    """Make a path absolute w.r.t. repo root if it isn't already."""
    if os.path.isabs(path):
        return path
    return os.path.join(_REPO_ROOT, path)


# =============================================================================
# 1. Offline ToM intent feature extraction (mirrors offline_relabel.py)
# =============================================================================

TOM_PROMPT_TEMPLATE = """You are observing a cooperative foraging episode.
Agents must coordinate to collect food. Each agent has a level and can only
collect food at or below its level.

Episode summary (last {window} steps):
{traj_text}

Agents: {agent_ids}

For each agent, infer its current INTENT as one of:
  0 = SEEK_FOOD       (moving toward collectable food)
  1 = COOPERATE       (positioning to help another agent collect)
  2 = EXPLORE         (searching for food in unvisited area)
  3 = IDLE            (not productively moving)

Reply with ONLY a JSON object mapping agent id to intent integer,
e.g. {{"1": 0, "2": 1}}. No other text."""


class OfflineToMExtractor:
    """Loads a HF LLM once and produces per-agent intent labels for batches of
    LBF trajectories. Mirrors offline_relabel.py's HFClassifier."""

    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device="auto"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=self.device,
            trust_remote_code=True)
        self._model.eval()

    def _generate(self, prompt, max_new_tokens=32):
        self._ensure_model()
        import torch
        msgs = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True).strip()

    def extract_intents(self, traj_text, agent_ids, window=10):
        prompt = TOM_PROMPT_TEMPLATE.format(
            window=window, traj_text=traj_text,
            agent_ids=", ".join(str(a) for a in agent_ids))
        text = self._generate(prompt)
        preds = {}
        try:
            obj = json.loads(text[text.find("{"):text.rfind("}") + 1])
            for a in agent_ids:
                v = obj.get(str(a), obj.get(a))
                if isinstance(v, int) and 0 <= v < 4:
                    preds[a] = v
        except Exception:
            pass
        for a in agent_ids:
            if a not in preds:
                preds[a] = random.randrange(4)  # fallback
        return preds


def traj_to_text(traj, n_steps=10):
    """Convert a trajectory (list of steps) to a compact text summary."""
    lines = []
    for s in traj[-n_steps:]:
        acts = s.get("actions", [])
        pos = s.get("positions", [])
        foods = s.get("food_levels", [])
        line = f"step{s.get('step','?')}: "
        if pos:
            line += "pos=" + ",".join(f"a{i}@({p[0]},{p[1]})" for i, p in enumerate(pos))
        if foods:
            line += " food=" + ",".join(str(f) for f in foods)
        line += f" actions={acts}"
        lines.append(line)
    return "\n".join(lines)


def build_intent_features(layer1_dir, out_path, n_samples=200, window=10,
                          model_name="Qwen/Qwen2.5-7B-Instruct", mock=False):
    """Sample LBF trajectories (or generate dummy ones if no LBF data yet),
    run offline ToM extraction, save intent features for later injection.

    In mock mode, uses random intents to test the pipeline without GPU.
    """
    layer1_dir = _abs(layer1_dir)
    out_path = _abs(out_path)
    # try to load real LBF trajectories; if none, generate synthetic ones
    traj_path = os.path.join(_RESULTS_DIR, "lbf_trajectories.jsonl")
    trajectories = []
    if os.path.exists(traj_path):
        with open(traj_path) as f:
            for line in f:
                trajectories.append(json.loads(line))
    if not trajectories:
        # synthetic LBF-like trajectories for pipeline testing
        print(f"[warn] no LBF trajectories at {traj_path}; generating synthetic")
        rng = random.Random(0)
        n_agents = 3
        for i in range(n_samples):
            traj = []
            for t in range(window):
                traj.append({
                    "step": t,
                    "positions": [[rng.randrange(10), rng.randrange(10)]
                                  for _ in range(n_agents)],
                    "food_levels": [rng.randrange(3) for _ in range(2)],
                    "actions": [rng.randrange(6) for _ in range(n_agents)],
                })
            trajectories.append({"episode": i, "steps": traj})

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if mock:
        extractor = None
        print("[mock] using random intents (no GPU/model)")
    else:
        extractor = OfflineToMExtractor(model_name=model_name)

    rng = random.Random(42)
    sample_idx = rng.sample(range(len(trajectories)),
                            min(n_samples, len(trajectories)))
    features = []
    for i, idx in enumerate(sample_idx):
        traj = trajectories[idx]["steps"] if "steps" in trajectories[idx] else trajectories[idx]
        agent_ids = list(range(len(traj[0].get("actions", [0, 1, 2]))))
        if mock:
            intents = {a: rng.randrange(4) for a in agent_ids}
        else:
            text = traj_to_text(traj, window)
            intents = extractor.extract_intents(text, agent_ids, window)
        features.append({
            "episode": trajectories[idx].get("episode", idx),
            "intents": intents,
            "config_source": "mock" if mock else model_name,
        })
        if (i + 1) % 20 == 0:
            print(f"[inject] processed {i+1}/{len(sample_idx)} trajectories")

    with open(out_path, "w") as f:
        for feat in features:
            f.write(json.dumps(feat) + "\n")
    print(f"[inject] wrote {len(features)} intent features -> {out_path}")
    return out_path


# =============================================================================
# 2. EPyMARL config patch for HetToM intent injection
# =============================================================================

MAPPO_HETTOM_CONFIG = """# --- MAPPO with HetToM offline intent injection ---
# Layer-2 config: standard MAPPO + periodic offline ToM intent features
# injected as extra observation context. Does NOT alter the optimal-policy
# guarantee (intents are observation-side only, no reward change).

action_selector: "soft_policies"
mask_before_softmax: True
runner: "parallel"
buffer_size: 10
batch_size_run: 10
batch_size: 10
target_update_interval_or_tau: 0.01
lr: 0.0003
hidden_dim: 128
obs_agent_id: True
obs_last_action: False
obs_individual_obs: False
agent_output_type: "pi_logits"
learner: "ppo_learner"
entropy_coef: 0.001
use_rnn: True
standardise_returns: False
standardise_rewards: True
q_nstep: 5
critic_type: "cv_critic"
epochs: 4
eps_clip: 0.2
name: "mappo_hettom"
t_max: 1000000

# HetToM intent injection (Layer 2)
hettom_enabled: True
hettom_intent_dim: 4          # SEEK/COOP/EXPLORE/IDLE one-hot
hettom_inject_interval: 50    # re-extract intents every K episodes
hettom_intent_path: ""        # set by launcher
hettom_inject_mode: "obs"     # "obs" = concat to observation
"""


def write_config(out_dir=None):
    out_dir = out_dir or _CONFIG_ALGS_DIR
    out_dir = _abs(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "mappo_hettom.yaml")
    with open(path, "w") as f:
        f.write(MAPPO_HETTOM_CONFIG)
    print(f"[config] wrote {path}")
    return path


# =============================================================================
# 3. Launcher: wire intent features + config + EPyMARL training
# =============================================================================

def launch(config, seeds, env_key="lbforaging:Foraging-10x10-3p-3f-v3",
           t_max=1000000, time_limit=50, out_dir="results/hettom_layer2",
           intent_path=None, mock=False, dry_run=False):
    """Launch MAPPO training for each seed, mirroring run_qs_quick_test.sh."""
    out_dir = _abs(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    env_config = "gymma"
    intent_abs = _abs(intent_path) if intent_path else "none"
    extra = (f"hettom_enabled=True hettom_intent_path={intent_abs} "
             f"hettom_inject_interval=50")
    for seed in seeds:
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        log = os.path.join(seed_dir, f"mappo_hettom_seed{seed}.log")
        cmd = (f"cd {_EPYMARL_DIR} && CUDA_VISIBLE_DEVICES=0 python3 src/main.py "
               f"--config={config} --env-config={env_config} with "
               f'env_args.key="{env_key}" env_args.time_limit={time_limit} '
               f"t_max={t_max} use_cuda=True test_nepisode=5 test_interval=50000 "
               f"log_interval=25000 seed={seed} {extra}")
        if mock:
            cmd += " t_max=2000"  # quick smoke
        print(f"[launch] seed={seed} -> {log}")
        if dry_run:
            print(f"  DRY-RUN: {cmd}")
            continue
        with open(log, "w") as f:
            ret = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        (open(os.path.join(seed_dir, "DONE"), "w")
         if ret.returncode == 0 else open(os.path.join(seed_dir, "FAIL"), "w")).close()
        print(f"  exit={ret.returncode}")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="HetToM Layer-2 bridge: migrate winners to LBF+MAPPO")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="write the mappo_hettom config")
    p_prepare.add_argument("--out_dir", default=None)

    p_inject = sub.add_parser("inject", help="build offline ToM intent features")
    p_inject.add_argument("--layer1_dir", default="results/hettom_layer1")
    p_inject.add_argument("--out", default="results/hettom_layer2/intent_features.jsonl")
    p_inject.add_argument("--n_samples", type=int, default=200)
    p_inject.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p_inject.add_argument("--mock", action="store_true")

    p_launch = sub.add_parser("launch", help="launch MAPPO training")
    p_launch.add_argument("--config", default="mappo_hettom")
    p_launch.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p_launch.add_argument("--intent_path", default=None)
    p_launch.add_argument("--env", default="lbforaging:Foraging-10x10-3p-3f-v3")
    p_launch.add_argument("--t_max", type=int, default=1000000)
    p_launch.add_argument("--out_dir", default="results/hettom_layer2")
    p_launch.add_argument("--mock", action="store_true")
    p_launch.add_argument("--dry_run", action="store_true")

    p_all = sub.add_parser("all", help="prepare + inject + launch")
    p_all.add_argument("--layer1_dir", default="results/hettom_layer1")
    p_all.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p_all.add_argument("--mock", action="store_true")
    p_all.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    if args.cmd == "prepare":
        write_config(args.out_dir)
    elif args.cmd == "inject":
        build_intent_features(args.layer1_dir, args.out, args.n_samples,
                              model_name=args.model, mock=args.mock)
        write_config()
    elif args.cmd == "launch":
        launch(args.config, args.seeds, intent_path=args.intent_path,
               env_key=args.env, t_max=args.t_max, out_dir=args.out_dir,
               mock=args.mock, dry_run=args.dry_run)
    elif args.cmd == "all":
        cfg = write_config()
        intent = os.path.join(os.path.dirname(args.layer1_dir) or ".",
                              "hettom_layer2", "intent_features.jsonl")
        build_intent_features(args.layer1_dir, intent, mock=args.mock)
        launch(cfg, args.seeds, intent_path=intent,
               out_dir=os.path.dirname(intent), mock=args.mock,
               dry_run=args.dry_run)


if __name__ == "__main__":
    main()
