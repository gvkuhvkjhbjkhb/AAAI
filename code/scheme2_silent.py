#!/usr/bin/env python3
"""Scheme 2 — silent-anti-coord GSACA  (GPU, ~30 min on free GPUs).

Does NOT modify hettom_baseline.py or run_experiment_local.py, so the ongoing
run_final_fast.sh experiment is 100% undisturbed (running processes already
imported the modules; this script imports them read-only and adds its own
run function).

Mechanism (see EXPERIMENT_PLAN / user Scheme 2):
  Current GSACA on anti_coord detection keeps the full CGA stack: cheap-talk
  announcements are generated each round, the gate arbitrates signal vs ToM
  belief, and on a signal-belief conflict the gated belief (often = the
  trusted signal) is injected into the action prompt as "you predict action X".
  Cheap-talk's mechanism is to push CONVERGENCE — poison for anti-coordination
  games where the team wants a split profile.

  Scheme 2: after GSACA warmup detection, if structure == anti_coord, set
  ag.use_talk = False on every agent. This (a) stops cheap-talk generation,
  (b) leaves the gate with no signals to arbitrate, (c) makes agents act on
  pure ToM predictions (prompt_belief = teammate_preds). Coord detection is
  unchanged (switch to Gated mode). Only the prompt-assembly / signal path
  is touched — no model change, no new gate.

  Reruns the 3 anti-coord games x 20 seeds (chicken, hawk_dove, deadlock).
  coord games are unaffected (silent-GSACA == old GSACA there), so skipped.

Output: /data/lab/results/v2/exp_scheme2_silent/<game>/seed_<s>/het_gsaca_silent/
"""
import argparse, gc, json, os, random, sys, time
from datetime import datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hettom_baseline as hb
import run_experiment_local as rel   # reuse make_config, estimator, wrap_payoff_noise

OUT = "/data/lab/results/v2/exp_scheme2_silent"
ANTI_GAMES = ["chicken", "hawk_dove", "deadlock"]


def run_silent_gsaca_cell(cfg, n_episodes, out_dir, log_every=5, warmup=5):
    """Like rel.run_gsaca_cell but, on anti_coord detection, switches agents
    to use_talk=False (silent ToM-only) instead of keeping CGA."""
    cell = cfg["cell_name"]
    t0 = time.time()
    game, agents = hb.build_agents(cfg)
    horizon, memory = cfg["horizon"], cfg["memory"]
    estimator = rel.GameStructureEstimator()
    episodes = []

    for ep_idx in range(warmup):
        ep = hb.run_episode(game, agents, horizon, memory)
        episodes.append(ep)
        for step in ep:
            estimator.observe(step["actions"], step["rewards"])

    structure, split_score, n_obs = estimator.estimate()
    oracle = rel.ORACLE_STRUCTURE.get(cfg["game"], "unknown")
    detection_correct = (structure == oracle)

    if structure == "coord":
        # unchanged from GSACA: switch to Gated mode
        for ag in agents:
            ag.gated_talk_tom = False
            ag.diversity_preserving_gate = False
    else:
        # SCHEME 2: anti_coord -> go SILENT (block cheap-talk entirely)
        for ag in agents:
            ag.use_talk = False

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
    metrics["scheme2_mode"] = "silent_anti_coord" if structure != "coord" else "gated_coord"
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

    print(f"  [{cell}] detect={structure}(oracle={oracle},{'OK' if detection_correct else 'MISS'}) "
          f"split={split_score:.3f} mode={metrics['scheme2_mode']} "
          f"team={metrics['team_mean_payoff']:.4f} div={metrics['perspective_diversity']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del agents; gc.collect()
    return metrics


def run_game_seed(args, game, seed):
    seed_dir = os.path.join(args.out_dir, game, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    random.seed(seed); np.random.seed(seed)
    total_t0 = time.time()
    cell = "het_gsaca_silent"
    mpath = os.path.join(seed_dir, cell, "metrics.json")
    if (not args.force) and os.path.exists(mpath):
        print(f"  [skip] {cell} (exists)", flush=True); return
    cfg = rel.make_config(game, "het_dp_gated_atom_talk", seed,
                          horizon=args.horizon, memory=args.memory, args=args)
    cfg["cell_name"] = cell
    print(f"  [run]  {cell} @ {args.episodes}ep ...", flush=True)
    try:
        run_silent_gsaca_cell(cfg, args.episodes, seed_dir,
                              log_every=args.log_every, warmup=args.gsaca_warmup)
    except Exception as e:
        print(f"  [ERROR] {cell}: {e}"); import traceback; traceback.print_exc()
    print(f"  --- {game} seed {seed} done in {time.time()-total_t0:.0f}s ---", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Scheme 2 silent-anti-coord GSACA")
    ap.add_argument("--games", type=str, nargs="+", default=ANTI_GAMES)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 62)))
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--memory", type=int, default=2)
    ap.add_argument("--out_dir", type=str, default=OUT)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--gsaca_warmup", type=int, default=5)
    ap.add_argument("--gate_trust_threshold", type=float, default=0.6)
    ap.add_argument("--gate_ema_alpha", type=float, default=0.3)
    ap.add_argument("--atom_warmup", type=int, default=3)
    ap.add_argument("--models_het", type=str, nargs=2, default=rel.MODELS_HET)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total_t0 = time.time()
    for game in args.games:
        print(f"\n{'='*60}\nGAME: {game}\n{'='*60}", flush=True)
        for seed in args.seeds:
            print(f"\n--- {game} seed {seed} @ {datetime.now().strftime('%H:%M:%S')} ---", flush=True)
            run_game_seed(args, game, seed)
    print(f"\n{'='*60}\nScheme2 complete: {time.time()-total_t0:.0f}s\nOutput: {args.out_dir}\n{'='*60}")


if __name__ == "__main__":
    main()
