#!/usr/bin/env python3
"""Round-4 HetToM runner: cross-game generalization + diversity-preserving
gating + stronger heterogeneous models via Silicon Flow API.

Addresses all three fatal Round-3 gaps:
  1. Only 1 game (Stag Hunt) -> now 4 games (stag_hunt, battle_of_the_sexes,
     chicken, public_goods) for cross-environment generalization.
  2. Method doesn't beat baseline -> diversity-preserving gating that only
     intervenes on signal-belief conflict, preserving cognitive diversity.
  3. Model scale too weak (3B/1.5B same family) -> Qwen2.5-7B (Alibaba) +
     GLM-4-9B (Zhipu) for true architectural heterogeneity at 7-9B scale.

Also fixes:
  - Complete controls: all 15 cells (base 4 + talk 4 + atom 2 + combined 1
    + gated 2 + diversity-preserving 2) instead of Round-3's 8-cell subset.
  - n=5 seeds (expandable to 8 via --seeds).

Model selection rationale:
  - Qwen/Qwen2.5-7B-Instruct (Alibaba, 7B): strong instruction following,
    good game-theoretic reasoning, used as homogeneous baseline model.
  - THUDM/GLM-4-9B-0414 (Zhipu/Tsinghua, 9B): different architecture and
    training data, strong reasoning, gives true heterogeneity vs Qwen.
  - Both are FREE on Silicon Flow, so the 50 yuan budget is preserved.

Usage:
  # Quick pilot (verify API integration):
  python3 run_round4.py --games stag_hunt --seeds 42 --episodes 10 \
    --cells hom_notom het_notom het_dp_gated_atom_talk --out_dir results/round4_pilot

  # Full run (4 games × 15 cells × 5 seeds × 30 episodes):
  python3 run_round4.py --out_dir results/hettom_layer1/round4

  # Resume (skips cells with existing metrics.json):
  python3 run_round4.py --out_dir results/hettom_layer1/round4
"""
import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hettom_baseline as hb

SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"
SILICONFLOW_KEY = os.environ.get(
    "SILICONFLOW_API_KEY",
    "sk-hhpwsdsxkbcjxvgnpcdrfjdvwbqpmcltpvuowvxlkewczhwl")

# True architectural heterogeneity: different model families (Alibaba vs Zhipu)
DEFAULT_MODELS_HET = [
    "Qwen/Qwen2.5-7B-Instruct",
    "THUDM/GLM-4-9B-0414",
]
DEFAULT_MODEL_HOMO = "Qwen/Qwen2.5-7B-Instruct"

# All 15 cells for complete controls
ALL_CELLS = [
    # base 4-cell matrix
    "hom_notom", "hom_tom", "het_notom", "het_tom",
    # cheap-talk variants
    "hom_notom_talk", "hom_tom_talk", "het_notom_talk", "het_tom_talk",
    # adaptive ToM variants
    "hom_atom", "het_atom",
    # combined
    "het_atom_talk",
    # Round-3 gated
    "het_gated_talk_tom", "het_gated_atom_talk",
    # Round-4 diversity-preserving gated (NEW)
    "het_dp_gated_talk_tom", "het_dp_gated_atom_talk",
]

GAMES_2P = ["stag_hunt", "battle_of_the_sexes", "chicken"]
GAMES_ALL = GAMES_2P + ["public_goods"]


def main():
    ap = argparse.ArgumentParser(
        description="Round-4: cross-game + diversity-preserving gating + API models")
    ap.add_argument("--games", type=str, nargs="+", default=GAMES_ALL,
                    help="games to run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                    help="random seeds (5 for Mann-Whitney, 8 for paired Wilcoxon)")
    ap.add_argument("--episodes", type=int, default=30,
                    help="episodes per cell per seed")
    ap.add_argument("--horizon", type=int, default=5,
                    help="rounds per episode")
    ap.add_argument("--tom_order", type=int, default=1)
    ap.add_argument("--memory", type=int, default=2)
    ap.add_argument("--model_homo", type=str, default=DEFAULT_MODEL_HOMO)
    ap.add_argument("--models_het", type=str, nargs="+",
                    default=DEFAULT_MODELS_HET)
    ap.add_argument("--temp_homo", type=float, default=0.7)
    ap.add_argument("--temps_het", type=float, nargs="+",
                    default=[0.5, 0.8])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--cells", type=str, nargs="+", default=ALL_CELLS)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--force", action="store_true",
                    help="rerun cells even if metrics.json exists")
    ap.add_argument("--api_base", type=str, default=SILICONFLOW_BASE)
    ap.add_argument("--api_key", type=str, default=SILICONFLOW_KEY)
    args = ap.parse_args()

    args.extend = True
    os.makedirs(args.out_dir, exist_ok=True)

    total_start = time.time()
    total_cells = 0
    total_skipped = 0

    for game in args.games:
        args.game = game
        game_dir = os.path.join(args.out_dir, game)
        os.makedirs(game_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"GAME: {game}")
        print(f"{'='*60}")

        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)
            args.seed = seed
            args.n_agents = 4 if game == "public_goods" else None
            seed_dir = os.path.join(game_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            configs = hb.make_matrix_configs(args)
            print(f"\n--- {game} seed {seed}: {len(configs)} cells ---")
            for cfg in configs:
                cell = cfg["cell_name"]
                cfg["seed"] = seed
                mpath = os.path.join(seed_dir, cell, "metrics.json")
                if (not args.force) and os.path.exists(mpath):
                    print(f"  [skip] {cell} (exists)")
                    total_skipped += 1
                    continue
                t0 = time.time()
                print(f"  [run]  {cell} @ {args.episodes}ep ...", end="",
                      flush=True)
                try:
                    hb.run_config(cfg, args.episodes, seed_dir, args.log_every)
                    total_cells += 1
                    elapsed = time.time() - t0
                    print(f" done ({elapsed:.0f}s)")
                except Exception as e:
                    print(f" ERROR: {e}")
                    import traceback
                    traceback.print_exc()
            print(f"--- {game} seed {seed} done @ "
                  f"{datetime.now().strftime('%H:%M:%S')} ---")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Round-4 complete: {total_cells} cells run, "
          f"{total_skipped} skipped, {total_elapsed:.0f}s total")
    print(f"Output: {args.out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
