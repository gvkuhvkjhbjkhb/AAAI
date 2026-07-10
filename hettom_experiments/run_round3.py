#!/usr/bin/env python3
"""Round-3 HetToM runner: gated talk+ToM arbitration + improved A-ToM.

Runs a focused cell subset (default 8 cells) per seed, serially on one GPU,
into a fresh output dir so it never clobbers round-1/2 data. Resumable:
skips any cell that already has metrics.json (use --force to rerun).

Default Round-3 cells (all at the same episode budget for fair comparison):
  hom_notom            baseline (reasoning trap)
  het_notom            heterogeneity destroys cooperation
  het_tom              fixed 1st-order ToM
  het_notom_talk       cheap-talk only (round-2 strongest mechanism)
  het_tom_talk         naive ToM+talk (round-2 failure: interference)
  het_atom             improved A-ToM (per-order bandit, replaces coarse rule)
  het_gated_talk_tom   NEW: gated signal-belief arbitration (fixed ToM)
  het_gated_atom_talk  NEW: gated arbitration + improved A-ToM (full method)

The gated cell is the controlled test of the belief-signal arbitration
hypothesis: vs het_tom it adds arbitration, vs het_tom_talk it replaces
naive dual-feed with a single gated belief."""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hettom_baseline as hb

ROUND3_CELLS = [
    "hom_notom", "het_notom", "het_tom",
    "het_notom_talk", "het_tom_talk", "het_atom",
    "het_gated_talk_tom", "het_gated_atom_talk",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--game", type=str, default="stag_hunt")
    ap.add_argument("--n_agents", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--tom_order", type=int, default=1)
    ap.add_argument("--memory", type=int, default=2)
    ap.add_argument("--model_homo", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--models_het", type=str, nargs="+",
                    default=["Qwen/Qwen2.5-3B-Instruct",
                             "Qwen/Qwen2.5-1.5B-Instruct"])
    ap.add_argument("--temp_homo", type=float, default=0.7)
    ap.add_argument("--temps_het", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--cells", type=str, nargs="+", default=ROUND3_CELLS)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--force", action="store_true",
                    help="rerun cells even if metrics.json exists")
    args = ap.parse_args()

    # make_matrix_configs reads these attributes off args
    args.extend = True
    os.makedirs(args.out_dir, exist_ok=True)

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        args.seed = seed
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        configs = hb.make_matrix_configs(args)
        print(f"\n=== seed {seed}: {len(configs)} cells queued ===")
        for cfg in configs:
            cell = cfg["cell_name"]
            cfg["seed"] = seed
            mpath = os.path.join(seed_dir, cell, "metrics.json")
            if (not args.force) and os.path.exists(mpath):
                print(f"[skip] seed{seed} {cell} (metrics.json exists)")
                continue
            print(f"[run]  seed{seed} {cell} @ {args.episodes} ep")
            hb.run_config(cfg, args.episodes, seed_dir, args.log_every)
        print(f"=== seed {seed} done @ {__import__('datetime').datetime.now().time()} ===")


if __name__ == "__main__":
    main()
