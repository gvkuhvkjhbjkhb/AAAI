#!/usr/bin/env python3
"""Run only missing cells for a given seed. Skips cells that already have
metrics.json. Fastest way to resume an interrupted run on single GPU."""
import argparse, os, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hettom_baseline as hb

ALL_CELLS = ["hom_notom","hom_tom","het_notom","het_tom","hom_notom_talk",
             "hom_tom_talk","het_notom_talk","het_tom_talk","hom_atom",
             "het_atom","het_atom_talk"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--game", type=str, default="stag_hunt")
    ap.add_argument("--n_agents", type=int, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--tom_order", type=int, default=1)
    ap.add_argument("--memory", type=int, default=2)
    ap.add_argument("--model_homo", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--models_het", type=str, nargs="+",
                    default=["Qwen/Qwen2.5-3B-Instruct","Qwen/Qwen2.5-1.5B-Instruct"])
    ap.add_argument("--temp_homo", type=float, default=0.7)
    ap.add_argument("--temps_het", type=float, nargs="+", default=[0.5,0.8,1.0])
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--extend", action="store_true", default=True,
                    help="use extended 11-cell matrix (default True for round-2)")
    ap.add_argument("--force", action="store_true", help="rerun even if metrics exist")
    args = ap.parse_args()

    seed_dir = os.path.join(args.out_dir, f"seed_{args.seed}")
    configs = hb.make_matrix_configs(args)
    to_run = []
    for cfg in configs:
        cell = cfg["cell_name"]
        mpath = os.path.join(seed_dir, cell, "metrics.json")
        if args.force or not os.path.exists(mpath):
            to_run.append(cfg)
        else:
            print(f"[skip] {cell} (already has metrics.json)")
    if not to_run:
        print("nothing to run, all cells complete")
        return
    print(f"running {len(to_run)} missing cells: {[c['cell_name'] for c in to_run]}")
    import random, numpy as np
    random.seed(args.seed); np.random.seed(args.seed)
    for cfg in to_run:
        cfg["seed"] = args.seed
        hb.run_config(cfg, args.episodes, seed_dir, args.log_every)

if __name__ == "__main__":
    main()
