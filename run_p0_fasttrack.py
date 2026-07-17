#!/usr/bin/env python3
"""
P0 Fast-Track: Run deadlock n=8 extension + remaining game baselines
using the EXISTING run_round4.py infrastructure (no GSACA needed for baselines).

This covers:
  - Deadlock: 8-seed DP-gating vs gated (P0-A extension)
  - Public goods: 4-agent 8-seed (P0-B)
  - Re-run Stag Hunt + BoS + Chicken 8-seed with existing code

Runs on dual GPU: splits seeds across two processes.
Usage:
  bash run_p0_fasttrack.sh
"""
import argparse
import os
import subprocess
import sys
import time


def run_subset(label, games, cells, seeds, out_dir, gpu_id):
    """Run run_round4.py on a specific GPU."""
    cmd = [
        sys.executable,
        os.path.join("hettom_experiments", "run_round4.py"),
        "--games", *games,
        "--seeds", *[str(s) for s in seeds],
        "--episodes", "30",
        "--out-dir", out_dir,
    ]
    if cells:
        cmd.extend(["--cells", *cells])

    print(f"[GPU {gpu_id}] {label}: {' '.join(games)} x seeds {min(seeds)}-{max(seeds)}")
    print(f"[GPU {gpu_id}] CMD: {' '.join(cmd)}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    log_file = os.path.join(out_dir, f"p0_gpu{gpu_id}.log")
    os.makedirs(out_dir, exist_ok=True)
    with open(log_file, "w") as f:
        for line in proc.stdout:
            f.write(line)
            prefix = f"[GPU{gpu_id}] "
            sys.stdout.write(prefix + line)

    proc.wait()
    print(f"[GPU {gpu_id}] DONE (exit={proc.returncode})")
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="results/p0_fasttrack")
    parser.add_argument("--seeds-gpu0", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds-gpu1", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--episodes", type=int, default=25)
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__)) or "."

    # ─── Task allocation ───
    # GPU 0: Deadlock + Chicken (strongest anti-coord signal)
    # GPU 1: Stag Hunt + BoS (coordination + conflict)
    
    games_gpu0 = ["deadlock", "chicken"]
    games_gpu1 = ["stag_hunt", "battle_of_the_sexes"]
    
    # Core methods: vanilla gated (baseline), dp_gated (proposed), plus hom baseline
    baseline_cells = [
        "hom_notom",
        "het_notom",
        "het_dp_gated_atom_talk",  # DP-Gating
        "het_gated_atom_talk",     # Conventional gating
    ]

    print("=" * 60)
    print("P0 Fast-Track: 8-seed extension")
    print(f"GPU 0: {games_gpu0} seeds {args.seeds_gpu0[0]}-{args.seeds_gpu0[-1]}")
    print(f"GPU 1: {games_gpu1} seeds {args.seeds_gpu1[0]}-{args.seeds_gpu1[-1]}")
    print(f"Cells: {baseline_cells}")
    print("=" * 60)

    import threading

    def run_gpu0():
        run_subset("GPU0", games_gpu0, baseline_cells, args.seeds_gpu0,
                   os.path.join(args.out_dir, "gpu0"), 0)

    def run_gpu1():
        run_subset("GPU1", games_gpu1, baseline_cells, args.seeds_gpu1,
                   os.path.join(args.out_dir, "gpu1"), 1)

    t0 = threading.Thread(target=run_gpu0)
    t1 = threading.Thread(target=run_gpu1)
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    print("\n" + "=" * 60)
    print("P0 Fast-Track complete!")
    print(f"Results: {args.out_dir}/gpu0")
    print(f"Results: {args.out_dir}/gpu1")
    print("=" * 60)


if __name__ == "__main__":
    main()
