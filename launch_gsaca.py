#!/usr/bin/env python3
"""GSACA: Game-Structure-Adaptive Conditional Alignment - Compact Launcher.
Delegates to run_round4.py for baseline experiments.
Runs: Deadlock 8-seed + Chicken 8-seed on GPU0, BoS + StagHunt on GPU1.
"""
import os, sys, subprocess, time

RUNNER = os.path.join(os.path.dirname(__file__), "hettom_experiments", "run_round4.py")
BASE = os.path.join(os.path.dirname(__file__), "results", "gsaca_p0")
os.makedirs(BASE, exist_ok=True)

GAMES_GPU0 = ["deadlock", "chicken"]  
GAMES_GPU1 = ["stag_hunt", "battle_of_the_sexes"]
CELLS = ["hom_notom", "het_notom", "het_dp_gated_atom_talk", "het_gated_atom_talk"]
SEEDS = "1 2 3 4 5 6 7 8"

for gpu_id, games in [(0, GAMES_GPU0), (1, GAMES_GPU1)]:
    out_dir = f"{BASE}_gpu{gpu_id}"
    cmd = [
        "python3", RUNNER,
        "--games"] + games + [
        "--seeds"] + SEEDS.split() + [
        "--episodes", "30",
        "--cells"] + CELLS + [
        "--out-dir", out_dir,
    ]
    print(f"GPU {gpu_id}: {' '.join(games)}")
    subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)})
    time.sleep(1)

print("Launched! Poll results/gsaca_p0_gpu0/ and results/gsaca_p0_gpu1/")
print("Monitor: tail -f results/gsaca_p0_gpu0/*/seed_*/metrics.json 2>/dev/null || true")
print("Wait ~2h for completion (all 4 games x 4 cells x 8 seeds x 30 eps)")
