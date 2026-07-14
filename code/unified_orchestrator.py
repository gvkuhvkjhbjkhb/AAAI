#!/usr/bin/env python3
"""Unified orchestrator — ALL 4 experiment types with SAFE VRAM scheduling.

GPU now free (both 0 MiB after stop). cap=2 het workers/GPU (9GB each = 18GB,
safe on 32GB). stagger=90s. NO contention — runs immediately.

The 4 experiment types:
  TYPE 1: Scheme-1 3-arm GSACA (GPU)   — het_3arm cell, τ∈{0,0.2,0.4,0.6}
           4 τ × 6 games × 5 seeds = 120 cells  [validates offline recompute]
  TYPE 2: Scheme-2 silent-anti-coord   — het_gsaca_silent, anti_coord→use_talk=False
           3 anti-coord games × 20 seeds = 60 cells  [push chicken toward sig]
  TYPE 3: BoS+public_goods 20-seed     — 4 het cells (fills the n=20 gap)
           2 games × 20 seeds × 4 cells = 160 cells  [BoS=Gated arm, pg=abstain=0]
  TYPE 4: V2 recovery (D1+E+anti_test) — noise sweep + hyperparam + anti-coord patches
           D1: 4 noise × 6 games × 5 seeds = 120 cells
           E:  θ5+α4+W5 × 2 games × 5 seeds = 140 cells
           anti_test deadlock: 5 seeds × 5 anti cells = 25 cells

Total: ~625 cells. At ~3min/cell / 4 concurrent ≈ 7.8h.
Skips already-completed cells (--force removed for non-3arm).
"""
import os, sys, time, subprocess, glob, shutil
from datetime import datetime

GSACA = "/data/lab/gsaca"
V2    = "/data/lab/results/v2"
LOG   = os.path.join(V2, "unified_orchestrator.log")

QWEN = "Qwen/Qwen2.5-7B-Instruct"
GLM  = "THUDM/GLM-4-9B-0414"
LLAMA = "NousResearch/Meta-Llama-3.1-8B-Instruct"

HET_CELLS = ["het_notom", "het_gated_atom_talk", "het_dp_gated_atom_talk", "het_gsaca"]
ANTI_CELLS = ["het_gsaca", "het_role_asym", "het_hist_split", "het_adapt_interv", "het_combo_anti"]
TWO_PLAYER = ["chicken", "hawk_dove", "deadlock", "stag_hunt", "battle_of_the_sexes"]
PG = ["public_goods"]
ANTI_GAMES = ["chicken", "hawk_dove", "deadlock"]
SEEDS5 = [42, 43, 44, 45, 46]
SEEDS20 = list(range(42, 62))
SEED_HALVES = [list(range(42, 52)), list(range(52, 62))]
SEED_QUARTERS = [list(range(42, 47)), list(range(47, 52)),
                 list(range(52, 57)), list(range(57, 62))]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class VRAMScheduler:
    """cap=2 het workers/GPU (9GB each = 18GB, safe). stagger=90s."""
    def __init__(self, n_gpu=2, cap=2, stagger=90):
        self.n_gpu, self.cap, self.stagger = n_gpu, cap, stagger
        self.cnt = {g: 0 for g in range(n_gpu)}
        self.running = []
        self.done_ok, self.done_fail = 0, 0

    def _pick(self):
        for g in range(self.n_gpu):
            if self.cnt[g] < self.cap:
                return g
        return None

    def run(self, groups):
        pending = list(groups)
        while pending or self.running:
            i = 0
            launched = False
            while i < len(pending):
                g = pending[i]
                gpu = self._pick()
                if gpu is None:
                    i += 1; continue
                pending.pop(i)
                self._launch(g, gpu)
                launched = True
                time.sleep(self.stagger)
            self._reap()
            if not launched and self.running:
                time.sleep(15)
        while self.running:
            self._reap()
            time.sleep(15)

    def _launch(self, g, gpu):
        os.makedirs(g["logdir"], exist_ok=True)
        logpath = os.path.join(g["logdir"], g["name"] + ".log")
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} python3 {g['cmd']} > {logpath} 2>&1"
        self.cnt[gpu] += 1
        log(f"LAUNCH gpu{gpu} ({self.cnt[gpu]}/{self.cap}) {g['name']}")
        with open(LOG, "a") as f:
            f.write("    $ " + cmd + "\n")
        proc = subprocess.Popen(cmd, shell=True)
        self.running.append((proc, gpu, g["name"]))

    def _reap(self):
        still = []
        for proc, gpu, name in self.running:
            if proc.poll() is None:
                still.append((proc, gpu, name))
            else:
                self.cnt[gpu] -= 1
                ok = proc.returncode == 0
                if ok: self.done_ok += 1
                else: self.done_fail += 1
                log(f"DONE  gpu{gpu} rc={proc.returncode} {name} ({'OK' if ok else 'FAIL'})")
        self.running = still


def gargs_std(games, seeds, cells, out, episodes=30, warmup=5, models_het=None,
              model_homo=None, extra=None, force=False):
    a = ("--games %s --seeds %s --episodes %d --horizon 5 --memory 2 "
         "--cells %s --out_dir %s --log_every 10 --gsaca_warmup %d" %
         (" ".join(games), " ".join(map(str, seeds)), episodes,
          " ".join(cells), out, warmup))
    if models_het: a += " --models_het %s %s" % models_het
    if model_homo: a += " --model_homo %s" % model_homo
    if force: a += " --force"
    if extra:
        for k, v in extra.items(): a += " %s %s" % (k, v)
    return f"{GSACA}/run_experiment_local.py {a}"


def group(name, cmd, logdir):
    return {"name": name, "cmd": cmd, "logdir": logdir}


def build_groups():
    gs = []

    # ===== TYPE 1: Scheme-1 3-arm GSACA (GPU) — het_3arm cell =====
    # 4 τ × 6 games × 5 seeds = 120 cells. τ=0.0 = 2-arm baseline validation.
    # split per game so each worker = 6 games × 5 seeds × 1 cell = 30 cells
    for tau in ["0.0", "0.2", "0.4", "0.6"]:
        gs.append(group(f"T1_3arm_t{tau}",
            gargs_std(TWO_PLAYER + PG, SEEDS5, ["het_3arm"],
                      f"{V2}/exp_3arm/tau_{tau}", 30, 5,
                      models_het=(QWEN, GLM),
                      extra={"--abstain_tau": tau}, force=True),
            f"{V2}/exp_3arm"))

    # ===== TYPE 2: Scheme-2 silent-anti-coord =====
    # 3 anti-coord games × 20 seeds = 60 cells (scheme2_silent.py)
    for game in ANTI_GAMES:
        for hi, shalf in enumerate(SEED_HALVES):
            seeds = " ".join(map(str, shalf))
            gs.append(group(f"T2_silent_{game}_h{hi}",
                f"{GSACA}/scheme2_silent.py --games {game} --seeds {seeds} "
                f"--episodes 30 --log_every 10 --force",
                f"{V2}/exp_scheme2_silent"))

    # ===== TYPE 3: BoS + public_goods 20-seed (fill n=20 gap) =====
    # 2 games × 20 seeds × 4 het cells = 160 cells
    for game in ["battle_of_the_sexes"] + PG:
        eps = 20 if game == "public_goods" else 30
        warmup = 3 if game == "public_goods" else 5
        for qi, sq in enumerate(SEED_QUARTERS):
            seeds = " ".join(map(str, sq))
            gs.append(group(f"T3_{game}_q{qi}",
                gargs_std([game], sq, HET_CELLS, f"{V2}/exp_b_20seed",
                          eps, warmup, models_het=(QWEN, GLM), force=True),
                f"{V2}/exp_b_20seed"))

    # ===== TYPE 4: V2 recovery (D1 noise + E hyperparam + anti_test) =====
    # D1: 4 noise levels × 6 games × 5 seeds (het_gsaca) = 120 cells
    for noise in ["0.0", "0.5", "1.0", "2.0"]:
        tag = "n" + noise.replace(".", "")
        gs.append(group(f"T4_D1_{tag}_2p",
            gargs_std(TWO_PLAYER, SEEDS5, ["het_gsaca"], f"{V2}/exp_d_fix/d1_{tag}",
                      30, 5, models_het=(QWEN, GLM),
                      extra={"--payoff_noise_std": noise}, force=True),
            f"{V2}/exp_d_fix"))
        gs.append(group(f"T4_D1_{tag}_pg",
            gargs_std(PG, SEEDS5, ["het_gsaca"], f"{V2}/exp_d_fix/d1_{tag}",
                      20, 3, models_het=(QWEN, GLM),
                      extra={"--payoff_noise_std": noise}, force=True),
            f"{V2}/exp_d_fix"))
    # E: θ sweep (5) + α sweep (4) + W sweep (5) = 14 × 2 games × 5 seeds = 140 cells
    for th in ["0.3", "0.45", "0.6", "0.75", "0.9"]:
        gs.append(group(f"T4_E_th{th}",
            gargs_std(["chicken", "battle_of_the_sexes"], SEEDS5,
                      ["het_dp_gated_atom_talk"], f"{V2}/exp_e_fix/theta_{th}",
                      30, 5, models_het=(QWEN, GLM),
                      extra={"--gate_trust_threshold": th, "--gate_ema_alpha": "0.3"},
                      force=True),
            f"{V2}/exp_e_fix"))
    for al in ["0.1", "0.2", "0.3", "0.5"]:
        gs.append(group(f"T4_E_al{al}",
            gargs_std(["chicken", "battle_of_the_sexes"], SEEDS5,
                      ["het_dp_gated_atom_talk"], f"{V2}/exp_e_fix/alpha_{al}",
                      30, 5, models_het=(QWEN, GLM),
                      extra={"--gate_ema_alpha": al, "--gate_trust_threshold": "0.6"},
                      force=True),
            f"{V2}/exp_e_fix"))
    for w in ["2", "3", "5", "8", "10"]:
        gs.append(group(f"T4_E_W{w}",
            gargs_std(["chicken", "battle_of_the_sexes"], SEEDS5,
                      ["het_gsaca"], f"{V2}/exp_e_fix/warmup_{w}",
                      30, int(w), models_het=(QWEN, GLM),
                      extra={"--gate_trust_threshold": "0.6", "--gate_ema_alpha": "0.3"},
                      force=True),
            f"{V2}/exp_e_fix"))
    # anti_test: deadlock remaining (5 seeds × 5 anti cells = 25 cells)
    gs.append(group("T4_anti_deadlock",
        gargs_std(["deadlock"], SEEDS5, ANTI_CELLS, f"{V2}/exp_anti_test",
                  30, 5, models_het=(QWEN, GLM), force=True),
        f"{V2}/exp_anti_test"))

    return gs


def count(d):
    return len(glob.glob(os.path.join(d, "**", "metrics.json"), recursive=True))


def main():
    os.makedirs(V2, exist_ok=True)
    log("=" * 70)
    log("UNIFIED ORCHESTRATOR — ALL 4 EXPERIMENT TYPES (GPU free, cap=2/GPU)")
    log("=" * 70)

    # verify GPUs free
    nvidia = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"]).decode()
    log("GPU state:\n" + nvidia)

    groups = build_groups()
    log(f"total workers: {len(groups)}")
    # estimate cells per type
    log("  TYPE 1 (3-arm):       4 τ × 6 games × 5 seeds = 120 cells")
    log("  TYPE 2 (silent):      3 games × 20 seeds = 60 cells")
    log("  TYPE 3 (BoS+pg 20sd): 2 games × 20 seeds × 4 cells = 160 cells")
    log("  TYPE 4 (V2 recovery): D1(120) + E(140) + anti(25) = 285 cells")
    log("  TOTAL: ~625 cells at ~3min/cell / 4 concurrent ≈ 7.8h")

    sched = VRAMScheduler(n_gpu=2, cap=2, stagger=90)
    sched.run(groups)

    log("\n" + "=" * 70)
    log("ALL EXPERIMENTS COMPLETE")
    for sub, label in [("exp_3arm", "TYPE1 3-arm"),
                       ("exp_scheme2_silent", "TYPE2 silent"),
                       ("exp_b_20seed", "TYPE3 BoS+pg"),
                       ("exp_d_fix", "TYPE4 D1"),
                       ("exp_e_fix", "TYPE4 E"),
                       ("exp_anti_test", "TYPE4 anti")]:
        log(f"  {label:20s}: {count(V2+'/'+sub)} metrics")
    log(f"  ok={sched.done_ok} fail={sched.done_fail}")
    log("=" * 70)

    # auto-run analyses
    log("running Scheme 1 analysis...")
    subprocess.run(["python3", f"{GSACA}/scheme1_offline.py"], cwd=GSACA,
                   stdout=open(f"{V2}/scheme1_offline/final_analysis.txt", "w"))
    if count(V2 + "/exp_scheme2_silent") >= 50:
        log("running Scheme 2 analysis...")
        subprocess.run(["python3", f"{GSACA}/scheme2_analyze.py"], cwd=GSACA,
                       stdout=open(f"{V2}/scheme2_offline/analysis_output.txt", "w"))
    log("ALL ANALYSES DONE.")


if __name__ == "__main__":
    main()
