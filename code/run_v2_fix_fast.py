#!/usr/bin/env python3
"""V2 FIX FAST — per-game splitting + VRAM-only scheduling (no proc cap).

Speedup vs v1: splits each pair/sweep into per-game workers (20 cells each
~35min) instead of monolithic 100-cell workers (~3h). VRAM scheduler packs
hom workers (4-5GB) 5-6/GPU and het workers (9GB) 3/GPU = up to 12 concurrent.

Skips already-completed cells (--force removed) to preserve 104 A metrics.
"""
import os, sys, time, subprocess, glob, shutil
from datetime import datetime

GSACA = "/data/lab/gsaca"
V2    = "/data/lab/results/v2"
LOG   = os.path.join(V2, "orchestrator_fix_fast.log")

QWEN  = "Qwen/Qwen2.5-7B-Instruct"
GLM   = "THUDM/GLM-4-9B-0414"
LLAMA = "NousResearch/Meta-Llama-3.1-8B-Instruct"

HET_CELLS = ["het_notom", "het_gated_atom_talk", "het_dp_gated_atom_talk", "het_gsaca"]
HOM_CELLS = ["hom_notom", "hom_gated_atom_talk", "hom_dp_gated_atom_talk", "hom_gsaca"]
TWO_PLAYER = ["chicken", "hawk_dove", "deadlock", "stag_hunt", "battle_of_the_sexes"]
PG         = ["public_goods"]
SEEDS5     = [42, 43, 44, 45, 46]
SEEDS20    = list(range(42, 62))
EXISTING_QG = "/data/lab/results/gsaca_full_20260712_120138"


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class VRAMScheduler:
    """No proc cap — packs workers by VRAM only. vram_cap=27 leaves 5GB headroom
    on 32GB cards for loading spikes. stagger=90s avoids simultaneous loads."""
    def __init__(self, n_gpu=2, vram_cap=27, stagger=90):
        self.n_gpu, self.vram_cap, self.stagger = n_gpu, vram_cap, stagger
        self.vram = {g: 0 for g in range(n_gpu)}
        self.cnt = {g: 0 for g in range(n_gpu)}
        self.running = []

    def _pick_gpu(self, gb):
        best, best_free = None, -1
        for g in range(self.n_gpu):
            if self.vram[g] + gb <= self.vram_cap:
                free = self.vram_cap - self.vram[g]
                if free > best_free:
                    best, best_free = g, free
        return best

    def run(self, groups):
        pending = list(groups)
        while pending or self.running:
            i = 0
            launched = False
            while i < len(pending):
                g = pending[i]
                gpu = self._pick_gpu(g["gb"])
                if gpu is None:
                    i += 1
                    continue
                pending.pop(i)
                self._launch(g, gpu)
                launched = True
                time.sleep(self.stagger)
            self._reap()
            if not launched and self.running:
                time.sleep(10)
        while self.running:
            self._reap()
            time.sleep(10)

    def _launch(self, g, gpu):
        os.makedirs(g["logdir"], exist_ok=True)
        logpath = os.path.join(g["logdir"], g["name"] + ".log")
        cmd = "CUDA_VISIBLE_DEVICES=%d python3 %s/run_experiment_local.py %s" % (gpu, GSACA, g["args"])
        cmd += " > %s 2>&1" % logpath
        self.cnt[gpu] += 1
        self.vram[gpu] += g["gb"]
        log("LAUNCH gpu%d (+%dGB -> %d/%dGB, %d procs)  %s" %
            (gpu, g["gb"], self.vram[gpu], self.vram_cap, self.cnt[gpu], g["name"]))
        proc = subprocess.Popen(cmd, shell=True)
        self.running.append((proc, gpu, g["gb"], g["name"]))

    def _reap(self):
        still = []
        for proc, gpu, gb, name in self.running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, gpu, gb, name))
            else:
                self.cnt[gpu] -= 1
                self.vram[gpu] -= gb
                log("DONE  gpu%d rc=%d  %s  (%s)" % (gpu, rc, name, "OK" if rc == 0 else "FAIL"))
        self.running = still


def gargs(games, seeds, cells, out, episodes=30, warmup=5,
          models_het=None, model_homo=None, extra=None, force=False):
    """force=False — skips completed cells to preserve prior progress."""
    a = ("--games %s --seeds %s --episodes %d --horizon 5 --memory 2 "
         "--cells %s --out_dir %s --log_every 10 --gsaca_warmup %d" %
         (" ".join(games), " ".join(map(str, seeds)), episodes,
          " ".join(cells), out, warmup))
    if models_het:
        a += " --models_het %s %s" % models_het
    if model_homo:
        a += " --model_homo %s" % model_homo
    if force:
        a += " --force"
    if extra:
        for k, v in extra.items():
            a += " %s %s" % (k, v)
    return a


def group(name, gb, args, logdir):
    return {"name": name, "gb": gb, "args": args, "logdir": logdir}


# ================================================================ per-game split

def build_all_groups():
    """Build ALL groups for A+D+E+B, split by game for max parallelism.
    Each worker = 1 game × N seeds × M cells (~20-40 cells, ~35-70min)."""
    gs = []

    # --- Exp A: 3 pairs × 5 games = 15 workers ---
    base_a = V2 + "/exp_a_fix"
    for game in TWO_PLAYER:
        gs.append(group("A_QQ_%s" % game, 4,
            gargs([game], SEEDS5, HOM_CELLS, base_a + "/QQ", 30, 5, model_homo=QWEN), base_a + "/QQ"))
        gs.append(group("A_GG_%s" % game, 5,
            gargs([game], SEEDS5, HOM_CELLS, base_a + "/GG", 30, 5, model_homo=GLM), base_a + "/GG"))
        gs.append(group("A_QL_%s" % game, 9,
            gargs([game], SEEDS5, HET_CELLS, base_a + "/QL", 30, 5, models_het=(QWEN, LLAMA)), base_a + "/QL"))

    # --- Exp D1: 4 noise × (5 two-player + 1 pg) = 24 workers ---
    base_d = V2 + "/exp_d_fix"
    for noise in ["0.0", "0.5", "1.0", "2.0"]:
        tag = "n" + noise.replace(".", "")
        for game in TWO_PLAYER:
            gs.append(group("D1_%s_%s" % (tag, game), 9,
                gargs([game], SEEDS5, ["het_gsaca"], base_d + "/d1_%s" % tag, 30, 5,
                      models_het=(QWEN, GLM), extra={"--payoff_noise_std": noise}), base_d + "/d1_%s" % tag))
        gs.append(group("D1_%s_pg" % tag, 9,
            gargs(PG, SEEDS5, ["het_gsaca"], base_d + "/d1_%s" % tag, 20, 3,
                  models_het=(QWEN, GLM), extra={"--payoff_noise_std": noise}), base_d + "/d1_%s" % tag))

    # --- Exp D2: 3 pairs × 1 game = 3 workers ---
    for nm, mh, mho, cells in [("QG", (QWEN, GLM), None, HET_CELLS),
                                ("QL", (QWEN, LLAMA), None, HET_CELLS),
                                ("QQ", None, QWEN, HOM_CELLS)]:
        gs.append(group("D2_mp_%s" % nm, (9 if mh else 4),
            gargs(["matching_pennies"], SEEDS5, cells, base_d + "/d2_%s" % nm, 30, 5,
                  models_het=mh, model_homo=mho), base_d + "/d2_%s" % nm))

    # --- Exp E: 14 workers ---
    base_e = V2 + "/exp_e_fix"
    for th in ["0.3", "0.45", "0.6", "0.75", "0.9"]:
        gs.append(group("E_th_%s" % th.replace(".", ""), 9,
            gargs(["chicken", "battle_of_the_sexes"], SEEDS5, ["het_dp_gated_atom_talk"],
                  base_e + "/theta_%s" % th, 30, 5, models_het=(QWEN, GLM),
                  extra={"--gate_trust_threshold": th, "--gate_ema_alpha": "0.3"}), base_e + "/theta_%s" % th))
    for al in ["0.1", "0.2", "0.3", "0.5"]:
        gs.append(group("E_al_%s" % al.replace(".", ""), 9,
            gargs(["chicken", "battle_of_the_sexes"], SEEDS5, ["het_dp_gated_atom_talk"],
                  base_e + "/alpha_%s" % al, 30, 5, models_het=(QWEN, GLM),
                  extra={"--gate_ema_alpha": al, "--gate_trust_threshold": "0.6"}), base_e + "/alpha_%s" % al))
    for w in ["2", "3", "5", "8", "10"]:
        gs.append(group("E_W_%s" % w, 9,
            gargs(["chicken", "battle_of_the_sexes"], SEEDS5, ["het_gsaca"],
                  base_e + "/warmup_%s" % w, 30, int(w), models_het=(QWEN, GLM),
                  extra={"--gate_trust_threshold": "0.6", "--gate_ema_alpha": "0.3"}), base_e + "/warmup_%s" % w))

    # --- Exp B: 2 games × 4 seed-groups = 8 workers ---
    base_b = V2 + "/exp_b_20seed"
    bos_seeds = [SEEDS20[i:i+5] for i in range(0, 20, 5)]
    pg_seeds  = [SEEDS20[i:i+5] for i in range(0, 20, 5)]
    for i, sg in enumerate(bos_seeds):
        gs.append(group("B_bos_s%d" % i, 9,
            gargs(["battle_of_the_sexes"], sg, HET_CELLS, base_b, 30, 5, models_het=(QWEN, GLM)), base_b))
    for i, sg in enumerate(pg_seeds):
        gs.append(group("B_pg_s%d" % i, 9,
            gargs(PG, sg, HET_CELLS, base_b, 20, 3, models_het=(QWEN, GLM)), base_b))

    return gs


def copy_qg_fix():
    base = V2 + "/exp_a_fix/QG"
    n = 0
    for game in TWO_PLAYER:
        for s in SEEDS5:
            for cell in HET_CELLS:
                sp = os.path.join(EXISTING_QG, game, "seed_%d" % s, cell, "metrics.json")
                if os.path.exists(sp):
                    dp = os.path.join(base, game, "seed_%d" % s, cell)
                    os.makedirs(dp, exist_ok=True)
                    shutil.copy2(sp, os.path.join(dp, "metrics.json"))
                    tp = os.path.join(EXISTING_QG, game, "seed_%d" % s, cell, "trajectories.jsonl")
                    if os.path.exists(tp):
                        shutil.copy2(tp, os.path.join(dp, "trajectories.jsonl"))
                    n += 1
    log("  copied %d QG-het files -> %s" % (n, base))


def count_metrics(d):
    return len(glob.glob(os.path.join(d, "**", "metrics.json"), recursive=True))


def main():
    os.makedirs(V2, exist_ok=True)
    log("=" * 64)
    log("V2 FIX FAST START  (VRAM-only scheduling, per-game split, no --force)")
    log("=" * 64)
    copy_qg_fix()

    gs = build_all_groups()
    log("Total workers: %d" % len(gs))
    log("  A: 15, D: 27, E: 14, B: 8 = 64 workers")
    log("  Expected peak concurrency: ~6-10 (VRAM-limited)")

    sched = VRAMScheduler(n_gpu=2, vram_cap=27, stagger=90)
    sched.run(gs)

    log("\n" + "=" * 64)
    log("V2 FIX FAST COMPLETE")
    log("  exp_c_payoff_prompt:  %d (original)" % count_metrics(V2 + "/exp_c_payoff_prompt"))
    log("  exp_a_fix:            %d" % count_metrics(V2 + "/exp_a_fix"))
    log("  exp_d_fix:            %d" % count_metrics(V2 + "/exp_d_fix"))
    log("  exp_e_fix:            %d" % count_metrics(V2 + "/exp_e_fix"))
    log("  exp_b_20seed:         %d" % count_metrics(V2 + "/exp_b_20seed"))
    log("=" * 64)


if __name__ == "__main__":
    main()
