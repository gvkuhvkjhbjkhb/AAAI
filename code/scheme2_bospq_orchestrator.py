#!/usr/bin/env python3
"""Combined orchestrator — runs Scheme 2 + BoS/pg 20-seed AFTER the ongoing
run_final_fast.sh fully drains. SAFE VRAM scheduling (cap=2 het workers/GPU =
18GB) avoids the OOM that killed D1/E/B in run_final_fast.sh (which ran 8+/GPU).

Waits for: run_final_fast.sh gone AND 0 run_experiment_local procs.
Then launches (14 workers, 4 concurrent):
  Scheme 2:  3 anti-coord games x 20 seeds x het_gsaca_silent = 60 cells  (6 wkrs)
  BoS+pg:    2 games x 20 seeds x 4 het cells = 160 cells                (8 wkrs)
  Total 220 cells. At ~3min/cell / 4 concurrent ~= 2.75h.
"""
import os, sys, time, subprocess, glob
from datetime import datetime

GSACA = "/data/lab/gsaca"
V2    = "/data/lab/results/v2"
LOG   = os.path.join(V2, "scheme2_bospq_orchestrator.log")

QWEN = "Qwen/Qwen2.5-7B-Instruct"
GLM  = "THUDM/GLM-4-9B-0414"
HET_CELLS = ["het_notom", "het_gated_atom_talk", "het_dp_gated_atom_talk", "het_gsaca"]
ANTI_GAMES = ["chicken", "hawk_dove", "deadlock"]
BOSPG_GAMES = ["battle_of_the_sexes", "public_goods"]
SEEDS20 = list(range(42, 62))
SEED_HALVES = [list(range(42, 52)), list(range(52, 62))]
SEED_QUARTERS = [list(range(42, 47)), list(range(47, 52)),
                 list(range(52, 57)), list(range(57, 62))]


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def driver_alive():
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "run_final_fast.sh"], stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return False


def exp_proc_count():
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "run_experiment_local.py"], stderr=subprocess.DEVNULL)
        return len([l for l in out.decode().split() if l.strip()])
    except Exception:
        return 0


class VRAMScheduler:
    """cap=2 het workers/GPU (9GB each = 18GB, safe on 32GB). stagger=90s."""
    def __init__(self, n_gpu=2, cap=2, gb=9, stagger=90):
        self.n_gpu, self.cap, self.gb, self.stagger = n_gpu, cap, gb, stagger
        self.cnt = {g: 0 for g in range(n_gpu)}
        self.running = []

    def _pick(self):
        best = None
        for g in range(self.n_gpu):
            if self.cnt[g] < self.cap:
                best = g; break   # fill GPU0 then GPU1 then back to GPU0
        return best

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
                log(f"DONE  gpu{gpu} rc={proc.returncode} {name}")
        self.running = still


def build_groups():
    gs = []
    # --- Scheme 2: 3 games x 2 seed-halves = 6 workers (scheme2_silent.py) ---
    for game in ANTI_GAMES:
        for hi, shalf in enumerate(SEED_HALVES):
            seeds = " ".join(map(str, shalf))
            name = f"S2_{game}_h{hi}"
            cmd = (f"{GSACA}/scheme2_silent.py --games {game} --seeds {seeds} "
                   f"--episodes 30 --log_every 10 --force")
            gs.append({"name": name, "cmd": cmd,
                       "logdir": f"{V2}/exp_scheme2_silent"})
    # --- BoS+pg 20-seed: 2 games x 4 seed-quarters = 8 workers (run_experiment_local.py) ---
    for game in BOSPG_GAMES:
        eps = 20 if game == "public_goods" else 30
        warmup = 3 if game == "public_goods" else 5
        for qi, sq in enumerate(SEED_QUARTERS):
            seeds = " ".join(map(str, sq))
            cells = " ".join(HET_CELLS)
            name = f"B_{game}_q{qi}"
            cmd = (f"{GSACA}/run_experiment_local.py --games {game} --seeds {seeds} "
                   f"--episodes {eps} --horizon 5 --memory 2 --cells {cells} "
                   f"--out_dir {V2}/exp_b_20seed --log_every 10 --gsaca_warmup {warmup} "
                   f"--models_het {QWEN} {GLM} --force")
            gs.append({"name": name, "cmd": cmd, "logdir": f"{V2}/exp_b_20seed"})
    return gs


def count(d):
    return len(glob.glob(os.path.join(d, "**", "metrics.json"), recursive=True))


def main():
    os.makedirs(V2, exist_ok=True)
    log("=" * 70)
    log("COMBINED ORCHESTRATOR (Scheme 2 + BoS/pg 20-seed)")
    log("=" * 70)

    # Phase 0: wait for ongoing run to FULLY drain
    log("waiting for run_final_fast.sh + all run_experiment_local procs to drain...")
    waited = 0
    while True:
        drv = driver_alive()
        nproc = exp_proc_count()
        if not drv and nproc == 0:
            log("drained! GPUs should be free.")
            break
        if waited % 600 == 0:   # log every 10 min
            log(f"  still busy: driver={'yes' if drv else 'no'} exp_procs={nproc}")
        time.sleep(30); waited += 30
    time.sleep(30)   # let VRAM release after procs exit

    nvidia = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"]).decode()
    log("GPU state after drain:\n" + nvidia)

    # Phase 1: run combined batch
    groups = build_groups()
    log(f"launching {len(groups)} workers (6 Scheme2 + 8 BoS/pg), cap=2/GPU")
    sched = VRAMScheduler(n_gpu=2, cap=2, gb=9, stagger=90)
    sched.run(groups)

    # Phase 2: summary + trigger analysis
    log("=" * 70)
    log("BATCH COMPLETE")
    log(f"  exp_scheme2_silent: {count(V2+'/exp_scheme2_silent')} metrics (expect 60)")
    log(f"  exp_b_20seed Bos:   {count(V2+'/exp_b_20seed/battle_of_the_sexes')} (expect 80)")
    log(f"  exp_b_20seed pg:    {count(V2+'/exp_b_20seed/public_goods')} (expect 80)")
    log("=" * 70)

    # auto-run analyses
    log("running Scheme 2 analysis...")
    subprocess.run(["python3", f"{GSACA}/scheme2_analyze.py"],
                   cwd=GSACA, stdout=open(f"{V2}/scheme2_offline/analysis_output.txt", "w"))
    log("running Scheme 1 (now with full 6-game n=20)...")
    subprocess.run(["python3", f"{GSACA}/scheme1_offline.py"],
                   cwd=GSACA, stdout=open(f"{V2}/scheme1_offline/analysis_output_full.txt", "w"))
    log("ALL ANALYSES DONE.")


if __name__ == "__main__":
    main()
