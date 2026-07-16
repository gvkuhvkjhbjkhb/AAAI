#!/usr/bin/env python3
"""V2 FIX re-run — repairs the path-overwrite + VRAM-OOM data loss from the
first orchestrator run.

Root causes fixed:
  1. Each group (pair / noise-level / sweep-value) now gets its OWN output
     subdir so parallel workers never overwrite each other.
  2. cap=2 workers/GPU (was 3) + stagger=120s (was 60s) so the 3rd worker
     never OOMs during model loading.

Only re-runs the broken experiments:
  - Exp A: all 4 pairs into exp_a_fix/<PAIR>/  (old exp_a_pairs is contaminated)
  - Exp D: D1 noise into exp_d_fix/d1_<NOISE>/, D2 MP into exp_d_fix/d2_<PAIR>/
  - Exp E: each sweep value into exp_e_fix/<param>/<value>/
  - Exp B: only battle_of_the_sexes + public_goods (the 4 OK games are kept)

Exp C (30, OK) and B's 4 games (320, OK) are NOT re-run.
"""
import os, sys, time, subprocess, glob
from datetime import datetime

GSACA = "/data/lab/gsaca"
V2    = "/data/lab/results/v2"
LOG   = os.path.join(V2, "orchestrator_fix.log")

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


class Scheduler:
    """cap=2 per GPU, stagger=120s — prevents VRAM OOM during model loading."""
    def __init__(self, n_gpu=2, cap=2, vram_cap=20, stagger=120):
        self.n_gpu, self.cap, self.vram_cap, self.stagger = n_gpu, cap, vram_cap, stagger
        self.cnt = {g: 0 for g in range(n_gpu)}
        self.vram = {g: 0 for g in range(n_gpu)}
        self.running = []

    def _pick_gpu(self, gb):
        best, best_free = None, -1
        for g in range(self.n_gpu):
            if self.cnt[g] < self.cap and self.vram[g] + gb <= self.vram_cap:
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
                time.sleep(15)
        while self.running:
            self._reap()
            time.sleep(15)

    def _launch(self, g, gpu):
        os.makedirs(g["logdir"], exist_ok=True)
        logpath = os.path.join(g["logdir"], g["name"] + ".log")
        cmd = "CUDA_VISIBLE_DEVICES=%d python3 %s/run_experiment_local.py %s" % (gpu, GSACA, g["args"])
        cmd += " > %s 2>&1" % logpath
        self.cnt[gpu] += 1
        self.vram[gpu] += g["gb"]
        log("LAUNCH gpu%d (+%.0fGB -> %.0f/%.0fGB, %d procs)  %s" %
            (gpu, g["gb"], self.vram[gpu], self.vram_cap, self.cnt[gpu], g["name"]))
        with open(LOG, "a") as f:
            f.write("    $ " + cmd + "\n")
        proc = subprocess.Popen(cmd, shell=True)
        self.running.append((proc, gpu, g["gb"], g["name"], logpath))

    def _reap(self):
        still = []
        for proc, gpu, gb, name, logpath in self.running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, gpu, gb, name, logpath))
            else:
                self.cnt[gpu] -= 1
                self.vram[gpu] -= gb
                log("DONE  gpu%d rc=%d  %s  (%s)" % (gpu, rc, name, "OK" if rc == 0 else "FAIL"))
        self.running = still


def gargs(games, seeds, cells, out, episodes=30, warmup=5,
          models_het=None, model_homo=None, extra=None, force=True):
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


# ================================================================ phases

def phase_A_fix():
    """Re-run all 4 pairs into SEPARATE subdirs."""
    base = V2 + "/exp_a_fix"
    gs = []
    # QG-het: reuse existing data (copy into QG subdir)
    # QQ-hom
    gs.append(group("A_qq", 4,
        gargs(TWO_PLAYER, SEEDS5, HOM_CELLS, base + "/QQ", 30, 5, model_homo=QWEN), base + "/QQ"))
    # GG-hom
    gs.append(group("A_gg", 5,
        gargs(TWO_PLAYER, SEEDS5, HOM_CELLS, base + "/GG", 30, 5, model_homo=GLM), base + "/GG"))
    # QL-het
    gs.append(group("A_ql", 9,
        gargs(TWO_PLAYER, SEEDS5, HET_CELLS, base + "/QL", 30, 5, models_het=(QWEN, LLAMA)), base + "/QL"))
    return base, gs


def copy_qg_fix(base):
    import shutil
    n = 0
    for game in TWO_PLAYER:
        for s in SEEDS5:
            for cell in HET_CELLS:
                sp = os.path.join(EXISTING_QG, game, "seed_%d" % s, cell, "metrics.json")
                if os.path.exists(sp):
                    dp = os.path.join(base, "QG", game, "seed_%d" % s, cell)
                    os.makedirs(dp, exist_ok=True)
                    shutil.copy2(sp, os.path.join(dp, "metrics.json"))
                    tp = os.path.join(EXISTING_QG, game, "seed_%d" % s, cell, "trajectories.jsonl")
                    if os.path.exists(tp):
                        shutil.copy2(tp, os.path.join(dp, "trajectories.jsonl"))
                    n += 1
    log("  copied %d QG-het files -> %s/QG" % (n, base))


def phase_D_fix():
    """Re-run D1 (4 noise levels) + D2 (3 pairs) into SEPARATE subdirs."""
    base = V2 + "/exp_d_fix"
    gs = []
    # D1: noise sweep, each level into its own dir
    for noise in ["0.0", "0.5", "1.0", "2.0"]:
        tag = "n" + noise.replace(".", "")
        gs.append(group("D1_%s_2p" % tag, 9,
            gargs(TWO_PLAYER, SEEDS5, ["het_gsaca"], base + "/d1_%s" % tag, 30, 5,
                  models_het=(QWEN, GLM), extra={"--payoff_noise_std": noise}), base + "/d1_%s" % tag))
        gs.append(group("D1_%s_pg" % tag, 9,
            gargs(PG, SEEDS5, ["het_gsaca"], base + "/d1_%s" % tag, 20, 3,
                  models_het=(QWEN, GLM), extra={"--payoff_noise_std": noise}), base + "/d1_%s" % tag))
    # D2: matching_pennies, 3 pairs into separate dirs
    for nm, mh, mho, cells in [("QG", (QWEN, GLM), None, HET_CELLS),
                                ("QL", (QWEN, LLAMA), None, HET_CELLS),
                                ("QQ", None, QWEN, HOM_CELLS)]:
        gs.append(group("D2_mp_%s" % nm, (9 if mh else 4),
            gargs(["matching_pennies"], SEEDS5, cells, base + "/d2_%s" % nm, 30, 5,
                  models_het=mh, model_homo=mho), base + "/d2_%s" % nm))
    return base, gs


def phase_E_fix():
    """Re-run all sweep values into SEPARATE subdirs."""
    base = V2 + "/exp_e_fix"
    gs = []
    # theta sweep
    for th in ["0.3", "0.45", "0.6", "0.75", "0.9"]:
        gs.append(group("E_th_%s" % th.replace(".", ""), 9,
            gargs(["chicken", "battle_of_the_sexes"], SEEDS5, ["het_dp_gated_atom_talk"],
                  base + "/theta_%s" % th, 30, 5, models_het=(QWEN, GLM),
                  extra={"--gate_trust_threshold": th, "--gate_ema_alpha": "0.3"}), base + "/theta_%s" % th))
    # alpha sweep
    for al in ["0.1", "0.2", "0.3", "0.5"]:
        gs.append(group("E_al_%s" % al.replace(".", ""), 9,
            gargs(["chicken", "battle_of_the_sexes"], SEEDS5, ["het_dp_gated_atom_talk"],
                  base + "/alpha_%s" % al, 30, 5, models_het=(QWEN, GLM),
                  extra={"--gate_ema_alpha": al, "--gate_trust_threshold": "0.6"}), base + "/alpha_%s" % al))
    # W sweep
    for w in ["2", "3", "5", "8", "10"]:
        gs.append(group("E_W_%s" % w, 9,
            gargs(["chicken", "battle_of_the_sexes"], SEEDS5, ["het_gsaca"],
                  base + "/warmup_%s" % w, 30, int(w), models_het=(QWEN, GLM),
                  extra={"--gate_trust_threshold": "0.6", "--gate_ema_alpha": "0.3"}), base + "/warmup_%s" % w))
    return base, gs


def phase_B_fix():
    """Re-run only the 2 failed B games."""
    base = V2 + "/exp_b_20seed"   # write into same dir (different games, no overwrite)
    gs = []
    gs.append(group("B_bos", 9,
        gargs(["battle_of_the_sexes"], SEEDS20, HET_CELLS, base, 30, 5, models_het=(QWEN, GLM)), base))
    gs.append(group("B_pg", 9,
        gargs(PG, SEEDS20, HET_CELLS, base, 20, 3, models_het=(QWEN, GLM)), base))
    return base, gs


def count_metrics(d):
    return len(glob.glob(os.path.join(d, "**", "metrics.json"), recursive=True))


def main():
    os.makedirs(V2, exist_ok=True)
    log("=" * 64)
    log("V2 FIX RE-RUN START  (cap=2/GPU, stagger=120s, separate dirs)")
    log("=" * 64)
    sched = Scheduler(n_gpu=2, cap=2, vram_cap=20, stagger=120)

    # Phase A fix
    log("\n##### FIX Exp A — 4 pairs into separate subdirs #####")
    out, gs = phase_A_fix(); copy_qg_fix(out); sched.run(gs)
    for sub in ["QG", "QQ", "GG", "QL"]:
        log("  A/%s: %d metrics" % (sub, count_metrics(os.path.join(out, sub))))

    # Phase D fix
    log("\n##### FIX Exp D — noise + MP into separate subdirs #####")
    out, gs = phase_D_fix(); sched.run(gs)
    log("  D total: %d metrics" % count_metrics(out))

    # Phase E fix
    log("\n##### FIX Exp E — sweep values into separate subdirs #####")
    out, gs = phase_E_fix(); sched.run(gs)
    log("  E total: %d metrics" % count_metrics(out))

    # Phase B fix (only 2 failed games)
    log("\n##### FIX Exp B — re-run battle_of_the_sexes + public_goods #####")
    out, gs = phase_B_fix(); sched.run(gs)
    log("  B total: %d metrics" % count_metrics(V2 + "/exp_b_20seed"))

    log("\n" + "=" * 64)
    log("V2 FIX RE-RUN COMPLETE")
    log("  exp_c_payoff_prompt:  %d (original, OK)" % count_metrics(V2 + "/exp_c_payoff_prompt"))
    log("  exp_a_fix:            %d" % count_metrics(V2 + "/exp_a_fix"))
    log("  exp_d_fix:            %d" % count_metrics(V2 + "/exp_d_fix"))
    log("  exp_e_fix:            %d" % count_metrics(V2 + "/exp_e_fix"))
    log("  exp_b_20seed:         %d" % count_metrics(V2 + "/exp_b_20seed"))
    log("=" * 64)


if __name__ == "__main__":
    main()
