#!/usr/bin/env python3
"""V2 experiment orchestrator — runs Exp A-E from EXPERIMENT_PLAN_V2.md.

Phases (sequential, each parallelizes across 2 GPUs, <=3 workers/GPU, <=28GB/GPU):
  P1  Exp C  payoff-in-prompt baseline        (~15 min)
  P2  Exp A  homogeneous controls + QL-het    (~2 h, reuses existing QG-het@42-46)
  P3  Exp D  stress test (noise + matching_pennies)  (~2 h)
  P4  Exp E  hyperparam sweep theta/alpha/W   (~1.2 h)
  P5  Exp B  20-seed full rerun (auto-chain)  (~5 h)

Each worker = one `run_experiment_local.py` invocation loading ONE model pair
once (4-bit) and looping its assigned games x seeds x cells. Scheduler picks
the GPU with the most free VRAM for each launch, capping 3 procs / 28GB per GPU,
staggering launches 60s so concurrent 4-bit loads don't spike VRAM.

All output under /data/lab/results/v2/<exp>/. Master log: v2/orchestrator.log.
"""
import os, sys, time, subprocess, json, shutil, glob
from datetime import datetime

GSACA = "/data/lab/gsaca"
V2    = "/data/lab/results/v2"
LOG   = os.path.join(V2, "orchestrator.log")

QWEN  = "Qwen/Qwen2.5-7B-Instruct"
GLM   = "THUDM/GLM-4-9B-0414"
LLAMA = "NousResearch/Meta-Llama-3.1-8B-Instruct"

HET_CELLS = ["het_notom", "het_gated_atom_talk", "het_dp_gated_atom_talk", "het_gsaca"]
HOM_CELLS = ["hom_notom", "hom_gated_atom_talk", "hom_dp_gated_atom_talk", "hom_gsaca"]
TWO_PLAYER = ["chicken", "hawk_dove", "deadlock", "stag_hunt", "battle_of_the_sexes"]
PG         = ["public_goods"]
SEEDS5     = [42, 43, 44, 45, 46]
SEEDS20    = list(range(42, 62))

EXISTING_QG = "/data/lab/results/gsaca_full_20260712_120138"   # QG-het @ seeds 42-46 (reuse for Exp A)


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class Scheduler:
    def __init__(self, n_gpu=2, cap=3, vram_cap=28, stagger=60):
        self.n_gpu, self.cap, self.vram_cap, self.stagger = n_gpu, cap, vram_cap, stagger
        self.cnt = {g: 0 for g in range(n_gpu)}
        self.vram = {g: 0 for g in range(n_gpu)}
        self.running = []   # (proc, gpu, gb, name, logpath)

    def _pick_gpu(self, gb):
        best, best_free = None, -1
        for g in range(self.n_gpu):
            if self.cnt[g] < self.cap and self.vram[g] + gb <= self.vram_cap:
                free = self.vram_cap - self.vram[g]
                if free > best_free:
                    best, best_free = g, free
        return best

    def run(self, groups):
        # groups: list of dict(name, gb, cmd(game-agnostic), ) where cmd has {GPU} placeholder
        pending = list(groups)
        done_ok, done_fail = 0, []
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
            self._reap()   # free finished slots
            if not launched and self.running:
                time.sleep(15)
        # drain remaining
        while self.running:
            self._reap()
            time.sleep(15)
        return done_ok, done_fail

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
                # count produced metrics as success signal
                ok = True
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


# ---------------------------------------------------------------- phases
def phase_C():
    out = V2 + "/exp_c_payoff_prompt"
    gs = []
    gs.append(group("C_2p", 9,
        gargs(TWO_PLAYER, SEEDS5, ["het_payoff_prompt"], out, 30, 5, models_het=(QWEN, GLM)), out))
    gs.append(group("C_pg", 9,
        gargs(PG, SEEDS5, ["het_payoff_prompt"], out, 20, 3, models_het=(QWEN, GLM)), out))
    return out, gs


def phase_A():
    out = V2 + "/exp_a_pairs"
    logdir = out
    gs = []
    # QQ-hom (model_homo=Qwen)
    gs.append(group("A_qq_g0", 4, gargs(TWO_PLAYER[:3], SEEDS5, HOM_CELLS, out, 30, 5, model_homo=QWEN), logdir))
    gs.append(group("A_qq_g1", 4, gargs(TWO_PLAYER[3:], SEEDS5, HOM_CELLS, out, 30, 5, model_homo=QWEN), logdir))
    # GG-hom (model_homo=GLM)
    gs.append(group("A_gg_g0", 5, gargs(TWO_PLAYER[:3], SEEDS5, HOM_CELLS, out, 30, 5, model_homo=GLM), logdir))
    gs.append(group("A_gg_g1", 5, gargs(TWO_PLAYER[3:], SEEDS5, HOM_CELLS, out, 30, 5, model_homo=GLM), logdir))
    # QL-het (models_het=Qwen+Llama)
    gs.append(group("A_ql_g0", 9, gargs(TWO_PLAYER[:3], SEEDS5, HET_CELLS, out, 30, 5, models_het=(QWEN, LLAMA)), logdir))
    gs.append(group("A_ql_g1", 9, gargs(TWO_PLAYER[3:], SEEDS5, HET_CELLS, out, 30, 5, models_het=(QWEN, LLAMA)), logdir))
    # reuse existing QG-het @ 42-46 for the 5 two-player games (matched seeds)
    return out, gs


def copy_existing_qg(out):
    n = 0
    for game in TWO_PLAYER:
        for s in SEEDS5:
            src = os.path.join(EXISTING_QG, game, "seed_%d" % s)
            dst = os.path.join(out, game, "seed_%d" % s)
            if os.path.isdir(src):
                for cell in HET_CELLS:
                    sp = os.path.join(src, cell, "metrics.json")
                    if os.path.exists(sp):
                        os.makedirs(os.path.join(dst, cell), exist_ok=True)
                        shutil.copy2(sp, os.path.join(dst, cell, "metrics.json"))
                        tp = os.path.join(src, cell, "trajectories.jsonl")
                        if os.path.exists(tp):
                            shutil.copy2(tp, os.path.join(dst, cell, "trajectories.jsonl"))
                        n += 1
    log("  reused %d existing QG-het metric files -> %s" % (n, out))


def phase_D():
    out = V2 + "/exp_d_stress"
    gs = []
    # D1: noise sweep on het_gsaca, 6 games, 5 seeds, 4 noise levels
    for noise in ["0.0", "0.5", "1.0", "2.0"]:
        tag = "n" + noise.replace(".", "")
        gs.append(group("D1_%s_2p" % tag, 9,
            gargs(TWO_PLAYER, SEEDS5, ["het_gsaca"], out + "/d1_noise", 30, 5,
                  models_het=(QWEN, GLM), extra={"--payoff_noise_std": noise}), out + "/d1_noise"))
        gs.append(group("D1_%s_pg" % tag, 9,
            gargs(PG, SEEDS5, ["het_gsaca"], out + "/d1_noise", 20, 3,
                  models_het=(QWEN, GLM), extra={"--payoff_noise_std": noise}), out + "/d1_noise"))
    # D2: matching_pennies, 5 seeds, 4 het cells, 3 pairs (QG, QL, QQ)
    for nm, mh, mho in [("QG", (QWEN, GLM), None), ("QL", (QWEN, LLAMA), None), ("QQ", None, QWEN)]:
        gs.append(group("D2_mp_%s" % nm, (9 if mh else 4),
            gargs(["matching_pennies"], SEEDS5, (HET_CELLS if mh else HOM_CELLS), out + "/d2_mp", 30, 5,
                  models_het=mh, model_homo=mho), out + "/d2_mp"))
    return out, gs


def phase_E():
    out = V2 + "/exp_e_hyperparam"
    gs = []
    # theta sweep (het_dp_gated_atom_talk)
    for th in ["0.3", "0.45", "0.6", "0.75", "0.9"]:
        gs.append(group("E_th_%s" % th.replace(".", ""), 9,
            gargs(["chicken", "battle_of_the_sexes"], SEEDS5, ["het_dp_gated_atom_talk"], out + "/theta", 30, 5,
                  models_het=(QWEN, GLM),
                  extra={"--gate_trust_threshold": th, "--gate_ema_alpha": "0.3"}), out + "/theta"))
    # alpha sweep (het_dp_gated_atom_talk)
    for al in ["0.1", "0.2", "0.3", "0.5"]:
        gs.append(group("E_al_%s" % al.replace(".", ""), 9,
            gargs(["chicken", "battle_of_the_sexes"], SEEDS5, ["het_dp_gated_atom_talk"], out + "/alpha", 30, 5,
                  models_het=(QWEN, GLM),
                  extra={"--gate_ema_alpha": al, "--gate_trust_threshold": "0.6"}), out + "/alpha"))
    # W sweep (het_gsaca, gsaca_warmup is the W)
    for w in ["2", "3", "5", "8", "10"]:
        gs.append(group("E_W_%s" % w, 9,
            gargs(["chicken", "battle_of_the_sexes"], SEEDS5, ["het_gsaca"], out + "/warmup", 30, int(w),
                  models_het=(QWEN, GLM),
                  extra={"--gate_trust_threshold": "0.6", "--gate_ema_alpha": "0.3"}), out + "/warmup"))
    return out, gs


def phase_B():
    out = V2 + "/exp_b_20seed"
    gs = []
    for game in TWO_PLAYER:
        gs.append(group("B_%s" % game, 9,
            gargs([game], SEEDS20, HET_CELLS, out, 30, 5, models_het=(QWEN, GLM)), out))
    gs.append(group("B_public_goods", 9,
        gargs(PG, SEEDS20, HET_CELLS, out, 20, 3, models_het=(QWEN, GLM)), out))
    return out, gs


def wait_llama(timeout=2400):
    from huggingface_hub import snapshot_download
    log("waiting for Llama snapshot to be ready (up to %ds)..." % timeout)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            p = snapshot_download(LLAMA, allow_patterns=["config.json"],
                                  local_files_only=True)
            log("Llama snapshot ready: %s" % p)
            return True
        except Exception:
            time.sleep(20)
    log("WARN: Llama not ready after %ds — QL groups will attempt live download" % timeout)
    return False


def count_metrics(d):
    return len(glob.glob(os.path.join(d, "**", "metrics.json"), recursive=True))


def main():
    os.makedirs(V2, exist_ok=True)
    log("=" * 64)
    log("V2 ORCHESTRATOR START  (2 GPU, <=3 procs/GPU, <=28GB/GPU, 60s stagger)")
    log("=" * 64)
    sched = Scheduler(n_gpu=2, cap=3, vram_cap=28, stagger=60)

    # Phase 0: ensure Llama (needed by P2 QL + P3 D2 QL)
    wait_llama()

    # Phase 1: Exp C
    log("\n##### PHASE 1 / Exp C — payoff-in-prompt baseline #####")
    out, gs = phase_C(); sched.run(gs)
    log("P1 done: %d metrics in %s" % (count_metrics(out), out))

    # Phase 2: Exp A
    log("\n##### PHASE 2 / Exp A — hom controls + QL-het #####")
    out, gs = phase_A(); copy_existing_qg(out); sched.run(gs)
    log("P2 done: %d metrics in %s" % (count_metrics(out), out))

    # Phase 3: Exp D
    log("\n##### PHASE 3 / Exp D — stress test (noise + matching_pennies) #####")
    out, gs = phase_D(); sched.run(gs)
    log("P3 done: %d metrics in %s" % (count_metrics(out), out))

    # Phase 4: Exp E
    log("\n##### PHASE 4 / Exp E — hyperparam sweep #####")
    out, gs = phase_E(); sched.run(gs)
    log("P4 done: %d metrics in %s" % (count_metrics(out), out))

    # Phase 5: Exp B (auto-chain)
    log("\n##### PHASE 5 / Exp B — 20-seed full rerun (auto-chain) #####")
    out, gs = phase_B(); sched.run(gs)
    log("P5 done: %d metrics in %s" % (count_metrics(out), out))

    log("\n" + "=" * 64)
    log("V2 ORCHESTRATOR COMPLETE")
    for sub in ["exp_c_payoff_prompt", "exp_a_pairs", "exp_d_stress",
                "exp_e_hyperparam", "exp_b_20seed"]:
        log("  %-22s %d metrics" % (sub, count_metrics(os.path.join(V2, sub))))
    log("=" * 64)


if __name__ == "__main__":
    main()
