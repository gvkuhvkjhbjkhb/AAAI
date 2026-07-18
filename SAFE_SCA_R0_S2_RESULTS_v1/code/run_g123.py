#!/usr/bin/env python3
"""Unified parallel driver for G1 (end-to-end bandit), G2 (top_p=1.0 ablation),
and G3 (online-detection validation). Shards each group's (game, seed) pairs
across N worker subprocesses that all hit the two shared vLLM servers
(Qwen:8000 / GLM:8001), so the GPUs stay saturated. Resumable: the runner
skips cells whose metrics.json already exists.

Groups:
  G1  exp_g1_bandit  het_bandit            6 games x 20 seeds  top_p=0.9 bandit_k=5
  G2  exp_g2_topp1   het_notom + Gated     3 anti x 10 seeds   top_p=1.0
  G3  exp_g3_detect  het_gsaca             6 games x 5 seeds   top_p=0.9

Usage:  python3 run_g123.py            # run all three concurrently
        python3 run_g123.py G1 G3      # only G1 and G3
"""
import os, sys, time, subprocess
from datetime import datetime

ROOT = os.environ.get("GSACA_ROOT", "/data/lab/AAAI")
RUNNER = f"{ROOT}/code/run_experiment_local.py"
OUT_BASE = f"{ROOT}/v2_results"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
GLM = "THUDM/GLM-4-9B-0414"
SEEDS20 = list(range(42, 62))
SEEDS10 = list(range(42, 52))
SEEDS5 = list(range(42, 47))
TWO = ["chicken", "deadlock", "hawk_dove", "stag_hunt", "battle_of_the_sexes"]
ANTI3 = ["chicken", "deadlock", "hawk_dove"]
ALL6 = TWO + ["public_goods"]


def shard(seeds, n):
    return [seeds[i::n] for i in range(n) if seeds[i::n]]


def base_cmd(top_p, extra):
    c = ["python3", RUNNER, "--use_vllm", "--gen_seed_base", "1000",
         "--auto_episodes", "--horizon", "5", "--memory", "2", "--log_every", "100",
         "--models_het", QWEN, GLM,
         "--gate_trust_threshold", "0.6", "--gate_ema_alpha", "0.3", "--atom_warmup", "3",
         "--top_p", str(top_p)] + extra
    return c


procs, logfiles = [], []


def launch(group, games, seeds, cells, top_p, nshard, extra=None):
    extra = extra or []
    out = f"{OUT_BASE}/{group}"
    os.makedirs(f"{out}/logs", exist_ok=True)
    for k, s in enumerate(shard(seeds, nshard)):
        cmd = base_cmd(top_p, extra) + ["--games"] + games + \
              ["--seeds"] + [str(x) for x in s] + ["--cells"] + cells + \
              ["--out_dir", out]
        lf = f"{out}/logs/{group}_shard{k}.log"
        f = open(lf, "w")
        procs.append((group, k, subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)))
        logfiles.append((f, lf))
    print(f"[{group}] launched {len(shard(seeds, nshard))} workers "
          f"({len(seeds)} seeds x {len(games)} games x {len(cells)} cells)", flush=True)


which = set(sys.argv[1:]) if len(sys.argv) > 1 else {"G1", "G2", "G3"}

if "G1" in which:
    launch("exp_g1_bandit", ALL6, SEEDS20, ["het_bandit"], 0.9, nshard=10,
           extra=["--bandit_k", "5"])
if "G2" in which:
    launch("exp_g2_topp1", ANTI3, SEEDS10, ["het_notom", "het_gated_atom_talk"], 1.0,
           nshard=6)
if "G3" in which:
    launch("exp_g3_detect", ALL6, SEEDS5, ["het_gsaca"], 0.9, nshard=4)

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ALL {len(procs)} workers launched; "
      f"waiting...\n", flush=True)
t0 = time.time()
done = 0
for group, k, p in procs:
    p.wait()
    done += 1
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {done}/{len(procs)} "
          f"{group}/shard{k} rc={p.returncode} elapsed={time.time()-t0:.0f}s", flush=True)
for f, _ in logfiles:
    try:
        f.close()
    except Exception:
        pass
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ALL DONE in {time.time()-t0:.0f}s", flush=True)
