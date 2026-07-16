#!/usr/bin/env python3
"""Parallel work-queue driver for the W1 same-batch main table + Exp-A gap-fill.

Runs one subprocess per (out_dir, game, seed, cell) work item, all against
shared vLLM servers (OpenAI-compatible), with a bounded concurrency pool so the
two RTX 5090s stay saturated. Each subprocess is the already-validated
code/run_experiment_local.py in --use_vllm mode, invoked for exactly ONE cell,
so all mechanism logic (ToM, gating, GSACA, payoff-in-prompt, metrics) is
unchanged. Idempotent: skips any cell whose metrics.json already exists.

The vLLM endpoint per model is chosen by run_experiment_local's VLLM_API_BASE_MAP,
which we extend at runtime via env VLLM_ENDPOINTS (json: {model_name: url}).

Usage:
  python3 code/parallel_driver.py --workers 24 --manifest work_manifest.json
  # or generate the built-in manifest:
  python3 code/parallel_driver.py --emit_manifest work_manifest.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_experiment_local.py")


def episodes_for(game):
    return 20 if game == "public_goods" else 30


def run_one(item):
    """item: dict(out_dir, game, seed, cell, models_het[list]|model_homo, homogeneous)."""
    out_dir = item["out_dir"]
    game = item["game"]
    seed = item["seed"]
    cell = item["cell"]
    mpath = os.path.join(out_dir, game, f"seed_{seed}", cell, "metrics.json")
    if os.path.exists(mpath):
        return (item, "skip", 0.0)
    ep = episodes_for(game)
    cmd = [sys.executable, RUNNER,
           "--games", game, "--seeds", str(seed),
           "--episodes", str(ep), "--horizon", "5", "--memory", "2",
           "--cells", cell, "--out_dir", out_dir,
           "--log_every", "999", "--use_vllm", "--force"]
    if item.get("homogeneous"):
        cmd += ["--model_homo", item["model_homo"]]
    else:
        cmd += ["--models_het"] + item["models_het"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        ok = os.path.exists(mpath)
        status = "done" if ok else "FAIL"
        if not ok:
            # keep last lines of stderr for debugging
            tail = (r.stderr or r.stdout or "")[-400:]
            return (item, "FAIL:" + tail.replace("\n", " "), time.time() - t0)
        return (item, status, time.time() - t0)
    except subprocess.TimeoutExpired:
        return (item, "TIMEOUT", time.time() - t0)
    except Exception as e:
        return (item, "ERR:" + str(e)[:200], time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--emit_manifest", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--progress_every", type=int, default=10)
    args = ap.parse_args()

    if args.emit_manifest:
        items = build_manifest()
        json.dump(items, open(args.emit_manifest, "w"), indent=1)
        print(f"wrote {len(items)} items -> {args.emit_manifest}")
        summarize(items)
        return

    items = json.load(open(args.manifest))
    # filter out already-done
    todo = [it for it in items
            if not os.path.exists(os.path.join(it["out_dir"], it["game"],
                                                f"seed_{it['seed']}", it["cell"],
                                                "metrics.json"))]
    print(f"[driver] {len(items)} items, {len(todo)} to run, {args.workers} workers",
          flush=True)
    t0 = time.time()
    done = fail = skip = 0
    times = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, it) for it in todo]
        for i, f in enumerate(as_completed(futs), 1):
            it, st, dt = f.result()
            if st == "done":
                done += 1; times.append(dt)
            elif st == "skip":
                skip += 1
            else:
                fail += 1
                print(f"  [FAIL] {it['out_dir'].split('/')[-1]}/{it['game']}/"
                      f"seed_{it['seed']}/{it['cell']} :: {st[:160]}", flush=True)
            if i % args.progress_every == 0 or i == len(todo):
                el = time.time() - t0
                rate = done / el * 3600 if el > 0 else 0
                rem = len(todo) - i
                eta = rem / (done / el) / 60 if done > 0 and el > 0 else float("nan")
                avg = sum(times) / len(times) if times else 0
                print(f"  [{i}/{len(todo)}] done={done} fail={fail} "
                      f"avg={avg:.0f}s rate={rate:.0f}/h ETA~{eta:.0f}min "
                      f"elapsed={el/60:.1f}min", flush=True)
    print(f"[driver] COMPLETE done={done} fail={fail} skip={skip} "
          f"in {(time.time()-t0)/60:.1f}min", flush=True)


def build_manifest():
    """Enumerate all W1 + gap-fill work items."""
    GSACA = os.environ.get("GSACA_ROOT", "/data/lab/gsaca")
    V2 = os.path.join(GSACA, "v2_results")
    QWEN = "Qwen/Qwen2.5-7B-Instruct"
    GLM = "THUDM/GLM-4-9B-0414"
    LLAMA = os.environ.get("LLAMA_MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct")
    SEEDS20 = list(range(42, 62))
    SEEDS5 = [42, 43, 44, 45, 46]
    GAMES6 = ["chicken", "deadlock", "hawk_dove", "stag_hunt",
              "battle_of_the_sexes", "public_goods"]
    items = []

    # ---------- W1 SAME-BATCH MAIN TABLE ----------
    # one fresh tree; per game the SCA arm(s) + payoff-in-prompt baseline.
    # We run all three het arms everywhere so analysis can pick per-game and
    # also compute drift; the baseline is het_payoff_prompt.
    sb = os.path.join(V2, "exp_samebatch_maintable")
    W1_ARMS = ["het_notom", "het_gated_atom_talk", "het_payoff_prompt"]
    for g in GAMES6:
        for s in SEEDS20:
            for c in W1_ARMS:
                items.append(dict(out_dir=sb, game=g, seed=s, cell=c,
                                  homogeneous=False, models_het=[QWEN, GLM]))

    # ---------- EXP-A GAP-FILL (n=5) ----------
    HOM = ["hom_notom", "hom_gated_atom_talk", "hom_dp_gated_atom_talk", "hom_gsaca"]
    HET = ["het_notom", "het_gated_atom_talk", "het_dp_gated_atom_talk", "het_gsaca"]
    exp_a = os.path.join(V2, "exp_a_fix")

    # QQ (Qwen x Qwen homogeneous): stag_hunt, BoS, public_goods
    for g in ["stag_hunt", "battle_of_the_sexes", "public_goods"]:
        for s in SEEDS5:
            for c in HOM:
                items.append(dict(out_dir=os.path.join(exp_a, "QQ"), game=g,
                                  seed=s, cell=c, homogeneous=True, model_homo=QWEN))
    # GG (GLM x GLM homogeneous): deadlock, stag_hunt, BoS, public_goods
    for g in ["deadlock", "stag_hunt", "battle_of_the_sexes", "public_goods"]:
        for s in SEEDS5:
            for c in HOM:
                items.append(dict(out_dir=os.path.join(exp_a, "GG"), game=g,
                                  seed=s, cell=c, homogeneous=True, model_homo=GLM))
    # QL (Qwen x Llama heterogeneous): deadlock only (per spec)
    for g in ["deadlock"]:
        for s in SEEDS5:
            for c in HET:
                items.append(dict(out_dir=os.path.join(exp_a, "QL"), game=g,
                                  seed=s, cell=c, homogeneous=False,
                                  models_het=[QWEN, LLAMA]))
    return items


def summarize(items):
    from collections import Counter
    by = Counter()
    for it in items:
        tree = it["out_dir"].split("/")[-1]
        by[(tree, it["game"])] += 1
    print("  work items by (tree, game):")
    for k in sorted(by):
        print(f"    {k[0]:26s} {k[1]:20s} {by[k]}")
    print(f"  TOTAL = {len(items)} cells")


if __name__ == "__main__":
    main()
