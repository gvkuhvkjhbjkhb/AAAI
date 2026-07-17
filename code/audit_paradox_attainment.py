#!/usr/bin/env python3
"""Post-run audit for the 3-arm paradox/attainment study.

Checks:
  (1) completeness: 360/360 metrics.json present.
  (2) balance: each (game, arm) has exactly the 20 expected seeds.
  (3) trajectory re-computation: for a random >=10% sample of cells, reload
      trajectories.jsonl and recompute cooperation_payoff exactly as
      hettom_baseline.compute_metrics does (mean over episodes of the per-step
      mean of rewards[0]); flag any |recomputed - stored| > 1e-6.
  (4) arm_order manifests: present for every (game, seed) and match the
      pre-registered Latin square (seed % 3 rotation).

Exit 0 iff all checks pass. Writes AUDIT_REPORT.txt into --root.
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np

CELLS = ["het_notom", "het_gated_atom_talk", "het_dp_gated_atom_talk"]
GAMES = ["chicken", "deadlock", "hawk_dove", "stag_hunt",
         "battle_of_the_sexes", "public_goods"]
SEEDS = list(range(42, 62))


def latin(seed):
    k = len(CELLS); shift = seed % k
    return [CELLS[(i + shift) % k] for i in range(k)]


def recompute_coop_payoff(traj_path):
    per_ep = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)["steps"]
            if not ep:
                continue
            per_ep.append(float(np.mean([s["rewards"][0] for s in ep])))
    return float(np.mean(per_ep)) if per_ep else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sample_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    R = args.root
    out = []
    def emit(s=""):
        print(s); out.append(s)

    ok = True
    emit("=" * 80)
    emit(f"POST-RUN AUDIT  root={R}")
    emit("=" * 80)

    # (1) completeness + (2) balance
    total = 0
    missing = []
    for g in GAMES:
        for c in CELLS:
            present = set()
            for s in SEEDS:
                p = os.path.join(R, g, f"seed_{s}", c, "metrics.json")
                if os.path.exists(p):
                    present.add(s); total += 1
                else:
                    missing.append((g, s, c))
            miss = sorted(set(SEEDS) - present)
            tag = "OK" if len(present) == 20 else f"MISSING {miss}"
            if len(present) != 20:
                ok = False
            emit(f"  {g:22s} {c:24s} {len(present):2d}/20  {tag}")
    emit(f"\n(1)+(2) total metrics = {total}/360   "
         f"{'PASS' if total == 360 and not missing else 'FAIL'}")
    if missing:
        ok = False
        emit(f"    MISSING: {missing[:20]}")

    # (4) arm_order manifests vs Latin square
    emit("\n(4) arm_order manifest check (Latin square seed%3 rotation)")
    bad_manifest = 0
    for g in GAMES:
        for s in SEEDS:
            mp = os.path.join(R, g, f"seed_{s}", "arm_order.json")
            if not os.path.exists(mp):
                emit(f"    MISSING manifest {g} seed_{s}"); bad_manifest += 1; continue
            m = json.load(open(mp))
            if m.get("arm_order") != latin(s):
                emit(f"    MISMATCH {g} seed_{s}: {m.get('arm_order')} != {latin(s)}")
                bad_manifest += 1
    emit(f"    manifests bad = {bad_manifest}   {'PASS' if bad_manifest == 0 else 'FAIL'}")
    if bad_manifest:
        ok = False

    # (3) trajectory re-computation on >=10% sample
    all_cells = []
    for g in GAMES:
        for s in SEEDS:
            for c in CELLS:
                mp = os.path.join(R, g, f"seed_{s}", c, "metrics.json")
                tp = os.path.join(R, g, f"seed_{s}", c, "trajectories.jsonl")
                if os.path.exists(mp) and os.path.exists(tp):
                    all_cells.append((g, s, c, mp, tp))
    random.seed(args.seed)
    k = max(36, int(len(all_cells) * args.sample_frac))
    sample = random.sample(all_cells, min(k, len(all_cells)))
    emit(f"\n(3) trajectory audit: recomputing cooperation_payoff for "
         f"{len(sample)}/{len(all_cells)} cells ({len(sample)/max(len(all_cells),1):.0%})")
    max_err = 0.0
    n_bad = 0
    for g, s, c, mp, tp in sample:
        stored = json.load(open(mp)).get("cooperation_payoff")
        recomp = recompute_coop_payoff(tp)
        if stored is None or recomp != recomp:
            continue
        err = abs(stored - recomp)
        max_err = max(max_err, err)
        if err > 1e-6:
            n_bad += 1
            emit(f"    MISMATCH {g}/{s}/{c}: stored={stored:.6f} recomp={recomp:.6f} err={err:.2e}")
    emit(f"    sampled={len(sample)}  max|err|={max_err:.2e}  mismatches={n_bad}   "
         f"{'PASS' if n_bad == 0 else 'FAIL'}")
    if n_bad:
        ok = False

    emit("\n" + "=" * 80)
    emit(f"AUDIT RESULT: {'ALL PASS' if ok else 'FAILURES DETECTED'}")
    emit("=" * 80)
    with open(os.path.join(R, "AUDIT_REPORT.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
