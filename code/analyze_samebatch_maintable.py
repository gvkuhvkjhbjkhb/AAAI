#!/usr/bin/env python3
"""W1 fix -- SAME-BATCH main-table recompute + cross-batch drift diagnostics.

Reads a single same-batch out_dir (produced by run_samebatch_maintable.sh) that
contains, for every (game, seed), the SCA arm(s) AND the payoff-in-prompt
baseline generated back-to-back in one run. Because the pairing is now
same-batch, the paired difference no longer confounds method effect with the
0.4-0.9 absolute-payoff batch drift that motivated reviewer W1.

It produces three blocks:

  (A) SAME-BATCH main table -- the honest replacement for Table 2:
      ours (SCA) vs payoff-in-prompt, n=20 paired Wilcoxon, Cohen's dz, win rate.
      Arm per game: coord -> Gated (het_gated_atom_talk); anti/boundary ->
      NoToM/abstain (het_notom), matching recompute_maintable.py.

  (B) PER-GAME DRIFT diagnostic -- the number the paper SHOULD compare each
      effect against. For each game and each arm present in BOTH the old
      cross-batch data and this new same-batch run, we report the absolute
      shift of the SAME arm between batches (|mean_new - mean_old|). That is the
      empirically measured drift FOR THAT GAME, replacing the blanket "0.4-0.9"
      band. We then flag whether the same-batch effect exceeds that game's own
      measured drift.

  (C) OLD vs NEW comparison -- side-by-side of the submitted (cross-batch)
      delta and the same-batch delta, so the rebuttal can state exactly how
      much of each headline effect survives once the batch confound is removed.

Zero deps beyond numpy + scipy. Offline.

Usage:
  python3 code/analyze_samebatch_maintable.py \
      --root v2_results/exp_samebatch_maintable \
      [--old_base ../PAPER_V5_PACKAGE/02_raw_data]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

CELL = {"het_notom": "NoToM", "het_gated_atom_talk": "Gated",
        "het_dp_gated_atom_talk": "CGA", "het_gsaca": "GSACA",
        "het_payoff_prompt": "Payoff"}
GROUP = {"chicken": "anti", "hawk_dove": "anti", "deadlock": "anti",
         "stag_hunt": "coord", "battle_of_the_sexes": "coord",
         "public_goods": "boundary"}
OUR_ARM = {"anti": "NoToM", "boundary": "NoToM", "coord": "Gated"}
ORDER = ["chicken", "deadlock", "hawk_dove", "stag_hunt",
         "battle_of_the_sexes", "public_goods"]


def load(root, cells):
    """root/<game>/seed_<n>/<cell>/metrics.json -> data[game][label][seed]=payoff."""
    data = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(f"{root}/*/seed_*/*/metrics.json"):
        p = f.split("/")
        cd, seed, game = p[-2], p[-3], p[-4]
        if cd not in cells:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cp = d.get("cooperation_payoff")
        if cp is None or (isinstance(cp, float) and np.isnan(cp)):
            continue
        data[game][cells[cd]][seed] = float(cp)
    return data


def paired(a_map, b_map):
    s = sorted(set(a_map) & set(b_map))
    if len(s) < 3:
        return None
    a = np.array([a_map[x] for x in s])
    b = np.array([b_map[x] for x in s])
    diff = a - b
    sd = diff.std(ddof=1)
    dz = diff.mean() / sd if sd > 0 else 0.0
    try:
        p = stats.wilcoxon(a, b).pvalue
    except Exception:
        p = float("nan")
    return diff.mean(), len(s), p, dz, float((diff > 0).mean()), a.mean(), b.mean()


def sig(p):
    if p != p:
        return "?"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def load_old(old_base):
    """Reconstruct the SUBMITTED (cross-batch) cell means, exactly as
    recompute_maintable.py assembled them."""
    cells = defaultdict(lambda: defaultdict(dict))
    for root in [f"{old_base}/v2_results/exp_b_20seed",
                 f"{old_base}/results_v2/exp_b_20seed"]:
        d = load(root, CELL)
        for g in d:
            for c in d[g]:
                cells[g][c].update(d[g][c])
    pay = load(f"{old_base}/v2_results/exp_c_payoff_prompt",
               {"het_payoff_prompt": "Payoff"})
    for g in pay:
        for c in pay[g]:
            cells[g][c].update(pay[g][c])
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="same-batch out_dir (exp_samebatch_maintable)")
    ap.add_argument("--old_base", default=None,
                    help="PAPER_V5_PACKAGE/02_raw_data for drift diagnostics")
    args = ap.parse_args()

    new = load(args.root, CELL)
    old = load_old(args.old_base) if args.old_base and os.path.isdir(args.old_base) else None

    # ---------------- (A) SAME-BATCH MAIN TABLE ----------------
    print("=" * 104)
    print("(A) SAME-BATCH MAIN TABLE  --  ours (SCA) vs payoff-in-prompt "
          "(n=20 paired Wilcoxon, same batch)")
    print("=" * 104)
    hdr = (f"  {'game':20s} {'grp':8s} {'arm':6s} {'ours':>7s} {'payoff':>7s} "
           f"{'delta':>8s} {'p':>9s} {'sig':>4s} {'dz':>8s} {'win':>5s} {'n':>3s}")
    print(hdr)
    samebatch = {}
    for g in ORDER:
        grp = GROUP[g]
        arm = OUR_ARM[grp]
        ours = new.get(g, {}).get(arm, {})
        pay = new.get(g, {}).get("Payoff", {})
        r = paired(ours, pay)
        if not r:
            print(f"  {g:20s} {grp:8s} {arm:6s}   -- missing cells (ours={len(ours)}, "
                  f"payoff={len(pay)}) --")
            continue
        d, n, p, dz, win, om, pm = r
        samebatch[g] = (d, p, dz, win, n, arm, grp)
        print(f"  {g:20s} {grp:8s} {arm:6s} {om:7.3f} {pm:7.3f} {d:+8.3f} "
              f"{p:9.4f} {sig(p):>4s} {dz:+8.2f} {win:5.0%} {n:3d}")

    wins = sum(1 for g in samebatch if samebatch[g][1] < 0.05 and samebatch[g][0] > 0)
    negs = sum(1 for g in samebatch if samebatch[g][1] < 0.05 and samebatch[g][0] < 0)
    print(f"\n  SAME-BATCH headline: {wins}/6 significant wins, {negs} significant losses.")

    # ---------------- (B) PER-GAME MEASURED DRIFT ----------------
    if old is not None:
        print("\n" + "=" * 104)
        print("(B) PER-GAME MEASURED DRIFT  --  |mean_new - mean_old| of the SAME arm "
              "across batches (replaces blanket 0.4-0.9)")
        print("=" * 104)
        print(f"  {'game':20s} {'arm':6s} {'old_mean':>9s} {'new_mean':>9s} "
              f"{'drift':>8s}   {'effect':>8s}   {'effect>drift?':>13s}")
        for g in ORDER:
            arm = OUR_ARM[GROUP[g]]
            om = list(old.get(g, {}).get(arm, {}).values()) if old else []
            nm = list(new.get(g, {}).get(arm, {}).values())
            if not om or not nm:
                continue
            drift = abs(np.mean(nm) - np.mean(om))
            eff = samebatch.get(g, (float('nan'),))[0]
            flag = "YES" if (eff == eff and abs(eff) > drift) else "no"
            print(f"  {g:20s} {arm:6s} {np.mean(om):9.3f} {np.mean(nm):9.3f} "
                  f"{drift:8.3f}   {eff:+8.3f}   {flag:>13s}")
        # payoff arm drift too
        print("\n  (payoff-in-prompt arm drift across batches)")
        print(f"  {'game':20s} {'arm':6s} {'old_mean':>9s} {'new_mean':>9s} {'drift':>8s}")
        for g in ORDER:
            om = list(old.get(g, {}).get("Payoff", {}).values()) if old else []
            nm = list(new.get(g, {}).get("Payoff", {}).values())
            if not om or not nm:
                continue
            print(f"  {g:20s} {'Payoff':6s} {np.mean(om):9.3f} {np.mean(nm):9.3f} "
                  f"{abs(np.mean(nm)-np.mean(om)):8.3f}")

    # ---------------- (C) OLD (cross-batch) vs NEW (same-batch) ----------------
    if old is not None:
        print("\n" + "=" * 104)
        print("(C) SUBMITTED (cross-batch) vs SAME-BATCH delta  --  how much of "
              "each headline effect survives")
        print("=" * 104)
        print(f"  {'game':20s} {'old_delta':>10s} {'old_sig':>8s}   "
              f"{'new_delta':>10s} {'new_sig':>8s}   {'shrinkage':>10s}")
        for g in ORDER:
            arm = OUR_ARM[GROUP[g]]
            ro = paired(old.get(g, {}).get(arm, {}), old.get(g, {}).get("Payoff", {}))
            if g not in samebatch or ro is None:
                continue
            od, on, op, odz, owin, oo, opm = ro
            nd, np_, ndz, nwin, nn, _, _ = samebatch[g]
            shrink = (od - nd)
            print(f"  {g:20s} {od:+10.3f} {sig(op):>8s}   "
                  f"{nd:+10.3f} {sig(np_):>8s}   {shrink:+10.3f}")

    print("\nDone. Block (A) is the honest same-batch Table 2; (B) gives each "
          "game's own measured drift to compare against; (C) shows survival.")


if __name__ == "__main__":
    main()
