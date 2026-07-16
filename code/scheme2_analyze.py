#!/usr/bin/env python3
"""Scheme 2 analysis — silent-anti-coord GSACA vs baselines.

Reads the Scheme 2 output (exp_scheme2_silent/het_gsaca_silent) and compares
against the existing 20-seed baselines (NoToM / Gated / CGA / old-GSACA) from
exp_b_20seed, on the 3 anti-coord games (chicken, hawk_dove, deadlock).

Key question: does blocking cheap-talk in anti_coord mode (silent ToM-only)
beat the old GSACA, and does it push chicken from +0.046(ns) toward significance?

Paired Wilcoxon (one-sided), Cohen's dz, BCa 95% CI, win rate. dev/holdout split.
"""
import glob, json, os
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
import warnings; warnings.filterwarnings("ignore")

ROOT20 = "/data/lab/results/v2/exp_b_20seed"
ROOTS2 = "/data/lab/results/v2/exp_scheme2_silent"
OUT    = "/data/lab/results/v2/scheme2_offline"

NOTOM = "het_notom"; GATED = "het_gated_atom_talk"
CGA   = "het_dp_gated_atom_talk"; GSACA = "het_gsaca"; SILENT = "het_gsaca_silent"
GAMES = ["chicken", "hawk_dove", "deadlock"]
DEV = list(range(42, 52)); HOLD = list(range(52, 62)); ALL20 = DEV + HOLD


def load(root, cell, metric="team_mean_payoff"):
    out = defaultdict(dict)
    for mp in glob.glob(os.path.join(root, "**", cell, "metrics.json"), recursive=True):
        rel = mp.replace(root, "").strip("/").split(os.sep)
        if len(rel) < 4 or not rel[1].startswith("seed_"):
            continue
        g, s = rel[0], int(rel[1].replace("seed_", ""))
        m = json.load(open(mp))
        if metric in m:
            out[g][s] = float(m[metric])
    return out


def wpvalue(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) < 2 or np.all(d == 0):
        return 1.0 if len(d) else float("nan")
    return float(wilcoxon(d, zero_method="wilcox", correction=True,
                 alternative="greater" if d.mean() > 0 else "less").pvalue)


def bca(d, n_boot=10000, seed=0):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n < 2: return (np.nan, np.nan)
    rng = np.random.RandomState(seed); xb = d.mean()
    boot = np.array([d[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    from scipy.stats import norm
    z0 = norm.ppf(min(max(np.mean(boot < xb), 1e-6), 1 - 1e-6))
    jack = np.array([np.delete(d, i).mean() for i in range(n)]); jb = jack.mean()
    a = np.sum((jb - jack) ** 3) / (6.0 * (np.sum((jack - jb) ** 2) ** 1.5) or 1)
    lo = norm.cdf(z0 + (z0 + norm.ppf(0.025)) / (1 - a * (z0 + norm.ppf(0.025))))
    hi = norm.cdf(z0 + (z0 + norm.ppf(0.975)) / (1 - a * (z0 + norm.ppf(0.975))))
    return (float(np.quantile(boot, lo)), float(np.quantile(boot, hi)))


def sig(p):
    if p is None or np.isnan(p): return "  "
    return "***" if p < 0.01 else ("** " if p < 0.05 else ("*  " if p < 0.10 else "ns "))


def cmp(treat_vals, ctrl_vals, seeds):
    t = np.array([treat_vals.get(s, np.nan) for s in seeds], float)
    c = np.array([ctrl_vals.get(s, np.nan) for s in seeds], float)
    d = t - c; m = ~np.isnan(d); d = d[m]
    if len(d) < 2: return None
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan")
    return dict(n=len(d), tmean=float(t[m].mean()), cmean=float(c[m].mean()),
                diff=float(d.mean()), dz=float(dz), win=float((d > 0).mean()),
                p=wpvalue(d), ci=bca(t[m] - c[m]))


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 95)
    print("SCHEME 2 ANALYSIS — silent-anti-coord GSACA vs baselines (3 anti-coord games)")
    print("=" * 95)

    silent = load(ROOTS2, SILENT)
    notom = load(ROOT20, NOTOM); gated = load(ROOT20, GATED)
    cga = load(ROOT20, CGA); gsaca = load(ROOT20, GSACA)

    n_silent = sum(len(silent[g]) for g in GAMES)
    print(f"silent metrics loaded: {n_silent} (expect 60 = 3 games x 20 seeds)")
    for g in GAMES:
        print(f"  {g}: {len(silent.get(g, {}))} seeds")

    for label, seeds in [("FULL n=20", ALL20), ("DEV 42-51", DEV), ("HOLDOUT 52-61", HOLD)]:
        print(f"\n##### {label} — silent-GSACA vs each baseline #####")
        print("  %-12s %-10s %4s %8s %8s %8s %6s %5s %9s %4s  %16s" %
              ("game", "vs", "n", "silent", "ctrl", "diff", "dz", "win", "p(1s)", "sig", "BCa95"))
        for g in GAMES:
            if g not in silent or len(silent[g]) < 2:
                print("  %-12s  (no data)" % g); continue
            for nm, ctrl in [("NoToM", notom), ("Gated", gated), ("CGA", cga), ("oldGSACA", gsaca)]:
                r = cmp(silent[g], ctrl.get(g, {}), seeds)
                if r is None: continue
                print("  %-12s %-10s %4d %8.3f %8.3f %+8.3f %6.2f %5.2f %9.3g %4s  [%+6.3f,%+6.3f]" %
                      (g, nm, r["n"], r["tmean"], r["cmean"], r["diff"],
                       r["dz"], r["win"], r["p"], sig(r["p"]), r["ci"][0], r["ci"][1]))

    # head-to-head: silent vs old GSACA (does blocking cheap-talk help?)
    print("\n" + "=" * 95)
    print("HEAD-TO-HEAD: silent-GSACA vs old-GSACA  (does blocking cheap-talk in anti_coord help?)")
    print("=" * 95)
    print("  %-12s %4s %8s %8s %8s %6s %5s %9s %4s  %16s" %
          ("game", "n", "silent", "oldGSA", "diff", "dz", "win", "p(2s)", "sig", "BCa95"))
    for g in GAMES:
        if g not in silent: continue
        t = np.array([silent[g].get(s, np.nan) for s in ALL20], float)
        c = np.array([gsaca.get(g, {}).get(s, np.nan) for s in ALL20], float)
        d = t - c; m = ~np.isnan(d); d = d[m]
        if len(d) < 2: continue
        p2 = float(wilcoxon(d, zero_method="wilcox", correction=True, alternative="two-sided").pvalue)
        print("  %-12s %4d %8.3f %8.3f %+8.3f %6.2f %5.2f %9.3g %4s  [%+6.3f,%+6.3f]" %
              (g, len(d), t[m].mean(), c[m].mean(), d.mean(),
               d.mean()/d.std(ddof=1) if d.std(ddof=1) else float("nan"),
               float((d>0).mean()), p2, sig(p2), *bca(d)))

    print(f"\n  -> results in {OUT}/")


if __name__ == "__main__":
    main()
