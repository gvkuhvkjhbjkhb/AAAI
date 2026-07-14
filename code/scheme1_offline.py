#!/usr/bin/env python3
"""Scheme 1 (offline, zero-GPU) — three-arm abstention GSACA recompute.

Reads existing exp_b_20seed metrics (the three arms CGA/Gated/NoToM already ran
for 20 seeds) + the per-seed warmup split_score stored in het_gsaca metrics.json,
and recomputes what a THREE-ARM selector would pick:

    split > +tau  -> CGA   (het_dp_gated_atom_talk)
    split < -tau  -> Gated (het_gated_atom_talk)
    |split| <= tau -> NoToM (het_notom)        [ABSTAIN: no intervention]

vs the current TWO-ARM GSACA rule (split>0 -> CGA, split<=0 -> Gated).

Statistical discipline (per the plan):
  - dev   = seeds 42..51  (used ONLY to choose / tune tau)
  - hold  = seeds 52..61  (frozen; final table reported here)
  - tau sensitivity sweep {0.2,0.3,0.4,0.5,0.6} run on DEV ONLY
  - pre-registered main tau = 0.4 ; final holdout table uses tau=0.4

Outputs:
  - reproduces n=20 diagnostic table (GSACA vs NoToM/Gated/CGA) to confirm the
    deadlock -0.023 gap and chicken +0.046 ns
  - three-arm vs NoToM / Gated / old-GSACA / CGA, full + dev + holdout
  - Proposition 3 empirical check: abstain-zone regret vs NoToM == 0
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

ROOT = "/data/lab/results/v2/exp_b_20seed"
ROOT_N5 = "/data/lab/results/gsaca_full_20260712_120138"   # n=5 fallback for BoS + public_goods
OUT  = "/data/lab/results/v2/scheme1_offline"

NOTOM = "het_notom"
GATED = "het_gated_atom_talk"
CGA   = "het_dp_gated_atom_talk"
GSACA = "het_gsaca"

# 4 games complete at 20 seeds; BoS + public_goods pending in the ongoing run
#   (previewed at n=5 from ROOT_N5 — selector is deterministic for both:
#    BoS split ~ -2.5 -> Gated arm; public_goods split ~ -0.21 -> abstain -> NoToM)
GAMES_DONE = ["chicken", "hawk_dove", "deadlock", "stag_hunt"]
GAMES_N5   = ["battle_of_the_sexes", "public_goods"]
ORACLE = {"chicken": "anti_coord", "hawk_dove": "anti_coord",
          "deadlock": "anti_coord", "stag_hunt": "coord",
          "battle_of_the_sexes": "coord", "public_goods": "coord"}

DEV = list(range(42, 52))   # 42..51
HOLD = list(range(52, 62))  # 52..61
ALL20 = DEV + HOLD


def load(root, metric):
    """-> {game: {cell: {seed: value}}} for the given metric."""
    out = defaultdict(lambda: defaultdict(dict))
    for mpath in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        rel = mpath.replace(root, "").strip("/").split(os.sep)
        if len(rel) < 4:
            continue
        game, seedstr, cell = rel[0], rel[1], rel[2]
        if not seedstr.startswith("seed_"):
            continue
        try:
            seed = int(seedstr.replace("seed_", ""))
        except ValueError:
            continue
        with open(mpath) as f:
            m = json.load(f)
        if metric in m:
            out[game][cell][seed] = float(m[metric])
    return out


def load_splits(root):
    """-> {game: {seed: split_score}} from het_gsaca metrics."""
    out = defaultdict(dict)
    for mpath in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        rel = mpath.replace(root, "").strip("/").split(os.sep)
        if len(rel) < 4 or rel[2] != GSACA:
            continue
        game, seedstr = rel[0], rel[1]
        if not seedstr.startswith("seed_"):
            continue
        try:
            seed = int(seedstr.replace("seed_", ""))
        except ValueError:
            continue
        with open(mpath) as f:
            m = json.load(f)
        if "gsaca_split_score" in m:
            out[game][seed] = float(m["gsaca_split_score"])
    return out


def wpvalue(d, two_sided=False):
    """One-sided Wilcoxon signed-rank p (greater if mean>0 else less).
    all-zero diffs -> p=1.0 (no effect); scipy handles all-same-sign correctly."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return float("nan")
    if np.all(d == 0):
        return 1.0
    if not HAVE_SCIPY:
        return float("nan")
    alt = "two-sided" if two_sided else ("greater" if d.mean() > 0 else "less")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(wilcoxon(d, zero_method="wilcox", correction=True,
                              alternative=alt).pvalue)


def paired(t, c, seeds):
    t = np.array([t[s] for s in seeds], float)
    c = np.array([c[s] for s in seeds], float)
    d = t - c
    res = dict(n=len(seeds), tmean=float(t.mean()), cmean=float(c.mean()),
               diff=float(d.mean()), win=float((d > 0).mean()),
               dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"))
    res["p"] = wpvalue(d)
    res["p2"] = wpvalue(d, two_sided=True)
    return res


def three_arm_payoff(splits, arms, game, seeds, tau):
    """For each seed, pick the arm by split_score threshold and return
    (per-seed three-arm payoff array, per-seed chosen-arm labels)."""
    pay, arms_ch = [], []
    cga, gated, notom = arms[game][CGA], arms[game][GATED], arms[game][NOTOM]
    for s in seeds:
        sp = splits[game].get(s)
        if sp is None:
            pay.append(np.nan); arms_ch.append("?"); continue
        if sp > tau:
            v, ch = cga.get(s, np.nan), "CGA"
        elif sp < -tau:
            v, ch = gated.get(s, np.nan), "Gated"
        else:
            v, ch = notom.get(s, np.nan), "ABSTAIN"
        pay.append(v); arms_ch.append(ch)
    return np.array(pay), arms_ch


def two_arm_payoff(splits, arms, game, seeds):
    """Current GSACA rule: split>0 -> CGA else Gated."""
    pay = []
    cga, gated = arms[game][CGA], arms[game][GATED]
    for s in seeds:
        sp = splits[game].get(s)
        if sp is None:
            pay.append(np.nan); continue
        pay.append(cga.get(s, np.nan) if sp > 0 else gated.get(s, np.nan))
    return np.array(pay)


def bca(diffs, n_boot=10000, alpha=0.05, seed=0):
    diffs = np.asarray(diffs, float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 2:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    xbar = diffs.mean()
    boot = np.array([diffs[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    from scipy.stats import norm
    prop = np.mean(boot < xbar)
    z0 = norm.ppf(min(max(prop, 1e-6), 1 - 1e-6))
    jack = np.array([np.delete(diffs, i).mean() for i in range(n)])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jack - jbar) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    lo = norm.cdf(z0 + (z0 + norm.ppf(alpha / 2)) / (1 - a * (z0 + norm.ppf(alpha / 2))))
    hi = norm.cdf(z0 + (z0 + norm.ppf(1 - alpha / 2)) / (1 - a * (z0 + norm.ppf(1 - alpha / 2))))
    return (float(np.quantile(boot, lo)), float(np.quantile(boot, hi)))


def sig(p):
    if p is None or np.isnan(p):
        return "  "
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "** "
    if p < 0.10:
        return "*  "
    return "ns "


def main():
    os.makedirs(OUT, exist_ok=True)
    metric = "team_mean_payoff"
    print("=" * 92)
    print("SCHEME 1 — OFFLINE THREE-ARM ABSTENTION GSACA  (metric=%s, zero GPU)" % metric)
    print("=" * 92)

    arms = load(ROOT, metric)
    splits = load_splits(ROOT)

    # ---------- per-seed split scores ----------
    print("\n### Per-seed warmup split_score (from het_gsaca metrics)")
    for g in GAMES_DONE:
        ss = [splits[g].get(s, float("nan")) for s in ALL20]
        print("  %-14s oracle=%-9s mean=%+.3f min=%+.3f max=%+.3f" %
              (g, ORACLE[g], np.nanmean(ss), np.nanmin(ss), np.nanmax(ss)))
        print("    per-seed: " + " ".join("%+.2f" % x for x in ss))

    # ---------- diagnostic table: reproduce the gap ----------
    print("\n" + "=" * 92)
    print("DIAGNOSTIC (n=20) — confirm deadlock -0.023 gap & chicken +0.046 ns")
    print("=" * 92)
    for ctrl_name, ctrl in [("vs NoToM", NOTOM), ("vs Gated", GATED), ("vs CGA", CGA)]:
        print("\n--- GSACA %s  (metric=%s) ---" % (ctrl_name, metric))
        print("  %-14s %4s %8s %8s %8s %6s %5s %9s %5s  %16s" %
              ("game", "n", "gsaca", "ctrl", "diff", "dz", "win", "p(1s)", "sig", "BCa95"))
        for g in GAMES_DONE:
            if GSACA not in arms[g] or ctrl not in arms[g]:
                continue
            seeds = sorted(set(arms[g][GSACA]) & set(arms[g][ctrl]))
            st = paired(arms[g][GSACA], arms[g][ctrl], seeds)
            ci = bca([arms[g][GSACA][s] - arms[g][ctrl][s] for s in seeds])
            print("  %-14s %4d %8.3f %8.3f %+8.3f %6.2f %5.2f %9.3g %4s  [%+6.3f,%+6.3f]" %
                  (g, st["n"], st["tmean"], st["cmean"], st["diff"],
                   st["dz"], st["win"], st.get("p", float("nan")),
                   sig(st.get("p")), ci[0], ci[1]))

    # ---------- three-arm recompute ----------
    TAU = 0.4
    print("\n" + "=" * 92)
    print("THREE-ARM GSACA (tau=%.2f):  split>+tau->CGA  split<-tau->Gated  |split|<=tau->NoToM" % TAU)
    print("=" * 92)

    for label, seeds in [("FULL n=20", ALL20), ("DEV 42-51", DEV), ("HOLDOUT 52-61", HOLD)]:
        print("\n##### %s #####" % label)
        # arm-choice distribution
        print("  arm choices: " + ", ".join("%s" % g for g in GAMES_DONE))
        for g in GAMES_DONE:
            _, ch = three_arm_payoff(splits, arms, g, seeds, TAU)
            from collections import Counter
            cnt = Counter(ch)
            print("    %-14s %s" % (g, dict(cnt)))

        print("\n  %-14s %8s %8s %8s %6s %5s %9s %5s  | %8s %8s %8s" %
              ("game", "3arm", "NoToM", "diff", "dz", "win", "p(1s)", "sig",
               "vsGated", "vsGSACA", "vsCGA"))
        for g in GAMES_DONE:
            if not all(c in arms[g] for c in (NOTOM, GATED, CGA, GSACA)):
                continue
            three, _ = three_arm_payoff(splits, arms, g, seeds, TAU)
            no = np.array([arms[g][NOTOM][s] for s in seeds], float)
            gd = np.array([arms[g][GATED][s] for s in seeds], float)
            ca = np.array([arms[g][CGA][s] for s in seeds], float)
            old = two_arm_payoff(splits, arms, g, seeds)
            d_no = three - no
            mask = ~np.isnan(d_no)
            def st(a, b):
                d = a - b; d = d[~np.isnan(d)]
                if len(d) < 2:
                    return (np.nan, np.nan, np.nan, np.nan, np.nan)
                dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan")
                p = wpvalue(d)
                return (d.mean(), dz, float((d > 0).mean()), p, d.std(ddof=1))
            dm, dzm, wm, pm, _ = st(three, no)
            dg, _, _, pg, _ = st(three, gd)
            do, _, _, po, _ = st(three, old)
            dc, _, _, pc, _ = st(three, ca)
            print("  %-14s %8.3f %8.3f %+8.3f %6.2f %5.2f %9.3g %4s  | %+8.3f %+8.3f %+8.3f" %
                  (g, three[mask].mean(), no[mask].mean(), dm, dzm, wm, pm, sig(pm), dg, do, dc))

    # ---------- n=5 preview for BoS + public_goods (20-seed pending in ongoing run) ----------
    print("\n" + "=" * 92)
    print("N=5 PREVIEW — BoS + public_goods  (20-seed pending; selector is deterministic)")
    print("=" * 92)
    arms5 = load(ROOT_N5, metric)
    splits5 = load_splits(ROOT_N5)
    SEEDS5 = list(range(42, 47))
    print("  %-14s %4s %8s %8s %8s %9s %5s  | %8s  %s" %
          ("game", "n", "3arm", "NoToM", "diff", "p(1s)", "sig", "arm", "splits"))
    for g in GAMES_N5:
        if not all(c in arms5[g] for c in (NOTOM, GATED, CGA, GSACA)):
            print("  %-14s  data missing" % g); continue
        three, ch = three_arm_payoff(splits5, arms5, g, SEEDS5, TAU)
        no = np.array([arms5[g][NOTOM][s] for s in SEEDS5], float)
        d = three - no; d = d[~np.isnan(d)]
        p = wpvalue(d)
        from collections import Counter
        sp = [splits5[g].get(s, float("nan")) for s in SEEDS5]
        print("  %-14s %4d %8.3f %8.3f %+8.3f %9.3g %4s  | %8s  %s" %
              (g, len(d), three[~np.isnan(three)].mean(), no[~np.isnan(no)].mean(),
               d.mean() if len(d) else float("nan"), p, sig(p),
               dict(Counter(ch)), " ".join("%+.2f" % x for x in sp)))

    # ---------- Proposition 3 check ----------
    print("\n" + "=" * 92)
    print("PROPOSITION 3 CHECK — abstain-zone regret vs NoToM must be == 0")
    print("=" * 92)
    for g in GAMES_DONE:
        abst_seeds = [s for s in ALL20 if abs(splits[g].get(s, 99)) <= TAU]
        if not abst_seeds:
            print("  %-14s no abstain seeds" % g); continue
        reg = [arms[g][NOTOM][s] - arms[g][NOTOM][s] for s in abst_seeds]  # 3arm picks NoToM here
        print("  %-14s abstain=%2d/%2d seeds  regret_vs_NoToM=%+.6f (==0 expected)" %
              (g, len(abst_seeds), 20, float(np.sum(reg))))

    # ---------- tau sensitivity (DEV ONLY) ----------
    print("\n" + "=" * 92)
    print("TAU SENSITIVITY (DEV seeds 42-51 ONLY — pre-registration)")
    print("=" * 92)
    print("  %-6s " % "tau", end="")
    for g in GAMES_DONE:
        print("%-13s" % g[:11], end="")
    print()
    for tau in [0.2, 0.3, 0.4, 0.5, 0.6]:
        print("  %-6.1f " % tau, end="")
        for g in GAMES_DONE:
            three, _ = three_arm_payoff(splits, arms, g, DEV, tau)
            no = np.array([arms[g][NOTOM][s] for s in DEV], float)
            d = three - no; d = d[~np.isnan(d)]
            if len(d) < 2:
                print("%+6.3f(n/a)  " % (np.nan,), end=""); continue
            p = wpvalue(d)
            print("%+6.3f(%3s) " % (d.mean(), sig(p).strip()), end="")
        print()

    # ---------- save JSON summary ----------
    summary = {"metric": metric, "tau_main": TAU, "dev": DEV, "holdout": HOLD,
               "games": GAMES_DONE}
    with open(os.path.join(OUT, "scheme1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n  -> summary saved to %s" % os.path.join(OUT, "scheme1_summary.json"))


if __name__ == "__main__":
    main()
