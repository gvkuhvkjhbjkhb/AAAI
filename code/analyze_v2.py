#!/usr/bin/env python3
"""V2 analysis — paired statistics replacing the n=5 MWU.

For a given results dir laid out as <game>/seed_<S>/<cell>/metrics.json, computes
per-game paired comparisons of a treatment cell vs a control cell across seeds:
  - Wilcoxon signed-rank (scipy) with zero_method='wilcox', correction=True
  - BCa 95% CI on the mean per-seed difference
  - Cohen's d (paired, dz = mean_diff / sd_diff)
  - paired win rate (treatment > control)
  - mean +/- std per cell, mean diff

Usage:
  python3 analyze_v2.py --dir /data/lab/results/v2/exp_b_20seed \
      --treatment het_dp_gated_atom_talk --control het_notom
  python3 analyze_v2.py --dir /data/lab/results/v2/exp_a_pairs \
      --treatment het_gsaca --control hom_gsaca --metric team_mean_payoff
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np


def load_cell_values(root, metric="team_mean_payoff"):
    """-> {game: {cell: {seed: value}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    for mpath in glob.glob(os.path.join(root, "**", "metrics.json"), recursive=True):
        parts = mpath.replace(root, "").strip("/").split(os.sep)
        if len(parts) < 4 or parts[1] != "seed_"[:5]:
            pass
        try:
            game = parts[0]
            seed = int(parts[1].replace("seed_", ""))
            cell = parts[2]
        except (IndexError, ValueError):
            continue
        with open(mpath) as f:
            m = json.load(f)
        if metric in m:
            out[game][cell][seed] = float(m[metric])
    return out


def bca_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    """Bias-corrected accelerated bootstrap CI on the mean of diffs."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n < 2:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    xbar = np.mean(diffs)
    boot = np.array([np.mean(diffs[rng.randint(0, n, n)]) for _ in range(n_boot)])
    # bias-correction z0
    prop = np.mean(boot < xbar)
    from scipy.stats import norm
    z0 = norm.ppf(min(max(prop, 1e-6), 1 - 1e-6))
    # acceleration via jackknife
    jack = np.array([np.mean(np.delete(diffs, i)) for i in range(n)])
    jbar = np.mean(jack)
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jack - jbar) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    lo_p = norm.cdf(z0 + (z0 + norm.ppf(alpha / 2)) / (1 - a * (z0 + norm.ppf(alpha / 2))))
    hi_p = norm.cdf(z0 + (z0 + norm.ppf(1 - alpha / 2)) / (1 - a * (z0 + norm.ppf(1 - alpha / 2))))
    return (float(np.quantile(boot, lo_p)), float(np.quantile(boot, hi_p)))


def paired_stats(t_vals, c_vals):
    """t_vals, c_vals: dicts seed->value. Returns dict of paired stats."""
    seeds = sorted(set(t_vals) & set(c_vals))
    if len(seeds) < 2:
        return {"n": len(seeds), "note": "insufficient paired seeds"}
    t = np.array([t_vals[s] for s in seeds])
    c = np.array([c_vals[s] for s in seeds])
    d = t - c
    res = {
        "n": len(seeds), "seeds": seeds,
        "treatment_mean": float(np.mean(t)), "treatment_std": float(np.std(t, ddof=1)),
        "control_mean": float(np.mean(c)), "control_std": float(np.std(c, ddof=1)),
        "mean_diff": float(np.mean(d)),
        "win_rate": float(np.mean(d > 0)),
        "tie_rate": float(np.mean(d == 0)),
        "cohens_dz": float(np.mean(d) / np.std(d, ddof=1)) if np.std(d, ddof=1) > 0 else float("nan"),
        "bca95_ci": bca_ci(d),
    }
    # Wilcoxon signed-rank
    try:
        from scipy.stats import wilcoxon
        nonpos = np.sum(d <= 0)
        if nonpos == 0 or nonpos == len(d):
            res["wilcoxon_p"] = 1.0
            res["wilcoxon_note"] = "all diffs same sign"
        else:
            w = wilcoxon(d, zero_method="wilcox", correction=True,
                         alternative="greater" if np.mean(d) > 0 else "less")
            res["wilcoxon_stat"] = float(w.statistic)
            res["wilcoxon_p"] = float(w.pvalue)
            # also two-sided
            w2 = wilcoxon(d, zero_method="wilcox", correction=True, alternative="two-sided")
            res["wilcoxon_p_twosided"] = float(w2.pvalue)
    except Exception as e:
        res["wilcoxon_error"] = str(e)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--metric", default="team_mean_payoff")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = load_cell_values(args.dir, args.metric)
    results = {}
    print(f"\n=== {args.treatment} vs {args.control}  (metric={args.metric}) ===")
    print(f"    dir: {args.dir}")
    print(f"    {'game':<20} {'n':>3} {'treat':>8} {'ctrl':>8} {'diff':>8} "
          f"{'d_z':>6} {'win':>5} {'wilcoxon_p':>11} {'bca95':>18}")
    for game in sorted(data):
        if args.treatment not in data[game] or args.control not in data[game]:
            continue
        st = paired_stats(data[game][args.treatment], data[game][args.control])
        st["treatment_cell"] = args.treatment
        st["control_cell"] = args.control
        st["metric"] = args.metric
        results[game] = st
        if "wilcoxon_p" in st:
            ci = st.get("bca95_ci", (np.nan, np.nan))
            print(f"    {game:<20} {st['n']:>3} {st['treatment_mean']:>8.3f} "
                  f"{st['control_mean']:>8.3f} {st['mean_diff']:>+8.3f} "
                  f"{st['cohens_dz']:>6.2f} {st['win_rate']:>5.2f} "
                  f"{st['wilcoxon_p']:>11.2e} [{ci[0]:+.3f},{ci[1]:+.3f}]")
        else:
            print(f"    {game:<20} {st}")
    out = args.out or os.path.join(args.dir, "paired_stats_%s_vs_%s.json" %
                                   (args.treatment, args.control))
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
