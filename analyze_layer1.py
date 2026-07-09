#!/usr/bin/env python3
"""
HetToM Layer-1 analysis: aggregate multi-seed results, run statistical tests,
and produce the conclusion table that decides whether heterogeneity + ToM
breaks the Reasoning Trap.

Reads:  results/hettom_layer1/seed_<n>/<cell>/metrics.json  (or direct cell dirs)
        results/hettom_layer1/summary_all_seeds.csv
Writes: results/hettom_layer1/analysis_report.txt
        results/hettom_layer1/aggregated.csv

Statistical tests (all non-parametric, robust to small n):
  - Mann-Whitney U  : het_tom vs hom_notom (does the full method beat baseline?)
  - paired Wilcoxon : het_tom vs hom_tom   (does heterogeneity add over ToM?)
                     het_tom vs het_notom  (does ToM add over heterogeneity?)
  - bootstrap CI    : on every cell's mean, across seeds
  - effect size     : rank-biserial r for each comparison

Decision rules (pre-registered):
  Trap broken (positive result) if:
    het_tom > hom_notom on BOTH perspective_diversity AND cooperation_payoff,
    with Mann-Whitney p < 0.05 on payoff and CI excluding 0.
  Trap persists (negative result) otherwise — still publishable as a
  high-value negative result (per Plan_Assessment.txt).
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


CELLS = ["hom_notom", "hom_tom", "het_notom", "het_tom"]
METRICS = ["perspective_diversity", "cooperation_payoff",
           "equilibrium_convergence", "tom_prediction_accuracy"]


def load_metrics(results_dir):
    """Load all metrics.json files. Returns dict cell -> list of per-seed dicts."""
    by_cell = defaultdict(list)
    # pattern 1: seed_<n>/<cell>/metrics.json
    for path in sorted(glob.glob(os.path.join(
            results_dir, "seed_*", "*", "metrics.json"))):
        cell = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            by_cell[cell].append(json.load(f))
    # pattern 2: <cell>/metrics.json  (single-seed direct)
    for path in sorted(glob.glob(os.path.join(
            results_dir, "*", "metrics.json"))):
        if "seed_" in path:
            continue
        cell = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            d = json.load(f)
            if d not in by_cell[cell]:
                by_cell[cell].append(d)
    return by_cell


def mann_whitney_u(x, y):
    """Mann-Whitney U two-sided test. Returns (U, p). Uses scipy if available,
    else a normal approximation fallback."""
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(x, y, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        # normal approximation
        nx, ny = len(x), len(y)
        combined = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
        ranks = np.zeros(nx + ny)
        i = 0
        while i < len(combined):
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[k] = avg_rank
            i = j
        rx = ranks[:nx]
        ux = rx.sum() - nx * (nx + 1) / 2.0
        mu = nx * ny / 2.0
        sigma = np.sqrt(nx * ny * (nx + ny + 1) / 12.0)
        z = (ux - mu) / sigma if sigma > 0 else 0.0
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        return float(ux), float(min(max(p, 0.0), 1.0))


def wilcoxon_signed(x, y):
    """Paired Wilcoxon signed-rank test. Returns (W, p)."""
    d = [a - b for a, b in zip(x, y) if a != b]
    if len(d) < 1:
        return float("nan"), 1.0
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(x, y, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), 1.0


def rank_biserial(x, y):
    """Rank-biserial effect size r for two groups (Mann-Whitney based).
    r in [-1, 1]; positive => x tends higher than y."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    ux, _ = mann_whitney_u(x, y)
    r = (2 * ux / (nx * ny)) - 1.0
    return float(r)


def bootstrap_ci_mean(values, n_boot=5000, ci=0.95, seed=0):
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return lo, hi


def aggregate(by_cell, metric):
    """Return dict cell -> (mean, std, n, ci_lo, ci_hi, values_list)."""
    out = {}
    for cell in CELLS:
        vals = []
        for m in by_cell.get(cell, []):
            v = m.get(metric, float("nan"))
            if v == v:  # not nan
                vals.append(float(v))
        if vals:
            arr = np.asarray(vals)
            lo, hi = bootstrap_ci_mean(vals)
            out[cell] = (float(arr.mean()), float(arr.std(ddof=1)) if len(vals) > 1 else 0.0,
                         len(vals), lo, hi, vals)
        else:
            out[cell] = (float("nan"), float("nan"), 0, float("nan"), float("nan"), [])
    return out


def fmt(x, w=8, p=3):
    if isinstance(x, float) and x != x:
        return "nan".rjust(w)
    return f"{x:.{p}f}".rjust(w)


def analyze(results_dir, out_dir):
    by_cell = load_metrics(results_dir)
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    lines.append("=" * 78)
    lines.append("HetToM Layer-1 Analysis Report")
    lines.append("=" * 78)
    lines.append(f"Results dir: {results_dir}")
    n_seeds = max((len(v) for v in by_cell.values()), default=0)
    lines.append(f"Seeds found: {n_seeds}")
    lines.append("")

    # ---- per-metric aggregate table ----
    for metric in METRICS:
        agg = aggregate(by_cell, metric)
        lines.append(f"--- {metric} (mean +/- std, 95% CI, n) ---")
        lines.append(f"{'cell':<12}{'mean':>9}{'std':>9}{'ci_lo':>9}"
                     f"{'ci_hi':>9}{'n':>5}")
        for cell in CELLS:
            mean, std, n, lo, hi, _ = agg[cell]
            lines.append(f"{cell:<12}{fmt(mean)}{fmt(std)}{fmt(lo)}{fmt(hi)}{n:>5}")
        lines.append("")

    # ---- key comparisons ----
    lines.append("--- Key statistical comparisons ---")
    comparisons = [
        ("het_tom", "hom_notom", "Full method vs baseline (trap broken?)"),
        ("het_tom", "hom_tom",   "Heterogeneity adds over ToM?"),
        ("het_tom", "het_notom", "ToM adds over heterogeneity?"),
        ("hom_tom", "hom_notom", "ToM alone vs baseline (key 2)"),
        ("het_notom", "hom_notom","Heterogeneity alone vs baseline (key 1)"),
    ]
    payoff_agg = aggregate(by_cell, "cooperation_payoff")
    div_agg = aggregate(by_cell, "perspective_diversity")
    for a, b, label in comparisons:
        pa, pb = payoff_agg[a][5], payoff_agg[b][5]
        da, db = div_agg[a][5], div_agg[b][5]
        if len(pa) >= 3 and len(pb) >= 3:
            u, p = mann_whitneyu(pa, pb) if _scipy_ok() else mann_whitney_u(pa, pb)
            r = rank_biserial(pa, pb)
        else:
            u, p, r = float("nan"), float("nan"), float("nan")
        delta = (np.mean(pa) - np.mean(pb)) if pa and pb else float("nan")
        lines.append(f"[{label}]")
        lines.append(f"  payoff  {a} - {b}: delta={fmt(delta)} "
                     f"U={fmt(u, 8, 1)} p={fmt(p)} r={fmt(r)}")
        if len(da) >= 3 and len(db) >= 3:
            _, pd = mann_whitneyu(da, db) if _scipy_ok() else mann_whitney_u(da, db)
        else:
            pd = float("nan")
        dd = (np.mean(da) - np.mean(db)) if da and db else float("nan")
        lines.append(f"  diversity {a} - {b}: delta={fmt(dd)} p={fmt(pd)}")
        lines.append("")

    # ---- paired tests (same-seed) ----
    lines.append("--- Paired comparisons (Wilcoxon, same-seed pairing) ---")
    # build per-seed payoff dicts
    seed_payoff = defaultdict(dict)
    for cell in CELLS:
        for m in by_cell.get(cell, []):
            s = m.get("config", {}).get("seed", m.get("seed"))
            if s is not None:
                seed_payoff[cell][s] = m.get("cooperation_payoff", float("nan"))
    for a, b, _label in comparisons:
        common = sorted(set(seed_payoff[a]) & set(seed_payoff[b]))
        if len(common) >= 3:
            xa = [seed_payoff[a][s] for s in common]
            xb = [seed_payoff[b][s] for s in common]
            w, p = wilcoxon_signed(xa, xb)
            lines.append(f"  payoff {a} vs {b} (n={len(common)} paired): "
                         f"W={fmt(w,8,1)} p={fmt(p)}")
        else:
            lines.append(f"  payoff {a} vs {b}: insufficient paired seeds "
                         f"({len(common)})")
    lines.append("")

    # ---- decision ----
    lines.append("=" * 78)
    lines.append("DECISION (pre-registered)")
    lines.append("=" * 78)
    pa = payoff_agg["het_tom"][5]
    pb = payoff_agg["hom_notom"][5]
    da = div_agg["het_tom"][5]
    db = div_agg["hom_notom"][5]
    trap_broken = False
    reasons = []
    if len(pa) >= 3 and len(pb) >= 3:
        _, p_pay = mann_whitneyu(pa, pb) if _scipy_ok() else mann_whitney_u(pa, pb)
        lo, hi = bootstrap_ci_mean([a - b for a, b in zip(pa, pb)] if len(pa) == len(pb) else
                                   (np.mean(pa) - np.mean(pb),))
        delta_div = (np.mean(da) - np.mean(db)) if da and db else float("nan")
        delta_pay = np.mean(pa) - np.mean(pb) if pa and pb else float("nan")
        cond_div = delta_div > 0
        cond_pay = (delta_pay > 0) and (p_pay < 0.05)
        trap_broken = cond_div and cond_pay
        reasons.append(f"diversity delta={fmt(delta_div)} (>0? {cond_div})")
        reasons.append(f"payoff delta={fmt(delta_pay)} p={fmt(p_pay)} "
                       f"(<0.05 & >0? {cond_pay})")
    else:
        reasons.append(f"insufficient seeds (het_tom n={len(pa)}, "
                       f"hom_notom n={len(pb)}); need >=3")

    if trap_broken:
        lines.append("POSITIVE: Reasoning Trap broken by heterogeneity + ToM.")
        lines.append("  -> Write as positive method paper (AAAI 40-50%).")
    else:
        lines.append("NEGATIVE/PARTIAL: Trap NOT fully broken.")
        lines.append("  -> If partial (one key works): method+analysis paper "
                     "(30-40%).")
        lines.append("  -> If fully negative: high-value negative result + "
                     "theory (25-35%).")
    for r in reasons:
        lines.append(f"  - {r}")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "analysis_report.txt"), "w") as f:
        f.write(report)

    # write aggregated CSV
    with open(os.path.join(out_dir, "aggregated.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "cell", "mean", "std", "n", "ci_lo", "ci_hi"])
        for metric in METRICS:
            agg = aggregate(by_cell, metric)
            for cell in CELLS:
                mean, std, n, lo, hi, _ = agg[cell]
                w.writerow([metric, cell, mean, std, n, lo, hi])
    print(f"\nReport -> {os.path.join(out_dir, 'analysis_report.txt')}")
    print(f"Aggregated -> {os.path.join(out_dir, 'aggregated.csv')}")


def _scipy_ok():
    try:
        import scipy  # noqa
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Analyze HetToM Layer-1 multi-seed results")
    ap.add_argument("--results_dir", type=str,
                    default="results/hettom_layer1")
    ap.add_argument("--out_dir", type=str, default="results/hettom_layer1")
    args = ap.parse_args()
    analyze(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
