#!/usr/bin/env python3
"""Paired statistical analysis of Exp B (20-seed) results.

Compares the 4 heterogeneous cells with PROPER paired tests (Wilcoxon
signed-rank, same seed across cells) instead of independent t-tests, as
required by EXPERIMENT_PLAN_V2 (critique #2).

Cells:
  het_notom              = NoToM   (no alignment, baseline)
  het_gated_atom_talk    = Gated   (always-inject forced alignment)
  het_dp_gated_atom_talk = CGA     (Conditional Gated Arbitration)
  het_gsaca              = GSACA   (Game-Structure-Adaptive CGA)
"""
import json, os, glob, sys
from collections import defaultdict, OrderedDict
import numpy as np
from scipy.stats import wilcoxon

V2 = "/data/lab/results/v2/exp_b_20seed"
CELLS = OrderedDict([
    ("het_notom",              "NoToM"),
    ("het_gated_atom_talk",    "Gated"),
    ("het_gated_atom_talk",    "Gated"),
    ("het_dp_gated_atom_talk", "CGA"),
    ("het_gsaca",              "GSACA"),
])
# unique cell order (dedup preserving order)
_seen = set()
CELL_ORDER = []
CELL_LABEL = {}
for c, lab in CELLS.items():
    if c not in _seen:
        _seen.add(c); CELL_ORDER.append(c); CELL_LABEL[c] = lab

GAMES = ["battle_of_the_sexes", "public_goods"]

METRICS = ["cooperation_payoff", "perspective_diversity",
           "equilibrium_convergence", "tom_prediction_accuracy"]


def load_paired():
    """Return {game: {seed: {cell: metrics_dict}}}."""
    data = defaultdict(lambda: defaultdict(dict))
    for mpath in sorted(glob.glob(f"{V2}/*/*/*/metrics.json")):
        try:
            d = json.load(open(mpath))
        except Exception:
            continue
        parts = mpath.split("/")
        game = None; seed = None; cell = None
        for i, p in enumerate(parts):
            if p in ("battle_of_the_sexes", "public_goods"):
                game = p
            if p.startswith("seed_"):
                seed = int(p.replace("seed_", ""))
            if p in CELL_ORDER:
                cell = p
        if game is None or seed is None or cell is None:
            continue
        data[game][seed][cell] = d
    return data


def desc(vals):
    a = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))], float)
    if len(a) == 0:
        return None
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    rng = np.random.RandomState(42)
    if len(a) > 1:
        boots = np.array([float(np.mean(rng.choice(a, len(a)))) for _ in range(10000)])
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        lo = hi = mean
    return {"n": len(a), "mean": mean, "std": std, "ci": [lo, hi]}


def boot_diff_ci(x, y):
    """Percentile bootstrap 95% CI on mean(paired difference)."""
    x = np.array(x, float); y = np.array(y, float)
    d = x - y
    rng = np.random.RandomState(42)
    boots = np.array([float(np.mean(rng.choice(d, len(d)))) for _ in range(10000)])
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def paired_test(x, y, alt_label=""):
    """Wilcoxon signed-rank (paired) + Cohen's d (paired) + win rate + CI."""
    x = np.array(x, float); y = np.array(y, float)
    d = x - y
    n = len(d)
    if n == 0:
        return None
    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    cohen_d = md / sd if sd > 0 else (float("inf") if md > 0 else (-float("inf") if md < 0 else 0.0))
    win = int(np.sum(d > 0)); lose = int(np.sum(d < 0)); tie = int(np.sum(d == 0))
    winrate = win / n if n else 0.0
    ci = boot_diff_ci(x, y) if n > 1 else [md, md]
    # Wilcoxon
    nonzero = d[d != 0]
    p = None; W = None; method = "n/a"
    if len(nonzero) >= 1:
        try:
            if len(nonzero) <= 25:
                res = wilcoxon(x, y, zero_method="wilcox", correction=True,
                               method="exact", alternative="two-sided")
                method = "exact"
            else:
                res = wilcoxon(x, y, zero_method="wilcox", correction=True,
                               method="approx", alternative="two-sided")
                method = "approx"
            W = float(res.statistic); p = float(res.pvalue)
        except Exception as e:
            # all-zero differences or degenerate
            if len(nonzero) == 0:
                p = 1.0; W = 0.0; method = "degenerate(all-equal)"
            else:
                try:
                    res = wilcoxon(x, y, zero_method="wilcox", correction=True,
                                   method="approx", alternative="two-sided")
                    W = float(res.statistic); p = float(res.pvalue); method = "approx(fallback)"
                except Exception:
                    p = None; W = None; method = f"err:{e}"
    return {"n": n, "mean_diff": md, "cohen_d": cohen_d, "win": win,
            "lose": lose, "tie": tie, "winrate": winrate, "ci": ci,
            "W": W, "p": p, "method": method}


def sig(p):
    if p is None:
        return "  -"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "** "
    if p < 0.05:
        return "*  "
    if p < 0.10:
        return ".  "
    return "ns "


def fmt_desc(s, prec=3):
    if s is None:
        return "n/a"
    return f"{s['mean']:.{prec}f}±{s['std']:.{prec}f} [{s['ci'][0]:.{prec}f},{s['ci'][1]:.{prec}f}] (n={s['n']})"


def main():
    out = []
    def w(s=""):
        out.append(s); print(s)

    data = load_paired()

    w("# Exp B — 20-Seed Paired Statistical Analysis (Wilcoxon Signed-Rank)")
    w("")
    w(f"> Generated: {os.popen('date -u').read().strip()} UTC")
    w(f"> Data: `{V2}`")
    w("> Paired by seed across cells. Stats: Wilcoxon signed-rank (two-sided),")
    w("> Cohen's d (paired), 10k-resample percentile bootstrap 95% CI, paired win-rate.")
    w("> Cells: **NoToM**=het_notom (baseline), **Gated**=het_gated_atom_talk (forced alignment),")
    w("> **CGA**=het_dp_gated_atom_talk (conditional gated arbitration), **GSACA**=het_gsaca (structure-adaptive).")
    w("")

    # completeness check
    w("## 0. Data completeness")
    for g in GAMES:
        seeds = sorted(data.get(g, {}).keys())
        for c in CELL_ORDER:
            n = sum(1 for s in seeds if c in data[g][s])
            w(f"- {g} / {CELL_LABEL[c]}: {n}/{len(seeds)} seeds")
    w("")

    grand_summary = []

    for g in GAMES:
        w("---")
        w(f"## {g.upper().replace('_', ' ')}")
        seeds = sorted(data.get(g, {}).keys())
        if not seeds:
            w("_no data_"); continue

        # per-cell descriptives
        w("")
        w("### Descriptive statistics (cooperation payoff + secondary metrics)")
        hdr = f"| Cell | n | cooperation_payoff | perspective_diversity | equilibrium_conv | tom_pred_acc |"
        w(hdr); w("|" + "---|" * 6)
        cell_payoff = {}
        for c in CELL_ORDER:
            rows = [data[g][s][c] for s in seeds if c in data[g][s]]
            if not rows:
                w(f"| {CELL_LABEL[c]} | 0 | - | - | - | - |"); continue
            ps = desc([r.get("cooperation_payoff") for r in rows])
            ds = desc([r.get("perspective_diversity") for r in rows])
            cs = desc([r.get("equilibrium_convergence") for r in rows])
            ts = desc([r.get("tom_prediction_accuracy") for r in rows])
            cell_payoff[c] = [r.get("cooperation_payoff") for r in rows]
            w(f"| {CELL_LABEL[c]} | {ps['n']} | {fmt_desc(ps)} | {fmt_desc(ds)} | {fmt_desc(cs,3)} | {fmt_desc(ts,3)} |")
        w("")

        # GSACA detection
        gsaca_rows = [data[g][s]["het_gsaca"] for s in seeds if "het_gsaca" in data[g][s]]
        if gsaca_rows:
            det = [r.get("gsaca_detection_correct") for r in gsaca_rows]
            ndet = sum(1 for x in det if x is True)
            oracle = [r.get("gsaca_oracle_structure") for r in gsaca_rows]
            detected = [r.get("gsaca_detected_structure") for r in gsaca_rows]
            from collections import Counter
            w(f"- **GSACA structure detection**: {ndet}/{len(det)} = {100*ndet/max(1,len(det)):.1f}% correct")
            w(f"- Oracle structure (per seed): {Counter(oracle)}")
            w(f"- Detected structure: {Counter(detected)}")
            w("")

        # pairwise paired tests on cooperation_payoff
        w("### Pairwise paired comparisons (cooperation_payoff, Wilcoxon signed-rank)")
        w("")
        pairs = [
            ("het_gated_atom_talk",    "het_notom",              "Gated vs NoToM"),
            ("het_dp_gated_atom_talk", "het_notom",              "CGA vs NoToM"),
            ("het_dp_gated_atom_talk", "het_gated_atom_talk",    "CGA vs Gated"),
            ("het_gsaca",              "het_notom",              "GSACA vs NoToM"),
            ("het_gsaca",              "het_gated_atom_talk",    "GSACA vs Gated"),
            ("het_gsaca",              "het_dp_gated_atom_talk", "GSACA vs CGA"),
        ]
        w("| Comparison | n | mean Δ | 95% CI | Cohen's d | win/lose/tie | W | p | sig |")
        w("|" + "---|" * 9)
        for ca, cb, label in pairs:
            if ca not in cell_payoff or cb not in cell_payoff:
                continue
            # align by seed
            xs, ys = [], []
            for s in seeds:
                if ca in data[g][s] and cb in data[g][s]:
                    xs.append(data[g][s][ca].get("cooperation_payoff"))
                    ys.append(data[g][s][cb].get("cooperation_payoff"))
            r = paired_test(xs, ys)
            if r is None:
                continue
            ci = f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}]"
            cd = f"{r['cohen_d']:.3f}" if abs(r['cohen_d']) != float('inf') else ("+inf" if r['cohen_d']>0 else "-inf")
            W = f"{r['W']:.1f}" if r['W'] is not None else "-"
            p = f"{r['p']:.4f}" if r['p'] is not None else "-"
            w(f"| {label} | {r['n']} | {r['mean_diff']:+.3f} | {ci} | {cd} | {r['win']}/{r['lose']}/{r['tie']} | {W} | {p} | {sig(r['p'])} |")
            grand_summary.append((g, label, r))
        w("")

        # GSACA gate / DP metrics (mechanism diagnostics)
        w("### Mechanism diagnostics (GSACA cell, mean across seeds)")
        w("")
        if gsaca_rows:
            for k in ["gate_trust_rate", "gated_prediction_accuracy", "signal_accuracy",
                      "dp_conflict_rate", "dp_intervention_rate", "gsaca_split_score"]:
                vals = [r.get(k) for r in gsaca_rows if r.get(k) is not None]
                s = desc(vals)
                if s:
                    w(f"- {k}: {fmt_desc(s, 4)}")
        w("")

    # ---- cross-game summary ----
    w("---")
    w("## Cross-game summary")
    w("")
    w("### GSACA vs baselines — does structure-adaptive alignment win? (Δ = GSACA − baseline, payoff)")
    w("")
    w("| Game | vs NoToM Δ (p) | vs Gated Δ (p) | vs CGA Δ (p) |")
    w("|" + "---|" * 4)
    by_game = defaultdict(dict)
    for g, label, r in grand_summary:
        by_game[g][label] = r
    for g in GAMES:
        row = []
        for key in ["GSACA vs NoToM", "GSACA vs Gated", "GSACA vs CGA"]:
            r = by_game.get(g, {}).get(key)
            if r:
                row.append(f"{r['mean_diff']:+.3f} ({sig(r['p']).strip()})")
            else:
                row.append("-")
        w(f"| {g} | {row[0]} | {row[1]} | {row[2]} |")
    w("")

    # interpretation
    w("### Interpretation key")
    w("- These two games are **coordination** games (symmetric/coordination NE).")
    w("- Expected pattern: forced alignment (Gated) helps coordination; CGA (conditional)")
    w("  should not hurt; **GSACA** detects 'coord' structure and selects the Gated arm,")
    w("  recovering Gated-level payoff while keeping the safety of abstaining when CGA would harm.")
    w("- sig codes: `***` p<0.001, `**` p<0.01, `*` p<0.05, `.` p<0.10, `ns` otherwise.")
    w("")

    # write report file
    rep = "/data/lab/gsaca/ANALYSIS_EXP_B_PAIRED.md"
    with open(rep, "w") as f:
        f.write("\n".join(out))
    print(f"\n[REPORT WRITTEN] {rep}", file=sys.stderr)


if __name__ == "__main__":
    main()
