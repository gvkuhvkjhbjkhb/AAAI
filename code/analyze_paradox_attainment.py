#!/usr/bin/env python3
"""Analysis for the independent same-batch 3-arm paradox / attainment study.

Design (pre-registered):
  6 games x 20 seeds x 3 arms (NoToM, Gated, CGA), all cells for a given
  (game, seed) produced back-to-back in ONE vLLM batch -> same-seed pairing
  is also same-batch. Arms:
    het_notom              -> NoToM  (independent baseline / abstain control)
    het_gated_atom_talk    -> Gated  (forced alignment)
    het_dp_gated_atom_talk -> CGA    (mild / diversity-preserving alignment)

Three paired contrasts per game (same seed, n<=20):
    Paradox            : CGA  - Gated
    Baseline attainment: Gated - NoToM
    Mild-intervention  : CGA  - NoToM

For each contrast/game we report: mean difference, paired Wilcoxon p,
Cohen's d_z, paired win rate, and a 95% bootstrap CI of the mean difference.
The six *primary* comparisons (CGA-Gated across the six games) receive a
Holm-Bonferroni correction. Effect direction and CIs are emphasized over any
single significance flag, per the pre-registration.

Outputs (into --root):
  paradox_attainment_summary.csv
  paradox_attainment_summary.json
  paradox_attainment_table.tex
  paradox_attainment_audit.log

Zero deps beyond numpy + scipy. Offline.

Usage:
  python3 code/analyze_paradox_attainment.py --root v2_results/exp_vllm_paradox_attainment_v1
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy import stats

CELL = {"het_notom": "NoToM",
        "het_gated_atom_talk": "Gated",
        "het_dp_gated_atom_talk": "CGA"}
GROUP = {"chicken": "anti", "hawk_dove": "anti", "deadlock": "anti",
         "stag_hunt": "coord", "battle_of_the_sexes": "coord",
         "public_goods": "boundary"}
ORDER = ["chicken", "deadlock", "hawk_dove", "stag_hunt",
         "battle_of_the_sexes", "public_goods"]
# pre-registered directional expectations for the paradox (CGA - Gated)
EXPECT_PARADOX = {"chicken": ">0", "deadlock": "<=0(indep-baseline test)",
                  "hawk_dove": "<=0(indep-baseline test)",
                  "stag_hunt": "<0", "battle_of_the_sexes": "<0",
                  "public_goods": "exploratory"}

METRIC = "cooperation_payoff"


def load(root):
    """root/<game>/seed_<n>/<cell>/metrics.json -> data[game][arm][seed]=payoff."""
    data = defaultdict(lambda: defaultdict(dict))
    files = glob.glob(f"{root}/*/seed_*/*/metrics.json")
    for f in files:
        p = f.split("/")
        cd, seed, game = p[-2], p[-3], p[-4]
        if cd not in CELL:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        v = d.get(METRIC)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        data[game][CELL[cd]][int(seed.replace("seed_", ""))] = float(v)
    return data, len(files)


def boot_ci(diff, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def paired(a_map, b_map, boot_seed=0):
    """a - b paired on shared seeds."""
    s = sorted(set(a_map) & set(b_map))
    if len(s) < 3:
        return None
    a = np.array([a_map[x] for x in s], dtype=float)
    b = np.array([b_map[x] for x in s], dtype=float)
    diff = a - b
    sd = diff.std(ddof=1)
    dz = float(diff.mean() / sd) if sd > 0 else 0.0
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        p = float("nan")
    lo, hi = boot_ci(diff, seed=boot_seed)
    return {"mean_diff": float(diff.mean()), "n": len(s), "wilcoxon_p": p,
            "cohen_dz": dz, "win_rate": float((diff > 0).mean()),
            "ci_lo": lo, "ci_hi": hi, "mean_a": float(a.mean()),
            "mean_b": float(b.mean()), "seeds": s}


def holm(pvals):
    """Holm-Bonferroni. pvals: list of (key, p). Returns dict key->adj_p."""
    items = [(k, p) for k, p in pvals if p == p]  # drop nan
    items.sort(key=lambda kp: kp[1])
    m = len(items)
    adj = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        a = max(a, prev)  # enforce monotonicity
        adj[k] = a
        prev = a
    for k, p in pvals:
        if p != p:
            adj[k] = float("nan")
    return adj


def sig(p):
    if p != p:
        return "?"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    data, nfiles = load(args.root)

    log_lines = []
    def emit(s=""):
        print(s)
        log_lines.append(s)

    emit("=" * 108)
    emit(f"INDEPENDENT SAME-BATCH 3-ARM PARADOX / ATTAINMENT STUDY")
    emit(f"root={args.root}   metrics.json found={nfiles}   generated={datetime.now().isoformat()}")
    emit(f"metric={METRIC}   arms: NoToM / Gated / CGA")
    emit("=" * 108)

    contrasts = [("paradox", "CGA", "Gated"),
                 ("attainment", "Gated", "NoToM"),
                 ("mild_cost", "CGA", "NoToM")]

    results = {}   # results[contrast][game] = paired dict
    for cname, hi, lo in contrasts:
        results[cname] = {}
        for gi, g in enumerate(ORDER):
            r = paired(data.get(g, {}).get(hi, {}),
                       data.get(g, {}).get(lo, {}), boot_seed=1000 + gi)
            if r:
                results[cname][g] = r

    # Holm across the 6 primary CGA-Gated comparisons
    prim = [(g, results["paradox"][g]["wilcoxon_p"])
            for g in ORDER if g in results["paradox"]]
    holm_adj = holm(prim)

    # ---- per-contrast tables ----
    for cname, hi, lo in contrasts:
        emit("")
        emit("-" * 108)
        title = {"paradox": "PARADOX  (CGA - Gated)",
                 "attainment": "BASELINE ATTAINMENT  (Gated - NoToM)",
                 "mild_cost": "MILD-INTERVENTION COST  (CGA - NoToM)"}[cname]
        emit(f"{title}")
        emit("-" * 108)
        hdr = (f"  {'game':20s} {'grp':9s} {hi+'':>7s} {lo+'':>7s} {'delta':>8s} "
               f"{'95% CI':>18s} {'p':>8s} {'sig':>4s}")
        if cname == "paradox":
            hdr += f" {'p_holm':>8s} {'sigH':>5s}"
        hdr += f" {'dz':>7s} {'win':>5s} {'n':>3s}  expect"
        emit(hdr)
        for g in ORDER:
            if g not in results[cname]:
                emit(f"  {g:20s} -- missing --")
                continue
            r = results[cname][g]
            ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
            line = (f"  {g:20s} {GROUP[g]:9s} {r['mean_a']:7.3f} {r['mean_b']:7.3f} "
                    f"{r['mean_diff']:+8.3f} {ci:>18s} {r['wilcoxon_p']:8.4f} "
                    f"{sig(r['wilcoxon_p']):>4s}")
            if cname == "paradox":
                ph = holm_adj.get(g, float("nan"))
                line += f" {ph:8.4f} {sig(ph):>5s}"
            line += f" {r['cohen_dz']:+7.2f} {r['win_rate']:5.0%} {r['n']:3d}"
            if cname == "paradox":
                line += f"  {EXPECT_PARADOX.get(g,'')}"
            emit(line)

    # ---- headline ----
    emit("")
    emit("=" * 108)
    p = results["paradox"]
    pos = [g for g in p if p[g]["mean_diff"] > 0]
    neg = [g for g in p if p[g]["mean_diff"] < 0]
    sig_pos = [g for g in pos if holm_adj.get(g, 1) < 0.05]
    sig_neg = [g for g in neg if holm_adj.get(g, 1) < 0.05]
    emit(f"PARADOX headline (CGA-Gated): {len(pos)}/6 games CGA>Gated "
         f"({len(sig_pos)} Holm-sig), {len(neg)}/6 CGA<Gated ({len(sig_neg)} Holm-sig).")
    att = results["attainment"]
    att_pos = [g for g in att if att[g]["mean_diff"] > 0]
    emit(f"ATTAINMENT headline (Gated-NoToM): Gated>=NoToM in {len(att_pos)}/6 games "
         f"(tests whether alignment gain is baseline-attainment-driven).")
    emit("Emphasis: effect DIRECTION + CI over single-test significance (pre-registered).")
    emit("=" * 108)

    # ---- write CSV / JSON / TeX ----
    os.makedirs(args.root, exist_ok=True)
    csv_path = os.path.join(args.root, "paradox_attainment_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "game", "group", "mean_hi", "mean_lo",
                    "mean_diff", "ci_lo", "ci_hi", "wilcoxon_p", "holm_p",
                    "cohen_dz", "win_rate", "n"])
        for cname, _, _ in contrasts:
            for g in ORDER:
                if g not in results[cname]:
                    continue
                r = results[cname][g]
                ph = holm_adj.get(g, "") if cname == "paradox" else ""
                w.writerow([cname, g, GROUP[g], f"{r['mean_a']:.4f}",
                            f"{r['mean_b']:.4f}", f"{r['mean_diff']:.4f}",
                            f"{r['ci_lo']:.4f}", f"{r['ci_hi']:.4f}",
                            f"{r['wilcoxon_p']:.4f}",
                            (f"{ph:.4f}" if ph != "" else ""),
                            f"{r['cohen_dz']:.3f}", f"{r['win_rate']:.3f}", r["n"]])
    emit(f"\n[wrote] {csv_path}")

    json_path = os.path.join(args.root, "paradox_attainment_summary.json")
    out = {"root": args.root, "metric": METRIC, "n_files": nfiles,
           "holm_primary": holm_adj, "results": results,
           "generated": datetime.now().isoformat()}
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    emit(f"[wrote] {json_path}")

    tex_path = os.path.join(args.root, "paradox_attainment_table.tex")
    with open(tex_path, "w") as f:
        f.write("% same-batch 3-arm paradox/attainment table (auto-generated)\n")
        f.write("\\begin{table}[tb]\n\\centering\n\\small\n")
        f.write("\\caption{Same-batch 3-arm study (bf16/vLLM). Paired differences "
                "in cooperation payoff over $n{\\le}20$ shared seeds, with paired "
                "Wilcoxon $p$ (Holm-corrected across the six paradox tests), "
                "Cohen's $d_z$, paired win rate, and 95\\% bootstrap CI.}\n")
        f.write("\\label{tab:paradox_attainment}\n")
        f.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
        f.write("Game & Grp & $\\Delta_{\\text{CGA-Gated}}$ & 95\\% CI & $p_{\\text{Holm}}$ & $d_z$ & win \\\\\n")
        f.write("\\midrule\n")
        for g in ORDER:
            if g not in results["paradox"]:
                continue
            r = results["paradox"][g]
            ph = holm_adj.get(g, float("nan"))
            f.write(f"{g.replace('_',' ')} & {GROUP[g]} & {r['mean_diff']:+.3f} & "
                    f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] & {ph:.3f} & "
                    f"{r['cohen_dz']:+.2f} & {r['win_rate']:.0%} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    emit(f"[wrote] {tex_path}")

    audit_path = os.path.join(args.root, "paradox_attainment_audit.log")
    with open(audit_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"[wrote] {audit_path}")


if __name__ == "__main__":
    main()
