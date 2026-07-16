#!/usr/bin/env python3
"""Homogeneity / mediation analysis for reviewer W1 supplement.

Reads exp_a_fix (pair-partitioned) results laid out as
    <root>/<PAIR>/<game>/seed_<S>/<cell>/metrics.json
with PAIR in {QQ, GG, QL, QG}, cells in
    {*_notom (No-align), *_gated_atom_talk (Gated),
     *_dp_gated_atom_talk (CGA), *_gsaca (GSACA)}
(prefix hom_ for homogeneous QQ/GG, het_ for heterogeneous QL/QG).

Produces, per pair:
  1. No-align behavioral diversity per game (perspective_diversity of the
     *_notom cell), mean/min/max over seeds  -> "diversity" side of the chain.
  2. The alignment-paradox effect = CGA (dp) - Gated (gated) on
     cooperation_payoff, per game, paired over seeds with Wilcoxon + dz.
     Positive on anti-coordination games = the paradox (forced alignment via
     the diversity-preserving gate raises anti-coordination payoff relative to
     plain gating). We report both anti-coordination and coordination games.
  3. The mediation scatter table: x = per-(pair,game) No-align diversity,
     y = per-(pair,game) paradox effect, over anti-coordination games for all
     four pairs. Reports Spearman & Pearson correlation across the points.

With n=5 we make DIRECTION + EFFECT-SIZE statements only; significance is
deferred to the n=20 primary heterogeneous pair (per the analysis protocol).

Outputs a JSON summary and (if matplotlib available) a scatter PDF.

Usage:
  python3 code/analyze_homogeneity_mediation.py \
      --root v2_results/exp_a_fix \
      --out_json v2_results/exp_a_fix/homogeneity_mediation.json \
      --out_fig figures/mediation_scatter.pdf
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

# cell suffix -> arm label (prefix hom_/het_ stripped before lookup)
ARM = {
    "notom": "NoAlign",
    "gated_atom_talk": "Gated",
    "dp_gated_atom_talk": "CGA",
    "gsaca": "GSACA",
}
ANTI = ["chicken", "deadlock", "hawk_dove"]
COORD = ["stag_hunt", "battle_of_the_sexes"]
BOUNDARY = ["public_goods"]
GAME_ORDER = ANTI + COORD + BOUNDARY
PAIR_ORDER = ["QG", "QL", "QQ", "GG"]  # het (diverse) -> hom (homogeneous)


def arm_of(cell):
    for pref in ("hom_", "het_"):
        if cell.startswith(pref):
            return ARM.get(cell[len(pref):])
    return ARM.get(cell)


def load(root):
    """-> data[pair][game][arm][seed] = {payoff, diversity}."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for f in glob.glob(f"{root}/*/*/seed_*/*/metrics.json"):
        p = f.split("/")
        pair, game, seed_s, cell = p[-5], p[-4], p[-3], p[-2]
        arm = arm_of(cell)
        if arm is None:
            continue
        try:
            seed = int(seed_s.replace("seed_", ""))
            d = json.load(open(f))
        except Exception:
            continue
        cp = d.get("cooperation_payoff")
        dv = d.get("perspective_diversity")
        if cp is None:
            continue
        data[pair][game][arm][seed] = {
            "payoff": float(cp),
            "diversity": float(dv) if dv is not None else float("nan"),
        }
    return data


def paired(a, b):
    """a,b: dict seed->value. Return (mean_diff,n,p,dz,win) or None."""
    s = sorted(set(a) & set(b))
    if len(s) < 2:
        return None
    da = np.array([a[x] for x in s])
    db = np.array([b[x] for x in s])
    diff = da - db
    sd = diff.std(ddof=1)
    dz = diff.mean() / sd if sd > 0 else 0.0
    try:
        nonzero = diff[diff != 0]
        p = stats.wilcoxon(nonzero).pvalue if len(nonzero) >= 1 and \
            not (np.all(nonzero > 0) or np.all(nonzero < 0)) else float("nan")
    except Exception:
        p = float("nan")
    return float(diff.mean()), len(s), p, float(dz), float((diff > 0).mean()), \
        float(da.mean()), float(db.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--out_fig", default=None)
    args = ap.parse_args()

    data = load(args.root)
    summary = {"diversity": {}, "paradox": {}, "mediation_points": []}

    # ---------- 1. No-align diversity per pair ----------
    print("=" * 92)
    print("(1) NO-ALIGN BEHAVIORAL DIVERSITY  (perspective_diversity of *_notom)")
    print("=" * 92)
    print(f"  {'pair':4s} {'game':20s} {'n':>3s} {'mean':>7s} {'min':>7s} {'max':>7s}")
    for pair in PAIR_ORDER:
        if pair not in data:
            continue
        summary["diversity"][pair] = {}
        for g in GAME_ORDER:
            arm = data[pair].get(g, {}).get("NoAlign", {})
            vals = [v["diversity"] for v in arm.values() if v["diversity"] == v["diversity"]]
            if not vals:
                continue
            summary["diversity"][pair][g] = {
                "n": len(vals), "mean": float(np.mean(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
            }
            print(f"  {pair:4s} {g:20s} {len(vals):3d} {np.mean(vals):7.3f} "
                  f"{np.min(vals):7.3f} {np.max(vals):7.3f}")
        # pair-level aggregate over anti-coordination games
        allv = []
        for g in ANTI:
            arm = data[pair].get(g, {}).get("NoAlign", {})
            allv += [v["diversity"] for v in arm.values() if v["diversity"] == v["diversity"]]
        if allv:
            summary["diversity"][pair]["_anti_pooled"] = {
                "n": len(allv), "mean": float(np.mean(allv)),
                "min": float(np.min(allv)), "max": float(np.max(allv)),
            }
            print(f"  {pair:4s} {'[anti pooled]':20s} {len(allv):3d} "
                  f"{np.mean(allv):7.3f} {np.min(allv):7.3f} {np.max(allv):7.3f}")

    # ---------- 2. Paradox effect CGA(dp) - Gated ----------
    print("\n" + "=" * 92)
    print("(2) ALIGNMENT-PARADOX EFFECT  =  CGA(dp) - Gated  on cooperation_payoff "
          "(paired over seeds)")
    print("=" * 92)
    print(f"  {'pair':4s} {'game':20s} {'grp':6s} {'CGA':>7s} {'Gated':>7s} "
          f"{'delta':>8s} {'dz':>7s} {'win':>5s} {'p':>8s} {'n':>3s}")
    grp_of = {**{g: "anti" for g in ANTI}, **{g: "coord" for g in COORD},
              **{g: "bound" for g in BOUNDARY}}
    for pair in PAIR_ORDER:
        if pair not in data:
            continue
        summary["paradox"][pair] = {}
        for g in GAME_ORDER:
            cga = {s: v["payoff"] for s, v in data[pair].get(g, {}).get("CGA", {}).items()}
            gat = {s: v["payoff"] for s, v in data[pair].get(g, {}).get("Gated", {}).items()}
            r = paired(cga, gat)
            if not r:
                continue
            d, n, p, dz, win, cm, gm = r
            summary["paradox"][pair][g] = {
                "delta": d, "n": n, "p": p, "dz": dz, "win": win,
                "cga_mean": cm, "gated_mean": gm, "group": grp_of[g],
            }
            ps = f"{p:8.3f}" if p == p else "     n/a"
            print(f"  {pair:4s} {g:20s} {grp_of[g]:6s} {cm:7.3f} {gm:7.3f} "
                  f"{d:+8.3f} {dz:+7.2f} {win:5.0%} {ps} {n:3d}")

    # ---------- 3. Mediation scatter (anti-coordination games) ----------
    print("\n" + "=" * 92)
    print("(3) MEDIATION SCATTER  --  x=No-align diversity, y=paradox effect "
          "(anti-coordination games, all pairs)")
    print("=" * 92)
    print(f"  {'pair':4s} {'game':20s} {'diversity_x':>12s} {'paradox_y':>10s}")
    xs, ys, labels = [], [], []
    for pair in PAIR_ORDER:
        for g in ANTI:
            dv = summary["diversity"].get(pair, {}).get(g)
            pe = summary["paradox"].get(pair, {}).get(g)
            if not dv or not pe:
                continue
            x = dv["mean"]
            y = pe["delta"]
            xs.append(x); ys.append(y); labels.append(f"{pair}/{g}")
            summary["mediation_points"].append(
                {"pair": pair, "game": g, "diversity": x, "paradox_effect": y})
            print(f"  {pair:4s} {g:20s} {x:12.3f} {y:+10.3f}")
    if len(xs) >= 3:
        xs_a, ys_a = np.array(xs), np.array(ys)
        sp = stats.spearmanr(xs_a, ys_a)
        pe = stats.pearsonr(xs_a, ys_a)
        summary["mediation_correlation"] = {
            "n_points": len(xs),
            "spearman_rho": float(sp.correlation), "spearman_p": float(sp.pvalue),
            "pearson_r": float(pe[0]), "pearson_p": float(pe[1]),
        }
        print(f"\n  n_points={len(xs)}  Spearman rho={sp.correlation:+.3f} "
              f"(p={sp.pvalue:.3f})  Pearson r={pe[0]:+.3f} (p={pe[1]:.3f})")
        print("  Positive correlation => paradox strength grows with team behavioral"
              " diversity\n  (mechanism chain: heterogeneity -> diversity -> "
              "forced-alignment paradox).")

    # ---------- optional figure ----------
    if args.out_fig and len(xs) >= 3:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # Okabe-Ito colorblind-safe palette per pair
            pal = {"QG": "#0072B2", "QL": "#009E73", "QQ": "#E69F00", "GG": "#D55E00"}
            mrk = {"chicken": "o", "deadlock": "s", "hawk_dove": "^"}
            fig, ax = plt.subplots(figsize=(5.2, 4.0))
            for x, y, lab in zip(xs, ys, labels):
                pr, gm = lab.split("/")
                ax.scatter(x, y, c=pal.get(pr, "#555555"), marker=mrk.get(gm, "o"),
                           s=70, edgecolors="k", linewidths=0.5, zorder=3)
            # trend line
            xs_a, ys_a = np.array(xs), np.array(ys)
            if xs_a.std() > 0:
                b, a = np.polyfit(xs_a, ys_a, 1)
                xx = np.linspace(xs_a.min(), xs_a.max(), 50)
                ax.plot(xx, a + b * xx, "--", c="0.35", lw=1.3, zorder=2)
            ax.axhline(0, color="0.7", lw=0.8, zorder=1)
            ax.set_xlabel("No-align behavioral diversity (perspective KL)")
            ax.set_ylabel(r"Paradox effect: CGA $-$ Gated payoff")
            # legends
            from matplotlib.lines import Line2D
            ph = [Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[p],
                         markeredgecolor="k", markersize=8, label=p) for p in PAIR_ORDER
                  if p in pal]
            gh = [Line2D([0], [0], marker=mrk[g], color="w", markerfacecolor="0.5",
                         markeredgecolor="k", markersize=8, label=g) for g in ANTI]
            leg1 = ax.legend(handles=ph, title="pair", loc="upper left", fontsize=8,
                             title_fontsize=8, frameon=False)
            ax.add_artist(leg1)
            ax.legend(handles=gh, title="game", loc="lower right", fontsize=8,
                      title_fontsize=8, frameon=False)
            if "mediation_correlation" in summary:
                mc = summary["mediation_correlation"]
                ax.set_title(f"Spearman "r"$\rho$"f"={mc['spearman_rho']:+.2f}",
                             fontsize=10)
            fig.tight_layout()
            os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)
            fig.savefig(args.out_fig, bbox_inches="tight")
            print(f"\n  figure -> {args.out_fig}")
        except Exception as e:
            print(f"\n  [fig skipped] {e}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        json.dump(summary, open(args.out_json, "w"), indent=2)
        print(f"  json  -> {args.out_json}")


if __name__ == "__main__":
    main()
