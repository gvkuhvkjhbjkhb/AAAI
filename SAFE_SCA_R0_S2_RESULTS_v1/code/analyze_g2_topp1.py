#!/usr/bin/env python3
"""G2 analysis: top_p single-factor ablation (top_p=1.0 vs frozen 0.9).

Reads the top_p=1.0 cells (exp_g2_topp1: het_notom + het_gated_atom_talk on the
3 anti-coordination games, seeds 42-51) and the top_p=0.9 ground truth from
ground_truth_paradox.json (which holds the same arms at the frozen config).

Question: is the anti-coordination "flip" (NoToM >= Gated, i.e. forced alignment
hurts) at top_p=0.9 driven by nucleus-sampling truncation? If at top_p=1.0 the
flip disappears (Gated >= NoToM), sampling truncation is the main cause; if it
persists, the flip is robust to sampling.

Outputs g2_topp1_summary.{json,md}.
"""
import argparse, json, glob, os
from collections import defaultdict
import numpy as np
from scipy import stats

GAMES = ["chicken", "deadlock", "hawk_dove"]


def load(root, cells):
    """root/<game>/seed_<n>/<cell>/metrics.json -> data[game][arm][seed]=payoff."""
    data = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(f"{root}/*/seed_*/*/metrics.json"):
        p = f.split("/")
        cell, seed, game = p[-2], p[-3], p[-4]
        if cell not in cells:
            continue
        arm = cells[cell]
        try:
            d = json.load(open(f))
        except Exception:
            continue
        v = d.get("cooperation_payoff")
        if v is None:
            continue
        data[game][arm][int(seed.replace("seed_", ""))] = float(v)
    return data


def paired(a_map, b_map):
    s = sorted(set(a_map) & set(b_map))
    if len(s) < 3:
        return None
    a = np.array([a_map[x] for x in s])
    b = np.array([b_map[x] for x in s])
    diff = a - b
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        p = float("nan")
    sd = diff.std(ddof=1)
    dz = float(diff.mean() / sd) if sd > 0 else 0.0
    return {"n": len(s), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta": float(diff.mean()), "p": p, "dz": dz,
            "win": float((diff > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topp1", required=True, help="exp_g2_topp1 root")
    ap.add_argument("--gt", required=True, help="ground_truth_paradox.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.topp1

    cells = {"het_notom": "NoToM", "het_gated_atom_talk": "Gated"}
    d1 = load(args.topp1, cells)                       # top_p=1.0
    gt = json.load(open(args.gt))
    # top_p=0.9 ground truth per (game,seed)
    d0 = defaultdict(lambda: defaultdict(dict))
    for c in gt["cells"]:
        if c["game"] in GAMES:
            d0[c["game"]]["NoToM"][c["seed"]] = c["notom"]
            d0[c["game"]]["Gated"][c["seed"]] = c["gated"]

    md = ["# G2 — top_p single-factor ablation (1.0 vs frozen 0.9)\n"]
    md.append("3 anti-coordination games, seeds 42-51. Arms NoToM (NoAlign) & Gated.\n")
    md.append("\n## Per-game: NoToM vs Gated at each top_p (paired Wilcoxon)\n")
    md.append("| game | top_p | NoToM | Gated | Δ(Gated-NoToM) | p | dz | flip(NoToM>=Gated) |\n"
              "|---|---|---|---|---|---|---|---|")
    summary = {}
    for g in GAMES:
        summary[g] = {}
        for tag, d, tp in [("0.9", d0, 0.9), ("1.0", d1, 1.0)]:
            r = paired(d[g].get("NoToM", {}), d[g].get("Gated", {}))
            if r is None:
                continue
            flip = r["mean_a"] >= r["mean_b"]
            summary[g][tag] = r
            md.append(f"| {g} | {tp} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | "
                      f"{r['delta']:+.3f} | {r['p']:.4f} | {r['dz']:+.2f} | "
                      f"{'YES' if flip else 'no'} |")
    md.append("\n## Attribution\n")
    flips0 = [g for g in GAMES if summary[g].get("0.9") and
              summary[g]["0.9"]["mean_a"] >= summary[g]["0.9"]["mean_b"]]
    flips1 = [g for g in GAMES if summary[g].get("1.0") and
              summary[g]["1.0"]["mean_a"] >= summary[g]["1.0"]["mean_b"]]
    md.append(f"- top_p=0.9 flip (NoToM>=Gated) in: {flips0}\n")
    md.append(f"- top_p=1.0 flip (NoToM>=Gated) in: {flips1}\n")
    if not flips1 and flips0:
        verdict = "SAMPLING-DRIVEN: flip disappears at top_p=1.0 -> top_p=0.9 truncation is the main cause"
    elif flips1:
        verdict = "SAMPLING-ROBUST: flip persists at top_p=1.0 -> attributable to precision/template/other factors"
    else:
        verdict = "no flip at either top_p"
    md.append(f"- **verdict: {verdict}**\n")
    # cross-config shift per arm
    md.append("\n## Per-arm shift top_p 0.9 -> 1.0 (same seeds)\n")
    md.append("| game | arm | 0.9 | 1.0 | shift |\n|---|---|---|---|---|")
    for g in GAMES:
        for arm in ["NoToM", "Gated"]:
            s = sorted(set(d0[g].get(arm, {})) & set(d1[g].get(arm, {})))
            if not s:
                continue
            m0 = np.mean([d0[g][arm][x] for x in s])
            m1 = np.mean([d1[g][arm][x] for x in s])
            md.append(f"| {g} | {arm} | {m0:.3f} | {m1:.3f} | {m1-m0:+.3f} |")

    json.dump({"summary": summary, "flips_0.9": flips0, "flips_1.0": flips1,
               "verdict": verdict},
              open(f"{out}/g2_topp1_summary.json", "w"), indent=2)
    open(f"{out}/g2_topp1_summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[wrote] {out}/g2_topp1_summary.{{json,md}}")


if __name__ == "__main__":
    main()
