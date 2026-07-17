#!/usr/bin/env python3
"""G1 analysis: end-to-end 2-arm attainment bandit vs offline reconstruction.

Reads the het_bandit cells (exp_g1_bandit) and the compact paradox ground-truth
(ground_truth_paradox.json: per (game,seed) NoToM/Gated cooperation_payoff,
true_best arm, oracle upper bound, SCA = two-arm-abstain policy).

Pre-registered decision rule:
  - PASS (write into §6.5, contribution 2 gains the end-to-end clause) if
    selection accuracy >= 0.85 AND bandit commit mean >= SCA mean.
  - Otherwise report honestly: probe/commit decoupling weakened the offline
    estimate; offline result retained as an upper bound.

Outputs g1_bandit_summary.{json,md}.
"""
import argparse, json, glob, os
from collections import defaultdict
import numpy as np
from scipy import stats

GROUP = {"chicken": "anti", "hawk_dove": "anti", "deadlock": "anti",
         "stag_hunt": "coord", "battle_of_the_sexes": "coord",
         "public_goods": "boundary"}
ORDER = ["chicken", "deadlock", "hawk_dove", "stag_hunt",
         "battle_of_the_sexes", "public_goods"]


def load_gt(path):
    gt = json.load(open(path))
    m = {(c["game"], c["seed"]): c for c in gt["cells"]}
    return gt, m


def load_bandit(root):
    rows = []
    for f in glob.glob(f"{root}/*/seed_*/het_bandit/metrics.json"):
        p = f.split("/")
        seed = int(p[-3].replace("seed_", ""))
        game = p[-4]
        d = json.load(open(f))
        rows.append({"game": game, "seed": seed,
                     "chosen": d.get("bandit_chosen_arm"),
                     "probe_N": d.get("bandit_probe_mean_NoAlign"),
                     "probe_G": d.get("bandit_probe_mean_Gated"),
                     "commit_payoff": d.get("cooperation_payoff"),
                     "team_mean": d.get("team_mean_payoff"),
                     "probe_order": d.get("bandit_probe_order"),
                     "wall_s": d.get("wall_time_s")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandit", required=True, help="exp_g1_bandit root")
    ap.add_argument("--gt", required=True, help="ground_truth_paradox.json")
    ap.add_argument("--acc_thresh", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt, gtm = load_gt(args.gt)
    rows = load_bandit(args.bandit)
    out = args.out or args.bandit

    per = []
    for r in rows:
        c = gtm.get((r["game"], r["seed"]))
        if c is None:
            continue
        correct = int(r["chosen"] == c["true_best"])
        per.append({**r, "true_best": c["true_best"], "correct": correct,
                    "oracle": c["oracle"], "notom": c["notom"], "gated": c["gated"],
                    "group": GROUP[r["game"]]})

    n = len(per)
    n_correct = sum(p["correct"] for p in per)
    acc = n_correct / n if n else 0.0
    bandit_mean = float(np.mean([p["commit_payoff"] for p in per]))
    oracle_mean = float(np.mean([p["oracle"] for p in per]))
    sca_mean = gt["sca_mean"]
    # probe regret: oracle_commit_proxy - probe payoff (exploration cost)
    probe_payoffs = [p["probe_N"] if p["chosen"] == "NoAlign" else p["probe_G"]
                     for p in per]
    probe_mean = float(np.mean(probe_payoffs))

    pass_acc = acc >= args.acc_thresh
    pass_mean = bandit_mean >= sca_mean
    verdict = "PASS" if (pass_acc and pass_mean) else "PARTIAL/FAIL"

    # per-game table
    pg = {}
    for g in ORDER:
        gp = [p for p in per if p["game"] == g]
        if not gp:
            continue
        pg[g] = {
            "group": GROUP[g], "n": len(gp),
            "acc": sum(p["correct"] for p in gp) / len(gp),
            "bandit_mean": float(np.mean([p["commit_payoff"] for p in gp])),
            "oracle_mean": float(np.mean([p["oracle"] for p in gp])),
            "frac_Gated": float(np.mean([1 for p in gp if p["chosen"] == "Gated"]))}

    # paired test bandit vs SCA (reconstruct SCA payoff per same seed)
    sca_pay = np.array([p["gated"] if p["group"] == "coord" else p["notom"]
                        for p in per])
    band_pay = np.array([p["commit_payoff"] for p in per])
    try:
        pwil = float(stats.wilcoxon(band_pay, sca_pay).pvalue)
    except Exception:
        pwil = float("nan")
    dz = float((band_pay - sca_pay).mean() / (band_pay - sca_pay).std(ddof=1)) \
        if (band_pay - sca_pay).std(ddof=1) > 0 else 0.0

    md = []
    md.append("# G1 — End-to-end 2-arm attainment bandit\n")
    md.append(f"cells: {n}/{gt['n']}  |  metric: cooperation_payoff (commit phase, "
              f"20 ep for 2-player / 10 ep for public_goods)\n")
    md.append(f"\n## Headline (pre-registered: accuracy>={args.acc_thresh:.0%} AND bandit>=SCA)\n")
    md.append(f"- selection accuracy: **{n_correct}/{n} = {acc:.1%}** "
              f"(offline probe-argmax reconstruction was 116/120=96.7%)\n")
    md.append(f"- bandit commit mean: **{bandit_mean:.4f}** vs SCA {sca_mean:.4f} "
              f"(Δ={bandit_mean-sca_mean:+.4f}, p={pwil:.4f}, dz={dz:+.2f}) "
              f"vs oracle {oracle_mean:.4f}\n")
    md.append(f"- **verdict: {verdict}** "
              f"(acc>={args.acc_thresh:.0%}: {'yes' if pass_acc else 'no'}; "
              f"bandit>=SCA: {'yes' if pass_mean else 'no'})\n")
    md.append(f"- probe exploration payoff: {probe_mean:.4f} "
              f"(commit {bandit_mean:.4f}; probe-commit gap = {bandit_mean-probe_mean:+.4f})\n")
    md.append("\n## Per-game\n")
    md.append("| game | grp | n | acc | bandit | oracle | frac Gated |\n|---|---|---|---|---|---|---|")
    for g in ORDER:
        if g in pg:
            r = pg[g]
            md.append(f"| {g} | {r['group']} | {r['n']} | {r['acc']:.0%} | "
                      f"{r['bandit_mean']:.3f} | {r['oracle_mean']:.3f} | {r['frac_Gated']:.0%} |")
    md.append("\n## Decision log (misses)\n")
    misses = [p for p in per if not p["correct"]]
    md.append(f"{len(misses)} misses: " + "; ".join(
        f"{p['game']}/s{p['seed']}: chose {p['chosen']} (true {p['true_best']}, "
        f"probe N={p['probe_N']:.2f}/G={p['probe_G']:.2f})" for p in misses[:30]))

    summary = {"n": n, "n_correct": n_correct, "accuracy": acc,
               "bandit_mean": bandit_mean, "sca_mean": sca_mean,
               "oracle_mean": oracle_mean, "probe_mean": probe_mean,
               "wilcoxon_p_bandit_vs_sca": pwil, "cohen_dz": dz,
               "acc_thresh": args.acc_thresh, "verdict": verdict,
               "pass_acc": pass_acc, "pass_mean": pass_mean,
               "per_game": pg, "per_cell": per}
    json.dump(summary, open(f"{out}/g1_bandit_summary.json", "w"), indent=2)
    open(f"{out}/g1_bandit_summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[wrote] {out}/g1_bandit_summary.{{json,md}}")


if __name__ == "__main__":
    main()
