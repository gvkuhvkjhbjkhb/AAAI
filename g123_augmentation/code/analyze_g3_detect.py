#!/usr/bin/env python3
"""G3 analysis: stack-B online-detection validation (het_gsaca).

Reads the het_gsaca cells (exp_g3_detect: 6 games x 5 seeds = 30 cells) and
reports the per-cell split_score sign, detected vs oracle structure, and the
detection accuracy. This closes the v6 limitation "detector not re-run on
stack B" (Table 2 caption). Prop 1 predicts 30/30 correct (matrices unchanged).

If low-diversity warm-up profiles cause misses, that is itself evidence of the
Prop 1 precondition failing, to be written into §6.6 detection boundary.

Outputs g3_detect_summary.{json,md}.
"""
import argparse, json, glob, os
from collections import defaultdict
import numpy as np

ORACLE = {"chicken": "anti_coord", "hawk_dove": "anti_coord", "deadlock": "anti_coord",
          "stag_hunt": "coord", "battle_of_the_sexes": "coord", "public_goods": "coord"}
GROUP = {"chicken": "anti", "hawk_dove": "anti", "deadlock": "anti",
         "stag_hunt": "coord", "battle_of_the_sexes": "coord", "public_goods": "boundary"}
ORDER = ["chicken", "deadlock", "hawk_dove", "stag_hunt",
         "battle_of_the_sexes", "public_goods"]


def load(root):
    rows = []
    for f in glob.glob(f"{root}/*/seed_*/het_gsaca/metrics.json"):
        p = f.split("/")
        seed = int(p[-3].replace("seed_", ""))
        game = p[-4]
        d = json.load(open(f))
        rows.append({"game": game, "seed": seed,
                     "split": d.get("gsaca_split_score"),
                     "detected": d.get("gsaca_detected_structure"),
                     "oracle": d.get("gsaca_oracle_structure", ORACLE.get(game)),
                     "correct": d.get("gsaca_detection_correct"),
                     "payoff": d.get("cooperation_payoff"),
                     "wall_s": d.get("wall_time_s")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", required=True, help="exp_g3_detect root")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.detect
    rows = load(args.detect)
    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])

    md = ["# G3 — Stack-B online-detection validation (het_gsaca)\n"]
    md.append(f"cells: {n}/30  |  detection accuracy: **{n_correct}/{n}**\n")
    md.append("\n## Per-cell (split sign + correctness)\n")
    md.append("| game | grp | seed | split | detected | oracle | correct |\n|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (ORDER.index(x["game"]) if x["game"] in ORDER else 99, x["seed"])):
        md.append(f"| {r['game']} | {GROUP.get(r['game'],'?')} | {r['seed']} | "
                  f"{r['split']:+.3f} | {r['detected']} | {r['oracle']} | "
                  f"{'OK' if r['correct'] else 'MISS'} |")
    # per-game split summary
    md.append("\n## Per-game split (mean ± std) and accuracy\n")
    md.append("| game | grp | n | split mean | split std | acc |\n|---|---|---|---|---|---|")
    per_game = {}
    for g in ORDER:
        gr = [r for r in rows if r["game"] == g]
        if not gr:
            continue
        sp = [r["split"] for r in gr]
        acc = sum(1 for r in gr if r["correct"]) / len(gr)
        per_game[g] = {"n": len(gr), "split_mean": float(np.mean(sp)),
                       "split_std": float(np.std(sp)), "acc": acc}
        md.append(f"| {g} | {GROUP[g]} | {len(gr)} | {np.mean(sp):+.3f} | "
                  f"{np.std(sp):.3f} | {acc:.0%} |")
    verdict = ("30/30 correct (Prop 1 holds on stack B; matrices unchanged)"
               if n_correct == n else
               f"{n_correct}/{n}: misses present -> Prop 1 precondition (warm-up profile coverage) fails")
    md.append(f"\n## Verdict\n- **{verdict}**\n")

    json.dump({"n": n, "n_correct": n_correct, "accuracy": n_correct / n if n else 0,
               "per_game": per_game, "per_cell": rows, "verdict": verdict},
              open(f"{out}/g3_detect_summary.json", "w"), indent=2)
    open(f"{out}/g3_detect_summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n[wrote] {out}/g3_detect_summary.{{json,md}}")


if __name__ == "__main__":
    main()
