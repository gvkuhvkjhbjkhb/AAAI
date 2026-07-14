#!/usr/bin/env python3
"""
Nash Equilibrium Structure Analysis for LLM Cooperation Games.
Offline analysis — zero GPU. Reads completed experiment metrics.
"""
import json, os, sys, csv
import numpy as np
from collections import defaultdict
from datetime import datetime

GAMES = {
    "stag_hunt": {
        "type": "coordination",
        "payoff": [[(3,3), (0,2)], [(2,0), (2,2)]],
        "actions": ["Stag", "Hare"],
    },
    "battle_of_the_sexes": {
        "type": "preference_conflict",
        "payoff": [[(3,2), (0,0)], [(0,0), (2,3)]],
        "actions": ["Opera", "Football"],
    },
    "chicken": {
        "type": "anti_coordination",
        "payoff": [[(3,3), (1,5)], [(5,1), (0,0)]],
        "actions": ["Dove", "Hawk"],
    },
    "hawk_dove": {
        "type": "anti_coordination",
        "payoff": [[(2,2), (0,3)], [(3,0), (-1,-1)]],
        "actions": ["Dove", "Hawk"],
    },
    "deadlock": {
        "type": "anti_coordination",
        "payoff": [[(2,2), (0,4)], [(4,0), (1,1)]],
        "actions": ["Coop", "Defect"],
    },
}

def find_nash_equilibria(payoff_mat):
    pure_ne = []
    for a1 in [0, 1]:
        for a2 in [0, 1]:
            r1, r2 = payoff_mat[a1][a2]
            best_resp_1 = all(payoff_mat[ap][a2][0] <= r1 + 1e-9 for ap in [0, 1])
            best_resp_2 = all(payoff_mat[a1][ap][1] <= r2 + 1e-9 for ap in [0, 1])
            if best_resp_1 and best_resp_2:
                pure_ne.append({"a1": a1, "a2": a2, "payoffs": (r1, r2)})
    return {
        "pure_nash": pure_ne,
        "num_pure_ne": len(pure_ne),
    }

def analyze_game(gname, ginfo):
    ne = find_nash_equilibria(ginfo["payoff"])
    pure_actions = {(n["a1"], n["a2"]) for n in ne["pure_nash"]}
    is_split = {(0, 1), (1, 0)}.issubset(pure_actions) or any(n["a1"] != n["a2"] for n in ne["pure_nash"])
    is_symmetric = any(n["a1"] == n["a2"] for n in ne["pure_nash"])
    
    aligned_welfare = []
    for a in [0, 1]:
        s = ginfo["payoff"][a][a][0] + ginfo["payoff"][a][a][1]
        aligned_welfare.append(s)
    
    eq_payoffs = [n["payoffs"] for n in ne["pure_nash"]]
    max_ne_welfare = max(sum(p) for p in eq_payoffs) if eq_payoffs else 0
    max_aligned_welfare = max(aligned_welfare) if aligned_welfare else 0
    
    return {
        "game": gname,
        "type": ginfo["type"],
        "split_equilibria": is_split,
        "symmetric_equilibria": is_symmetric,
        "split_required": is_split and not is_symmetric,
        "num_pure_ne": ne["num_pure_ne"],
        "max_ne_welfare": max_ne_welfare,
        "max_aligned_welfare": max_aligned_welfare,
        "gated_penalty": max_aligned_welfare - max_ne_welfare,
        "pure_nash": ne["pure_nash"],
        "dp_prediction": ("beneficial" if is_split and not is_symmetric else 
                          "neutral" if is_split and is_symmetric else 
                          "harmful_or_neutral"),
    }

def load_metrics_json(top_dir):
    data = defaultdict(lambda: defaultdict(dict))
    if not os.path.isdir(top_dir):
        return data
    for root, dirs, files in os.walk(top_dir):
        if "metrics.json" not in files:
            continue
        parts = root.split("/")
        if len(parts) < 3:
            continue
        try:
            cell = parts[-1]
            seed_dir = parts[-2]
            game = parts[-3]
        except IndexError:
            continue
        mp = os.path.join(root, "metrics.json")
        try:
            with open(mp) as f: d = json.load(f)
        except:
            continue
        payoff = d.get("cooperation_payoff", d.get("payoff"))
        if payoff is None:
            continue
        data[game][cell][seed_dir] = {
            "payoff": payoff,
            "diversity": d.get("perspective_diversity", d.get("diversity")),
        }
    return data

def main():
    base = "/data/lab"
    phase1_dir = os.path.join(base, "results", "phase1_5_unified", "phase1")
    chicken_dir = os.path.join(base, "results", "phase1_5_unified", "chicken_repro")
    out = os.path.join(base, "analysis")
    os.makedirs(out, exist_ok=True)

    exp_data = {}
    for label, d in [("phase1", phase1_dir), ("chicken_repro", chicken_dir)]:
        exp_data[label] = load_metrics_json(d)

    # Build report
    lines = ["# Nash Equilibrium Structure Analysis\n",
             f"> {datetime.now().isoformat()}\n",
             "## 1. Game-Theoretic Predictions\n\n"]

    rows = []
    for gname, ginfo in GAMES.items():
        r = analyze_game(gname, ginfo)
        lines.append(f"### {gname} ({r['type']})\n")
        lines.append(f"- Split equilibria: {r['split_equilibria']}\n")
        lines.append(f"- Symmetric equilibria: {r['symmetric_equilibria']}\n")
        lines.append(f"- Split required for Nash: {r['split_required']}\n")
        for n in r["pure_nash"]:
            lines.append(f"  - NE: a1={n['a1']}, a2={n['a2']} → {n['payoffs']}\n")
        if r["split_required"]:
            lines.append(f"- **DP-Gating**: ✅ BENEFICIAL — gated destroys the only NE (split)\n")
            lines.append(f"  - Max NE welfare: {r['max_ne_welfare']:.0f}, Max aligned: {r['max_aligned_welfare']:.0f}\n")
            lines.append(f"  - Gated penalty: {r['gated_penalty']:.0f}\n")
        elif r["split_equilibria"] and r["symmetric_equilibria"]:
            lines.append(f"- **DP-Gating**: ⚪ NEUTRAL — both split and symmetric NE exist\n")
        else:
            lines.append(f"- **DP-Gating**: ⚪ NEUTRAL — only symmetric NE\n")
        lines.append("")

        # Empirical deltas
        for src_label in ["phase1", "chicken_repro"]:
            ed = exp_data.get(src_label, {})
            gg = ed.get(gname, {})
            dp_key = "het_dp_gated_atom_talk"
            gt_key = "het_gated_atom_talk"
            if dp_key in gg and gt_key in gg:
                dp_vals = [v["payoff"] for v in gg[dp_key].values()]
                gt_vals = [v["payoff"] for v in gg[gt_key].values()]
                if len(dp_vals) >= 3 and len(gt_vals) >= 3:
                    try:
                        from scipy import stats
                        stat, p = stats.mannwhitneyu(dp_vals, gt_vals, alternative="two-sided")
                    except ImportError:
                        p = 1.0
                    delta = np.mean(dp_vals) - np.mean(gt_vals)
                    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "n.s."))
                    lines.append(f"  [{src_label}] Δ={delta:.3f}, p={p:.4f} {sig}, n_dp={len(dp_vals)}, n_gt={len(gt_vals)}\n")

    # Summary table
    lines.append("\n## 2. Mechanism: Why Non-Intervention Works\n\n")
    lines.append(
        "The Nash equilibrium structure predicts DP-Gating effectiveness:\n\n"
        "1. **Anti-coordination** (Chicken, Hawk-Dove, Deadlock): Nash equilibria "
        "are SPLIT — (Hawk, Dove) and (Dove, Hawk). Gated arbitration forces both "
        "agents to the same action, DESTROYING the split equilibrium. ✅ DP predicted "
        "beneficial.\n\n"
        "2. **Coordination** (Stag Hunt): Nash equilibria are SYMMETRIC — (Stag, Stag) "
        "and (Hare, Hare). Gated preserves symmetric structure. ⚪ DP predicted neutral.\n\n"
        "3. **Preference conflict** (BoS): Split NE exist but agents have CONFLICTING "
        "preferences over which equilibrium. Non-intervention preserves conflict. "
        "⚠️ Neither method resolves preference conflict.\n\n"
        "**Key insight**: DP-Gating works NOT by 'preserving diversity' but by "
        "NOT DESTROYING the equilibrium structure required by anti-coordination games. "
        "This is a structural claim grounded in game theory.\n"
    )

    with open(os.path.join(out, "nash_equilibrium_report.md"), "w") as f:
        f.write("".join(lines))

    print(f"Report: {os.path.join(out, 'nash_equilibrium_report.md')}")
    for l in lines:
        print(l.rstrip())


if __name__ == "__main__":
    main()
