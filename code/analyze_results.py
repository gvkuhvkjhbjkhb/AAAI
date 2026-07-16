#!/usr/bin/env python3
"""Analyze all Phase 1-5 experiment results and produce a report."""
import json, os, glob, sys
import numpy as np
from collections import defaultdict
from datetime import datetime

BASE = "/data/lab/results/phase1_5_unified"
CELLS = ["het_dp_gated_atom_talk", "het_gated_atom_talk", "het_notom", "hom_notom"]

def load_metrics(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def safe_mean(arr):
    arr = [x for x in arr if x is not None and not np.isnan(x)]
    return float(np.mean(arr)) if arr else float("nan")

def safe_std(arr):
    arr = [x for x in arr if x is not None and not np.isnan(x)]
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = np.sqrt(((nx-1)*np.var(x,ddof=1)**2 + (ny-1)*np.var(y,ddof=1)**2) / (nx+ny-2))
    return float((np.mean(x)-np.mean(y))/pooled) if pooled > 0 else 0.0

def mw_p(x, y):
    """One-sided Mann-Whitney U p-value (simplified normal approx for small n)."""
    from scipy.stats import mannwhitneyu
    try:
        _, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(p)
    except:
        return 1.0

def get_source_metrics(source_dir, game):
    metrics = []
    for seed in range(42, 50):
        mpath = os.path.join(source_dir, game, f"seed_{seed}")
        if os.path.isdir(mpath):
            m = {}
            for cell in CELLS:
                cm = load_metrics(os.path.join(mpath, cell, "metrics.json"))
                if cm:
                    m[cell] = cm
            if m:
                metrics.append(m)
    return metrics

def analyze_source(name, source_dir, games):
    print(f"\n{'='*80}")
    print(f"  {name.upper()}")
    print(f"{'='*80}")

    for game in games:
        ms = get_source_metrics(source_dir, game)
        if not ms:
            continue

        gated_payoffs = []
        dp_payoffs = []
        gated_divs = []
        dp_divs = []
        gated_conflicts = []
        dp_interventions = []

        for m in ms:
            g = m.get("het_gated_atom_talk", {})
            d = m.get("het_dp_gated_atom_talk", {})

            gp = g.get("cooperation_payoff")
            dp = d.get("cooperation_payoff")
            gd = g.get("perspective_diversity")
            dd = d.get("perspective_diversity")
            gc = g.get("dp_conflict_rate", g.get("gate_trust_rate"))
            di = d.get("dp_intervention_rate")

            if gp is not None: gated_payoffs.append(gp)
            if dp is not None: dp_payoffs.append(dp)
            if gd is not None: gated_divs.append(gd)
            if dd is not None: dp_divs.append(dd)
            if gc is not None: gated_conflicts.append(gc)
            if di is not None: dp_interventions.append(di)

        n = len(dp_payoffs)
        if n < 3:
            print(f"\n--- {game.upper()} (n={n}, insufficient) ---")
            if n > 0:
                print(f"  DP-gating payoff: {safe_mean(dp_payoffs):.3f}±{safe_std(dp_payoffs):.3f}")
                print(f"  Gated    payoff: {safe_mean(gated_payoffs):.3f}±{safe_std(gated_payoffs):.3f}")
            continue

        delta = safe_mean(dp_payoffs) - safe_mean(gated_payoffs)
        delta_div = safe_mean(dp_divs) - safe_mean(gated_divs)
        p_val = mw_p(dp_payoffs, gated_payoffs) if n >= 3 else 1.0
        p_div = mw_p(dp_divs, gated_divs) if n >= 3 and len(dp_divs)>=3 else 1.0
        dps_wins = sum(1 for dp, gp in zip(dp_payoffs, gated_payoffs) if dp > gp)
        ci = (safe_std([a-b for a,b in zip(dp_payoffs,gated_payoffs)]) * 1.96 / np.sqrt(n)) if n > 1 else 0
        ci_lo = delta - ci
        ci_hi = delta + ci

        dp_mean = safe_mean(dp_payoffs)
        gated_mean = safe_mean(gated_payoffs)
        stats = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else "n.s."))

        print(f"\n--- {game.upper()} (n={n}) ---")
        print(f"  PAYOFF | DP-gating {dp_mean:.3f}±{safe_std(dp_payoffs):.3f} | Gated {gated_mean:.3f}±{safe_std(gated_payoffs):.3f}")
        print(f"         | Δ={delta:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}]  p={p_val:.4f}  {dps_wins}/{n} wins  {stats}")
        print(f"  DIVER  | DP-gating {safe_mean(dp_divs):.4f}±{safe_std(dp_divs):.4f} | Gated {safe_mean(gated_divs):.4f}±{safe_std(gated_divs):.4f}")
        print(f"         | Δ={delta_div:+.4f}  p={p_div:.4f}  conflict_rate: {safe_mean(dp_interventions) if dp_interventions else safe_mean(gated_conflicts):.3f}")


def analyze_thresholds():
    base = "/data/lab/results/phase1_5_unified/phase3_threshold"
    print(f"\n{'='*80}")
    print(f"  PHASE 3: THRESHOLD ABLATION")
    print(f"{'='*80}")

    for thr_dir in sorted(os.listdir(base)):
        thr_path = os.path.join(base, thr_dir)
        if not os.path.isdir(thr_path):
            continue
        thr = thr_dir.replace("thr_", "")

        for game in ["chicken", "hawk_dove"]:
            payoffs = []
            divs = []
            conflicts = []
            interventions = []
            for seed in range(42, 50):
                m_path = os.path.join(thr_path, game, f"seed_{seed}", "het_dp_gated_atom_talk", "metrics.json")
                m = load_metrics(m_path)
                if m:
                    payoffs.append(m.get("cooperation_payoff"))
                    divs.append(m.get("perspective_diversity"))
                    interventions.append(m.get("dp_intervention_rate"))
                    conflicts.append(m.get("dp_conflict_rate"))

            n = len(payoffs)
            if n < 3:
                continue

            print(f"\n  {thr} | {game} (n={n})")
            print(f"    payoff: {safe_mean(payoffs):.3f}±{safe_std(payoffs):.3f}  div: {safe_mean(divs):.4f}  interv: {safe_mean(interventions):.3f}  conflict: {safe_mean(conflicts):.3f}")


def final_summary():
    print(f"\n{'='*80}")
    print(f"  CROSS-GAME SUMMARY")
    print(f"{'='*80}")

    results = {}
    for name, source_dir, games in [
        ("Deadlock", f"{BASE}/phase1", ["deadlock"]),
        ("StagHunt", f"{BASE}/phase1", ["stag_hunt"]),
        ("Hawk-Dove", f"{BASE}/phase1", ["hawk_dove"]),
        ("BoS", f"{BASE}/phase1", ["battle_of_the_sexes"]),
        ("Chicken", f"{BASE}/chicken_repro", ["chicken"]),
    ]:
        for game in games:
            ms = get_source_metrics(source_dir, game)
            if len(ms) < 3:
                continue
            dp_p = [m.get("het_dp_gated_atom_talk",{}).get("cooperation_payoff") for m in ms if m.get("het_dp_gated_atom_talk",{}).get("cooperation_payoff") is not None]
            ga_p = [m.get("het_gated_atom_talk",{}).get("cooperation_payoff") for m in ms if m.get("het_gated_atom_talk",{}).get("cooperation_payoff") is not None]
            results[game] = {
                "n": len(dp_p), "dp_mean": safe_mean(dp_p), "dp_std": safe_std(dp_p),
                "ga_mean": safe_mean(ga_p), "ga_std": safe_std(ga_p),
                "delta": safe_mean(dp_p) - safe_mean(ga_p),
                "p": mw_p(dp_p, ga_p) if len(dp_p)>=3 and len(ga_p)>=3 else 1.0,
                "wins": sum(1 for a,b in zip(dp_p,ga_p) if a>b),
                "total": len(dp_p),
            }

    print(f"\n{'Game':<15} {'n':>3} {'Gated':<16} {'DP-gating':<18} {'Delta':<9} {'p':<9} {'Wins':<8} {'Verdict'}")
    print("-" * 110)
    for game, r in sorted(results.items()):
        gated_str = f"{r['ga_mean']:.3f}±{r['ga_std']:.3f}"
        dp_str = f"{r['dp_mean']:.3f}±{r['dp_std']:.3f}"
        delta_str = f"{r['delta']:+.3f}"
        p_str = f"{r['p']:.4f}"
        wins_str = f"{r['wins']}/{r['total']}"
        if r['p'] < 0.01: v = "★★★ SIG"
        elif r['p'] < 0.05: v = "★★  SIG"
        elif r['p'] < 0.10: v = "★   WEAK"
        else: v = "n.s."
        print(f"{game:<15} {r['n']:>3} {gated_str:<16} {dp_str:<18} {delta_str:<9} {p_str:<9} {wins_str:<8} {v}")

    sig_games = [g for g, r in results.items() if r['p'] < 0.05 and r['delta'] > 0]
    neg_games = [g for g, r in results.items() if r['delta'] < -0.05]
    ns_games = [g for g, r in results.items() if r['p'] >= 0.05 and abs(r['delta']) < 0.1]

    print(f"\n  Strong wins (p<0.05, delta>0): {len(sig_games)} — {', '.join(sig_games) or 'none'}")
    print(f"  Negative (delta<-0.05): {len(neg_games)} — {', '.join(neg_games) or 'none'}")
    print(f"  Neutral (n.s.): {len(ns_games)} — {', '.join(ns_games) or 'none'}")

    return results


if __name__ == "__main__":
    analyze_source("Phase 1 — Deadlock & StagHunt", f"{BASE}/phase1", ["deadlock", "stag_hunt"])
    analyze_source("Phase 1 — HawkDove & BoS", f"{BASE}/phase1", ["hawk_dove", "battle_of_the_sexes"])
    analyze_source("Phase 1 — Chicken Reproduction", f"{BASE}/chicken_repro", ["chicken"])
    analyze_thresholds()
    final_summary()
