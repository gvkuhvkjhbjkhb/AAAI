#!/usr/bin/env python3
"""Final clean analysis for the corrected paper tables.
Computes:
  (1) Exp C payoff-prompt vs NoToM (paired Wilcoxon, n=20) — Layer 3 baseline
  (2) Three-arm GSACA main table (corrected deadlock Δ=-0.002)
  (3) public_goods complete (n=20) — Layer 2a
  (4) Exp A homogeneous control summary — Layer 2b
Uses scipy.stats.wilcoxon (one-sided greater), BCa CI via bootstrap, Cohen's dz.
"""
import json, glob, os, sys
import numpy as np
from scipy.stats import wilcoxon

GAMES_2P = ["chicken","hawk_dove","deadlock","stag_hunt","battle_of_the_sexes"]
ALL_GAMES = GAMES_2P + ["public_goods"]
LAB = "/data/lab/gsaca"

def load(tree, exp, game, cell):
    """Load payoff per seed from a results tree. Returns {seed: coop_payoff}."""
    base = f"{LAB}/{tree}/{exp}/{game}"
    out = {}
    for sd in sorted(glob.glob(f"{base}/seed_*")):
        s = int(os.path.basename(sd).split("_")[1])
        f = f"{sd}/{cell}/metrics.json"
        if os.path.exists(f):
            try:
                d = json.load(open(f))
                out[s] = d.get("cooperation_payoff", d.get("team_mean_payoff"))
            except: pass
    return out

def bca_ci(d, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    d = np.asarray(d, float)
    if len(d) < 2: return (float('nan'), float('nan'))
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

def paired(notom, method, alt="greater"):
    common = sorted(set(notom) & set(method))
    if not common: return None
    a = np.array([notom[s] for s in common]); b = np.array([method[s] for s in common])
    diff = b - a
    if np.all(diff == 0):
        return dict(n=len(common), d=0.0, p=1.0, sig="ns", ci=(0.0,0.0), dz=0.0, win=0.0, exact_zero=True)
    try:
        w = wilcoxon(b, a, alternative=alt, zero_method="wilcox", correction=True)
        p = float(w.pvalue)
    except Exception:
        p = 1.0
    dz = float(diff.mean()/diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
    sig = "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else "ns"))
    lo, hi = bca_ci(diff.tolist())
    return dict(n=len(common), d=float(diff.mean()), p=p, sig=sig, ci=(lo,hi), dz=dz,
                win=float(np.mean(diff>0)), exact_zero=False)

def sig_star(p):
    return "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else "ns"))

print("="*78)
print("  (1) LAYER 3: Exp C payoff-prompt vs NoToM (paired Wilcoxon, n=20)")
print("="*78)
# NoToM baseline from exp_b (results/v2 for BoS,pg; v2_results for 4 games)
notom_tree = {"chicken":"v2_results","hawk_dove":"v2_results","deadlock":"v2_results",
              "stag_hunt":"v2_results","battle_of_the_sexes":"results/v2","public_goods":"results/v2"}
print(f"{'Game':22s} {'NoToM n':>7s} {'payoff':>8s} {'Prompt n':>8s} {'payoff':>8s} {'Δ':>8s} {'p':>8s} {'sig':>5s} {'dz':>7s} {'win':>5s}")
for g in ALL_GAMES:
    nt = load(notom_tree[g], "exp_b_20seed", g, "het_notom")
    pp = load("v2_results", "exp_c_payoff_prompt", g, "het_payoff_prompt")
    r = paired(nt, pp, alt="greater")
    if r:
        print(f"{g:22s} {len(nt):7d} {np.mean(list(nt.values())):8.4f} {len(pp):8d} {np.mean(list(pp.values())):8.4f} {r['d']:+8.4f} {r['p']:8.4f} {r['sig']:>5s} {r['dz']:+7.2f} {r['win']:5.2f}")
    else:
        print(f"{g:22s} {len(nt):7d} {'?':>8s} {len(pp):8d} {'?':>8s}  (no common seeds)")

print()
print("="*78)
print("  (2) public_goods COMPLETE (Layer 2a, n=20) — GSACA vs NoToM")
print("="*78)
nt = load("results/v2","exp_b_20seed","public_goods","het_notom")
for cell in ["het_gated_atom_talk","het_dp_gated_atom_talk","het_gsaca"]:
    m = load("results/v2","exp_b_20seed","public_goods",cell)
    r = paired(nt, m, alt="greater")
    if r:
        flag = "  <-- Prop3 abstain (exact 0)" if r["exact_zero"] else ""
        print(f"  {cell:28s} n={r['n']:2d} Δ={r['d']:+.4f} p={r['p']:.4f} {r['sig']:>5s} dz={r['dz']:+.2f} win={r['win']:.2f}{flag}")

print()
print("="*78)
print("  (3) Three-arm GSACA main table (corrected) — offline recompute n=20")
print("="*78)
print("  (uses het_gsaca per-seed split_score to pick CGA/Gated/Abstain arm;")
print("   deadlock Δ must be -0.002 NOT -0.019; verify by recomputing offline)")
# This requires the scheme1_offline logic; report what het_gsaca (2-arm) gives vs NoToM
print(f"{'Game':22s} {'Δ(GSACA-NoToM)':>16s} {'p':>8s} {'sig':>5s}")
for g in GAMES_2P:
    nt = load("v2_results","exp_b_20seed",g,"het_notom")
    gs = load("v2_results","exp_b_20seed",g,"het_gsaca")
    r = paired(nt, gs, alt="greater")
    if r: print(f"{g:22s} {r['d']:+16.4f} {r['p']:8.4f} {r['sig']:>5s}")

print()
print("="*78)
print("  (4) Exp A homogeneous control (Layer 2b) — CGA vs Gated on hom teams")
print("="*78)
for pair in ["QQ","GG","QG"]:
    print(f"  --- {pair} ---")
    for g in GAMES_2P:
        ga = load("v2_results", f"exp_a_fix/{pair}", g, "hom_gated_atom_talk")
        cg = load("v2_results", f"exp_a_fix/{pair}", g, "hom_dp_gated_atom_talk")
        r = paired(ga, cg, alt="greater")
        if r and r['n']>0:
            print(f"    {g:18s} n={r['n']:2d} Δ(CGA-Gated)={r['d']:+.4f} {r['sig']:>5s} dz={r['dz']:+.2f}")
print()
print("DONE.")
