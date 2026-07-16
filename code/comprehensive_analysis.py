#!/usr/bin/env python3
"""Comprehensive analysis of ALL 4 experiment types.

Type 1: 3-arm abstention GSACA (exp_3arm) — τ sweep, arm selection, deadlock gap
Type 2: silent-anti-coord (exp_scheme2_silent) — silent vs baselines on anti-coord
Type 3: BoS+pg 20-seed (exp_b_20seed) — paired Wilcoxon, 4-cell comparison
Type 4: V2 recovery (exp_d_fix + exp_e_fix + exp_anti_test) — noise/hyperparam/anti
"""
import json, os, glob, sys, math
from collections import defaultdict, OrderedDict
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

V2 = "/data/lab/results/v2"
RNG = np.random.RandomState(42)

CELL_LABEL = {
    "het_notom": "NoToM",
    "het_gated_atom_talk": "Gated",
    "het_dp_gated_atom_talk": "CGA",
    "het_gsaca": "GSACA",
    "het_gsaca_silent": "Silent",
    "het_3arm": "3-arm",
    "het_role_asym": "RoleAsym",
    "het_hist_split": "HistSplit",
    "het_adapt_interv": "AdaptInterv",
    "het_combo_anti": "ComboAnti",
    "het_payoff_prompt": "PayoffPrompt",
}
CELL_ORDER = ["het_notom", "het_gated_atom_talk", "het_dp_gated_atom_talk", "het_gsaca"]

def safe(x):
    if x is None: return None
    if isinstance(x, float) and math.isnan(x): return None
    return float(x)

def desc(vals):
    a = np.array([v for v in vals if v is not None], float)
    if len(a) == 0: return None
    m = float(np.mean(a)); s = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    if len(a) > 1:
        boots = np.array([float(np.mean(RNG.choice(a, len(a)))) for _ in range(5000)])
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        lo = hi = m
    return {"n": len(a), "mean": m, "std": s, "ci": [lo, hi], "sem": s/math.sqrt(len(a)) if len(a)>1 else 0}

def fmt_d(d, prec=3):
    if d is None: return "  N/A"
    return f"{d['mean']:.{prec}f}±{d['std']:.{prec}f} [n={d['n']}]"

def fmt_ci(d, prec=3):
    if d is None: return "N/A"
    return f"{d['mean']:.{prec}f} [{d['ci'][0]:.{prec}f},{d['ci'][1]:.{prec}f}]"

def wilcoxon_paired(x, y):
    """Paired Wilcoxon signed-rank. x,y = paired arrays (same seeds)."""
    x = [v for v in x if v is not None]
    y = [v for v in y if v is not None]
    n = min(len(x), len(y))
    if n < 5: return None
    diff = np.array(x[:n]) - np.array(y[:n])
    diff = diff[diff != 0]
    if len(diff) < 5: return {"p": 1.0, "n": n, "stat": 0, "effect": 0.0}
    try:
        stat, p = wilcoxon(diff, alternative="greater")
    except Exception:
        return None
    eff = float(np.mean(diff))
    return {"p": float(p), "n": n, "stat": float(stat), "effect": eff}

def load_dir(base, pattern="*/*/*/metrics.json"):
    """Load all metrics from a directory. Returns list of (path_parts, dict)."""
    out = []
    for mpath in sorted(glob.glob(os.path.join(base, pattern))):
        try:
            d = json.load(open(mpath))
        except Exception:
            continue
        out.append((mpath, d))
    return out

def extract_game_seed_cell(mpath):
    parts = mpath.split("/")
    game = seed = cell = None
    for p in parts:
        if p in ("chicken","hawk_dove","deadlock","stag_hunt","battle_of_the_sexes","public_goods","matching_pennies"):
            game = p
        if p.startswith("seed_"):
            seed = int(p.replace("seed_",""))
        if p.startswith("het_"):
            cell = p
    return game, seed, cell

# ============================================================
print("=" * 70)
print("COMPREHENSIVE EXPERIMENT ANALYSIS — 4 Types, 831 metrics")
print("=" * 70)

# ============================================================
# TYPE 3: BoS + public_goods 20-seed (MAIN RESULTS)
# ============================================================
print("\n" + "=" * 70)
print("TYPE 3: BoS + public_goods 20-seed (n=20, paired)")
print("=" * 70)

t3_data = defaultdict(lambda: defaultdict(dict))  # game -> seed -> cell
for mpath, d in load_dir(f"{V2}/exp_b_20seed"):
    game, seed, cell = extract_game_seed_cell(mpath)
    if game and seed is not None and cell:
        t3_data[game][seed][cell] = d

for game in ["battle_of_the_sexes", "public_goods"]:
    print(f"\n--- {game} (n={len(t3_data[game])} seeds) ---")
    print(f"{'Cell':<12} {'coop_payoff':>22} {'persp_div':>22} {'eq_conv':>22} {'tom_acc':>22}")
    cell_vals = {}
    for cell in CELL_ORDER:
        vals = {m: [] for m in ["cooperation_payoff","perspective_diversity","equilibrium_convergence","tom_prediction_accuracy"]}
        for seed in sorted(t3_data[game]):
            d = t3_data[game][seed].get(cell)
            if d:
                for m in vals: vals[m].append(safe(d.get(m)))
        cell_vals[cell] = vals
        ds = {m: desc(vals[m]) for m in vals}
        print(f"{CELL_LABEL[cell]:<12} {fmt_d(ds['cooperation_payoff']):>22} {fmt_d(ds['perspective_diversity']):>22} {fmt_d(ds['equilibrium_convergence']):>22} {fmt_d(ds['tom_prediction_accuracy']):>22}")

    # Paired Wilcoxon: each cell vs NoToM (baseline)
    print(f"\n  Paired Wilcoxon (vs NoToM baseline, one-sided greater):")
    print(f"  {'Comparison':<25} {'Δmean':>8} {'p-value':>10} {'n':>4} {'sig':>5}")
    baseline = cell_vals["het_notom"]
    for cell in ["het_gated_atom_talk", "het_dp_gated_atom_talk", "het_gsaca"]:
        cv = cell_vals[cell]
        for metric in ["cooperation_payoff"]:
            seeds = sorted(set(t3_data[game]) & set(s for s in t3_data[game] if cell in t3_data[game][s] and "het_notom" in t3_data[game][s]))
            x = [safe(t3_data[game][s][cell].get(metric)) for s in seeds]
            y = [safe(t3_data[game][s]["het_notom"].get(metric)) for s in seeds]
            r = wilcoxon_paired(x, y)
            if r:
                sig = "***" if r["p"]<0.001 else "**" if r["p"]<0.01 else "*" if r["p"]<0.05 else "ns"
                print(f"  {CELL_LABEL[cell]:<25} {r['effect']:>+8.3f} {r['p']:>10.4f} {r['n']:>4} {sig:>5}")

    # GSACA structure detection
    gsaca_seeds = [s for s in t3_data[game] if "het_gsaca" in t3_data[game][s]]
    if gsaca_seeds:
        correct = sum(1 for s in gsaca_seeds if t3_data[game][s]["het_gsaca"].get("gsaca_detection_correct"))
        splits = [safe(t3_data[game][s]["het_gsaca"].get("gsaca_split_score")) for s in gsaca_seeds]
        sp = [x for x in splits if x is not None]
        print(f"\n  GSACA detection: {correct}/{len(gsaca_seeds)} correct, split_score={np.mean(sp):.3f}±{np.std(sp):.3f}")

# ============================================================
# TYPE 1: 3-arm abstention GSACA (τ sweep)
# ============================================================
print("\n" + "=" * 70)
print("TYPE 1: 3-arm abstention GSACA — τ sweep (4 games × 5 seeds × 4 τ)")
print("=" * 70)

t1_by_tau = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # tau -> game -> seed -> metrics
for tau_dir in sorted(glob.glob(f"{V2}/exp_3arm/tau_*")):
    tau = os.path.basename(tau_dir).replace("tau_", "")
    for mpath, d in load_dir(tau_dir):
        game, seed, cell = extract_game_seed_cell(mpath)
        if game and seed is not None:
            t1_by_tau[tau][game][seed] = d

print(f"\n{'τ':<6} {'Game':<20} {'coop_payoff':>22} {'arm':>15} {'det_correct':>12}")
for tau in sorted(t1_by_tau):
    for game in sorted(t1_by_tau[tau]):
        vals = [safe(d.get("cooperation_payoff")) for d in t1_by_tau[tau][game].values()]
        arms = [d.get("gsaca_3arm_selected","?") for d in t1_by_tau[tau][game].values()]
        arm_counts = defaultdict(int)
        for a in arms: arm_counts[a] += 1
        arm_str = ", ".join(f"{k}:{v}" for k,v in sorted(arm_counts.items()))
        correct = sum(1 for d in t1_by_tau[tau][game].values() if d.get("gsaca_detection_correct"))
        ds = desc(vals)
        print(f"{tau:<6} {game:<20} {fmt_d(ds):>22} {arm_str:>15} {correct}/{len(vals):>10}")

# Key: deadlock gap elimination at τ=0.4
print("\n  Deadlock: 3-arm τ=0.4 vs NoToM (gap elimination check):")
# Load baseline NoToM deadlock from exp_b_20seed or v2_results
deadlock_notom = []
for mpath, d in load_dir(f"{V2}/exp_anti_test"):
    game, seed, cell = extract_game_seed_cell(mpath)
    if game == "deadlock" and cell == "het_notom":
        deadlock_notom.append(safe(d.get("cooperation_payoff")))
# Also from v2_results
for mpath, d in load_dir(f"/data/lab/gsaca/v2_results/exp_b_20seed", "*/*/*/metrics.json"):
    game, seed, cell = extract_game_seed_cell(mpath)
    if game == "deadlock" and cell == "het_notom":
        deadlock_notom.append(safe(d.get("cooperation_payoff")))

deadlock_3arm_04 = [safe(d.get("cooperation_payoff")) for d in t1_by_tau.get("0.4",{}).get("deadlock",{}).values()]
deadlock_gsaca_old = []
for mpath, d in load_dir(f"/data/lab/gsaca/v2_results/exp_b_20seed", "*/*/*/metrics.json"):
    game, seed, cell = extract_game_seed_cell(mpath)
    if game == "deadlock" and cell == "het_gsaca":
        deadlock_gsaca_old.append(safe(d.get("cooperation_payoff")))

print(f"  NoToM baseline:      {fmt_d(desc(deadlock_notom))}")
print(f"  Old GSACA (2-arm):   {fmt_d(desc(deadlock_gsaca_old))}")
print(f"  3-arm τ=0.4:         {fmt_d(desc(deadlock_3arm_04))}")
if deadlock_3arm_04 and deadlock_notom:
    r = wilcoxon_paired(deadlock_3arm_04[:len(deadlock_notom)], deadlock_notom[:len(deadlock_3arm_04)])
    if r: print(f"  3-arm vs NoToM: Δ={r['effect']:+.4f}, p={r['p']:.4f} ({'***' if r['p']<0.001 else '**' if r['p']<0.01 else '*' if r['p']<0.05 else 'ns'})")

# ============================================================
# TYPE 2: silent-anti-coord
# ============================================================
print("\n" + "=" * 70)
print("TYPE 2: silent-anti-coord (3 anti-coord games × 20 seeds)")
print("=" * 70)

t2_data = defaultdict(lambda: defaultdict(dict))  # game -> seed -> cell
for mpath, d in load_dir(f"{V2}/exp_scheme2_silent"):
    game, seed, cell = extract_game_seed_cell(mpath)
    if game and seed is not None and cell:
        t2_data[game][seed][cell] = d

# Load baselines from v2_results for same games
t2_baseline = defaultdict(lambda: defaultdict(dict))
for mpath, d in load_dir(f"/data/lab/gsaca/v2_results/exp_b_20seed", "*/*/*/metrics.json"):
    game, seed, cell = extract_game_seed_cell(mpath)
    if game in ("chicken","hawk_dove","deadlock") and seed is not None and cell:
        t2_baseline[game][seed][cell] = d

print(f"\n{'Game':<15} {'Silent':>22} {'NoToM':>22} {'GSACA':>22} {'CGA':>22}")
for game in sorted(t2_data):
    sil = [safe(d.get("cooperation_payoff")) for d in t2_data[game].values()]
    nom = [safe(t2_baseline[game][s].get("cooperation_payoff")) for s in sorted(t2_baseline[game]) if "het_notom" in t2_baseline[game][s]]
    gsa = [safe(t2_baseline[game][s].get("cooperation_payoff")) for s in sorted(t2_baseline[game]) if "het_gsaca" in t2_baseline[game][s]]
    cga = [safe(t2_baseline[game][s].get("cooperation_payoff")) for s in sorted(t2_baseline[game]) if "het_dp_gated_atom_talk" in t2_baseline[game][s]]
    print(f"{game:<15} {fmt_d(desc(sil)):>22} {fmt_d(desc(nom)):>22} {fmt_d(desc(gsa)):>22} {fmt_d(desc(cga)):>22}")
    # Silent vs NoToM
    seeds_s = sorted(t2_data[game])
    if nom:
        r = wilcoxon_paired(sil[:len(nom)], nom[:len(sil)])
        if r: print(f"  {'':>15} Silent vs NoToM: Δ={r['effect']:+.4f}, p={r['p']:.4f} ({'***' if r['p']<0.001 else '**' if r['p']<0.01 else '*' if r['p']<0.05 else 'ns'})")

# ============================================================
# TYPE 4: V2 recovery (D1 noise + E hyperparam + anti_test)
# ============================================================
print("\n" + "=" * 70)
print("TYPE 4a: Noise robustness (exp_d_fix, d1 noise sweep)")
print("=" * 70)

t4d = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # noise_tag -> game -> seed -> metrics
for ndir in sorted(glob.glob(f"{V2}/exp_d_fix/d1_*")):
    ntag = os.path.basename(ndir)
    for mpath, d in load_dir(ndir):
        game, seed, cell = extract_game_seed_cell(mpath)
        if game and seed is not None:
            t4d[ntag][game][seed] = d

print(f"\n{'Noise':<10} {'Game':<20} {'coop_payoff':>22} {'persp_div':>22} {'eq_conv':>22}")
for ntag in sorted(t4d):
    for game in sorted(t4d[ntag]):
        vals_cp = [safe(d.get("cooperation_payoff")) for d in t4d[ntag][game].values()]
        vals_pd = [safe(d.get("perspective_diversity")) for d in t4d[ntag][game].values()]
        vals_ec = [safe(d.get("equilibrium_convergence")) for d in t4d[ntag][game].values()]
        print(f"{ntag:<10} {game:<20} {fmt_d(desc(vals_cp)):>22} {fmt_d(desc(vals_pd)):>22} {fmt_d(desc(vals_ec)):>22}")

print("\n" + "=" * 70)
print("TYPE 4b: Hyperparameter sensitivity (exp_e_fix)")
print("=" * 70)

t4e = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # hp_tag -> game -> seed -> metrics
for hdir in sorted(glob.glob(f"{V2}/exp_e_fix/*")):
    htag = os.path.basename(hdir)
    for mpath, d in load_dir(hdir):
        game, seed, cell = extract_game_seed_cell(mpath)
        if game and seed is not None:
            t4e[htag][game][seed] = d

print(f"\n{'Hyperparam':<15} {'Game':<20} {'coop_payoff':>22}")
for htag in sorted(t4e):
    for game in sorted(t4e[htag]):
        vals = [safe(d.get("cooperation_payoff")) for d in t4e[htag][game].values()]
        print(f"{htag:<15} {game:<20} {fmt_d(desc(vals)):>22}")

# Summary: range of means across hyperparams (sensitivity)
print("\n  Hyperparameter sensitivity summary (range of coop_payoff means):")
for game in sorted(set(g for h in t4e for g in t4e[h])):
    means = []
    for htag in sorted(t4e):
        if game in t4e[htag]:
            vals = [safe(d.get("cooperation_payoff")) for d in t4e[htag][game].values()]
            ds = desc(vals)
            if ds: means.append(ds["mean"])
    if means:
        print(f"  {game}: min={min(means):.3f} max={max(means):.3f} range={max(means)-min(means):.3f} (across {len(means)} configs)")

print("\n" + "=" * 70)
print("TYPE 4c: Anti-coord enhancement patches (exp_anti_test, deadlock)")
print("=" * 70)

t4a = defaultdict(lambda: defaultdict(dict))  # cell -> seed -> metrics
for mpath, d in load_dir(f"{V2}/exp_anti_test"):
    game, seed, cell = extract_game_seed_cell(mpath)
    if cell and seed is not None:
        t4a[cell][seed] = d

print(f"\n{'Cell':<20} {'coop_payoff':>22} {'persp_div':>22} {'eq_conv':>22}")
anti_cells = sorted(t4a)
for cell in anti_cells:
    vals_cp = [safe(d.get("cooperation_payoff")) for d in t4a[cell].values()]
    vals_pd = [safe(d.get("perspective_diversity")) for d in t4a[cell].values()]
    vals_ec = [safe(d.get("equilibrium_convergence")) for d in t4a[cell].values()]
    lbl = CELL_LABEL.get(cell, cell)
    print(f"{lbl:<20} {fmt_d(desc(vals_cp)):>22} {fmt_d(desc(vals_pd)):>22} {fmt_d(desc(vals_ec)):>22}")

# ============================================================
# OVERALL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)
total = 0
for mpath in glob.glob(f"{V2}/**/metrics.json", recursive=True):
    total += 1
print(f"Total metrics analyzed: {total}")
print(f"  Type 1 (3-arm):         {sum(1 for _ in glob.glob(f'{V2}/exp_3arm/**/metrics.json', recursive=True))}")
print(f"  Type 2 (silent):        {sum(1 for _ in glob.glob(f'{V2}/exp_scheme2_silent/**/metrics.json', recursive=True))}")
print(f"  Type 3 (BoS+pg 20seed): {sum(1 for _ in glob.glob(f'{V2}/exp_b_20seed/**/metrics.json', recursive=True))}")
print(f"  Type 4 (D+E+anti):      {sum(1 for _ in glob.glob(f'{V2}/exp_d_fix/**/metrics.json', recursive=True)) + sum(1 for _ in glob.glob(f'{V2}/exp_e_fix/**/metrics.json', recursive=True)) + sum(1 for _ in glob.glob(f'{V2}/exp_anti_test/**/metrics.json', recursive=True))}")
