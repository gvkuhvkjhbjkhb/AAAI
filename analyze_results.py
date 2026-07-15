#!/usr/bin/env python3
"""Comprehensive analysis of all 4 experiment types in /data/lab/results/v2/."""
import json, os, glob, sys
from collections import defaultdict
import numpy as np
from datetime import datetime

V2 = "/data/lab/results/v2"

def load_all():
    """Load every metrics.json. Returns list of dicts with extracted fields."""
    rows = []
    for mpath in sorted(glob.glob(f"{V2}/**/metrics.json", recursive=True)):
        try:
            d = json.load(open(mpath))
        except: continue
        # infer experiment type from path
        rel = os.path.relpath(mpath, V2)
        exp_type = "unknown"
        if "exp_3arm" in rel:        exp_type = "T1_3arm"
        elif "exp_scheme2_silent" in rel: exp_type = "T2_silent"
        elif "exp_b_20seed" in rel:  exp_type = "T3_bos_pg"
        elif "exp_d_fix" in rel:     exp_type = "T4_D1_noise"
        elif "exp_e_fix" in rel:     exp_type = "T4_E_sweep"
        elif "exp_anti_test" in rel: exp_type = "T4_anti"
        # parse game/seed/cell from path components
        parts = rel.split("/")
        game, seed, cell = None, None, None
        for p in parts:
            if p.startswith("seed_"): seed = int(p.replace("seed_",""))
        # game is usually the dir before seed_
        for p in parts:
            if p in ("chicken","hawk_dove","deadlock","stag_hunt",
                     "battle_of_the_sexes","public_goods"):
                game = p
        # for 3-arm: path = exp_3arm/tau_X/<game>/seed_Y/<cell>/metrics.json
        # cell is the dir name right before metrics.json
        cell_dir = os.path.basename(os.path.dirname(mpath))
        cfg = d.get("config", {})
        row = {
            "exp_type": exp_type,
            "path": rel,
            "game": game or cfg.get("game"),
            "seed": seed or cfg.get("seed"),
            "cell": cell_dir,
            "cell_label": d.get("cell", cell_dir),
            "cooperation_payoff": d.get("cooperation_payoff"),
            "team_mean_payoff": d.get("team_mean_payoff"),
            "perspective_diversity": d.get("perspective_diversity"),
            "equilibrium_convergence": d.get("equilibrium_convergence"),
            "tom_prediction_accuracy": d.get("tom_prediction_accuracy"),
            "gsaca_detected_structure": d.get("gsaca_detected_structure"),
            "gsaca_oracle_structure": d.get("gsaca_oracle_structure"),
            "gsaca_detection_correct": d.get("gsaca_detection_correct"),
            "gsaca_split_score": d.get("gsaca_split_score"),
            "gsaca_3arm_selected": d.get("gsaca_3arm_selected"),
            "gsaca_3arm_tau": d.get("gsaca_3arm_tau"),
            "gate_trust_rate": d.get("gate_trust_rate"),
            "gated_prediction_accuracy": d.get("gated_prediction_accuracy"),
            "signal_accuracy": d.get("signal_accuracy"),
            "dp_conflict_rate": d.get("dp_conflict_rate"),
            "dp_intervention_rate": d.get("dp_intervention_rate"),
            "n_episodes": d.get("n_episodes"),
            "wall_time_s": d.get("wall_time_s"),
            "config_game": cfg.get("game"),
            "config_cell": cfg.get("cell_name"),
            "config_gate_trust_threshold": cfg.get("gate_trust_threshold"),
            "config_gate_ema_alpha": cfg.get("gate_ema_alpha"),
            "config_gsaca_warmup": cfg.get("gsaca_warmup_episodes"),
            "config_abstain_tau": cfg.get("abstain_tau"),
            "config_payoff_noise_std": cfg.get("payoff_noise_std"),
        }
        rows.append(row)
    return rows

def stats(vals):
    a = np.array([v for v in vals if v is not None and not (isinstance(v,float) and np.isnan(v))])
    if len(a) == 0: return {"n":0,"mean":None,"std":None,"ci95":None}
    mean = float(np.mean(a)); std = float(np.std(a, ddof=1)) if len(a)>1 else 0.0
    # 95% CI (bootstrap, n_boot=2000)
    rng = np.random.RandomState(42)
    boots = [float(np.mean(rng.choice(a, len(a)))) for _ in range(2000)]
    lo, hi = float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5))
    return {"n":len(a),"mean":mean,"std":std,"ci95":[lo,hi]}

def ttest_ind(a, b):
    """Welch's t-test. Returns (t, p, df, cohen_d)."""
    a = np.array([v for v in a if v is not None]); b = np.array([v for v in b if v is not None])
    if len(a)<2 or len(b)<2: return None
    ma, mb = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va/na + vb/nb)
    if se == 0: return {"t":0,"p":1.0,"df":na+nb-2,"cohen_d":0}
    t = (ma - mb) / se
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    # two-tailed p via normal approx (good for df>30; rough for small)
    from math import erf, sqrt
    p = 2 * (1 - 0.5*(1+erf(abs(t)/sqrt(2))))
    # Cohen's d (pooled)
    sp = np.sqrt(((na-1)*va + (nb-1)*vb)/(na+nb-2))
    d = (ma-mb)/sp if sp>0 else 0
    return {"t":float(t),"p":float(p),"df":float(df),"cohen_d":float(d)}

def fmt_s(st, key="mean", prec=3, ci=True):
    if st["n"]==0: return "n/a"
    s = f"{st[key]:.{prec}f}"
    if ci and st.get("ci95"):
        s += f" [{st['ci95'][0]:.{prec}f}, {st['ci95'][1]:.{prec}f}]"
    s += f" (n={st['n']})"
    return s

print("=" * 78)
print(f"  COMPREHENSIVE EXPERIMENT ANALYSIS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 78)
rows = load_all()
print(f"\nTotal metrics.json files loaded: {len(rows)}")
by_exp = defaultdict(list)
for r in rows: by_exp[r["exp_type"]].append(r)
print("By experiment type:")
for k in ["T1_3arm","T2_silent","T3_bos_pg","T4_D1_noise","T4_E_sweep","T4_anti"]:
    print(f"  {k:15s}: {len(by_exp[k])}")

# =========================================================================
print("\n" + "=" * 78)
print("  SECTION 1: T1 — THREE-ARM ABSTENTION GSACA (τ sweep)")
print("=" * 78)
t1 = by_exp["T1_3arm"]
if t1:
    # group by tau
    by_tau = defaultdict(list)
    for r in t1:
        tau = r["config_abstain_tau"]
        if tau is None: tau = r["gsaca_3arm_tau"]
        by_tau[tau].append(r)
    print(f"\n{'τ':>6} {'n':>4} {'payoff':>22} {'diversity':>22} {'conv':>10} {'arm_split':>20}")
    print("-"*90)
    tau_payoffs = {}
    for tau in sorted(by_tau.keys(), key=lambda x: (x is None, x)):
        rs = by_tau[tau]
        ps = stats([r["cooperation_payoff"] for r in rs])
        ds = stats([r["perspective_diversity"] for r in rs])
        cs = stats([r["equilibrium_convergence"] for r in rs])
        tau_payoffs[tau] = [r["cooperation_payoff"] for r in rs]
        # arm selection
        arms = [r["gsaca_3arm_selected"] for r in rs if r["gsaca_3arm_selected"]]
        arm_str = "/".join(f"{a}:{arms.count(a)}" for a in set(arms)) if arms else "-"
        print(f"{str(tau):>6} {ps['n']:>4} {fmt_s(ps,prec=3):>22} {fmt_s(ds,prec=3):>22} {fmt_s(cs,prec=3,ci=False):>10} {arm_str:>20}")
    # τ=0 vs τ=0.4 comparison (key test: abstention helps?)
    if 0.0 in tau_payoffs and 0.4 in tau_payoffs:
        tt = ttest_ind(tau_payoffs[0.4], tau_payoffs[0.0])
        print(f"\n  τ=0.4 vs τ=0.0 payoff: t={tt['t']:.3f}, p={tt['p']:.4f}, d={tt['cohen_d']:.3f} (Welch)")
    if 0.0 in tau_payoffs and 0.6 in tau_payoffs:
        tt = ttest_ind(tau_payoffs[0.6], tau_payoffs[0.0])
        print(f"  τ=0.6 vs τ=0.0 payoff: t={tt['t']:.3f}, p={tt['p']:.4f}, d={tt['cohen_d']:.3f}")
    # by game × tau
    print(f"\n  By game × τ (cooperation_payoff mean):")
    games = sorted(set(r["game"] for r in t1 if r["game"]))
    print(f"  {'game':<22}", end="")
    for tau in sorted(by_tau.keys(), key=lambda x: (x is None, x)):
        print(f"  τ={str(tau):<5}", end="")
    print()
    for g in games:
        print(f"  {g:<22}", end="")
        for tau in sorted(by_tau.keys(), key=lambda x: (x is None, x)):
            vals = [r["cooperation_payoff"] for r in t1 if r["game"]==g and (r["config_abstain_tau"]==tau or r["gsaca_3arm_tau"]==tau)]
            st = stats(vals)
            print(f"  {st['mean']:.3f}" if st["n"]>0 else f"  {'n/a':<6}", end="")
        print()
    # arm selection distribution
    print(f"\n  Arm selection distribution:")
    for tau in sorted(by_tau.keys(), key=lambda x: (x is None, x)):
        arms = [r["gsaca_3arm_selected"] for r in by_tau[tau] if r["gsaca_3arm_selected"]]
        dist = {a: arms.count(a) for a in set(arms)}
        print(f"    τ={tau}: {dist}")
    # GSACA detection accuracy
    correct = [r["gsaca_detection_correct"] for r in t1 if r["gsaca_detection_correct"] is not None]
    if correct:
        print(f"\n  GSACA structure detection accuracy: {sum(correct)}/{len(correct)} = {sum(correct)/len(correct)*100:.1f}%")

# =========================================================================
print("\n" + "=" * 78)
print("  SECTION 2: T2 — SILENT-ANTI-COORD GSACA")
print("=" * 78)
t2 = by_exp["T2_silent"]
if t2:
    print(f"\n  {'game':<14} {'n':>4} {'payoff':>22} {'diversity':>22} {'conv':>10}")
    print("  " + "-"*78)
    by_game = defaultdict(list)
    for r in t2: by_game[r["game"]].append(r)
    for g in sorted(by_game.keys()):
        rs = by_game[g]
        ps = stats([r["cooperation_payoff"] for r in rs])
        ds = stats([r["perspective_diversity"] for r in rs])
        cs = stats([r["equilibrium_convergence"] for r in rs])
        print(f"  {g:<14} {ps['n']:>4} {fmt_s(ps,prec=3):>22} {fmt_s(ds,prec=3):>22} {fmt_s(cs,prec=3,ci=False):>10}")

# =========================================================================
print("\n" + "=" * 78)
print("  SECTION 3: T3 — BoS + PUBLIC_GOODS 20-SEED (4 HET CELLS)")
print("=" * 78)
t3 = by_exp["T3_bos_pg"]
if t3:
    cells_order = ["het_notom","het_gated_atom_talk","het_dp_gated_atom_talk","het_gsaca","het_3arm"]
    games = sorted(set(r["game"] for r in t3 if r["game"]))
    for g in games:
        print(f"\n  Game: {g}")
        print(f"  {'cell':<26} {'n':>4} {'payoff':>24} {'diversity':>24} {'conv':>10} {'ToM_acc':>10}")
        print("  " + "-"*92)
        for cell in cells_order:
            rs = [r for r in t3 if r["game"]==g and (r["cell"]==cell or r["cell_label"]==cell)]
            if not rs: continue
            ps = stats([r["cooperation_payoff"] for r in rs])
            ds = stats([r["perspective_diversity"] for r in rs])
            cs = stats([r["equilibrium_convergence"] for r in rs])
            ts = stats([r["tom_prediction_accuracy"] for r in rs])
            print(f"  {cell:<26} {ps['n']:>4} {fmt_s(ps,prec=3):>24} {fmt_s(ds,prec=3):>24} {fmt_s(cs,prec=3,ci=False):>10} {fmt_s(ts,prec=3,ci=False):>10}")
        # GSACA vs het_notom baseline comparison
        baseline = [r["cooperation_payoff"] for r in t3 if r["game"]==g and (r["cell"]=="het_notom" or r["cell_label"]=="het_notom")]
        gsaca = [r["cooperation_payoff"] for r in t3 if r["game"]==g and (r["cell"]=="het_gsaca" or r["cell_label"]=="het_gsaca")]
        if len(baseline)>=2 and len(gsaca)>=2:
            tt = ttest_ind(gsaca, baseline)
            print(f"  → GSACA vs NoToM: t={tt['t']:.3f}, p={tt['p']:.4f}, d={tt['cohen_d']:.3f}")

# =========================================================================
print("\n" + "=" * 78)
print("  SECTION 4: T4-D1 — NOISE ROBUSTNESS SWEEP")
print("=" * 78)
t4d = by_exp["T4_D1_noise"]
if t4d:
    # parse noise from path or config
    by_noise = defaultdict(list)
    for r in t4d:
        noise = r["config_payoff_noise_std"]
        if noise is None:
            # try path: exp_d_fix/d1_nXX/...
            for p in r["path"].split("/"):
                if p.startswith("d1_n"):
                    mapping = {"d1_n00":0.0,"d1_n05":0.5,"d1_n10":1.0,"d1_n20":2.0}
                    noise = mapping.get(p)
        by_noise[noise].append(r)
    print(f"\n  {'noise':>6} {'n':>4} {'payoff':>24} {'diversity':>24} {'conv':>10} {'detect_acc':>12}")
    print("  " + "-"*86)
    for noise in sorted([n for n in by_noise.keys() if n is not None]):
        rs = by_noise[noise]
        ps = stats([r["cooperation_payoff"] for r in rs])
        ds = stats([r["perspective_diversity"] for r in rs])
        cs = stats([r["equilibrium_convergence"] for r in rs])
        correct = [r["gsaca_detection_correct"] for r in rs if r["gsaca_detection_correct"] is not None]
        acc = f"{sum(correct)/len(correct)*100:.1f}%" if correct else "-"
        print(f"  {noise:>6} {ps['n']:>4} {fmt_s(ps,prec=3):>24} {fmt_s(ds,prec=3):>24} {fmt_s(cs,prec=3,ci=False):>10} {acc:>12}")
    # noise=0 vs noise=2.0 comparison
    if 0.0 in by_noise and 2.0 in by_noise:
        a = [r["cooperation_payoff"] for r in by_noise[0.0]]
        b = [r["cooperation_payoff"] for r in by_noise[2.0]]
        tt = ttest_ind(a, b)
        print(f"\n  noise=0 vs noise=2.0 payoff: t={tt['t']:.3f}, p={tt['p']:.4f}, d={tt['cohen_d']:.3f}")

# =========================================================================
print("\n" + "=" * 78)
print("  SECTION 5: T4-E — HYPERPARAMETER SWEEP (θ/α/W)")
print("=" * 78)
t4e = by_exp["T4_E_sweep"]
if t4e:
    # group by sweep type using path
    theta_rs, alpha_rs, warmup_rs = [], [], []
    for r in t4e:
        if "theta_" in r["path"]: theta_rs.append(r)
        elif "alpha_" in r["path"]: alpha_rs.append(r)
        elif "warmup_" in r["path"]: warmup_rs.append(r)
    # θ sweep
    print(f"\n  --- Gate trust threshold (θ) sweep ---")
    if theta_rs:
        by_theta = defaultdict(list)
        for r in theta_rs:
            th = r["config_gate_trust_threshold"]
            if th is None:
                for p in r["path"].split("/"):
                    if p.startswith("theta_"): th = float(p.replace("theta_",""))
            by_theta[th].append(r)
        print(f"  {'θ':>6} {'n':>4} {'payoff':>24} {'gate_trust':>12} {'dp_conflict':>14}")
        print("  " + "-"*68)
        for th in sorted(by_theta.keys()):
            rs = by_theta[th]
            ps = stats([r["cooperation_payoff"] for r in rs])
            gs = stats([r["gate_trust_rate"] for r in rs if r["gate_trust_rate"] is not None])
            dps = stats([r["dp_conflict_rate"] for r in rs if r["dp_conflict_rate"] is not None])
            print(f"  {th:>6} {ps['n']:>4} {fmt_s(ps,prec=3):>24} {fmt_s(gs,prec=3,ci=False):>12} {fmt_s(dps,prec=3,ci=False):>14}")
    # α sweep
    print(f"\n  --- Gate EMA alpha (α) sweep ---")
    if alpha_rs:
        by_alpha = defaultdict(list)
        for r in alpha_rs:
            al = r["config_gate_ema_alpha"]
            if al is None:
                for p in r["path"].split("/"):
                    if p.startswith("alpha_"): al = float(p.replace("alpha_",""))
            by_alpha[al].append(r)
        print(f"  {'α':>6} {'n':>4} {'payoff':>24} {'dp_interv':>14}")
        print("  " + "-"*62)
        for al in sorted(by_alpha.keys()):
            rs = by_alpha[al]
            ps = stats([r["cooperation_payoff"] for r in rs])
            dis = stats([r["dp_intervention_rate"] for r in rs if r["dp_intervention_rate"] is not None])
            print(f"  {al:>6} {ps['n']:>4} {fmt_s(ps,prec=3):>24} {fmt_s(dis,prec=3,ci=False):>14}")
    # W sweep
    print(f"\n  --- GSACA warmup (W) sweep ---")
    if warmup_rs:
        by_w = defaultdict(list)
        for r in warmup_rs:
            w = r["config_gsaca_warmup"]
            if w is None:
                for p in r["path"].split("/"):
                    if p.startswith("warmup_"): w = int(p.replace("warmup_",""))
            by_w[w].append(r)
        print(f"  {'W':>6} {'n':>4} {'payoff':>24} {'detect_acc':>12}")
        print("  " + "-"*56)
        for w in sorted(by_w.keys()):
            rs = by_w[w]
            ps = stats([r["cooperation_payoff"] for r in rs])
            correct = [r["gsaca_detection_correct"] for r in rs if r["gsaca_detection_correct"] is not None]
            acc = f"{sum(correct)/len(correct)*100:.1f}%" if correct else "-"
            print(f"  {w:>6} {ps['n']:>4} {fmt_s(ps,prec=3):>24} {acc:>12}")

# =========================================================================
print("\n" + "=" * 78)
print("  SECTION 6: T4-ANTI — ANTI-COORDINATION CELL VARIANTS (deadlock)")
print("=" * 78)
t4a = by_exp["T4_anti"]
if t4a:
    print(f"\n  {'cell':<22} {'n':>4} {'payoff':>24} {'diversity':>24} {'conv':>10}")
    print("  " + "-"*78)
    by_cell = defaultdict(list)
    for r in t4a: by_cell[r["cell"]].append(r)
    baseline_payoffs = None
    for cell in ["het_gsaca","het_role_asym","het_hist_split","het_adapt_interv","het_combo_anti"]:
        rs = by_cell.get(cell, [])
        if not rs: continue
        ps = stats([r["cooperation_payoff"] for r in rs])
        ds = stats([r["perspective_diversity"] for r in rs])
        cs = stats([r["equilibrium_convergence"] for r in rs])
        print(f"  {cell:<22} {ps['n']:>4} {fmt_s(ps,prec=3):>24} {fmt_s(ds,prec=3):>24} {fmt_s(cs,prec=3,ci=False):>10}")
        if cell == "het_gsaca": baseline_payoffs = [r["cooperation_payoff"] for r in rs]
    if baseline_payoffs:
        print(f"\n  vs het_gsaca baseline (Welch t-test):")
        for cell in ["het_role_asym","het_hist_split","het_adapt_interv","het_combo_anti"]:
            rs = by_cell.get(cell, [])
            if len(rs) < 2: continue
            tt = ttest_ind([r["cooperation_payoff"] for r in rs], baseline_payoffs)
            sig = "***" if tt["p"]<0.001 else ("**" if tt["p"]<0.01 else ("*" if tt["p"]<0.05 else "ns"))
            print(f"    {cell:<22}: t={tt['t']:.3f}, p={tt['p']:.4f} {sig}, d={tt['cohen_d']:.3f}")

# =========================================================================
print("\n" + "=" * 78)
print("  SECTION 7: CROSS-EXPERIMENT SUMMARY")
print("=" * 78)
# GSACA detection accuracy across all
all_detect = [r["gsaca_detection_correct"] for r in rows if r["gsaca_detection_correct"] is not None]
if all_detect:
    print(f"\n  GSACA structure detection accuracy (all): {sum(all_detect)}/{len(all_detect)} = {sum(all_detect)/len(all_detect)*100:.1f}%")
# by game
by_game_detect = defaultdict(lambda: [0,0])
for r in rows:
    if r["gsaca_detection_correct"] is not None and r["game"]:
        by_game_detect[r["game"]][0] += int(r["gsaca_detection_correct"])
        by_game_detect[r["game"]][1] += 1
print(f"\n  Detection accuracy by game:")
for g in sorted(by_game_detect.keys()):
    ok, tot = by_game_detect[g]
    print(f"    {g:<22}: {ok}/{tot} = {ok/tot*100:.1f}%")

# Game-structure effect: anti-coord vs coord
print(f"\n  Game structure effect on diversity (all data):")
anti = [r["perspective_diversity"] for r in rows if r["gsaca_oracle_structure"]=="anti_coord" and r["perspective_diversity"] is not None]
coord = [r["perspective_diversity"] for r in rows if r["gsaca_oracle_structure"]=="coord" and r["perspective_diversity"] is not None]
if anti and coord:
    st_a = stats(anti); st_c = stats(coord)
    print(f"    anti_coord: {fmt_s(st_a,prec=3)}")
    print(f"    coord:      {fmt_s(st_c,prec=3)}")
    tt = ttest_ind(anti, coord)
    print(f"    t={tt['t']:.3f}, p={tt['p']:.4f}, d={tt['cohen_d']:.3f}")

# GSACA vs baselines: pooled across all games where het_notom exists
print(f"\n  GSACA improvement over baselines (pooled across all games/cells):")
for baseline_cell in ["het_notom","het_gated_atom_talk","het_dp_gated_atom_talk"]:
    base = [r["cooperation_payoff"] for r in rows if (r["cell"]==baseline_cell or r["cell_label"]==baseline_cell) and r["cooperation_payoff"] is not None]
    gsaca = [r["cooperation_payoff"] for r in rows if (r["cell"]=="het_gsaca" or r["cell_label"]=="het_gsaca") and r["cooperation_payoff"] is not None and r["exp_type"]!="T1_3arm"]
    if len(base)>=2 and len(gsaca)>=2:
        tt = ttest_ind(gsaca, base)
        sig = "***" if tt["p"]<0.001 else ("**" if tt["p"]<0.01 else ("*" if tt["p"]<0.05 else "ns"))
        print(f"    GSACA vs {baseline_cell:<22}: Δ={np.mean(gsaca)-np.mean(base):+.3f}, t={tt['t']:.3f}, p={tt['p']:.4f} {sig}, d={tt['cohen_d']:.3f}")

print("\n" + "=" * 78)
print("  ANALYSIS COMPLETE")
print("=" * 78)
