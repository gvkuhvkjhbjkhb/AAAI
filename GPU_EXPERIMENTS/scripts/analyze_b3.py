"""Analyze B-3 unknown-payoff results with cluster-robust CIs."""
from __future__ import annotations
import json, sys, os, collections
import numpy as np
sys.path.insert(0, "/data/lab")
from cluster_ci import all_cis

def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs

for dist in ["uniform", "integer", "adversarial"]:
    path = f"/data/lab/res_b3_qwen_glm_{dist}.jsonl"
    if not os.path.exists(path):
        continue
    recs = load(path)
    by_K = collections.defaultdict(list)
    for r in recs:
        by_K[r["K"]].append(r)

    print(f"\n{'='*70}")
    print(f"B-3: {dist}  (total rows={len(recs)})")
    print(f"{'='*70}")

    for K in sorted(by_K.keys()):
        rs = by_K[K]
        permit = np.array([r["permit"] for r in rs])
        eff = np.array([r["effect"] for r in rs])
        grp = np.array([r["matrix_id"] for r in rs])
        fid = np.array([r["fidelity"] for r in rs])
        realized_harm = np.array([
            r["permit"] and min(r["role_effect"]) < 0 for r in rs])

        print(f"\n--- K={K} probe rounds ---")
        print(f"  rows={len(rs)}, matrices G={len(set(grp))}")
        print(f"  permit rate (on U_hat):    {permit.mean():.3f}")

        if permit.any():
            ci = all_cis(eff[permit], grp[permit])
            print(f"  realized effect|permitted : mean {ci['mean']:+.3f}"
                  f"  wild-CI [{ci['wild_cluster'][0]:+.3f}, {ci['wild_cluster'][1]:+.3f}]"
                  f"  G_eff={ci['G']}")
            print(f"  fidelity | permitted:       {fid[permit].mean():.3f}")
            print(f"  false-permit rate:          {realized_harm[permit].mean():.3f}"
                  f"  (permitted yet role realized below baseline)")
        else:
            print("  (no permitted rows)")

    # Also report: comparison across K for permit rate change
    if len(by_K) >= 2:
        print(f"\n--- K-sweep summary ---")
        for K in sorted(by_K.keys()):
            rs = by_K[K]
            permit = np.array([r["permit"] for r in rs])
            eff = np.array([r["effect"] for r in rs])
            fid = np.array([r["fidelity"] for r in rs])
            print(f"  K={K:2d}: permit={permit.mean():.3f}, "
                  f"effect={eff[permit].mean():+.3f} (permitted only), "
                  f"fidelity={fid[permit].mean():.3f}")

print(f"\n{'='*70}")
print("Analysis complete")
