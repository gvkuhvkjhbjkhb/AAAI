"""
analyze_rollouts.py -- Analyze results.jsonl from run_generalization_rollouts.py
with the SAME cluster-robust estimators as the 0-GPU B3 recomputation.

Reports, per distribution:
  * realized effect vs NoAlign with wild-cluster / cluster-t 95% CIs and G
  * realization fidelity (mean fraction of steps on the advised target)
  * certificate-level rates on the SAME games (permit rate; and, using the
    realized behaviour, realized false-permit = permitted AND a role realized
    strictly below its NoAlign baseline)
This closes the loop: does a certificate-permitted target actually help on
games the authors did not hand-label?
"""
from __future__ import annotations
import json, sys, os, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cluster_ci import all_cis   # noqa: E402


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def analyze(path):
    recs = load(path)
    by_dist = collections.defaultdict(list)
    for r in recs:
        by_dist[r["dist"]].append(r)
    for dist, rs in by_dist.items():
        permit = np.array([r["permit"] for r in rs])
        eff = np.array([r["effect"] for r in rs])
        grp = np.array([r["matrix_id"] for r in rs])
        fid = np.array([r["fidelity"] for r in rs])
        # realized false-permit: permitted but some role realized < its baseline
        realized_harm = np.array([
            r["permit"] and min(r["role_effect"]) < 0 for r in rs])
        print(f"\n### {dist}  (rows={len(rs)}, matrices G={len(set(grp))})")
        print(f"  certificate permit rate         : {permit.mean():.3f}")
        # realized effect on PERMITTED (routed) rows only
        if permit.any():
            ci = all_cis(eff[permit], grp[permit])
            print(f"  realized effect | permitted     : mean {ci['mean']:+.3f}"
                  f"  wild-CI [{ci['wild_cluster'][0]:+.3f},"
                  f" {ci['wild_cluster'][1]:+.3f}]  G={ci['G']}")
            print(f"  realization fidelity | permitted: {fid[permit].mean():.3f}")
            print(f"  realized false-permit rate      : "
                  f"{realized_harm[permit].mean():.3f}"
                  f"   (permitted yet a role realized below baseline)")
        else:
            print("  (no permitted rows)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python analyze_rollouts.py results.jsonl")
        sys.exit(1)
    analyze(sys.argv[1])
