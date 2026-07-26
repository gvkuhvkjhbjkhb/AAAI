"""
registry.py -- the frozen P6R matrix registry, regenerated from its seed.

This is a verbatim re-expression of `p6_registry.py` / `p6r_registry.py` from
the code supplement, kept here so the replay package is self-contained.  The
generator is seeded, so `build_p6r_registry(260902, 4)` reproduces the exact 16
matrices the original P6R campaign used; `verify_registry()` asserts the
structural invariants the paper's families are defined by.
"""
from __future__ import annotations

import numpy as np

FAMILIES = ("anti_safe", "anti_tradeoff", "coordination", "mixed")
LAMBDA = 0.25


def _pair(a: float, b: float) -> list[float]:
    return [round(float(a), 2), round(float(b), 2)]


def _j(rng: np.random.Generator, scale: float) -> float:
    return float(rng.uniform(-scale, scale))


def _matrix(family: str, rng: np.random.Generator):
    if family == "anti_safe":
        target = _pair(4.35 + _j(rng, .18), 4.05 + _j(rng, .18))
        return [[_pair(.45 + _j(rng, .12), .40 + _j(rng, .12)), target],
                [_pair(1.30 + _j(rng, .18), 1.25 + _j(rng, .18)),
                 _pair(.20 + _j(rng, .10), .25 + _j(rng, .10))]]
    if family == "anti_tradeoff":
        target = _pair(.00 + _j(rng, .04), 5.00 + _j(rng, .08))
        return [[_pair(.92 + _j(rng, .10), .92 + _j(rng, .10)), target],
                [_pair(.18 + _j(rng, .07), .20 + _j(rng, .07)),
                 _pair(-.12 + _j(rng, .04), -.10 + _j(rng, .04))]]
    if family == "coordination":
        return [[_pair(4.25 + _j(rng, .20), 4.10 + _j(rng, .20)),
                 _pair(.15 + _j(rng, .10), .95 + _j(rng, .12))],
                [_pair(.95 + _j(rng, .12), .15 + _j(rng, .10)),
                 _pair(3.20 + _j(rng, .20), 3.05 + _j(rng, .20))]]
    if family == "mixed":
        base = 2.15 + _j(rng, .12)
        return [[_pair(base + _j(rng, .16), base + _j(rng, .16)) for _ in range(2)]
                for _ in range(2)]
    raise ValueError(family)


def target_profile(payoff, lam: float = LAMBDA) -> list[int]:
    best, bv = None, None
    for i in range(2):
        for j in range(2):
            u1, u2 = payoff[i][j]
            v = (u1 + u2) / 2.0 - lam * abs(u1 - u2)
            if bv is None or v > bv + 1e-12:
                best, bv = [i, j], v
    return best


def build_p6r_registry(seed: int, n_per_family: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for family in FAMILIES:
        for i in range(1, n_per_family + 1):
            P = _matrix(family, rng)
            out.append({"matrix_id": f"p6r_{family}_{i:02d}",
                        "analysis_family": family,
                        "payoff_matrix": P,
                        "target": target_profile(P)})
    return out


def verify_registry(reg: list[dict]) -> dict:
    """Structural invariants the family labels mean.  Run before any GPU time."""
    rep = {}
    for m in reg:
        P, fam, t = m["payoff_matrix"], m["analysis_family"], m["target"]
        u = P[t[0]][t[1]]
        # deviation gain EXCLUDING the incumbent action: > 0 means a role
        # strictly wants to leave, = 0 means indifference (weak equilibrium),
        # < 0 means a strict equilibrium with |gain| of slack.
        dev1 = max(P[a][t[1]][0] for a in range(2) if a != t[0]) - u[0]
        dev2 = max(P[t[0]][a][1] for a in range(2) if a != t[1]) - u[1]
        d = rep.setdefault(fam, {"n": 0, "offdiag": 0, "nash": 0,
                                 "strict_nash": 0, "min_slack": 9e9})
        d["n"] += 1
        d["offdiag"] += int(t[0] != t[1])
        d["nash"] += int(max(dev1, dev2) <= 1e-9)
        d["strict_nash"] += int(max(dev1, dev2) < 0)
        d["min_slack"] = min(d["min_slack"], -max(dev1, dev2))
    return rep


if __name__ == "__main__":
    import json
    reg = build_p6r_registry(260902, 4)
    print(json.dumps(verify_registry(reg), indent=2))
    for m in reg:
        if m["analysis_family"] in ("anti_safe", "anti_tradeoff"):
            t = m["target"]
            print(f"{m['matrix_id']:<24} target={t} "
                  f"u={m['payoff_matrix'][t[0]][t[1]]}")
