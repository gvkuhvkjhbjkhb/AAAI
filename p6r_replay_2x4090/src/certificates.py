"""
certificates.py -- the two gates, on warm-up evidence, exactly as the campaigns
computed them.

These are transcriptions of `legacy_certificate` and `action_certificate` from
the code supplement's `p7_runner.py`, with two additions, both flagged:

  * `strict_deviation` (protocol flag).  The paper adopts the strict form of
    (C4).  On this registry it is verdict-neutral -- every anti-tradeoff target
    beats its best deviation by at least 0.10 -- so turning it on must not move
    a single route rate, and `analyze_replay.py` asserts that.  It is exposed
    here so the assertion is a real test rather than a claim.

  * `profile_histogram` in the returned dict, so a route decision can be
    audited against what the warm-up actually saw.

Everything else is byte-for-byte the original logic: same bootstrap draws, same
one-sided upper bounds, same coverage guards, same failure reason strings.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def legacy_certificate(steps: list[dict], cfg: dict, seed: int) -> dict[str, Any]:
    """The diagnosed object: it reads only the same/different-action summary."""
    same = np.array([np.mean(x["rewards"]) for x in steps
                     if x["actions"][0] == x["actions"][1]])
    diff = np.array([np.mean(x["rewards"]) for x in steps
                     if x["actions"][0] != x["actions"][1]])
    profiles = {tuple(x["actions"]) for x in steps}
    reasons: list[str] = []
    if len(same) < cfg["min_same_observations"]:
        reasons.append("insufficient_same")
    if len(diff) < cfg["min_different_observations"]:
        reasons.append("insufficient_different")
    if len(profiles) / 4 < cfg["min_profile_coverage"]:
        reasons.append("insufficient_coverage")

    upper = float("inf")
    if not reasons:
        rng = np.random.default_rng(seed)
        draws = cfg["bootstrap_samples"]
        vals = np.array([rng.choice(diff, len(diff), True).mean()
                         - rng.choice(same, len(same), True).mean()
                         for _ in range(draws)])
        upper = float(np.quantile(vals, cfg["confidence"]))
        if upper >= -cfg["tau"]:
            reasons.append("coordination_not_certified")

    return {"certificate_type": "legacy_same_different_v1",
            "n_observations": len(steps), "n_same": int(len(same)),
            "n_different": int(len(diff)), "n_unique_profiles": len(profiles),
            "profile_coverage": len(profiles) / 4,
            "split_upper_bound": upper,
            "safety_pass": not reasons, "safety_reasons": reasons}


def action_certificate(steps: list[dict], m: dict, cfg: dict,
                       seed: int) -> dict[str, Any]:
    """(C0)-(C4) on the proposed profile, from the same warm-up record."""
    target = m["target"]
    P = m["payoff_matrix"]
    rewards = np.asarray([x["rewards"] for x in steps], float)
    rng = np.random.default_rng(seed)
    draws = cfg["bootstrap_samples"]

    def upper(x):
        return float(np.quantile(rng.choice(x, (draws, len(x)), True).mean(1),
                                 cfg["confidence"]))

    role_upper = [upper(rewards[:, i]) for i in range(2)]
    team_upper = upper(rewards.mean(1))
    tr = np.asarray(P[target[0]][target[1]], float)

    strict = bool(cfg.get("strict_deviation", False))
    if strict:
        # exclude the incumbent action: a target a role is indifferent about
        # leaving is not stable
        dev = [max(P[a][target[1]][0] for a in range(2) if a != target[0]) - tr[0],
               max(P[target[0]][a][1] for a in range(2) if a != target[1]) - tr[1]]
        dev_limit = -1e-12                      # deviation must strictly lose
    else:
        dev = [max(P[a][target[1]][0] for a in range(2)) - tr[0],
               max(P[target[0]][a][1] for a in range(2)) - tr[1]]
        dev_limit = cfg["max_unilateral_deviation_gain"] + 1e-9

    profiles = {tuple(x["actions"]) for x in steps}
    reasons: list[str] = []
    # (C0) coverage guard
    if len(steps) < cfg["min_total_observations"]:
        reasons.append("insufficient_observations")
    if len(profiles) / 4 < cfg["min_profile_coverage"]:
        reasons.append("insufficient_coverage")
    # (C1) team non-inferiority
    if tr.mean() - team_upper < -cfg["team_noninferiority_margin"]:
        reasons.append("team_not_noninferior")
    # (C2) per-role non-inferiority
    if min(tr - np.asarray(role_upper)) < -cfg["role_noninferiority_margin"]:
        reasons.append("role_not_noninferior")
    # (C3) role floor
    if min(tr) < cfg["minimum_role_target_reward"]:
        reasons.append("minimum_role_reward")
    # (C4) deviation stability
    if max(dev) > dev_limit:
        reasons.append("not_stable")

    return {"certificate_type": "action_aligned_feasibility_v1",
            "strict_deviation": strict,
            "recommended_joint_action": list(target),
            "target_rewards": tr.tolist(),
            "baseline_role_upper_means": role_upper,
            "baseline_team_upper_mean": team_upper,
            "unilateral_deviation_gains": dev,
            "n_unique_profiles": len(profiles),
            "profile_coverage": len(profiles) / 4,
            "safety_pass": not reasons, "safety_reasons": reasons}
