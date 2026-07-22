"""
random_games_audit.py -- 0-GPU generalization audit for ActionSafe (review B2).

The reviewer's blocker B2: the paper only evaluates ActionSafe on hand-built
anti-safe / anti-tradeoff families that are *designed* to be separable by
(C1)-(C4), so 100%/0% there "verifies the certificate implements its own
definition." B2 asks for a family the authors did NOT hand-label -- randomly /
adversarially sampled 2x2 games -- with ActionSafe's false-permit / false-veto
rates reported there.

This audit is fully 0-GPU: the certificate verdict is a deterministic function
of the payoff table + warm-up reference, and here we use a table-only reference
proxy (expected payoff under uniform-random play) so no rollouts are needed.
Realized-utility generalization (does the LLM pair actually play a*?) DOES need
model inference and is specified in gpu_experiments/ instead.

KEY design choice that avoids the tautology: ground-truth labels are defined by
THRESHOLD-FREE semantics, independent of the certificate's eps margins:
  * GROUND-TRUTH SAFE    = a* is a strict pure NE AND both roles strictly
                           improve over the reference (u_i(a*) > rbar_i).
  * GROUND-TRUTH HARMFUL = some role is strictly worse than the reference
                           (u_i(a*) < rbar_i)  -- a genuine role trade-off.
  * (weak/borderline cases are 'ambiguous' and excluded from the two rates.)
Because these semantics do NOT use the certificate's (eps_T, eps_R, rho), the
error rates below are NOT built in -- they are an honest test of whether the
certificate's fixed margins generalize to games it never saw.

Reported quantities
-------------------
  false_veto      : SAFE games the certificate wrongly VETOES  (over-caution)
  false_permit    : HARMFUL games the certificate PERMITS      (tolerated harm)
  false_permit_unbounded : HARMFUL games with harm > eps_R that are permitted
                           -- MUST be ~0 (soundness; C2 rejects harm > eps_R)
  max_permitted_harm     : worst per-role shortfall among permitted targets
                           -- MUST be <= eps_R (soundness)
  prop1_holds_frac       : among (C2)-vetoed games with g>0, fraction where
                           favored_gain > 2g+eps_R (Proposition 1) -- MUST be 1.0
All rates come with game-clustered 95% bootstrap CIs (see cluster_ci.py).
"""
from __future__ import annotations
import numpy as np
from certificate import (actionsafe, target_profile, is_strict_pure_nash,
                         team_payoff, reference_uniform, EPS_R)
from cluster_ci import cluster_bootstrap_ci


def sample_uniform_game(rng) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(2, 2, 2))


def sample_integer_game(rng, lo=0, hi=5) -> np.ndarray:
    return rng.integers(lo, hi + 1, size=(2, 2, 2)).astype(float)


def sample_adversarial_game(rng, eps_R=EPS_R, tries=200):
    """Rejection-sample games whose target sits NEAR a certificate boundary
    (per-role margin within +/-0.15 of -eps_R, i.e. harm close to the C2 edge),
    where verdict errors are most likely -- an adversarial stress distribution."""
    for _ in range(tries):
        U = rng.uniform(0.0, 1.0, size=(2, 2, 2))
        rbar = reference_uniform(U)
        a1, a2 = target_profile(U)
        m = np.array([U[a1, a2, 0], U[a1, a2, 1]]) - rbar
        if np.any(np.abs(m + eps_R) < 0.15):     # margin near the C2 threshold
            return U
    return U                                      # fall back to last draw


def ground_truth_label(U, rbar):
    a1, a2 = target_profile(U)
    u = np.array([U[a1, a2, 0], U[a1, a2, 1]])
    harm = float(np.min(u - rbar))               # most negative role margin
    strict_ne = is_strict_pure_nash(U, a1, a2)
    if strict_ne and np.all(u - rbar > 0):
        return "safe", harm
    if np.any(u - rbar < 0):
        return "harmful", harm
    return "ambiguous", harm


def audit(sampler, n=2000, seed=0, ref_fn=reference_uniform):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        U = sampler(rng)
        rbar = ref_fn(U)
        label, harm = ground_truth_label(U, rbar)
        v = actionsafe(U, rbar)
        rows.append(dict(game=k, label=label, harm=harm, permit=v.permit,
                         failed=";".join(v.failed),
                         min_permit_margin=min(v.margins["role_margins"])))
    return rows


def summarize(rows, eps_R=EPS_R):
    import numpy as np
    lab = np.array([r["label"] for r in rows])
    permit = np.array([r["permit"] for r in rows])
    harm = np.array([r["harm"] for r in rows])
    n = len(rows)
    n_safe = int((lab == "safe").sum())
    n_harm = int((lab == "harmful").sum())
    n_amb = int((lab == "ambiguous").sum())

    # false-veto: among SAFE, fraction vetoed
    safe_mask = lab == "safe"
    fv = (~permit[safe_mask]).astype(float) if n_safe else np.array([])
    # false-permit: among HARMFUL, fraction permitted
    harm_mask = lab == "harmful"
    fp = (permit[harm_mask]).astype(float) if n_harm else np.array([])
    # unbounded false-permit: HARMFUL with harm < -eps_R that are permitted
    unb_mask = harm_mask & (harm < -eps_R)
    fp_unb = (permit[unb_mask]).astype(float) if unb_mask.sum() else np.array([])
    # worst per-role shortfall among ALL permitted targets (soundness)
    permitted_margins = [r["min_permit_margin"] for r in rows if r["permit"]]
    max_permit_harm = -min(permitted_margins) if permitted_margins else 0.0

    def rate_ci(x, groups):
        if len(x) == 0:
            return (float("nan"), (float("nan"), float("nan")), 0)
        lo, hi = cluster_bootstrap_ci(x, groups, n_boot=5000, seed=1)
        return (float(x.mean()), (lo, hi), len(x))

    fv_r = rate_ci(fv, np.arange(len(fv)))
    fp_r = rate_ci(fp, np.arange(len(fp)))
    fp_unb_r = rate_ci(fp_unb, np.arange(len(fp_unb)))
    return dict(
        n=n, n_safe=n_safe, n_harmful=n_harm, n_ambiguous=n_amb,
        false_veto=fv_r, false_permit=fp_r, false_permit_unbounded=fp_unb_r,
        max_permitted_harm=max_permit_harm, eps_R=eps_R,
    )


def prop1_soundness(sampler, n=4000, seed=7):
    """Fraction of (C2)-vetoed games with g>0 that satisfy Proposition 1."""
    from certificate import prop1_check
    rng = np.random.default_rng(seed)
    checked = 0; holds = 0
    for _ in range(n):
        U = sampler(rng)
        rbar = reference_uniform(U)
        res = prop1_check(U, rbar)
        if res is None:
            continue
        checked += 1
        holds += int(res["favored_gain_exceeds_bound"] and
                     res["harmed_loss_exceeds_epsR"] and res["identity_holds"])
    return checked, holds


def _fmt(rate_tuple):
    r, (lo, hi), nn = rate_tuple
    if nn == 0:
        return "n/a (0 cases)"
    return f"{r:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={nn})"


if __name__ == "__main__":
    samplers = {
        "uniform[0,1]": sample_uniform_game,
        "integer{0..5}": sample_integer_game,
        "adversarial(near-C2)": sample_adversarial_game,
    }
    print("=" * 74)
    print("0-GPU GENERALIZATION AUDIT of ActionSafe on non-hand-labeled games")
    print("ground truth is threshold-free (independent of eps margins)")
    print("=" * 74)
    for name, s in samplers.items():
        rows = audit(s, n=4000, seed=42)
        summ = summarize(rows)
        print(f"\n### distribution: {name}   (N={summ['n']} games)")
        print(f"  ground-truth mix: safe={summ['n_safe']}  "
              f"harmful={summ['n_harmful']}  ambiguous={summ['n_ambiguous']}")
        print(f"  false-veto  (safe wrongly vetoed)   : {_fmt(summ['false_veto'])}")
        print(f"  false-permit(harmful permitted, <=eps_R by design)")
        print(f"                                      : {_fmt(summ['false_permit'])}")
        print(f"  false-permit UNBOUNDED(harm>eps_R)  : {_fmt(summ['false_permit_unbounded'])}"
              f"   [soundness -> must be 0]")
        print(f"  max harm among PERMITTED targets    : {summ['max_permitted_harm']:.3f}"
              f"   [soundness -> must be <= eps_R={summ['eps_R']}]")
        chk, hold = prop1_soundness(s)
        frac = (hold / chk) if chk else float("nan")
        print(f"  Proposition 1 on (C2)-vetoed g>0    : {hold}/{chk} hold "
              f"({frac:.4f})   [must be 1.0]")
