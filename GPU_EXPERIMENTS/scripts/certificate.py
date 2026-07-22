"""
certificate.py -- Reference implementation of the U-SafeSCA ActionSafe
certificate (C1-C4), the payoff-aware target a* (Eq. 1), and the
Proposition 1 veto-price check.

Everything here is a DETERMINISTIC function of a 2x2 payoff table plus a
warm-up reference vector. It requires NO GPU and NO model inference: given the
tables and references, the certificate's permit/veto verdict is fully
determined. This is the object the paper's Observation 1 and Proposition 1 are
about, and it is what the 0-GPU audits in this kit exercise.

Conventions
-----------
A 2x2 game is stored as U with shape (2, 2, 2):
    U[a1, a2, i] = payoff to role i when role 1 plays a1 and role 2 plays a2,
with a1, a2 in {0, 1} and i in {0, 1} (role index).

The warm-up reference is rbar with shape (2,): rbar[i] is role i's reference
reward (in the deployed system, the 95% bootstrap UPPER bound of role i's
NoAlign warm-up reward; for a table-only audit you may pass a payoff-table
proxy -- see reference_uniform() / reference_maxmin()).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

PROFILES = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Frozen margins from the paper (eps_T, eps_R, rho); lambda for the target.
EPS_T = 0.10
EPS_R = 0.25
RHO = 0.25
LAMBDA = 0.10


def team_payoff(U: np.ndarray, a1: int, a2: int) -> float:
    """u_T(a) = 1/2 (u_1 + u_2)."""
    return 0.5 * (U[a1, a2, 0] + U[a1, a2, 1])


def target_profile(U: np.ndarray, lam: float = LAMBDA) -> tuple[int, int]:
    """a* = argmax_a [ u_T(a) - lam * |u_1(a) - u_2(a)| ]   (Eq. 1).

    Ties are broken deterministically by profile order in PROFILES, so the map
    is a well-defined function of the table.
    """
    best, best_val = None, -np.inf
    for (a1, a2) in PROFILES:
        val = team_payoff(U, a1, a2) - lam * abs(U[a1, a2, 0] - U[a1, a2, 1])
        if val > best_val + 1e-12:
            best, best_val = (a1, a2), val
    return best


def is_pure_nash(U: np.ndarray, a1: int, a2: int, tol: float = 0.0) -> bool:
    """Pure-strategy Nash: no unilateral deviation strictly improves a role."""
    role1_ok = U[a1, a2, 0] >= U[1 - a1, a2, 0] - tol
    role2_ok = U[a1, a2, 1] >= U[a1, 1 - a2, 1] - tol
    return bool(role1_ok and role2_ok)


def is_strict_pure_nash(U: np.ndarray, a1: int, a2: int) -> bool:
    role1_ok = U[a1, a2, 0] > U[1 - a1, a2, 0]
    role2_ok = U[a1, a2, 1] > U[a1, 1 - a2, 1]
    return bool(role1_ok and role2_ok)


def reference_uniform(U: np.ndarray) -> np.ndarray:
    """Table-only proxy for the warm-up reference: each role's expected payoff
    under uniform-random joint play. Use ONLY for table-level audits; pass the
    real bootstrap-upper NoAlign reference when you have warm-up rollouts."""
    return U.mean(axis=(0, 1))


def reference_maxmin(U: np.ndarray) -> np.ndarray:
    """Alternative table-only proxy: each role's pure maxmin value."""
    r = np.zeros(2)
    # role 1 chooses a1 to maximize the worst-case over a2
    r[0] = max(min(U[a1, a2, 0] for a2 in (0, 1)) for a1 in (0, 1))
    r[1] = max(min(U[a1, a2, 1] for a1 in (0, 1)) for a2 in (0, 1))
    return r


@dataclass
class Verdict:
    permit: bool
    astar: tuple[int, int]
    failed: list[str]          # which of C1..C4 failed (empty if permitted)
    margins: dict              # diagnostic quantities


def actionsafe(U: np.ndarray, rbar: np.ndarray,
               eps_T: float = EPS_T, eps_R: float = EPS_R, rho: float = RHO,
               lam: float = LAMBDA) -> Verdict:
    """ActionSafe certificate on the payoff-aware target a* (Eq. 3).

    Permits iff all four hold:
      (C1) u_T(a*) - rbar_T   >= -eps_T           (team non-inferiority)
      (C2) u_i(a*) - rbar_i   >= -eps_R for all i (per-role non-inferiority)
      (C3) u_i(a*)            >=  rho    for all i (minimum role reward)
      (C4) a* is a pure Nash equilibrium of G       (deviation stability)
    where rbar_T = 1/2 (rbar_0 + rbar_1).
    """
    a1, a2 = target_profile(U, lam)
    uT = team_payoff(U, a1, a2)
    rbar_T = 0.5 * (rbar[0] + rbar[1])
    u = np.array([U[a1, a2, 0], U[a1, a2, 1]])

    c1 = (uT - rbar_T) >= -eps_T
    c2 = bool(np.all((u - rbar) >= -eps_R))
    c3 = bool(np.all(u >= rho))
    c4 = is_pure_nash(U, a1, a2)

    failed = []
    if not c1: failed.append("C1")
    if not c2: failed.append("C2")
    if not c3: failed.append("C3")
    if not c4: failed.append("C4")

    margins = dict(
        uT=uT, rbar_T=rbar_T, team_margin=uT - rbar_T,
        role_margins=(u - rbar).tolist(), roles=u.tolist(),
        forgone_team_gain=uT - rbar_T,
    )
    return Verdict(permit=(len(failed) == 0), astar=(a1, a2),
                   failed=failed, margins=margins)


def prop1_check(U: np.ndarray, rbar: np.ndarray,
                eps_R: float = EPS_R, lam: float = LAMBDA):
    """Numerically verify Proposition 1 on a single game whose target is
    vetoed through (C2) with a positive forgone team gain g > 0.

    Returns None if the hypotheses (C2-veto and g>0) do not hold, else a dict
    verifying:  favored role gain  >  2g + eps_R   and   harmed role loss > eps_R,
    and the exact identity  2g = sum_i (u_i(a*) - rbar_i).
    """
    a1, a2 = target_profile(U, lam)
    u = np.array([U[a1, a2, 0], U[a1, a2, 1]])
    m = u - rbar                      # per-role margins
    g = team_payoff(U, a1, a2) - 0.5 * (rbar[0] + rbar[1])
    c2_violated = bool(np.any(m < -eps_R))
    if not (g > 0 and c2_violated):
        return None
    i = int(np.argmin(m))             # most-harmed role
    j = 1 - i                         # favored role
    identity_lhs = 2 * g
    identity_rhs = float(m.sum())
    return dict(
        astar=(a1, a2), g=float(g), margins=m.tolist(),
        harmed_role=i, favored_role=j,
        identity_holds=bool(abs(identity_lhs - identity_rhs) < 1e-9),
        favored_gain=float(m[j]),
        bound_2g_plus_epsR=float(2 * g + eps_R),
        favored_gain_exceeds_bound=bool(m[j] > 2 * g + eps_R),
        harmed_loss=float(-m[i]),
        harmed_loss_exceeds_epsR=bool(-m[i] > eps_R),
    )


if __name__ == "__main__":
    # Observation 1 witness game from the paper (Figure 1): rbar = (1,1).
    # (L,R)=(0,1)->(3,2) strict eq, helps both; (R,L)=(1,0)->(4,0) harms role 2.
    U = np.zeros((2, 2, 2))
    U[0, 0] = (1, 1); U[0, 1] = (3, 2); U[1, 0] = (4, 0); U[1, 1] = (1, 1)
    rbar = np.array([1.0, 1.0])
    v = actionsafe(U, rbar)
    print("target a* =", v.astar, "(expect (0,1))")
    print("permit =", v.permit, "failed:", v.failed)
    print("strict NE at (0,1)?", is_strict_pure_nash(U, 0, 1))
    # Force a (C2) veto with g>0 to exercise Proposition 1:
    U2 = np.zeros((2, 2, 2))
    U2[0, 0] = (1, 1); U2[0, 1] = (0.2, 3.8); U2[1, 0] = (1, 1); U2[1, 1] = (1, 1)
    print("prop1:", prop1_check(U2, np.array([1.0, 1.0])))
