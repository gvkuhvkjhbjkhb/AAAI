"""
B-3: Unknown-payoff realized instantiation (full protocol per EXPERIMENT_PLAN)
===============================================================================
Uses the TWO vLLM API server setup from B-1.

Protocol (per game, per K, per seed):
  1. Sample a true payoff table U (kept hidden from the certificate).
  2. Play K noalign probe rounds with the LLM pair; record per-step
     (action, payoff) tuples.
  3. ESTIMATE payoff table U_hat from the K probe rounds via Laplace-smoothed
     cell means.  Reference rbar = reference_uniform(U_hat).
  4. Run actionsafe(U_hat, rbar) on the ESTIMATED table.  This tests whether
     the certificate is robust to estimation noise (real LLM noise, not
     simulated).
  5. If PERMITTED, play a 5-step gated episode (warmup 2 + advice for the
     a*_hat target found on U_hat).  Record realized team effect vs a
     separate noalign baseline, and realization fidelity.
  6. If VETOED (noise makes the estimated table look unsafe), record as veto.

The key question B-3 answers (blocker B2 depth): does the certificate's
graceful-degradation result hold with REAL estimation noise from K probe rounds,
where K ∈ {5, 10, 20}?

B-3 differs from B-1 in EXACTLY one way: the certificate sees U_hat instead of
the true U.  Everything else (prompts, warmup, gating) is identical to B-1.
This is the protocol EXACTLY as described in EXPERIMENT_PLAN.md §B-3.

Speed: 20 matrices × 10 seeds × 3 values of K = 600 episodes per model pair.
With warm inference (~100ms/LLM call), ~2 gens per step × 5 steps = 10 calls
per episode → ~6000 calls → ~10 min.

Usage:
    python run_b3.py \
        --url1 http://localhost:8000/v1 --url2 http://localhost:8001/v1 \
        --model1 Qwen/Qwen2.5-7B-Instruct \
        --model2 THUDM/glm-4-9b-chat \
        --dist uniform --n_matrices 20 --seeds 10 \
        --out res_b3_qwen_glm_uniform.jsonl
"""
from __future__ import annotations
import argparse, json, sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from certificate import (actionsafe, target_profile, team_payoff,  # noqa: E402
                         reference_uniform, Verdict, PROFILES,
                         EPS_T, EPS_R, RHO, LAMBDA)
from random_games_audit import (sample_uniform_game, sample_integer_game,  # noqa
                                sample_adversarial_game)

HORIZON = 5
WARMUP_STEPS = 2
TEMPERATURE = 0.7
TOP_P = 0.9

PROMPT_TMPL = """You are Player {role} in a repeated 2x2 game. Each round you \
pick action A or B. Your payoff table (your points, other's points):
  If you pick A and they pick A: you {AA_you}, them {AA_them}
  If you pick A and they pick B: you {AB_you}, them {AB_them}
  If you pick B and they pick A: you {BA_you}, them {BA_them}
  If you pick B and they pick B: you {BB_you}, them {BB_them}
History so far: {history}
{advice}
Reply with exactly one letter, A or B."""


def make_client(base_url):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key="x")


def generate(client, model, prompt):
    r = client.chat.completions.create(
        model=model, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=4,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content


def parse_action(text):
    t = (text or "").strip().upper()
    for ch in t:
        if ch in "AB":
            return 0 if ch == "A" else 1
    return None


def fmt_prompt(U, role, history, advice_profile):
    def cell(a1, a2):
        you = U[a1, a2, role]; them = U[a1, a2, 1 - role]
        return you, them
    kw = {}
    for name, (a1, a2) in [("AA", (0, 0)), ("AB", (0, 1)),
                            ("BA", (1, 0)), ("BB", (1, 1))]:
        y, th = (cell(a1, a2) if role == 0 else cell(a2, a1))
        kw[f"{name}_you"] = f"{y:.2f}"; kw[f"{name}_them"] = f"{th:.2f}"
    advice = ""
    if advice_profile is not None:
        my = advice_profile[role]
        advice = f"Advice: a good joint plan is for you to pick {'A' if my==0 else 'B'}."
    hist = "; ".join(f"r{t}:{'A' if a==0 else 'B'}/{'A' if b==0 else 'B'}"
                     for t, (a, b) in enumerate(history)) or "none"
    return PROMPT_TMPL.format(role=role + 1, history=hist, advice=advice, **kw)


# ---------------------------------------------------------------------------
# Payoff estimation from probe rounds
# ---------------------------------------------------------------------------
def estimate_payoff_table(probe_trajectory, n_cells=4):
    """Laplace-smoothed empirical cell means from a K-round probe trajectory.

    probe_trajectory: list of ((a1, a2), (pay0, pay1))
    Returns U_hat of shape (2, 2, 2) with values in [0, 1] (clipped).
    """
    sums = np.ones((2, 2, 2))       # pseudo-count of 1 in each cell
    counts = 2 * np.ones((2, 2))    # denominator = count + 2 (for two pseudo-counts)
    for (a1, a2), (r0, r1) in probe_trajectory:
        sums[a1, a2] += [r0, r1]
        counts[a1, a2] += 1.0
    U_hat = np.zeros((2, 2, 2))
    for a1 in range(2):
        for a2 in range(2):
            U_hat[a1, a2] = sums[a1, a2] / counts[a1, a2]
    return np.clip(U_hat, 0.0, 10.0)


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------
def run_probe_episode(client1, client2, model1, model2, U, seed, K):
    """Collect K probe rounds (noalign) and return [(action, payoff), ...]."""
    rng = np.random.default_rng(seed)
    history = []
    trajectory = []
    for step in range(K):
        p1 = fmt_prompt(U, 0, history, None)
        p2 = fmt_prompt(U, 1, history, None)
        a1 = parse_action(generate(client1, model1, p1))
        a2 = parse_action(generate(client2, model2, p2))
        if a1 is None: a1 = int(rng.integers(2))
        if a2 is None: a2 = int(rng.integers(2))
        payoff = (float(U[a1, a2, 0]), float(U[a1, a2, 1]))
        trajectory.append(((a1, a2), payoff))
        history.append((a1, a2))
    return trajectory


def run_gated_episode(client1, client2, model1, model2, U, seed, advice_profile):
    """5-step episode with advice starting after warmup; returns
    (per_role_total, fidelity_fraction)."""
    rng = np.random.default_rng(seed)
    history = []
    totals = np.zeros(2)
    realized_target = 0
    for step in range(HORIZON):
        advice = None
        if step >= WARMUP_STEPS and advice_profile is not None:
            advice = advice_profile
        p1 = fmt_prompt(U, 0, history, advice)
        p2 = fmt_prompt(U, 1, history, advice)
        a1 = parse_action(generate(client1, model1, p1))
        a2 = parse_action(generate(client2, model2, p2))
        if a1 is None: a1 = int(rng.integers(2))
        if a2 is None: a2 = int(rng.integers(2))
        totals += [U[a1, a2, 0], U[a1, a2, 1]]
        if advice_profile is not None and (a1, a2) == tuple(advice_profile):
            realized_target += 1
        history.append((a1, a2))
    return totals / HORIZON, realized_target / HORIZON


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run(args):
    from openai import OpenAI
    client1 = make_client(args.url1)
    client2 = make_client(args.url2)
    samplers = {"uniform": sample_uniform_game,
                "integer": sample_integer_game,
                "adversarial": sample_adversarial_game}
    sampler = samplers[args.dist]
    rng = np.random.default_rng(args.gen_seed)
    probe_ks = [int(k) for k in args.probe_ks.split(",")]

    with open(args.out, "w") as f:
        for mi in range(args.n_matrices):
            U = sampler(rng)
            for s in range(args.seeds):
                for K in probe_ks:
                    seed = 1_000_000 * mi + 10_000 * s + K

                    # --- 1. Play K probe rounds (noalign) ---
                    probe_traj = run_probe_episode(
                        client1, client2, args.model1, args.model2,
                        U, seed, K)

                    # --- 2. Estimate payoff table from probes ---
                    U_hat = estimate_payoff_table(probe_traj)
                    rbar = reference_uniform(U_hat)

                    # --- 3. Certificate on ESTIMATED table ---
                    v = actionsafe(U_hat, rbar)

                    # --- 4. If permitted, play gated episode on TRUE table ---
                    if v.permit:
                        astar_hat = target_profile(U_hat)
                        ga, fid = run_gated_episode(
                            client1, client2, args.model1, args.model2,
                            U, seed + 1000, astar_hat)
                        # noalign baseline on TRUE table (separate seed)
                        na_traj = run_probe_episode(
                            client1, client2, args.model1, args.model2,
                            U, seed + 2000, HORIZON)
                        na_payoffs = np.array([pay for _, pay in na_traj])
                        na = na_payoffs.mean(axis=0) if len(na_payoffs) > 0 else np.zeros(2)
                    else:
                        ga = np.zeros(2)
                        fid = 0.0
                        na = np.zeros(2)

                    rec = dict(
                        matrix_id=mi, seed=s, dist=args.dist,
                        K=K,
                        permit=v.permit,
                        failed=v.failed,
                        astar_hat=list(v.astar) if v.permit else [],
                        team_noalign=float(na.mean()),
                        team_gated=float(ga.mean()),
                        effect=float(ga.mean() - na.mean()),
                        role_effect=[float(ga[i] - na[i]) for i in range(2)],
                        fidelity=fid,
                    )
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url1", default="http://localhost:8000/v1")
    ap.add_argument("--url2", default="http://localhost:8001/v1")
    ap.add_argument("--model1", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--model2", default="THUDM/glm-4-9b-chat")
    ap.add_argument("--dist", default="uniform",
                    choices=["uniform", "integer", "adversarial"])
    ap.add_argument("--probe_ks", default="5,10,20",
                    help="comma-separated list of K values")
    ap.add_argument("--n_matrices", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--gen_seed", type=int, default=99)
    ap.add_argument("--out", default="res_b3.jsonl")
    run(ap.parse_args())
