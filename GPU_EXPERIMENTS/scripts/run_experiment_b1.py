"""
B-1 experiment runner: uses TWO vLLM API servers on different GPUs.
Model1 on localhost:8000, Model2 on localhost:8001.
"""
from __future__ import annotations
import argparse, json, sys, os
import numpy as np
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from certificate import actionsafe, target_profile  # noqa: E402
from random_games_audit import (sample_uniform_game, sample_integer_game,
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


def play_episode(client1, client2, model1, model2, U, seed, arm, advice_profile):
    rng = np.random.default_rng(seed)
    history = []
    totals = np.zeros(2)
    realized_target = 0
    for step in range(HORIZON):
        advice = None
        if arm == "gated" and step >= WARMUP_STEPS and advice_profile is not None:
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


def run(args):
    client1 = make_client(args.url1)
    client2 = make_client(args.url2)
    sampler = {"uniform": sample_uniform_game, "integer": sample_integer_game,
               "adversarial": sample_adversarial_game}[args.dist]
    rng = np.random.default_rng(args.gen_seed)
    with open(args.out, "w") as f:
        for mi in range(args.n_matrices):
            U = sampler(rng)
            from certificate import reference_uniform
            rbar = reference_uniform(U)
            v = actionsafe(U, rbar)
            astar = target_profile(U)
            for s in range(args.seeds):
                seed = 10_000 * mi + s
                na, _ = play_episode(client1, client2, args.model1, args.model2,
                                     U, seed, "noalign", None)
                if v.permit:
                    ga, fid = play_episode(client1, client2,
                                           args.model1, args.model2,
                                           U, seed, "gated", astar)
                else:
                    ga, fid = na, 0.0
                rec = dict(matrix_id=mi, seed=s, dist=args.dist,
                           permit=v.permit, failed=v.failed, astar=list(astar),
                           team_noalign=float(na.mean()),
                           team_gated=float(ga.mean()),
                           effect=float(ga.mean() - na.mean()),
                           role_effect=[float(ga[0]-na[0]), float(ga[1]-na[1])],
                           fidelity=fid)
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
    ap.add_argument("--n_matrices", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--gen_seed", type=int, default=42)
    ap.add_argument("--out", default="results.jsonl")
    run(ap.parse_args())
