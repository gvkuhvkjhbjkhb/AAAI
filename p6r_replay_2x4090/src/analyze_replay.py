"""
analyze_replay.py -- Phase 3: faithfulness gates, then the pre-registered verdict.

Order matters and is enforced.  The faithfulness gates run FIRST and can
invalidate the run: they are authorization-level results that depend only on the
payoff table and the warm-up record, not on whether the pair complies, so they
must reproduce whatever the fidelity turns out to be.  Only if they pass is the
fidelity statistic computed and compared against the pre-registered thresholds.

The verdict string this prints is meant to be pasted into the paper as-is.

Usage:
  python src/analyze_replay.py --runs out/replay_r0 out/replay_r1 out/replay_r2 \
      --fingerprint out/FINGERPRINT_*.json --original-fidelity 0.041 \
      --out out/REPLAY_VERDICT.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def load_cells(run_dirs: list[str]) -> list[dict]:
    cells = []
    for d in run_dirs:
        for p in sorted(Path(d).rglob("CELL.json")):
            cells.append(json.loads(p.read_text(encoding="utf-8")))
    return cells


def matrix_clustered_bootstrap(values_by_matrix: dict[str, list[float]],
                               draws: int, seed: int, conf: float) -> dict:
    """Cluster by matrix: average within a matrix first, then resample matrices.
    The seeds within a matrix are not independent samples."""
    keys = sorted(values_by_matrix)
    per_matrix = np.array([np.mean(values_by_matrix[k]) for k in keys])
    if len(per_matrix) == 0:
        return {"mean": None, "ci95": [None, None], "n_matrices": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_matrix), size=(draws, len(per_matrix)))
    boot = per_matrix[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return {"mean": float(per_matrix.mean()),
            "ci95": [float(lo), float(hi)],
            "n_matrices": len(per_matrix),
            "per_matrix": {k: float(np.mean(values_by_matrix[k])) for k in keys}}


def collect(cells: list[dict], family: str, policy: str, field: str):
    out: dict[str, list[float]] = {}
    for c in cells:
        if c["family"] != family:
            continue
        out.setdefault(c["matrix_id"], []).append(c["metrics"][policy][field])
    return out


def route_rate(cells: list[dict], family: str, policy: str) -> tuple[int, int]:
    sel = [c for c in cells if c["family"] == family]
    return sum(int(c["metrics"][policy]["intervened"]) for c in sel), len(sel)


def faithfulness(cells: list[dict], fingerprints: list[dict]) -> dict:
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    a_hit, a_n = route_rate(cells, "anti_safe", "ActionSafeFixed")
    t_hit, t_n = route_rate(cells, "anti_tradeoff", "ActionSafeFixed")
    l_hit, l_n = route_rate(cells, "anti_safe", "LegacyGateFixed")

    add("ActionSafe permits every anti-safe target",
        a_n > 0 and a_hit == a_n, f"{a_hit}/{a_n}")
    if t_n:
        add("ActionSafe permits no anti-tradeoff target",
            t_hit == 0, f"{t_hit}/{t_n}")
    add("OldGate permits no anti-safe target", l_hit == 0, f"{l_hit}/{l_n}")

    fams = {c["family"] for c in cells}
    for f in fams:
        n = len([c for c in cells if c["family"] == f])
        add(f"cells present: {f}", n > 0, str(n))

    strict_flags = {c["action_certificate"].get("strict_deviation") for c in cells}
    add("strict-(C4) flag consistent across cells", len(strict_flags) == 1,
        str(strict_flags))

    if len(fingerprints) > 1:
        h = {f["behaviour"]["completion_sha256"] for f in fingerprints}
        add("canary completion hash identical across fingerprints",
            len(h) == 1, str(h))
        h2 = {f["behaviour"]["logprob_sha256_2dp"] for f in fingerprints}
        add("canary logprob hash identical across fingerprints",
            len(h2) == 1, str(h2))

    return {"all_pass": all(c["pass"] for c in checks), "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--fingerprint", nargs="*", default=[])
    ap.add_argument("--original-fidelity", type=float, default=0.041)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cells = load_cells(a.runs)
    if not cells:
        raise SystemExit("no CELL.json found under " + " ".join(a.runs))
    fps = [json.loads(Path(p).read_text(encoding="utf-8"))
           for pat in a.fingerprint for p in sorted(glob.glob(pat))]

    print("=" * 78)
    print(f"P6R REPLAY ANALYSIS   ({len(cells)} cells from {len(a.runs)} run(s))")
    print("=" * 78)

    # ---- 1. faithfulness first --------------------------------------------
    faith = faithfulness(cells, fps)
    print("\nFaithfulness gates (authorization-level; must hold regardless of "
          "fidelity)")
    for c in faith["checks"]:
        print(f"  [{'ok ' if c['pass'] else 'FAIL'}] {c['check']:<52} {c['detail']}")

    # ---- 2. the pre-registered statistic ----------------------------------
    fid = matrix_clustered_bootstrap(
        collect(cells, "anti_safe", "Gated", "commit_target_action_rate"),
        a.draws, seed=20260726, conf=0.95)
    print(f"\nPRIMARY  anti-safe / Gated / commit_target_action_rate")
    print(f"  replay : {fid['mean']:.3f}  95% CI "
          f"[{fid['ci95'][0]:.3f}, {fid['ci95'][1]:.3f}]  "
          f"(G={fid['n_matrices']} matrices)")
    print(f"  original P6R : {a.original_fidelity:.3f}")

    # secondary endpoints, reported whatever the verdict
    sec = {}
    for fam in sorted({c["family"] for c in cells}):
        for pol in ("Gated", "ActionSafeFixed"):
            k = f"{fam}/{pol}"
            sec[k] = {
                "fidelity": matrix_clustered_bootstrap(
                    collect(cells, fam, pol, "commit_target_action_rate"),
                    a.draws, 7, 0.95),
                "team_effect_vs_noalign": matrix_clustered_bootstrap(
                    {m: [x - y for x, y in zip(
                        collect(cells, fam, pol, "total_horizon_team_mean_payoff")[m],
                        collect(cells, fam, "NoAlign",
                                "total_horizon_team_mean_payoff")[m])]
                     for m in collect(cells, fam, pol,
                                      "total_horizon_team_mean_payoff")},
                    a.draws, 11, 0.95),
            }

    # profile histogram on anti-safe: WHERE the pair actually played
    hist = {"00": 0, "01": 0, "10": 0, "11": 0}
    for c in cells:
        if c["family"] == "anti_safe":
            for k, v in c["metrics"]["Gated"]["profile_histogram"].items():
                hist[k] += v
    tot = sum(hist.values()) or 1
    print("\n  anti-safe joint-profile histogram (Gated arm, target 01):")
    for k, v in hist.items():
        print(f"    {k}: {v:>6}  {v/tot:6.1%}")

    # ---- 3. verdict --------------------------------------------------------
    m, lo, hi = fid["mean"], fid["ci95"][0], fid["ci95"][1]
    if not faith["all_pass"]:
        verdict, note = "INVALID_RUN", (
            "A faithfulness gate failed. The replay does not reproduce the "
            "authorization-level results, so its fidelity number must not be "
            "used. Fix the replay before interpreting anything below.")
    elif m >= 0.90 and lo > 0.80:
        verdict, note = "A_environment_artifact", (
            f"Under the current serving config the same pinned pair realizes the "
            f"same authorized target on {m:.3f} [{lo:.3f}, {hi:.3f}] of committed "
            f"steps, against {a.original_fidelity:.3f} in the original campaign. "
            f"With matrices, prompt and parser already excluded (P5c), the "
            f"original cell is attributable to the serving stack. The paper "
            f"retracts the {a.original_fidelity:.3f} cell and the realized effect "
            f"computed from it, and reports this replay in its place.")
    elif m <= 0.20 and hi < 0.30:
        verdict, note = "B_reproduces", (
            f"The low-fidelity cell reproduces: {m:.3f} [{lo:.3f}, {hi:.3f}] under "
            f"the current serving config. Since three other campaigns on the same "
            f"pair and the same registry generator reach 1.000, this is a real "
            f"pair-by-campaign interaction and a stronger result than the paper "
            f"currently claims: target fidelity is not a property of the pair, of "
            f"the matrices, of the prompt, or of the parser.")
    else:
        verdict, note = "C_inconclusive", (
            f"The replay gives {m:.3f} [{lo:.3f}, {hi:.3f}], which falls between "
            f"the pre-registered thresholds. The paper keeps its present wording "
            f"-- an unexplained campaign-level effect -- and reports this interval "
            f"alongside it.")

    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")
    print("=" * 78)
    print(note)

    out = {"verdict": verdict, "paper_sentence": note,
           "faithfulness": faith, "primary": fid,
           "original_fidelity": a.original_fidelity,
           "anti_safe_profile_histogram": hist,
           "secondary": sec, "n_cells": len(cells),
           "fingerprints": [f.get("behaviour", {}).get("completion_sha256")
                            for f in fps]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0 if faith["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
