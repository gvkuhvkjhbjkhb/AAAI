"""
forensics_reparse.py -- Phase 0: zero-GPU differential diagnosis.  RUN THIS FIRST.

The original P6R run archived joint ACTIONS but not raw completions, so we
cannot re-parse text.  But the actions themselves carry a lot: they say *which
profile* the pair actually played on those 4,000 committed steps, and the three
live hypotheses predict visibly different histograms.

  H1  ROLE / TRANSPOSE MIX-UP.  A convention slip between the runner and the
      metric (role order, or the role-relative matrix transpose) would show the
      pair concentrating on the TRANSPOSE of the target, (a*_2, a*_1).  On
      anti-safe that is (1,0), payoff about (1.30, 1.25).

  H2  TARGET INDEX MIX-UP.  A wrong target index shows mass concentrated on one
      fixed non-target cell with low entropy.

  H3  GENUINE NON-COMPLIANCE.  The pair stays in the warm-up basin (0,0),
      payoff about (0.45, 0.40), or spreads out with high entropy.

H1 and H2 are ANALYSIS bugs: they are fixable on CPU and would make the GPU
replay unnecessary -- the right response would be to correct the metric and
recompute, not to re-measure.  Only H3 leaves the serving stack as the
remaining explanation and justifies spending GPU time.

The archive layout varies between campaigns, so this reader accepts both known
shapes and reports what it found:
  A)  <root>/**/seed_<N>/<POLICY>/commit_trajectories.json     (P8 / P7C style)
  B)  <root>/**/<matrix>/seed_<N>/arms/<POLICY>/...            (P7 style)

Usage:
  python src/forensics_reparse.py --archive /path/to/P6R/results_live \
      --out out/PHASE0_FORENSICS.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import build_p6r_registry  # noqa: E402


def _iter_traj_files(root: Path):
    for p in root.rglob("*.json"):
        if p.name in {"commit_trajectories.json", "trajectories.json"}:
            yield p


def _steps_from(obj) -> list[dict]:
    """Flatten either {'episodes': [[step,...],...]} or {'commit_*': [[...]]}."""
    out = []
    if isinstance(obj, dict):
        for key in ("episodes", "commit_Gated", "commit", "commit_NoAlign"):
            if key in obj:
                for ep in obj[key]:
                    out.extend(ep)
                if out:
                    return out
    return out


def _meta_from_path(p: Path) -> dict:
    parts = list(p.parts)
    seed = next((int(re.sub(r"\D", "", x)) for x in parts
                 if x.startswith("seed_")), None)
    matrix = next((x for x in parts if "anti_safe" in x or "anti_tradeoff" in x
                   or "coordination" in x or "mixed" in x), None)
    policy = p.parent.name
    return {"matrix_id": matrix, "seed": seed, "policy": policy,
            "path": str(p)}


def entropy(counter: Counter, total: int) -> float:
    if total == 0:
        return 0.0
    h = 0.0
    for v in counter.values():
        if v:
            q = v / total
            h -= q * math.log(q, 2)
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True,
                    help="root of the original P6R results tree")
    ap.add_argument("--family", default="anti_safe")
    ap.add_argument("--policy-contains", default="Gated",
                    help="which arm to diagnose (default: the advised one)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    reg = {m["matrix_id"]: m for m in build_p6r_registry(260902, 4)}
    root = Path(a.archive)
    if not root.exists():
        raise SystemExit(f"archive not found: {root}")

    per_cell, files_seen = [], 0
    for p in _iter_traj_files(root):
        meta = _meta_from_path(p)
        if not meta["matrix_id"] or a.family not in meta["matrix_id"]:
            continue
        if a.policy_contains.lower() not in meta["policy"].lower():
            continue
        files_seen += 1
        steps = _steps_from(json.loads(p.read_text(encoding="utf-8")))
        if not steps:
            continue
        key = meta["matrix_id"]
        m = reg.get(key) or next((v for k, v in reg.items() if key.endswith(k[-2:])
                                  and v["analysis_family"] == a.family), None)
        if m is None:
            continue
        t = m["target"]
        c = Counter(tuple(s["actions"]) for s in steps)
        n = sum(c.values())
        per_cell.append({
            **meta, "target": t, "n_steps": n,
            "hist": {f"{i}{j}": c.get((i, j), 0) for i in range(2) for j in range(2)},
            "target_rate": c.get(tuple(t), 0) / n,
            "transpose_rate": c.get((t[1], t[0]), 0) / n,
            "diag00_rate": c.get((0, 0), 0) / n,
            "entropy_bits": entropy(c, n),
        })

    if not per_cell:
        raise SystemExit(
            f"no {a.family} / {a.policy_contains} trajectories found under {root} "
            f"({files_seen} candidate files scanned). Point --archive at the "
            f"directory that contains the per-cell trajectory JSONs.")

    n = len(per_cell)
    agg = {k: sum(c[k] for c in per_cell) / n
           for k in ("target_rate", "transpose_rate", "diag00_rate",
                     "entropy_bits")}
    total_hist = Counter()
    for c in per_cell:
        total_hist.update(c["hist"])
    tot = sum(total_hist.values()) or 1

    print("=" * 78)
    print(f"PHASE 0  archived-action forensics   family={a.family} "
          f"arm~{a.policy_contains}   cells={n}  steps={tot}")
    print("=" * 78)
    print("  joint-profile histogram (target is 01 on anti-safe):")
    for k in ("00", "01", "10", "11"):
        v = total_hist.get(k, 0)
        print(f"    {k}: {v:>7}  {v/tot:6.1%}")
    print(f"\n  mean target rate      {agg['target_rate']:.4f}")
    print(f"  mean transpose rate   {agg['transpose_rate']:.4f}   (H1)")
    print(f"  mean (0,0) basin rate {agg['diag00_rate']:.4f}   (H3)")
    print(f"  mean entropy          {agg['entropy_bits']:.3f} bits (max 2.0)")

    # ---- discriminate ----
    if agg["transpose_rate"] >= 0.60:
        h, verdict = "H1", (
            "The pair concentrated on the TRANSPOSE of the target. This is a "
            "role-order or matrix-transpose convention slip between the runner "
            "and the metric, i.e. an ANALYSIS bug. Correct the convention and "
            "recompute -- do NOT spend GPU time; the replay would answer a "
            "question that is not the one that went wrong.")
    elif (max(total_hist.get(k, 0) for k in ("00", "10", "11")) / tot >= 0.60
          and agg["entropy_bits"] < 0.8):
        h, verdict = "H2", (
            "Mass is concentrated on a single non-target cell with low entropy, "
            "which is what a wrong target index looks like. Check the target "
            "convention in the archived runner before spending GPU time.")
    else:
        h, verdict = "H3", (
            "The played profiles are consistent with genuine non-compliance "
            "(basin-seeking or high-entropy), not with a convention slip. The "
            "analysis-bug hypotheses are not supported, so the serving stack is "
            "the remaining explanation and the GPU replay is warranted.")
    print(f"\n  => {h}: {verdict}")

    out = {"family": a.family, "arm_filter": a.policy_contains,
           "n_cells": n, "n_steps": tot,
           "aggregate": agg, "total_histogram": dict(total_hist),
           "hypothesis": h, "recommendation": verdict,
           "per_cell": per_cell}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
