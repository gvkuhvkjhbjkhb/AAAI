"""
replay_runner.py -- Phase 2: the P6R replay under the current serving config.

Design notes that matter for the forensic purpose:

* TWO physical arms only.  `NoAlign` supplies the warm-up record W and the
  counterfactual commitment trajectory; `Gated` supplies the advised one.  The
  two gate policies (`LegacyGateFixed`, `ActionSafeFixed`) are read off those
  same sampled trajectories, so the gate comparison adds no model calls and
  cannot differ by sampling -- exactly the common-random-number construction
  P5/P6 used.

* CONCURRENCY is per cell, not per request.  Within a cell the episodes are
  sequential (each prompt shows the last `memory` steps) and the two roles are
  sequential within a step, so a cell is a chain of ~300 round trips.  The
  parallelism comes from running many cells at once; `--workers` is the number
  of concurrent cells and is what you tune.  With 64 workers each endpoint sees
  ~64 in-flight requests, which is a healthy vLLM batch for 12-token outputs.

* CHECKPOINTING is per cell.  A cell is written only when complete, so an
  interrupted run resumes by re-issuing the same command.

* DETERMINISM is not claimed.  Continuous batching makes bitwise reproduction
  impossible even with per-request seeds; that is why the protocol asks for
  three repeats of the decisive family rather than one "deterministic" run.
  Every request still carries a derived seed so the run is as reproducible as
  the serving stack allows.

Usage:
  python src/replay_runner.py --protocol protocols/p6r_replay_frozen.json \
      --out out/replay_r0 --workers 64 --repeat 0 \
      --p6r-code-dir /path/to/archived/p6r/code
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from certificates import action_certificate, legacy_certificate  # noqa: E402
from p6r_prompt import ParseFailure, resolve_prompt_impl         # noqa: E402
from registry import build_p6r_registry, verify_registry         # noqa: E402

try:
    import aiohttp
except ImportError:                                              # noqa: BLE001
    raise SystemExit("pip install aiohttp")


# --------------------------------------------------------------------------
def stable_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)                       # atomic: a cell file is never partial


# --------------------------------------------------------------------------
class Endpoints:
    """One bounded connection pool per role, so a slow endpoint throttles only
    its own role instead of starving the other."""

    def __init__(self, protocol: dict, workers: int):
        self.roles = protocol["model_pair"]["roles"]
        self.urls = {r: e.rstrip("/") + "/chat/completions"
                     for r, e in protocol["model_pair"]["endpoints"].items()}
        self.key = os.environ.get(
            protocol["model_pair"].get("api_key_env", "VLLM_API_KEY"), "")
        s = protocol["sampling"]
        self.body_base = {"temperature": s["temperature"], "top_p": s["top_p"],
                          "max_tokens": s["max_tokens"]}
        ex = protocol["execution"]
        self.timeout = aiohttp.ClientTimeout(total=ex["request_timeout_seconds"])
        self.retries = ex["max_retries"]
        self.backoff = ex["retry_backoff_seconds"]
        self.sem = {r: asyncio.Semaphore(workers + 8) for r in self.urls}
        self.session: aiohttp.ClientSession | None = None
        self.n_requests = 0
        self.n_retries = 0

    async def __aenter__(self):
        conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=600)
        self.session = aiohttp.ClientSession(connector=conn, timeout=self.timeout)
        return self

    async def __aexit__(self, *exc):
        assert self.session is not None
        await self.session.close()

    async def chat(self, role: int, messages: list[dict], seed: int) -> str:
        r = str(role)
        body = dict(self.body_base, model=self.roles[r], messages=messages,
                    seed=int(seed))
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.key}"}
        last: Exception | None = None
        async with self.sem[r]:
            for attempt in range(self.retries + 1):
                try:
                    assert self.session is not None
                    async with self.session.post(self.urls[r], json=body,
                                                 headers=headers) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                    self.n_requests += 1
                    return data["choices"][0]["message"]["content"]
                except Exception as e:                    # noqa: BLE001
                    last = e
                    self.n_retries += 1
                    await asyncio.sleep(self.backoff * (2 ** attempt))
        raise RuntimeError(f"role {role} endpoint failed after retries: {last}")


# --------------------------------------------------------------------------
async def run_arm(ep: Endpoints, m: dict, seed: int, arm: str, episodes: int,
                  protocol: dict, build_prompt, parse_action, repeat: int,
                  history_seed: list) -> list[list[dict]]:
    s = protocol["sampling"]
    horizon, memory = s["horizon"], s["memory"]
    stride = s["repeat_seed_stride"] * repeat
    history = list(history_seed)
    out = []
    for ep_i in range(episodes):
        steps = []
        for step in range(horizon):
            acts = []
            for role in (0, 1):
                msgs = build_prompt(m["payoff_matrix"], m["target"], role,
                                    history, arm, memory)
                gseed = (stable_seed(seed, ep_i, step, role, arm,
                                     m["matrix_id"]) + stride) % (2 ** 31 - 1)
                text = await ep.chat(role, msgs, gseed)
                try:
                    acts.append(parse_action(text))
                except ParseFailure:
                    raise ParseFailure(
                        f"{m['matrix_id']} seed={seed} arm={arm} ep={ep_i} "
                        f"step={step} role={role}: {text!r}")
            rewards = list(m["payoff_matrix"][acts[0]][acts[1]])
            rec = {"actions": acts, "rewards": rewards, "arm": arm}
            steps.append(rec)
            history.append(rec)
        out.append(steps)
    return out


def flat(eps: list[list[dict]]) -> list[dict]:
    return [x for e in eps for x in e]


def cell_metrics(warm: list[list[dict]], commit: list[list[dict]],
                 policy: str, target: list[int], intervened: bool) -> dict:
    def mean(steps):
        return float(np.mean([np.mean(x["rewards"]) for x in steps])) if steps else 0.0
    cm = flat(commit)
    return {
        "policy": policy,
        "intervened": bool(intervened),
        "warmup_team_mean_payoff": mean(flat(warm)),
        "commit_team_mean_payoff": mean(cm),
        "total_horizon_team_mean_payoff": mean(flat(warm) + cm),
        "commit_target_action_rate": float(
            np.mean([list(x["actions"]) == list(target) for x in cm])),
        "commit_steps": len(cm),
        "profile_histogram": {
            f"{i}{j}": int(sum(1 for x in cm if x["actions"] == [i, j]))
            for i in range(2) for j in range(2)},
        "test_time_probe_episodes": 0,
    }


async def run_cell(ep: Endpoints, protocol: dict, out_root: Path, m: dict,
                   seed: int, build_prompt, parse_action, repeat: int) -> str:
    cell_dir = out_root / m["matrix_id"] / f"seed_{seed}"
    done = cell_dir / "CELL.json"
    if done.exists():
        return "skip"

    s = protocol["sampling"]
    warm_eps = s["warmup_episodes"]
    commit_eps = s["total_episodes"] - warm_eps

    t0 = time.time()
    # --- warm-up (unmodified NoAlign) -------------------------------------
    warm = await run_arm(ep, m, seed, "NoAlign", warm_eps, protocol,
                         build_prompt, parse_action, repeat, [])
    warm_steps = flat(warm)
    warm_tail = warm_steps[-s["memory"]:]

    # --- certificates, computed from W exactly as the campaigns did -------
    leg = legacy_certificate(warm_steps, protocol["legacy_safety"],
                             stable_seed("leg", m["matrix_id"], seed, repeat))
    act = action_certificate(warm_steps, m, protocol["action_safety"],
                             stable_seed("act", m["matrix_id"], seed, repeat))

    # --- the two physical commitment arms ---------------------------------
    commit_noalign = await run_arm(ep, m, seed, "NoAlign", commit_eps, protocol,
                                   build_prompt, parse_action, repeat, warm_tail)
    commit_gated = await run_arm(ep, m, seed, "Gated", commit_eps, protocol,
                                 build_prompt, parse_action, repeat, warm_tail)

    # --- derived gate policies (no extra model calls) ----------------------
    tgt = m["target"]
    rows = {
        "NoAlign": cell_metrics(warm, commit_noalign, "NoAlign", tgt, False),
        "Gated": cell_metrics(warm, commit_gated, "Gated", tgt, True),
        "LegacyGateFixed": cell_metrics(
            warm, commit_gated if leg["safety_pass"] else commit_noalign,
            "LegacyGateFixed", tgt, leg["safety_pass"]),
        "ActionSafeFixed": cell_metrics(
            warm, commit_gated if act["safety_pass"] else commit_noalign,
            "ActionSafeFixed", tgt, act["safety_pass"]),
    }

    dump(cell_dir / "trajectories.json",
         {"warmup": warm, "commit_NoAlign": commit_noalign,
          "commit_Gated": commit_gated})
    dump(done, {"matrix_id": m["matrix_id"], "family": m["analysis_family"],
                "seed": seed, "repeat": repeat, "target": tgt,
                "payoff_matrix": m["payoff_matrix"],
                "legacy_certificate": leg, "action_certificate": act,
                "metrics": rows, "wall_seconds": round(time.time() - t0, 1)})
    return "done"


# --------------------------------------------------------------------------
async def main_async(a) -> int:
    protocol = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    if protocol["model_pair"].get("revision_lock_required"):
        for mdl, rev in protocol["model_pair"]["revisions"].items():
            if not rev or rev.startswith("PIN_EXACT_COMMIT"):
                raise SystemExit(f"revision not pinned for {mdl} -- see README 4.2")

    impl = resolve_prompt_impl(a.p6r_code_dir)
    if impl["provenance"]["reconstructed"] and not a.allow_reconstructed_prompt:
        raise SystemExit("Prompt/parser would be RECONSTRUCTED. Pass "
                         "--p6r-code-dir, or --allow-reconstructed-prompt to "
                         "accept it (the choice is recorded in RUN_MANIFEST).")

    reg = build_p6r_registry(protocol["p6r"]["registry_seed"],
                             protocol["p6r"]["n_per_family"])
    fams = a.families or protocol["p6r"]["families"]
    reg = [m for m in reg if m["analysis_family"] in fams]
    seeds = protocol["p6r"]["seeds"]

    cells = [(m, s) for m in reg for s in seeds]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        cells = [c for k, c in enumerate(cells) if k % n == i]

    out_root = Path(a.out)
    dump(out_root / "RUN_MANIFEST.json", {
        "campaign": protocol["campaign"],
        "repeat": a.repeat,
        "families": fams,
        "shard": a.shard,
        "workers": a.workers,
        "n_cells": len(cells),
        "protocol_sha256": hashlib.sha256(
            Path(a.protocol).read_bytes()).hexdigest(),
        "prompt_provenance": impl["provenance"],
        "registry_invariants": verify_registry(reg),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    print(f"cells to run: {len(cells)}  workers: {a.workers}  repeat: {a.repeat}")
    q: asyncio.Queue = asyncio.Queue()
    for c in cells:
        q.put_nowait(c)

    stats = {"done": 0, "skip": 0, "fail": 0}
    t0 = time.time()

    async with Endpoints(protocol, a.workers) as ep:
        async def worker(wid: int):
            while True:
                try:
                    m, seed = q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    r = await run_cell(ep, protocol, out_root, m, seed,
                                       impl["build_prompt"], impl["parse_action"],
                                       a.repeat)
                    stats[r] += 1
                except Exception as e:                    # noqa: BLE001
                    stats["fail"] += 1
                    dump(out_root / "FAILURES" /
                         f"{m['matrix_id']}_seed{seed}.json",
                         {"error": f"{type(e).__name__}: {e}"})
                finally:
                    q.task_done()
                    n = stats["done"] + stats["skip"] + stats["fail"]
                    if n % 10 == 0:
                        el = time.time() - t0
                        rate = stats["done"] / el * 3600 if el > 0 else 0
                        print(f"  {n}/{len(cells)} cells  "
                              f"({stats['done']} run, {stats['skip']} cached, "
                              f"{stats['fail']} failed)  "
                              f"{rate:.0f} cells/h  "
                              f"{ep.n_requests} reqs, {ep.n_retries} retries",
                              flush=True)

        await asyncio.gather(*(worker(i) for i in range(a.workers)))
        n_req, n_retry = ep.n_requests, ep.n_retries

    dump(out_root / "RUN_SUMMARY.json",
         {**stats, "requests": n_req, "retries": n_retry,
          "wall_seconds": round(time.time() - t0, 1),
          "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(json.dumps(stats), f"requests={n_req} retries={n_retry} "
          f"wall={time.time()-t0:.0f}s")
    return 1 if stats["fail"] else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=0,
                    help="seed-block index; 0 is the primary run")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--shard", default=None, help="i/N to split across hosts")
    ap.add_argument("--p6r-code-dir", default=None)
    ap.add_argument("--allow-reconstructed-prompt", action="store_true")
    a = ap.parse_args()
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    raise SystemExit(main())
