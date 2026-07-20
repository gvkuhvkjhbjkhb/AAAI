#!/usr/bin/env python3
"""P8 held-out calibration of an action-aligned intervention gate."""
from __future__ import annotations

import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import request

import numpy as np

from p8_shared import Matrix, action_certificate, dump, flat, matrix_text, stable_seed

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol" / "p8_gate_calibration_frozen.json"


def load(p: Path) -> Any: return json.loads(p.read_text(encoding="utf-8"))


def p8_registry(seed: int, n: int, families: list[str]) -> list[Matrix]:
    """Labels are saved for analysis but never passed to prompts or the gate."""
    rng = np.random.default_rng(seed); out = []
    for family in families:
        for k in range(1, n + 1):
            e = rng.uniform(-.16, .16, 8)
            if family == "anti_safe":
                p = (((.45+e[0], .45+e[1]), (4.35+e[2], 4.05+e[3])), ((1.15+e[4], 1.15+e[5]), (.20+e[6], .25+e[7])))
            elif family == "anti_tradeoff":
                p = (((.90+e[0], .88+e[1]), (-.02+e[2], 4.98+e[3])), ((.18+e[4], .20+e[5]), (-.12+e[6], -.08+e[7])))
            elif family == "boundary":
                # The first player is close to the frozen minimum-reward threshold.
                p = (((.70+e[0], .70+e[1]), (.125+e[2], 4.20+e[3])), ((.10+e[4], .10+e[5]), (.05+e[6], .08+e[7])))
            else: raise ValueError(family)
            out.append(Matrix(f"P8_{family}_{k:02d}", family, p))
    return out


def make_messages(m: Matrix, role: int, history: list[dict[str, Any]], advise: bool, memory: int) -> list[dict[str, str]]:
    own = "Row" if role == 0 else "Column"; target = "AB"[m.target()[role]]
    hist = "none" if not history else "; ".join(f"you={'AB'[s['actions'][role]]}, other={'AB'[s['actions'][1-role]]}" for s in history[-memory:])
    cue = f"Advisory: the transparent team-payoff rule recommends Action {target}. It is not forced." if advise else ""
    return [{"role": "system", "content": "Reply exactly ACTION: A or ACTION: B. No explanation."},
            {"role": "user", "content": f"You are {own} in a simultaneous two-action game. Entries are (your payoff, other payoff).\n{matrix_text(m, role)}\nRecent: {hist}\n{cue}\nChoose now."}]


def parse_action(raw: str) -> tuple[int, str]:
    upper = raw.upper()
    for token, value in (("ACTION: A", 0), ("ACTION:A", 0), ("ACTION: B", 1), ("ACTION:B", 1)):
        if token in upper: return value, "explicit_action"
    one = upper.strip().replace(".", "")
    if one in {"A", "0"}: return 0, "standalone_token"
    if one in {"B", "1"}: return 1, "standalone_token"
    import re
    m = re.search(r'\b(A|B|0|1)\b', upper)
    if m: return 0 if m.group(1) in {"A","0"} else 1, "loose_token"
    raise RuntimeError(f"unparseable completion: {raw!r}")


class Backend:
    def __init__(self, cfg: dict[str, Any], synthetic: bool):
        self.cfg, self.synthetic = cfg, synthetic
        self.models = [os.getenv("P8_ROW_MODEL", cfg["model_pair"]["row"]), os.getenv("P8_COL_MODEL", cfg["model_pair"]["column"])]
        self.urls = [os.getenv("P8_ROW_URL", ""), os.getenv("P8_COL_URL", "")]
        if not synthetic and not all(self.urls): raise RuntimeError("Set P8_ROW_URL and P8_COL_URL.")

    def complete(self, role: int, messages: list[dict[str, str]], req_seed: int, policy: str, advise: bool, m: Matrix) -> tuple[str, dict[str, Any]]:
        if self.synthetic:
            rng = np.random.default_rng(stable_seed("P8_synthetic", role, req_seed, policy, m.matrix_id))
            p = .20 if not advise else .72
            a = m.target()[role] if rng.random() < p else int(rng.integers(0, 2))
            return f"ACTION: {'AB'[a]}", {"synthetic": True, "model": self.models[role]}
        body = {"model": self.models[role], "messages": messages, "temperature": self.cfg["environment"]["temperature"], "top_p": self.cfg["environment"]["top_p"], "max_tokens": self.cfg["environment"]["max_tokens"], "seed": int(req_seed)}
        headers = {"Content-Type": "application/json"}; key = os.getenv("VLLM_API_KEY", "")
        if key: headers["Authorization"] = f"Bearer {key}"
        endpoint = self.urls[role].rstrip("/") + "/chat/completions"
        req = request.Request(endpoint, data=json.dumps(body).encode(), method="POST", headers=headers)
        with request.urlopen(req, timeout=180) as response: payload = json.loads(response.read().decode())
        return payload["choices"][0]["message"]["content"], {"endpoint": endpoint, "model": payload.get("model", self.models[role]), "usage": payload.get("usage", {})}


def rollout(backend: Backend, cfg: dict[str, Any], m: Matrix, seed: int, policy: str, advise: bool, episodes: int, phase: str) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    hist: list[dict[str, Any]] = []; eps: list[list[dict[str, Any]]] = []; raw_rows: list[dict[str, Any]] = []; env = cfg["environment"]
    for ep in range(episodes):
        steps = []
        for step in range(env["horizon"]):
            acts = []
            for role in range(2):
                req_seed = stable_seed("P8", m.matrix_id, seed, policy, phase, ep, step, role)
                msg = make_messages(m, role, hist, advise, env["memory_steps"]); raw, meta = backend.complete(role, msg, req_seed, policy, advise, m); action, matched = parse_action(raw); acts.append(action)
                raw_rows.append({"matrix_id": m.matrix_id, "seed": seed, "policy": policy, "phase": phase, "episode": ep, "step": step, "role": role, "request_seed": req_seed, "messages": msg, "raw_completion": raw, "parser": "canonical_v1", "parse_match_kind": matched, "parsed_action": action, "response_meta": meta})
            row = {"actions": acts, "rewards": list(m.payoff[acts[0]][acts[1]]), "policy": policy, "phase": phase}; hist.append(row); steps.append(row)
        eps.append(steps)
    return eps, raw_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows), encoding="utf-8")


def policy_advises(policy: str, decision: dict[str, Any], m: Matrix, seed: int, cfg: dict[str, Any]) -> bool:
    if policy == "NoAlign": return False
    if policy == "USafeSCA": return bool(decision["action_safety"]["safety_pass"])
    if policy == "AlwaysRecommend": return True
    if policy == "RateMatchedRandomGate":
        rng = np.random.default_rng(stable_seed(cfg["random_gate"]["seed_namespace"], m.matrix_id, seed)); return bool(rng.random() < cfg["random_gate"]["intervention_probability"])
    raise ValueError(policy)


def metric(warm: list[list[dict[str, Any]]], commit: list[list[dict[str, Any]]], m: Matrix, seed: int, policy: str, advise: bool) -> dict[str, Any]:
    cm = flat(commit); all_steps = flat(warm)+cm; target = m.target(); mean = lambda xs: float(np.mean([np.mean(x["rewards"]) for x in xs]))
    return {"matrix_id": m.matrix_id, "family": m.family, "seed": seed, "policy": policy, "intervened": advise, "target": list(target), "warmup_team_mean_payoff": mean(flat(warm)), "commit_team_mean_payoff": mean(cm), "total_horizon_team_mean_payoff": mean(all_steps), "commit_target_action_rate": float(np.mean([tuple(x["actions"]) == target for x in cm])), "commit_steps": len(cm)}


def run_context(cfg: dict[str, Any], out: Path, backend: Backend, m: Matrix, seed: int) -> None:
    base = out / "cells" / m.matrix_id / f"seed_{seed}"; complete = all((base / a / "metrics.json").exists() for a in cfg["arms"])
    if complete: return
    dump(base.parent / "matrix.json", m.audit())
    warm_path = base / "warmup_trajectories.json"
    if warm_path.exists(): warm = load(warm_path)["episodes"]
    else:
        warm, raw = rollout(backend, cfg, m, seed, "NoAlign", False, cfg["environment"]["warmup_episodes"], "warmup")
        dump(warm_path, {"episodes": warm}); write_jsonl(base / "warmup_raw_completions.jsonl", raw)
    decision_path = base / "decision.json"
    if decision_path.exists(): decision = load(decision_path)
    else:
        decision = {"matrix_id": m.matrix_id, "seed": seed, "action_safety": action_certificate(flat(warm), m, cfg["action_safety"], stable_seed("P8_cert", m.matrix_id, seed)), "labels_not_used_by_gate": True}
        dump(decision_path, decision)
    for policy in cfg["arms"]:
        path = base / policy / "metrics.json"
        if path.exists(): continue
        advise = policy_advises(policy, decision, m, seed, cfg)
        commit, raw = rollout(backend, cfg, m, seed, policy, advise, cfg["environment"]["commit_episodes"], "commit")
        dump(path, metric(warm, commit, m, seed, policy, advise)); dump(path.parent / "commit_trajectories.json", {"episodes": commit}); write_jsonl(path.parent / "raw_completions.jsonl", raw)


def preflight(cfg: dict[str, Any], out: Path, synthetic: bool) -> None:
    backend = Backend(cfg, synthetic); obj: dict[str, Any] = {"synthetic": synthetic, "configured_models": backend.models, "configured_urls": backend.urls, "responses": []}
    if synthetic: obj["ok"] = True
    else:
        for role, url in enumerate(backend.urls):
            endpoint = url.rstrip("/")+"/models"; headers = {}; key=os.getenv("VLLM_API_KEY", "")
            if key: headers["Authorization"] = f"Bearer {key}"
            with request.urlopen(request.Request(endpoint, headers=headers), timeout=60) as response: raw=json.loads(response.read().decode())
            ids=[x.get("id") for x in raw.get("data",[])]; obj["responses"].append({"role":role,"endpoint":endpoint,"model_ids":ids,"raw":raw,"configured_model_present":backend.models[role] in ids})
        obj["ok"] = all(x["configured_model_present"] for x in obj["responses"])
    dump(out / "P8_PREFLIGHT.json", obj)
    if not obj["ok"]: raise RuntimeError("Configured model missing from /v1/models; formal run blocked.")


def boot(values: list[float], draws: int, tag: str) -> dict[str, Any]:
    x=np.asarray(values,float)
    if not len(x): return {"mean":float("nan"),"ci95":[float("nan"),float("nan")],"n_contexts":0}
    rng=np.random.default_rng(stable_seed("P8_boot",tag)); b=x[rng.integers(0,len(x),(draws,len(x)))].mean(1)
    return {"mean":float(x.mean()),"ci95":[float(np.quantile(b,.025)),float(np.quantile(b,.975))],"n_contexts":len(x)}


def analyze(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    mats=p8_registry(cfg["registry"]["registry_seed"],cfg["registry"]["matrices_per_family"],cfg["registry"]["families"]); rows={}; missing=[]
    for m in mats:
        for seed in cfg["registry"]["seeds"]:
            for policy in cfg["arms"]:
                p=out/"cells"/m.matrix_id/f"seed_{seed}"/policy/"metrics.json"
                if not p.exists(): missing.append(str(p))
                else: rows[(m.matrix_id,seed,policy)]=load(p)
    res={"campaign":cfg["campaign"],"integrity":{"missing":missing,"expected_cells":len(mats)*len(cfg["registry"]["seeds"])*len(cfg["arms"])},"families":{}}
    for family in cfg["registry"]["families"]:
        mids=[m.matrix_id for m in mats if m.family==family]; res["families"][family]={}
        for policy in cfg["arms"]:
            fidelity=[]; effect=[]; intervention=[]
            for mid in mids:
                for seed in cfg["registry"]["seeds"]:
                    x=rows[(mid,seed,policy)]; no=rows[(mid,seed,"NoAlign")]; fidelity.append(x["commit_target_action_rate"]); effect.append(x["total_horizon_team_mean_payoff"]-no["total_horizon_team_mean_payoff"]); intervention.append(float(x["intervened"]))
            res["families"][family][policy]={"target_fidelity":boot(fidelity,cfg["analysis"]["bootstrap_resamples"],f"fid-{family}-{policy}"),"effect_vs_noalign":boot(effect,cfg["analysis"]["bootstrap_resamples"],f"eff-{family}-{policy}"),"intervention_rate":float(np.mean(intervention))}
    safe=res["families"]["anti_safe"]["USafeSCA"]; trade=res["families"]["anti_tradeoff"]["USafeSCA"]; res["discrimination_gap"]=safe["intervention_rate"]-trade["intervention_rate"]
    res["preregistered_success"]=bool(not missing and safe["intervention_rate"]>=.80 and safe["effect_vs_noalign"]["ci95"][0]>0 and trade["intervention_rate"]<=.10 and res["discrimination_gap"]>.70)
    res["interpretation_guardrail"]="A passing result supports selective intervention on this held-out matrix distribution and recorded serving stack only. AlwaysRecommend and RateMatchedRandomGate are mandatory comparators, not discardable ablations."
    dump(out/"P8_ANALYSIS.json",res); return res


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=["preflight","run","analyze"]); ap.add_argument("--out",type=Path,default=ROOT/"results_live"); ap.add_argument("--synthetic",action="store_true"); ap.add_argument("--workers",type=int,default=None); args=ap.parse_args(); cfg=load(PROTOCOL); args.out.mkdir(parents=True,exist_ok=True)
    if args.command=="preflight": preflight(cfg,args.out,args.synthetic); return
    if args.command=="analyze": print(json.dumps(analyze(cfg,args.out),indent=2)); return
    marker=args.out/"P8_PREFLIGHT.json"
    if not args.synthetic and (not marker.exists() or not load(marker).get("ok")): raise RuntimeError("Run successful preflight first.")
    backend=Backend(cfg,args.synthetic); mats=p8_registry(cfg["registry"]["registry_seed"],cfg["registry"]["matrices_per_family"],cfg["registry"]["families"]); tasks=[(m,s) for m in mats for s in cfg["registry"]["seeds"]]
    with ThreadPoolExecutor(max_workers=args.workers or cfg["execution"]["workers"]) as pool:
        futures=[pool.submit(run_context,cfg,args.out,backend,m,s) for m,s in tasks]
        for f in as_completed(futures): f.result()
    print(json.dumps(analyze(cfg,args.out),indent=2))


if __name__=="__main__": main()
