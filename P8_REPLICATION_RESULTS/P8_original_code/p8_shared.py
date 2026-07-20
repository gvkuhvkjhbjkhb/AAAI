#!/usr/bin/env python3
"""Standalone P7D/P7 runner.

P7D chooses one role-binding prompt on an isolated development registry.
P7 confirms that frozen prompt on a different registry.  Derived policies reuse
the same matched fixed-arm rollout; they never make extra LLM calls.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

import numpy as np


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(obj, sort_keys=True))
    import fcntl, tempfile
    for _attempt in range(30):
        try:
            fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    existing = path.read_text(encoding="utf-8").strip()
                    if existing:
                        old = json.loads(existing)
                        if old != normalized:
                            raise RuntimeError(f"immutable artifact differs: {path}")
                        return
                    tf = path.with_suffix(path.suffix + ".tmp")
                    tf.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    os.replace(tf, path)
                    return
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        except OSError:
            import time; time.sleep(0.1)
    raise RuntimeError(f"dump failed after retries: {path}")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**31 - 1)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Matrix:
    matrix_id: str
    family: str
    payoff: tuple[tuple[tuple[float, float], tuple[float, float]], tuple[tuple[float, float], tuple[float, float]]]

    def target(self) -> tuple[int, int]:
        # Transparent Gated rule: maximize team reward, then minimize inequity.
        candidates = []
        for r in range(2):
            for c in range(2):
                a, b = self.payoff[r][c]
                candidates.append(((a + b) / 2 - .25 * abs(a - b), r, c))
        _, r, c = max(candidates)
        return r, c

    def audit(self) -> dict[str, Any]:
        return {"matrix_id": self.matrix_id, "analysis_family": self.family, "payoff_matrix": self.payoff}


def registry(seed: int, n: int, families: list[str], prefix: str) -> list[Matrix]:
    """Deterministic, hidden analysis labels; labels never enter prompts/gates."""
    rng = np.random.default_rng(seed); out: list[Matrix] = []
    for family in families:
        for k in range(1, n + 1):
            eps = rng.uniform(-.18, .18, 8)
            if family == "anti_safe":
                # (A,B) is high, individually safe, and a strict equilibrium.
                p = (((.45+eps[0], .45+eps[1]), (4.35+eps[2], 4.05+eps[3])),
                     ((1.15+eps[4], 1.15+eps[5]), (.20+eps[6], .25+eps[7])))
            elif family == "anti_tradeoff":
                # Same target relation, but Row's target reward is below baseline.
                p = (((.90+eps[0], .88+eps[1]), (-.02+eps[2], 4.98+eps[3])),
                     ((.18+eps[4], .20+eps[5]), (-.12+eps[6], -.08+eps[7])))
            elif family == "coordination":
                p = (((4.15+eps[0], 4.05+eps[1]), (.15+eps[2], .95+eps[3])),
                     ((.95+eps[4], .15+eps[5]), (3.20+eps[6], 3.10+eps[7])))
            elif family == "mixed":
                p = (((2.10+eps[0], 2.15+eps[1]), (2.25+eps[2], 2.20+eps[3])),
                     ((2.20+eps[4], 2.10+eps[5]), (2.25+eps[6], 2.20+eps[7])))
            else: raise ValueError(family)
            out.append(Matrix(f"{prefix}_{family}_{k:02d}", family, p))
    return out


def matrix_text(m: Matrix, role: int) -> str:
    rows = []
    for own in range(2):
        cells = []
        for other in range(2):
            pair = m.payoff[own][other] if role == 0 else m.payoff[other][own]
            cells.append(f"({pair[0]:+.2f},{pair[1]:+.2f})")
        rows.append(" | ".join(cells))
    return "Other A | Other B\nSelf A " + rows[0] + "\nSelf B " + rows[1]


def prompt(m: Matrix, role: int, history: list[dict[str, Any]], intervention: str, memory: int) -> list[dict[str, str]]:
    target = m.target(); own, other = "Row", "Column"; assigned = "AB"[target[role]]; partner = "AB"[target[1-role]]
    hist = "none" if not history else "; ".join(f"you={'AB'[s['actions'][role]]}, other={'AB'[s['actions'][1-role]]}" for s in history[-memory:])
    cue = ""
    if intervention == "GatedBaseline":
        cue = f"Advisory: the transparent team-payoff rule recommends Action {assigned}. It is not forced."
    elif intervention == "RoleBoundConfirm":
        cue = f"Role-bound plan: you are {own}; commit to {assigned}. The other player ({other}) is assigned {partner}. Execute YOUR assigned action now."
    elif intervention == "RoleBoundAcknowledge":
        cue = f"Role-bound plan: you are {own}; your assigned action is {assigned}; the other player ({other}) is assigned {partner}. Internally verify the pair, then execute YOUR assigned action now."
    return [{"role":"system","content":"Reply exactly ACTION: A or ACTION: B. No explanation."},
            {"role":"user","content":f"You are {own} in a simultaneous two-action game. Entries are (your payoff, other payoff).\n{matrix_text(m,role)}\nRecent: {hist}\n{cue}\nChoose now."}]


def parse_action(text: str) -> int:
    upper = text.upper()
    for token, val in (("ACTION: A",0),("ACTION:A",0),("ACTION: B",1),("ACTION:B",1)):
        if token in upper: return val
    # Strict fallback for compliant one-token answers only.
    stripped = upper.strip().replace(".", "")
    if stripped in {"A", "0"}: return 0
    if stripped in {"B", "1"}: return 1
    raise RuntimeError(f"unparseable model response: {text!r}")


class Backend:
    def __init__(self, cfg: dict[str, Any], mode: str): self.cfg, self.mode = cfg, mode
    def choose(self, role: int, messages: list[dict[str,str]], seed: int, intervention: str, m: Matrix) -> int:
        if self.mode == "synthetic":
            rng = np.random.default_rng(stable_seed("synthetic", role, seed, intervention, m.matrix_id))
            target = m.target()[role]
            p = .20 if intervention == "NoAlign" else (.55 if intervention == "GatedBaseline" else .85)
            return target if rng.random() < p else int(rng.integers(0,2))
        model, endpoint = self.cfg["models"][role], self.cfg["endpoints"][role].rstrip("/") + "/chat/completions"
        key = os.environ.get(self.cfg.get("api_key_env", "VLLM_API_KEY"), "")
        body = {"model":model,"messages":messages,"temperature":.7,"top_p":.9,"max_tokens":20,"seed":int(seed)}
        req = request.Request(endpoint, data=json.dumps(body).encode(), method="POST", headers={"Content-Type":"application/json","Authorization":f"Bearer {key}"})
        with request.urlopen(req, timeout=180) as resp: data=json.loads(resp.read().decode())
        return parse_action(data["choices"][0]["message"]["content"])


def rollout(backend: Backend, m: Matrix, seed: int, intervention: str, episodes: int, horizon: int, memory: int) -> list[list[dict[str,Any]]]:
    history: list[dict[str,Any]]=[]; all_eps=[]
    for ep in range(episodes):
        steps=[]
        for step in range(horizon):
            acts=[backend.choose(role,prompt(m,role,history,intervention,memory),stable_seed(seed,ep,step,role,intervention,m.matrix_id),intervention,m) for role in range(2)]
            rewards=list(m.payoff[acts[0]][acts[1]]); record={"actions":acts,"rewards":rewards,"intervention":intervention}; steps.append(record); history.append(record)
        all_eps.append(steps)
    return all_eps


def flat(episodes: list[list[dict[str,Any]]]) -> list[dict[str,Any]]: return [x for e in episodes for x in e]


def legacy_certificate(steps: list[dict[str,Any]], cfg: dict[str,Any], seed: int) -> dict[str,Any]:
    same=np.array([np.mean(x["rewards"]) for x in steps if x["actions"][0]==x["actions"][1]]); diff=np.array([np.mean(x["rewards"]) for x in steps if x["actions"][0]!=x["actions"][1]])
    profiles={tuple(x["actions"]) for x in steps}; reasons=[]
    if len(same)<cfg["min_same_observations"]: reasons.append("insufficient_same")
    if len(diff)<cfg["min_different_observations"]: reasons.append("insufficient_different")
    if len(profiles)/4 < cfg["min_profile_coverage"]: reasons.append("insufficient_coverage")
    upper=float("inf")
    if not reasons:
        rng=np.random.default_rng(seed); draws=cfg["bootstrap_samples"]
        values=np.array([rng.choice(diff,len(diff),True).mean()-rng.choice(same,len(same),True).mean() for _ in range(draws)])
        upper=float(np.quantile(values,cfg["confidence"]))
        if upper >= -cfg["tau"]: reasons.append("coordination_not_certified")
    return {"certificate_type":"legacy_same_different_v1","n_observations":len(steps),"n_same":len(same),"n_different":len(diff),"n_unique_profiles":len(profiles),"profile_coverage":len(profiles)/4,"split_upper_bound":upper,"safety_pass":not reasons,"safety_reasons":reasons}


def action_certificate(steps: list[dict[str,Any]], m: Matrix, cfg: dict[str,Any], seed: int) -> dict[str,Any]:
    target=m.target(); rewards=np.asarray([x["rewards"] for x in steps],float); rng=np.random.default_rng(seed); draws=cfg["bootstrap_samples"]
    upper=lambda x:float(np.quantile(rng.choice(x,(draws,len(x)),True).mean(1),cfg["confidence"]))
    role_upper=[upper(rewards[:,i]) for i in range(2)]; team_upper=upper(rewards.mean(1)); tr=np.asarray(m.payoff[target[0]][target[1]],float)
    dev=[max(m.payoff[a][target[1]][0] for a in range(2))-tr[0],max(m.payoff[target[0]][a][1] for a in range(2))-tr[1]]
    profiles={tuple(x["actions"]) for x in steps}; reasons=[]
    if len(steps)<cfg["min_total_observations"]: reasons.append("insufficient_observations")
    if len(profiles)/4<cfg["min_profile_coverage"]: reasons.append("insufficient_coverage")
    if tr.mean()-team_upper < -cfg["team_noninferiority_margin"]: reasons.append("team_not_noninferior")
    if min(tr-np.asarray(role_upper)) < -cfg["role_noninferiority_margin"]: reasons.append("role_not_noninferior")
    if min(tr)<cfg["minimum_role_target_reward"]: reasons.append("minimum_role_reward")
    if max(dev)>cfg["max_unilateral_deviation_gain"]+1e-9: reasons.append("not_stable")
    return {"certificate_type":"action_aligned_feasibility_v1","recommended_joint_action":list(target),"target_rewards":tr.tolist(),"baseline_role_upper_means":role_upper,"baseline_team_upper_mean":team_upper,"unilateral_deviation_gains":dev,"safety_pass":not reasons,"safety_reasons":reasons}


def metrics(warm: list[list[dict[str,Any]]], commit: list[list[dict[str,Any]]], policy: str, selected: str, target: tuple[int,int]) -> dict[str,Any]:
    all_steps=flat(warm)+flat(commit); cm=flat(commit)
    mean=lambda s:float(np.mean([np.mean(x["rewards"]) for x in s]))
    fidelity=float(np.mean([tuple(x["actions"])==target for x in cm]))
    return {"policy":policy,"selected_intervention":selected,"warmup_team_mean_payoff":mean(flat(warm)),"commit_team_mean_payoff":mean(cm),"total_horizon_team_mean_payoff":mean(all_steps),"commit_target_action_rate":fidelity,"commit_steps":len(cm),"test_time_probe_episodes":0}


def run_cell(protocol: dict[str,Any], root: Path, phase: str, m: Matrix, seed: int, backend: Backend, selected_prompt: str|None=None) -> None:
    phase_cfg=protocol[phase]; base=root/phase.upper()/protocol["model_pair"]["profile"]/m.matrix_id/f"seed_{seed}"; dump(base/"matrix.json",m.audit())
    warm_path=base/"warmup.json"
    if warm_path.exists(): warm=load(warm_path)["episodes"]
    else:
        warm=rollout(backend,m,seed,"NoAlign",protocol["sampling"]["warmup_episodes"],protocol["sampling"]["horizon"],protocol["sampling"]["memory"])
        steps=flat(warm); legacy=legacy_certificate(steps,protocol["legacy_safety"],stable_seed("legacy",m.matrix_id,seed)); action=action_certificate(steps,m,protocol["action_safety"],stable_seed("action",m.matrix_id,seed))
        dump(warm_path,{"episodes":warm,"legacy_summary":legacy,"action_summary":action,"payoff_table_visible_to_all_policies":True})
    warm_obj=load(warm_path); target=m.target(); legacy=warm_obj["legacy_summary"]; action=warm_obj["action_summary"]
    if phase=="p7d": arms=phase_cfg["actual_arms"]
    else: arms=phase_cfg["actual_arms"]
    arm_map={"NoAlign":"NoAlign","GatedBaseline":"GatedBaseline","RoleBoundConfirm":"RoleBoundConfirm","RoleBoundAcknowledge":"RoleBoundAcknowledge","RoleBoundSelected":selected_prompt}
    for policy in arms:
        path=base/"arms"/policy/"metrics.json"
        if path.exists(): continue
        actual=arm_map[policy]; commit=rollout(backend,m,seed,actual,protocol["sampling"]["commit_episodes"],protocol["sampling"]["horizon"],protocol["sampling"]["memory"])
        dump(path,metrics(warm,commit,policy,actual,target)); dump(path.parent/"commit_trajectories.json",{"episodes":commit}); dump(path.parent/"decision.json",{"policy":policy,"selected_intervention":actual,"legacy_safety":legacy,"action_safety":action,"test_time_probe_episodes":0})
    if phase=="p7":
        for policy, allow in {"ActionSafeFixed":action["safety_pass"],"OldGateFixed":legacy["safety_pass"]}.items():
            path=base/"arms"/policy/"metrics.json"
            if path.exists(): continue
            selected="RoleBoundSelected" if allow else "NoAlign"; src=base/"arms"/selected/"metrics.json"; obj=dict(load(src)); obj.update({"policy":policy,"selected_intervention":selected,"counterfactual_source_policy":selected,"counterfactual_common_random_numbers":True})
            dump(path,obj); dump(path.parent/"decision.json",{"policy":policy,"selected_intervention":selected,"reason":"action_aligned_gate" if policy.startswith("Action") else "legacy_relation_gate","legacy_safety":legacy,"action_safety":action,"test_time_probe_episodes":0})


def bootstrap(matrix_values: dict[str,list[float]], draws: int, seed: int) -> dict[str,Any]:
    means=np.asarray([np.mean(v) for _,v in sorted(matrix_values.items())],float)
    if not len(means): return {"mean":float("nan"),"ci95":[float("nan"),float("nan")],"n_matrices":0}
    rng=np.random.default_rng(seed)
    # Vectorized resampling is identical to repeated clustered bootstrap draws,
    # but avoids a Python-loop bottleneck during the 20,000-draw final analysis.
    b=means[rng.integers(0,len(means),size=(draws,len(means)))].mean(axis=1)
    return {"mean":float(means.mean()),"ci95":[float(np.quantile(b,.025)),float(np.quantile(b,.975))],"n_matrices":len(means)}


def analyze(protocol: dict[str,Any], root: Path, phase: str) -> dict[str,Any]:
    base=root/phase.upper()/protocol["model_pair"]["profile"]; rows=[]; missing=[]
    cfg=protocol[phase]; fams=["anti_safe"] if phase=="p7d" else protocol["p7"]["families"]
    n=cfg.get("n_anti_safe_matrices",cfg.get("n_per_family")); expected=registry(cfg["registry_seed"],n,fams,phase)
    for m in expected:
        for seed in cfg["seeds"]:
            cell=base/m.matrix_id/f"seed_{seed}"/"arms"/"NoAlign"/"metrics.json"
            if not cell.exists(): missing.append(str(cell))
    for mp in sorted(base.glob("*")):
        for sp in sorted(mp.glob("seed_*")):
            try:
                mat=load(sp/"matrix.json"); no=load(sp/"arms"/"NoAlign"/"metrics.json")
                rows.append((mat,sp,no))
            except Exception: missing.append(str(sp))
    if phase=="p7d":
        result={"phase":"p7d","integrity":{"missing":missing},"variants":{}}
        for arm in protocol["p7d"]["candidate_prompts"]:
            vals={}
            for mat,sp,_ in rows: vals.setdefault(mat["matrix_id"],[]).append(load(sp/"arms"/arm/"metrics.json")["commit_target_action_rate"])
            result["variants"][arm]=bootstrap(vals,protocol["p7d"]["selection_bootstrap_draws"],stable_seed("p7d",arm))
        valid=[(v["mean"],name,v) for name,v in result["variants"].items() if v["ci95"][0]>=.50]
        result["selected_prompt"]=sorted(valid,key=lambda x:(-x[0],x[1]))[0][1] if valid else None; result["selection_success"]=bool(valid)
        dump(root/"P7D_ANALYSIS.json",result); dump(root/"P7D_SELECTION.json",{"selected_prompt":result["selected_prompt"],"selection_success":result["selection_success"],"protocol_sha256":sha(Path(sys.argv[sys.argv.index("--protocol")+1]))}); return result
    policies=["GatedBaseline","RoleBoundSelected","ActionSafeFixed","OldGateFixed"]; result={"phase":"p7","integrity":{"missing":missing},"families":{}}
    for family in protocol["p7"]["families"]:
        fam=[r for r in rows if r[0]["analysis_family"]==family]; result["families"][family]={}
        for policy in policies:
            effect={}; fidelity={}; rate=[]
            for mat,sp,no in fam:
                x=load(sp/"arms"/policy/"metrics.json"); effect.setdefault(mat["matrix_id"],[]).append(x["total_horizon_team_mean_payoff"]-no["total_horizon_team_mean_payoff"]); fidelity.setdefault(mat["matrix_id"],[]).append(x["commit_target_action_rate"]); rate.append(float(x["selected_intervention"]!="NoAlign"))
            result["families"][family][policy]={"effect_vs_noalign":bootstrap(effect,protocol["analysis"]["bootstrap_samples"],stable_seed("effect",family,policy)),"target_fidelity":bootstrap(fidelity,protocol["analysis"]["bootstrap_samples"],stable_seed("fidelity",family,policy)),"intervention_rate":float(np.mean(rate))}
    a=result["families"].get("anti_safe",{}).get("ActionSafeFixed",{}); t=result["families"].get("anti_tradeoff",{}).get("ActionSafeFixed",{}); old=result["families"].get("anti_safe",{}).get("OldGateFixed",{})
    result["confirmatory_success"]=bool(not missing and a and a["target_fidelity"]["ci95"][0]>.50 and a["effect_vs_noalign"]["ci95"][0]>0 and a["intervention_rate"]>=.80 and t["intervention_rate"]<=.10 and old["intervention_rate"]<=.10)
    dump(root/"P7_ANALYSIS.json",result); return result


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--protocol",type=Path,required=True); ap.add_argument("--results-root",type=Path,required=True); ap.add_argument("--phase",choices=["p7d","p7"],required=True); ap.add_argument("--backend",choices=["vllm","synthetic"],default="vllm"); ap.add_argument("--workers",type=int,default=12); ap.add_argument("--analyze",action="store_true")
    args=ap.parse_args(); p=load(args.protocol); root=args.results_root.resolve(); phase_cfg=p[args.phase]
    if args.analyze: print(json.dumps(analyze(p,root,args.phase),indent=2)); return
    if args.phase=="p7":
        selection=load(root/p["p7"]["selected_prompt_file"])
        if not selection.get("selection_success") or selection.get("selected_prompt") not in p["p7d"]["candidate_prompts"]: raise RuntimeError("P7 is blocked: P7D did not freeze a valid prompt")
        selected=selection["selected_prompt"]
    else: selected=None
    fams=["anti_safe"] if args.phase=="p7d" else p["p7"]["families"]; n=phase_cfg.get("n_anti_safe_matrices",phase_cfg.get("n_per_family")); specs=registry(phase_cfg["registry_seed"],n,fams,args.phase); dump(root/f"{args.phase.upper()}_REGISTRY.json",{"protocol_sha256":sha(args.protocol),"matrices":[m.audit() for m in specs]})
    backend=Backend(p["model_pair"],args.backend); tasks=[(m,s) for m in specs for s in phase_cfg["seeds"]]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures=[ex.submit(run_cell,p,root,args.phase,m,s,backend,selected) for m,s in tasks]
        for f in as_completed(futures): f.result()
    print(json.dumps({"phase":args.phase,"complete":True,"contexts":len(tasks),"next":"rerun with --analyze"},indent=2))

if __name__=="__main__": main()
