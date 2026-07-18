#!/usr/bin/env python3
"""S1 environment preflight with vLLM API-key support."""
from __future__ import annotations
import argparse, importlib.metadata, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

EXPECTED = {"vllm":"0.25.1","torch":"2.11.0+cu128","transformers":"5.14.1"}
ENDPOINTS = {"qwen":"http://localhost:8000/v1/models","glm":"http://localhost:8001/v1/models"}
EXPECTED_FRAGS = {"qwen":"Qwen2.5-7B-Instruct","glm":"GLM-4-9B-0414"}

def inst_ver(d):
    try: return importlib.metadata.version(d)
    except: return None

def ep_models(url, to):
    try:
        req = Request(url)
        req.add_header("Authorization", f"Bearer {os.environ.get('VLLM_API_KEY','dummy')}")
        with urlopen(req, timeout=to) as r:
            payload = json.loads(r.read().decode())
        ids = sorted(i.get("id") for i in payload.get("data",[]) if i.get("id"))
        return {"reachable":True,"models":ids,"error":None}
    except Exception as e:
        return {"reachable":False,"models":[],"error":str(e)}

def nvidia_smi():
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version,compute_cap",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        g = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        return {"available":True,"gpus":g,"error":None}
    except Exception as e:
        return {"available":False,"gpus":[],"error":str(e)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    vs = {n:inst_ver(n) for n in EXPECTED}
    m = {"schema_version":1,"created_at_utc":datetime.now(timezone.utc).isoformat(),
         "expected_versions":EXPECTED,"package_versions":vs,
         "python":sys.version,"platform":platform.platform(),
         "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
         "gpu":nvidia_smi(),
         "vllm_endpoints":{n:ep_models(u,a.timeout) for n,u in ENDPOINTS.items()},
         "allow_version_mismatch":False}
    fails = []
    for pkg,exp in EXPECTED.items():
        if vs.get(pkg)!=exp: fails.append(f"{pkg} ver {vs.get(pkg)!r} != {exp!r}")
    if not m["gpu"]["available"]:
        fails.append(f"nvidia-smi: {m['gpu']['error']}")
    elif len(m["gpu"]["gpus"])!=2:
        fails.append(f"GPU count: {len(m['gpu']['gpus'])}")
    for n,ep in m["vllm_endpoints"].items():
        if not ep["reachable"]:
            fails.append(f"{n} unreachable: {ep['error']}")
        elif not any(EXPECTED_FRAGS[n] in mo for mo in ep["models"]):
            fails.append(f"{n} missing {EXPECTED_FRAGS[n]!r}: {ep['models']}")
    m["preflight_passed"] = not fails
    m["failures"] = fails
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / "ENVIRONMENT_MANIFEST_S1.json"
    if out.exists() and not a.force:
        raise FileExistsError(f"Refusing to overwrite {out}; use --force")
    out.write_text(json.dumps(m,indent=2)+"\n")
    print(json.dumps(m,indent=2))
    if fails:
        raise SystemExit("preflight failed: "+"; ".join(fails))
if __name__=="__main__": main()
