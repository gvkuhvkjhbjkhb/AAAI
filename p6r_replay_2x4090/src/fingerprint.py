"""
fingerprint.py -- Phase 1: the serving fingerprint the original run lacked.

The P6R low-fidelity cell cannot be attributed today because nothing was
recorded that would let anyone tell, after the fact, whether the serving stack
differed.  This script produces that record, and it is the part of this package
worth keeping regardless of how the replay comes out.

Three layers, cheapest first:

  1. ENVIRONMENT   vLLM / torch / transformers versions, CUDA, driver, GPU name,
                   and what the endpoint says it is serving.
  2. IDENTITY      the model repo revision actually loaded, plus SHA-256 of the
                   tokenizer files and the chat template.  A chat-template change
                   is invisible in every metric yet moves every completion.
  3. BEHAVIOUR     a greedy canary: a fixed set of real P6R prompts sent at
                   temperature 0, ONE AT A TIME (so continuous batching cannot
                   perturb them), with the completions hashed; plus first-token
                   top-k logprobs, which detect dtype / quantization / template
                   drift far earlier than sampled text does.

Layer 3 is the sensitive one.  Two runs that agree on layers 1-2 but differ on
layer 3 have a real serving difference; two runs that agree on layer 3 are, for
this campaign's purposes, the same stack.

Usage:
  python src/fingerprint.py --protocol protocols/p6r_replay_frozen.json \
                            --out out/FINGERPRINT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p6r_prompt import resolve_prompt_impl  # noqa: E402
from registry import build_p6r_registry     # noqa: E402


def _post(url: str, body: dict, key: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url: str, key: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return ""


def environment_layer() -> dict:
    def ver(mod: str) -> str:
        try:
            return __import__(mod).__version__
        except Exception:                                 # noqa: BLE001
            return "absent"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "vllm": ver("vllm"),
        "torch": ver("torch"),
        "transformers": ver("transformers"),
        "nvidia_smi": _sh(["nvidia-smi",
                           "--query-gpu=name,driver_version,memory.total",
                           "--format=csv,noheader"]),
        "cuda": _sh(["nvcc", "--version"]).splitlines()[-1] if _sh(["nvcc", "--version"]) else "",
    }


def identity_layer(model: str, base: str, key: str, pinned_rev: str) -> dict:
    out = {"model": model, "pinned_revision": pinned_rev}
    try:
        out["served_models"] = [m["id"] for m in _get(f"{base}/models", key)["data"]]
    except urllib.error.URLError as e:
        out["served_models_error"] = str(e)

    # tokenizer + chat template hashes, read from the local HF cache snapshot
    try:
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(model, revision=pinned_rev or None,
                                      allow_patterns=["tokenizer*",
                                                      "*.jinja",
                                                      "generation_config.json",
                                                      "config.json"]))
        files = {}
        for p in sorted(snap.rglob("*")):
            if p.is_file():
                files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        out["file_sha256_prefix"] = files
        tmpl = None
        tok_cfg = snap / "tokenizer_config.json"
        if tok_cfg.exists():
            tmpl = json.loads(tok_cfg.read_text(encoding="utf-8")).get("chat_template")
        out["chat_template_sha256"] = (
            hashlib.sha256(tmpl.encode()).hexdigest() if tmpl else None)
    except Exception as e:                                # noqa: BLE001
        out["identity_error"] = f"{type(e).__name__}: {e}"
    return out


def canary_prompts(protocol: dict, build_prompt) -> list[dict]:
    """Real P6R prompts: every anti-safe matrix, both roles, three histories.
    Using real prompts (not toy ones) is the point -- a drift that only shows up
    on this distribution is exactly the drift we are hunting."""
    reg = build_p6r_registry(protocol["p6r"]["registry_seed"],
                             protocol["p6r"]["n_per_family"])
    memory = protocol["sampling"]["memory"]
    histories = [
        [],
        [{"actions": [0, 0]}, {"actions": [0, 0]}],
        [{"actions": [0, 1]}, {"actions": [1, 1]}],
    ]
    out = []
    for m in reg:
        if m["analysis_family"] != "anti_safe":
            continue
        for role in (0, 1):
            for hi, hist in enumerate(histories):
                for arm in ("NoAlign", "Gated"):
                    out.append({
                        "id": f"{m['matrix_id']}|role{role}|h{hi}|{arm}",
                        "role": role,
                        "messages": build_prompt(m["payoff_matrix"], m["target"],
                                                 role, hist, arm, memory),
                    })
    return out


def behaviour_layer(protocol: dict, build_prompt, key: str) -> dict:
    eps = protocol["model_pair"]["endpoints"]
    roles = protocol["model_pair"]["roles"]
    items = canary_prompts(protocol, build_prompt)
    records, digest = [], hashlib.sha256()
    for it in items:
        role = str(it["role"])
        url = eps[role].rstrip("/") + "/chat/completions"
        body = {"model": roles[role], "messages": it["messages"],
                "temperature": 0.0, "top_p": 1.0,
                "max_tokens": protocol["sampling"]["max_tokens"],
                "logprobs": True, "top_logprobs": 5, "seed": 0}
        data = _post(url, body, key)
        ch = data["choices"][0]
        text = ch["message"]["content"]
        top = []
        try:
            first = ch["logprobs"]["content"][0]
            top = [{"t": c["token"], "lp": round(c["logprob"], 4)}
                   for c in first["top_logprobs"]]
        except Exception:                                 # noqa: BLE001
            pass
        rec = {"id": it["id"], "completion": text, "first_token_top5": top}
        records.append(rec)
        digest.update(it["id"].encode())
        digest.update(text.encode())
    # a separate, coarser digest over logprobs rounded to 2dp: sensitive to
    # dtype/quantization drift but tolerant of last-bit kernel noise
    lp_digest = hashlib.sha256()
    for r in records:
        lp_digest.update(r["id"].encode())
        for c in r["first_token_top5"]:
            lp_digest.update(f"{c['t']}:{c['lp']:.2f}".encode())
    return {"n_canary": len(records),
            "completion_sha256": digest.hexdigest(),
            "logprob_sha256_2dp": lp_digest.hexdigest(),
            "records": records}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p6r-code-dir", default=None)
    ap.add_argument("--allow-reconstructed-prompt", action="store_true")
    a = ap.parse_args()

    protocol = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    if not a.p6r_code_dir and not a.allow_reconstructed_prompt:
        raise SystemExit("Pass --p6r-code-dir <archived P6R code> for a faithful "
                         "fingerprint, or --allow-reconstructed-prompt to accept "
                         "the reconstruction (recorded in the output).")
    impl = resolve_prompt_impl(a.p6r_code_dir)
    key = __import__("os").environ.get(
        protocol["model_pair"].get("api_key_env", "VLLM_API_KEY"), "")

    fp = {"campaign": protocol["campaign"],
          "protocol_sha256": hashlib.sha256(
              Path(a.protocol).read_bytes()).hexdigest(),
          "prompt_provenance": impl["provenance"],
          "environment": environment_layer(),
          "identity": {}, "behaviour": {}}

    for role, model in protocol["model_pair"]["roles"].items():
        fp["identity"][role] = identity_layer(
            model, protocol["model_pair"]["endpoints"][role], key,
            protocol["model_pair"]["revisions"].get(model, ""))

    fp["behaviour"] = behaviour_layer(protocol, impl["build_prompt"], key)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fp, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    print(f"  canary completions sha256 : {fp['behaviour']['completion_sha256']}")
    print(f"  canary logprob   sha256   : {fp['behaviour']['logprob_sha256_2dp']}")
    print("  Keep these two hashes.  Any future run that reproduces them is, for")
    print("  this campaign's purposes, running the same serving stack.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
