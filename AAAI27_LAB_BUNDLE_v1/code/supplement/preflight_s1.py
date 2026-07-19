#!/usr/bin/env python3
"""Record and validate the lab environment before an S1 GPU launch.

This is intentionally a local-only preflight: it reads package metadata,
``nvidia-smi``, and the two expected localhost vLLM ``/v1/models`` endpoints.
It never submits a generation request.  The resulting manifest is a required
artifact for separating the fresh vLLM 0.25.1 S1 run from earlier Stack-B runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from urllib.error import URLError
from urllib.request import urlopen


EXPECTED = {
    "vllm": "0.25.1",
    "torch": "2.11.0+cu128",
    "transformers": "5.14.1",
}
ENDPOINTS = {
    "qwen": "http://localhost:8000/v1/models",
    "glm": "http://localhost:8001/v1/models",
}
EXPECTED_ENDPOINT_MODEL_FRAGMENTS = {
    "qwen": "Qwen2.5-7B-Instruct",
    "glm": "GLM-4-9B-0414",
}
EXPECTED_GPU_NAME_FRAGMENT = "RTX 5090"


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def endpoint_models(url: str, timeout_seconds: float) -> dict:
    try:
        with urlopen(url, timeout=timeout_seconds) as response:  # nosec B310: fixed localhost endpoint
            payload = json.loads(response.read().decode("utf-8"))
        model_ids = sorted(item.get("id") for item in payload.get("data", []) if item.get("id"))
        return {"reachable": True, "models": model_ids, "error": None}
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"reachable": False, "models": [], "error": str(exc)}


def nvidia_smi() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "gpus": [], "error": str(exc)}
    if completed.returncode != 0:
        return {"available": False, "gpus": [], "error": completed.stderr.strip()}
    gpus = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {"available": True, "gpus": gpus, "error": None}


def evaluate(manifest: dict, allow_version_mismatch: bool) -> list[str]:
    version_failures: list[str] = []
    deployment_failures: list[str] = []
    versions = manifest["package_versions"]
    for package, expected in EXPECTED.items():
        actual = versions.get(package)
        if actual != expected:
            version_failures.append(f"{package} version is {actual!r}; expected {expected!r}")
    if not manifest["gpu"]["available"]:
        deployment_failures.append(f"nvidia-smi unavailable: {manifest['gpu']['error']}")
    elif len(manifest["gpu"]["gpus"]) != 2:
        deployment_failures.append(f"expected exactly two visible GPUs, found {len(manifest['gpu']['gpus'])}")
    else:
        unexpected = [gpu for gpu in manifest["gpu"]["gpus"] if EXPECTED_GPU_NAME_FRAGMENT not in gpu]
        if unexpected:
            deployment_failures.append(
                f"expected {EXPECTED_GPU_NAME_FRAGMENT} GPUs, found: {unexpected}"
            )
    for name, payload in manifest["vllm_endpoints"].items():
        if not payload["reachable"]:
            deployment_failures.append(f"{name} endpoint unreachable: {payload['error']}")
        elif not any(EXPECTED_ENDPOINT_MODEL_FRAGMENTS[name] in model for model in payload["models"]):
            deployment_failures.append(
                f"{name} endpoint does not advertise {EXPECTED_ENDPOINT_MODEL_FRAGMENTS[name]!r}: "
                f"{payload['models']}"
            )
    return deployment_failures + ([] if allow_version_mismatch else version_failures)


def main() -> None:
    parser = argparse.ArgumentParser(description="S1 environment preflight and manifest writer")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="S1 result directory that will contain ENVIRONMENT_MANIFEST_S1.json")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--allow-version-mismatch", action="store_true",
                        help="allow only package-version drift; GPU and endpoint checks remain strict; do not use for preregistered S1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")

    versions = {name: installed_version(name) for name in EXPECTED}
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_versions": EXPECTED,
        "package_versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": nvidia_smi(),
        "vllm_endpoints": {name: endpoint_models(url, args.timeout) for name, url in ENDPOINTS.items()},
        "allow_version_mismatch": args.allow_version_mismatch,
    }
    failures = evaluate(manifest, args.allow_version_mismatch)
    manifest["preflight_passed"] = not failures
    manifest["failures"] = failures
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "ENVIRONMENT_MANIFEST_S1.json"
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; use --force before launch only")
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"[wrote] {output}")
    if failures:
        raise SystemExit("S1 preflight failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
