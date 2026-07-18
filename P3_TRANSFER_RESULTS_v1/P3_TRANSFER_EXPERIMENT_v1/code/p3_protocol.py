"""Protocol parsing and immutable-artifact helpers for P3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from p3_matrices import matrix_ids, registry_sha256


REQUIRED_POLICIES = [
    "het_notom", "het_gated_atom_talk", "het_point_sca", "het_safe_sca",
]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    required = {
        "schema_version", "campaign", "seeds", "policies", "episodes", "horizon",
        "memory", "top_p", "gen_seed_base", "latin_square", "workers",
        "task_timeout_seconds", "max_retries", "models_het", "model_revisions",
        "safe_sca", "analysis", "matrix_ids", "controller_visibility",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"P3 protocol missing keys: {missing}")
    if protocol["campaign"] != "p3_transfer":
        raise ValueError("P3 protocol campaign must be 'p3_transfer'")
    if protocol["policies"] != REQUIRED_POLICIES:
        raise ValueError("P3 policies must be the frozen four-policy list in frozen order")
    if protocol["matrix_ids"] != matrix_ids():
        raise ValueError("P3 protocol matrix IDs do not match the frozen registry")
    if protocol["seeds"] != list(range(102, 112)):
        raise ValueError("P3 seeds must be exactly 102..111")
    if protocol["episodes"] != 30 or protocol["horizon"] != 5 or protocol["memory"] != 2:
        raise ValueError("P3 episode/horizon/memory design is frozen at 30/5/2")
    if protocol["top_p"] != 0.9 or protocol["gen_seed_base"] != 1000:
        raise ValueError("P3 sampling configuration is frozen")
    if protocol["workers"] != 32:
        raise ValueError("P3 worker topology is frozen at 32")
    if protocol["analysis"]["anti_noninferiority_margin"] != 0.1:
        raise ValueError("P3 anti non-inferiority margin is frozen at 0.10")
    if protocol["analysis"]["max_anti_false_align"] != 0:
        raise ValueError("P3 false-align budget is frozen at zero")
    if protocol["analysis"]["min_coordination_recovery"] != 0.3:
        raise ValueError("P3 recovery threshold is frozen at 0.30")
    if protocol["analysis"]["min_coordination_matrices"] != 2:
        raise ValueError("P3 requires two qualifying coordination matrices")
    if registry_sha256() != protocol.get("matrix_registry_sha256", registry_sha256()):
        raise ValueError("P3 protocol matrix registry hash does not match code")
    return protocol


def immutable_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != payload:
            raise RuntimeError(f"Refusing to overwrite immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
