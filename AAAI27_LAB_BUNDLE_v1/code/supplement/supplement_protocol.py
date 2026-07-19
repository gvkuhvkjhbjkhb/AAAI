"""Validation and immutable-artifact helpers for P0/P1/P2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from p3_label_variants import registry_sha256 as label_registry_sha256
from p3_matrices import matrix_ids, registry_sha256 as p3_registry_sha256


SOURCE_GAMES = [
    "chicken", "hawk_dove", "deadlock", "stag_hunt",
    "battle_of_the_sexes", "public_goods",
]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(path: Path) -> dict[str, Any]:
    p = read_json(path)
    required = {
        "schema_version", "campaign", "models_het", "model_revisions",
        "sampling", "execution", "p3", "source", "p0", "p1", "p2",
        "analysis",
    }
    missing = sorted(required - set(p))
    if missing:
        raise ValueError(f"Protocol missing keys: {missing}")
    if p["campaign"] != "aaai27_supplemental_controls_v1":
        raise ValueError("Unexpected campaign name")
    sampling = p["sampling"]
    if sampling != {"top_p": 0.9, "gen_seed_base": 1000,
                    "horizon": 5, "memory": 2}:
        raise ValueError("Sampling/horizon configuration differs from frozen S2/P3")
    if p["p3"]["matrix_ids"] != matrix_ids():
        raise ValueError("P3 matrix IDs differ from the frozen registry")
    if p["p3"]["seeds"] != list(range(102, 112)):
        raise ValueError("P3 seeds must be 102..111")
    if p["source"]["games"] != SOURCE_GAMES:
        raise ValueError("Source games differ from S2")
    if p["source"]["seeds"] != list(range(82, 102)):
        raise ValueError("Source seeds must be 82..101")
    if p["p0"]["new_policy"] != "het_payoff_prompt":
        raise ValueError("P0 must run the payoff-in-prompt policy")
    if p["p0"]["policies"] != [
        "het_notom", "het_gated_atom_talk", "het_safe_sca", "het_payoff_prompt"
    ]:
        raise ValueError("P0 must concurrently rerun fixed arms, Safe-SCA, and payoff prompt")
    if p["p1"]["variant"] != "label_swap":
        raise ValueError("P1 must be the frozen label-swap intervention")
    if p["p1"]["policies"] != [
        "het_notom", "het_gated_atom_talk", "het_safe_sca"
    ]:
        raise ValueError("P1 policies differ from the frozen list")
    if p["p2"]["cell"] != "het_bandit_teammean" or p["p2"]["bandit_k"] != 5:
        raise ValueError("P2 must use the team-mean bandit with K=5")
    if p["p3"]["registry_sha256"] != p3_registry_sha256():
        raise ValueError("P3 registry hash mismatch")
    if p["p1"]["variant_registry_sha256"] != label_registry_sha256():
        raise ValueError("P1 label-variant registry hash mismatch")
    if p["analysis"]["primary_endpoint"] != "team_mean_payoff":
        raise ValueError("Primary endpoint must be team_mean_payoff")
    return p


def immutable_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Refusing to overwrite immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
