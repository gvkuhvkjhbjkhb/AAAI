"""Prompt-surface controls for the frozen P3 payoff environments.

The ``label_swap`` condition changes only the strings attached to action
indices.  Payoff tensors, seeds, agent models, temperatures, and controller
inputs are unchanged.  Analysis categories never enter prompts/controllers.
"""
from __future__ import annotations

import hashlib
import json

from p3_matrices import PROMPT_GAME_NAME, SPECS, get_spec, matrix_ids


VARIANT = "label_swap"


def labels_for(matrix_id: str, variant: str = VARIANT) -> tuple[str, str]:
    if variant != VARIANT:
        raise ValueError(f"Unsupported label variant: {variant}")
    original = get_spec(matrix_id).action_labels
    return original[1], original[0]


def make_game(hb, matrix_id: str, variant: str = VARIANT):
    spec = get_spec(matrix_id)
    action_labels = labels_for(matrix_id, variant)

    def payoff(actions: tuple[int, int]) -> tuple[float, float]:
        a0, a1 = int(actions[0]), int(actions[1])
        return spec.payoff_matrix[a0][a1]

    return hb.MatrixGame(
        name=PROMPT_GAME_NAME,
        n_agents=2,
        n_actions=2,
        action_names=list(action_labels),
        payoff=payoff,
    )


def register_with_baseline(hb, variant: str = VARIANT) -> None:
    for matrix_id in matrix_ids():
        hb.GAMES[matrix_id] = (
            lambda mid=matrix_id, v=variant: make_game(hb, mid, v)
        )


def registry_audit() -> list[dict]:
    rows = []
    for spec in SPECS:
        rows.append({
            "matrix_id": spec.matrix_id,
            "variant": VARIANT,
            "original_action_labels": list(spec.action_labels),
            "variant_action_labels": list(labels_for(spec.matrix_id)),
            "payoff_matrix": [[list(pair) for pair in row]
                              for row in spec.payoff_matrix],
        })
    return rows


def registry_sha256() -> str:
    canonical = json.dumps(registry_audit(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

