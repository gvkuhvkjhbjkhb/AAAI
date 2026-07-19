"""Frozen P3 matrix registry.

Categories and matrix IDs are analysis metadata only.  ``make_game`` gives the
LLM agents the same generic game name for every matrix, neutral label surfaces,
and no payoff table.  Safe-SCA receives only realized action/reward histories.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable


PROMPT_GAME_NAME = "Anonymous interaction"


@dataclass(frozen=True)
class MatrixSpec:
    matrix_id: str
    analysis_category: str  # Never passed to the controller or LLM prompt.
    action_labels: tuple[str, str]
    payoff_matrix: tuple[tuple[tuple[float, float], tuple[float, float]],
                         tuple[tuple[float, float], tuple[float, float]]]

    def audit_dict(self) -> dict:
        raw = asdict(self)
        raw["payoff_matrix"] = [[list(pair) for pair in row] for row in self.payoff_matrix]
        raw["action_labels"] = list(self.action_labels)
        return raw


# Four coordination/boundary and four anti-coordination matrices.  None is an
# S1/S2 matrix.  The labels are deliberately opaque and category-neutral.
SPECS: tuple[MatrixSpec, ...] = (
    MatrixSpec("p3_m01", "coord_or_boundary", ("glyph-ivory", "glyph-slate"),
               (((4, 4), (0, 1)), ((1, 0), (2, 2)))),
    MatrixSpec("p3_m02", "coord_or_boundary", ("token-amber", "token-cyan"),
               (((4, 3), (0, 0)), ((0, 0), (3, 4)))),
    MatrixSpec("p3_m03", "coord_or_boundary", ("mark-lumen", "mark-umbra"),
               (((5, 5), (0, 2)), ((2, 0), (3, 3)))),
    MatrixSpec("p3_m04", "coord_or_boundary", ("sigil-cedar", "sigil-flint"),
               (((3, 3), (-1, 1)), ((1, -1), (4, 4)))),
    MatrixSpec("p3_m05", "anti", ("rune-coral", "rune-indigo"),
               (((2, 2), (0, 5)), ((5, 0), (-2, -2)))),
    MatrixSpec("p3_m06", "anti", ("node-silver", "node-ochre"),
               (((1, 1), (0, 4)), ((4, 0), (0, 0)))),
    MatrixSpec("p3_m07", "anti", ("tile-moss", "tile-rose"),
               (((-1, -1), (2, 3)), ((3, 2), (1, 1)))),
    MatrixSpec("p3_m08", "anti", ("form-north", "form-south"),
               (((0, 0), (1, 4)), ((4, 1), (2, 2)))),
)

BY_ID = {spec.matrix_id: spec for spec in SPECS}


def matrix_ids() -> list[str]:
    return [spec.matrix_id for spec in SPECS]


def get_spec(matrix_id: str) -> MatrixSpec:
    try:
        return BY_ID[matrix_id]
    except KeyError as exc:
        raise ValueError(f"Unknown P3 matrix {matrix_id!r}; expected one of {matrix_ids()}") from exc


def registry_audit() -> list[dict]:
    return [spec.audit_dict() for spec in SPECS]


def registry_sha256() -> str:
    canonical = json.dumps(registry_audit(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_game(hb, matrix_id: str):
    """Build a generic prompt-visible game from an analysis-only spec."""
    spec = get_spec(matrix_id)

    def payoff(actions: tuple[int, int]) -> tuple[float, float]:
        a0, a1 = (int(actions[0]), int(actions[1]))
        return spec.payoff_matrix[a0][a1]

    return hb.MatrixGame(
        name=PROMPT_GAME_NAME,
        n_agents=2,
        n_actions=2,
        action_names=list(spec.action_labels),
        payoff=payoff,
    )


def register_with_baseline(hb) -> None:
    """Register factories used by the existing runner; no categories leak."""
    for matrix_id in matrix_ids():
        hb.GAMES[matrix_id] = (lambda mid=matrix_id: make_game(hb, mid))


def verify_registry() -> None:
    if len(SPECS) != 8 or len(BY_ID) != 8:
        raise ValueError("P3 must contain exactly eight unique matrices")
    if sum(spec.analysis_category == "anti" for spec in SPECS) != 4:
        raise ValueError("P3 must contain exactly four anti-coordination matrices")
    if sum(spec.analysis_category == "coord_or_boundary" for spec in SPECS) != 4:
        raise ValueError("P3 must contain exactly four coordination/boundary matrices")
    for spec in SPECS:
        if len(spec.action_labels) != 2 or len(set(spec.action_labels)) != 2:
            raise ValueError(f"Invalid action surface for {spec.matrix_id}")
        if len(spec.payoff_matrix) != 2 or any(len(row) != 2 for row in spec.payoff_matrix):
            raise ValueError(f"{spec.matrix_id} is not a 2x2 matrix")
        for row in spec.payoff_matrix:
            for pair in row:
                if len(pair) != 2:
                    raise ValueError(f"{spec.matrix_id} has an invalid payoff pair")


verify_registry()
