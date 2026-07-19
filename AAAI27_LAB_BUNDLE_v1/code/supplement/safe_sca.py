"""Label-free, coverage-certified decision logic for S1 Safe-SCA.

The module deliberately contains no game names, payoff matrices, or oracle
labels.  It receives only warm-up action/reward observations and returns one
of two deployable post-warm-up arms:

* ``Gated``: enable the original forced talk+ToM arbitration.
* ``NoAlign``: disable all alignment machinery (the safe abstention arm).

An alignment decision requires (i) action-profile coverage, (ii) observations
from both same-action and differentiated-action strata, and (iii) a one-sided
bootstrap upper confidence bound below ``-tau``.  Consequently, ambiguous or
low-coverage warm-ups always abstain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import json

import numpy as np


@dataclass(frozen=True)
class SafeSCAConfig:
    """Frozen, label-free Safe-SCA hyperparameters.

    ``confidence`` is the one-sided confidence level for the upper bootstrap
    bound of ``mean(differentiated payoff) - mean(same-action payoff)``.
    Alignment is allowed only if that bound is less than ``-tau``.
    """

    warmup_episodes: int = 10
    tau: float = 0.10
    confidence: float = 0.95
    bootstrap_samples: int = 2_000
    min_profile_coverage: float = 0.25
    min_stratum_observations: int = 3

    def validate(self) -> None:
        if self.warmup_episodes <= 0:
            raise ValueError("warmup_episodes must be positive")
        if self.tau < 0:
            raise ValueError("tau must be non-negative")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0.5, 1.0)")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        if not 0.0 <= self.min_profile_coverage <= 1.0:
            raise ValueError("min_profile_coverage must be in [0, 1]")
        if self.min_stratum_observations < 1:
            raise ValueError("min_stratum_observations must be positive")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SafeSCAConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown Safe-SCA config field(s): {sorted(unknown)}")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> SafeSCAConfig:
    """Load and validate a frozen Safe-SCA JSON configuration."""
    with open(path, encoding="utf-8") as handle:
        return SafeSCAConfig.from_mapping(json.load(handle))


@dataclass(frozen=True)
class CoverageSummary:
    n_observations: int
    n_same: int
    n_different: int
    n_unique_profiles: int
    n_possible_profiles: int
    profile_coverage: float
    split_score: float | None
    split_upper_bound: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CoverageCertifiedEstimator:
    """Estimate a split score and the evidence needed to trust its sign."""

    def __init__(self) -> None:
        self._observations: list[tuple[tuple[int, ...], float]] = []

    def observe(self, actions: Iterable[int], rewards: Iterable[float]) -> None:
        action_tuple = tuple(int(action) for action in actions)
        if not action_tuple:
            raise ValueError("actions must not be empty")
        reward_values = [float(reward) for reward in rewards]
        if not reward_values:
            raise ValueError("rewards must not be empty")
        self._observations.append((action_tuple, float(np.mean(reward_values))))

    @property
    def observations(self) -> list[tuple[tuple[int, ...], float]]:
        """A defensive copy for audit logging; callers cannot mutate state."""
        return list(self._observations)

    def summarize(
        self,
        *,
        n_agents: int,
        n_actions: int,
        config: SafeSCAConfig,
        seed: int,
    ) -> CoverageSummary:
        config.validate()
        if n_agents <= 0 or n_actions <= 0:
            raise ValueError("n_agents and n_actions must be positive")

        same_payoffs = [payoff for actions, payoff in self._observations
                        if len(set(actions)) == 1]
        diff_payoffs = [payoff for actions, payoff in self._observations
                        if len(set(actions)) != 1]
        n_possible = n_actions ** n_agents
        n_unique = len({actions for actions, _ in self._observations})
        coverage = n_unique / n_possible

        split: float | None = None
        upper: float | None = None
        if same_payoffs and diff_payoffs:
            split = float(np.mean(diff_payoffs) - np.mean(same_payoffs))
            upper = _bootstrap_upper_bound(
                same_payoffs=same_payoffs,
                diff_payoffs=diff_payoffs,
                confidence=config.confidence,
                n_boot=config.bootstrap_samples,
                seed=seed,
            )

        return CoverageSummary(
            n_observations=len(self._observations),
            n_same=len(same_payoffs),
            n_different=len(diff_payoffs),
            n_unique_profiles=n_unique,
            n_possible_profiles=n_possible,
            profile_coverage=float(coverage),
            split_score=split,
            split_upper_bound=upper,
        )


def _bootstrap_upper_bound(
    *,
    same_payoffs: list[float],
    diff_payoffs: list[float],
    confidence: float,
    n_boot: int,
    seed: int,
) -> float:
    """One-sided bootstrap upper bound for a two-stratum mean difference."""
    rng = np.random.default_rng(seed)
    same = np.asarray(same_payoffs, dtype=float)
    diff = np.asarray(diff_payoffs, dtype=float)
    same_boot = rng.choice(same, size=(n_boot, len(same)), replace=True).mean(axis=1)
    diff_boot = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.quantile(diff_boot - same_boot, confidence))


def select_safe_arm(summary: CoverageSummary, config: SafeSCAConfig) -> tuple[str, list[str]]:
    """Return the label-free post-warm-up policy and an auditable rationale."""
    config.validate()
    reasons: list[str] = []
    if summary.n_same < config.min_stratum_observations:
        reasons.append("insufficient_same_action_observations")
    if summary.n_different < config.min_stratum_observations:
        reasons.append("insufficient_differentiated_observations")
    if summary.profile_coverage < config.min_profile_coverage:
        reasons.append("insufficient_profile_coverage")
    if summary.split_upper_bound is None:
        reasons.append("split_confidence_interval_unavailable")
    elif summary.split_upper_bound >= -config.tau:
        reasons.append("coordination_not_certified")

    return ("NoAlign", reasons) if reasons else ("Gated", ["coverage_certified_coordination"])


def select_point_estimate_arm(summary: CoverageSummary, tau: float = 0.0) -> tuple[str, list[str]]:
    """Unconstrained online baseline: align whenever the point split is low.

    This intentionally ignores coverage and uncertainty.  It is included in
    S1 as the direct comparator that Safe-SCA must improve upon.
    """
    if summary.split_score is None:
        return "Gated", ["point_estimate_unavailable_default_align"]
    if summary.split_score <= -tau:
        return "Gated", ["point_estimate_coordination"]
    return "NoAlign", ["point_estimate_anti_coordination"]


def configure_agents_for_arm(agents: Iterable[Any], arm: str) -> None:
    """Switch an existing agent pool without resetting warm-up state.

    Safe-SCA intentionally preserves agent state across the transition from
    NoAlign warm-up to Gated commit.  This makes Gated's trust-EMA transition
    cost a real, logged part of the end-to-end protocol rather than hiding it
    through a fresh-agent counterfactual.
    """
    if arm not in {"NoAlign", "Gated"}:
        raise ValueError(f"Unsupported deployable arm: {arm}")
    for agent in agents:
        if arm == "NoAlign":
            agent.use_tom = False
            agent.use_talk = False
            agent.adaptive_tom = False
            agent.gated_talk_tom = False
            agent.diversity_preserving_gate = False
        else:
            agent.use_tom = True
            agent.use_talk = True
            agent.adaptive_tom = True
            agent.gated_talk_tom = True
            agent.diversity_preserving_gate = False
