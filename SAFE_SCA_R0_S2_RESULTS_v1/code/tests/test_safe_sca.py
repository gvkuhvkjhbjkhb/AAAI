"""CPU-only tests for S1's label-free decision layer."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from safe_sca import (  # noqa: E402
    CoverageCertifiedEstimator,
    SafeSCAConfig,
    configure_agents_for_arm,
    select_point_estimate_arm,
    select_safe_arm,
)


class FakeAgent:
    def __init__(self):
        self.use_tom = False
        self.use_talk = False
        self.adaptive_tom = False
        self.gated_talk_tom = False
        self.diversity_preserving_gate = False


class SafeSCATest(unittest.TestCase):
    def setUp(self):
        self.config = SafeSCAConfig(
            warmup_episodes=5,
            tau=0.10,
            confidence=0.95,
            bootstrap_samples=500,
            min_profile_coverage=0.50,
            min_stratum_observations=3,
        )

    def summary(self, observations):
        estimator = CoverageCertifiedEstimator()
        for actions, reward in observations:
            estimator.observe(actions, [reward, reward])
        return estimator.summarize(
            n_agents=2, n_actions=2, config=self.config, seed=7,
        )

    def test_confident_coordination_aligns(self):
        # Same-action profiles outperform differentiated profiles by a stable
        # margin, both strata are covered, and 3/4 profiles are observed.
        observations = [
            ((0, 0), 3.0), ((1, 1), 3.0), ((0, 0), 3.0),
            ((0, 1), 1.0), ((1, 0), 1.0), ((0, 1), 1.0),
        ]
        arm, reasons = select_safe_arm(self.summary(observations), self.config)
        self.assertEqual(arm, "Gated")
        self.assertEqual(reasons, ["coverage_certified_coordination"])

    def test_low_coverage_abstains_even_with_negative_split(self):
        observations = [
            ((0, 0), 3.0), ((0, 0), 3.0), ((0, 0), 3.0),
            ((0, 1), 1.0), ((0, 1), 1.0), ((0, 1), 1.0),
        ]
        stricter = SafeSCAConfig(
            warmup_episodes=5, tau=0.10, confidence=0.95,
            bootstrap_samples=500, min_profile_coverage=0.75,
            min_stratum_observations=3,
        )
        estimator = CoverageCertifiedEstimator()
        for actions, reward in observations:
            estimator.observe(actions, [reward, reward])
        summary = estimator.summarize(n_agents=2, n_actions=2, config=stricter, seed=7)
        arm, reasons = select_safe_arm(summary, stricter)
        self.assertEqual(arm, "NoAlign")
        self.assertIn("insufficient_profile_coverage", reasons)

    def test_missing_stratum_abstains(self):
        observations = [((0, 0), 3.0), ((1, 1), 3.0), ((0, 0), 3.0)]
        arm, reasons = select_safe_arm(self.summary(observations), self.config)
        self.assertEqual(arm, "NoAlign")
        self.assertIn("insufficient_differentiated_observations", reasons)
        self.assertIn("split_confidence_interval_unavailable", reasons)

    def test_point_estimate_is_less_conservative(self):
        observations = [
            ((0, 0), 3.0), ((0, 0), 3.0), ((0, 0), 3.0),
            ((0, 1), 1.0), ((0, 1), 1.0), ((0, 1), 1.0),
        ]
        arm, _ = select_point_estimate_arm(self.summary(observations))
        self.assertEqual(arm, "Gated")

    def test_arm_switching_preserves_objects_and_changes_only_policy_flags(self):
        agents = [FakeAgent(), FakeAgent()]
        ids_before = [id(agent) for agent in agents]
        configure_agents_for_arm(agents, "Gated")
        self.assertTrue(all(agent.use_tom and agent.use_talk and agent.gated_talk_tom for agent in agents))
        configure_agents_for_arm(agents, "NoAlign")
        self.assertEqual(ids_before, [id(agent) for agent in agents])
        self.assertTrue(all(not agent.use_tom and not agent.use_talk and not agent.gated_talk_tom
                            for agent in agents))


if __name__ == "__main__":
    unittest.main()
