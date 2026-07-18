"""Development-only configuration selection tests."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from safe_sca import SafeSCAConfig  # noqa: E402
from select_s1_config import evaluate, ranking  # noqa: E402


def episode(actions, reward):
    return [{"actions": list(actions), "rewards": [reward, reward]}]


class SelectS1ConfigTest(unittest.TestCase):
    def test_evaluation_prioritizes_anti_false_aligns(self):
        # Chicken development behavior: differentiated profiles are better, so
        # a safe certificate must abstain. Stag Hunt: same-action profiles are
        # better and evidence is sufficient to permit alignment.
        data = [
            ("chicken", 42, [
                episode((0, 0), 1.0), episode((1, 1), 1.0), episode((0, 0), 1.0),
                episode((0, 1), 3.0), episode((1, 0), 3.0), episode((0, 1), 3.0),
            ]),
            ("stag_hunt", 42, [
                episode((0, 0), 3.0), episode((1, 1), 3.0), episode((0, 0), 3.0),
                episode((0, 1), 1.0), episode((1, 0), 1.0), episode((0, 1), 1.0),
            ]),
        ]
        config = SafeSCAConfig(
            warmup_episodes=6, tau=0.10, confidence=0.95,
            bootstrap_samples=500, min_profile_coverage=0.50,
            min_stratum_observations=3,
        )
        result = evaluate(config, data)
        self.assertEqual(result["anti_false_align_count"], 0)
        self.assertEqual(result["coord_align_count"], 1)
        self.assertIsInstance(ranking(result), tuple)


if __name__ == "__main__":
    unittest.main()
