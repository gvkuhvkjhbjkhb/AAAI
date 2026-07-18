"""Contract tests for R0 same-seed replay comparison."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from compare_safe_sca_replay import compare  # noqa: E402


def write_metrics(root: Path, cell: str, payoff: float, arm: str | None = None) -> None:
    path = root / "chicken" / "seed_62" / cell / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"team_mean_payoff": payoff}
    if cell == "het_safe_sca":
        payload.update({"s1_total_team_payoff": payoff, "s1_selected_post_warmup_arm": arm})
    path.write_text(json.dumps(payload), encoding="utf-8")


class ReplayComparisonTest(unittest.TestCase):
    def test_exact_match_passes(self):
        with tempfile.TemporaryDirectory() as reference_tmp, tempfile.TemporaryDirectory() as replay_tmp:
            reference = Path(reference_tmp)
            replay = Path(replay_tmp)
            for root in [reference, replay]:
                write_metrics(root, "het_notom", 2.0)
                write_metrics(root, "het_safe_sca", 2.1, "Gated")
            report = compare(
                reference, replay, games=["chicken"], seeds=[62],
                cells=["het_notom", "het_safe_sca"], payoff_tolerance=0.0,
                route_mismatch_budget=0,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["compared_rows"], 2)

    def test_route_or_payoff_change_fails(self):
        with tempfile.TemporaryDirectory() as reference_tmp, tempfile.TemporaryDirectory() as replay_tmp:
            reference = Path(reference_tmp)
            replay = Path(replay_tmp)
            write_metrics(reference, "het_notom", 2.0)
            write_metrics(reference, "het_safe_sca", 2.1, "Gated")
            write_metrics(replay, "het_notom", 2.0)
            write_metrics(replay, "het_safe_sca", 2.2, "NoAlign")
            report = compare(
                reference, replay, games=["chicken"], seeds=[62],
                cells=["het_notom", "het_safe_sca"], payoff_tolerance=0.0,
                route_mismatch_budget=0,
            )
            self.assertFalse(report["passed"])
            self.assertEqual(report["route_mismatches"], 1)
            self.assertEqual(report["payoff_violations"], 1)


if __name__ == "__main__":
    unittest.main()
