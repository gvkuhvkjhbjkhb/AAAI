"""Fixture-level tests for S1's total-horizon analysis contract."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from analyze_s1_safe_sca import aggregate, load, render_markdown  # noqa: E402


def write_metrics(root: Path, game: str, seed: int, cell: str, payload: dict) -> None:
    path = root / game / f"seed_{seed}" / cell / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class AnalyzeS1Test(unittest.TestCase):
    def test_total_horizon_pairing_and_routing_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for game, selected in [("chicken", "NoAlign"), ("stag_hunt", "Gated")]:
                for seed in [62, 63]:
                    write_metrics(root, game, seed, "het_notom", {"team_mean_payoff": 2.0})
                    write_metrics(root, game, seed, "het_gated_atom_talk", {"team_mean_payoff": 2.4})
                    write_metrics(root, game, seed, "het_safe_sca", {
                        "s1_total_team_payoff": 2.0 if game == "chicken" else 2.3,
                        "s1_selected_post_warmup_arm": selected,
                        "s1_decision_reasons": ["fixture"],
                        "s1_coverage": {"profile_coverage": 0.5},
                    })
            report = aggregate(load(root), safety_margin=0.10)
            self.assertTrue(report["safety_pass_all_anti_games"])
            self.assertFalse(report["utility_gate"]["passed"])
            self.assertFalse(report["method_paper_pass"])
            self.assertEqual(report["safe_sca_routing"]["anti_false_align"], 0)
            self.assertEqual(report["safe_sca_routing"]["coord_false_abstain"], 0)
            markdown = render_markdown(report)
            self.assertIn("total_horizon_team_payoff_including_warmup", markdown)
            self.assertIn("Safe-SCA", markdown)


if __name__ == "__main__":
    unittest.main()
