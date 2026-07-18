from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from analyze_p3_transfer import aggregate
from p3_matrices import PROMPT_GAME_NAME, SPECS, registry_sha256
from p3_protocol import load_protocol


class P3ProtocolAndAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT / "protocols" / "p3_frozen_protocol.json")

    def test_registry_is_frozen_and_balanced(self):
        self.assertEqual(len(SPECS), 8)
        self.assertEqual(sum(s.analysis_category == "anti" for s in SPECS), 4)
        self.assertEqual(sum(s.analysis_category == "coord_or_boundary" for s in SPECS), 4)
        self.assertEqual(self.protocol["matrix_registry_sha256"], registry_sha256())
        self.assertTrue(all(PROMPT_GAME_NAME == "Anonymous interaction" for _ in SPECS))

    def test_synthetic_data_passes_predeclared_gates(self):
        data = {}
        for spec in SPECS:
            data[spec.matrix_id] = {}
            for seed in self.protocol["seeds"]:
                if spec.analysis_category == "anti":
                    safe_payoff, selected = 1.0, "NoAlign"
                else:
                    safe_payoff, selected = 1.5, "Gated"
                data[spec.matrix_id][seed] = {
                    "het_notom": {"team_mean_payoff": 1.0},
                    "het_gated_atom_talk": {"team_mean_payoff": 2.0},
                    "het_point_sca": {"s1_total_team_payoff": safe_payoff},
                    "het_safe_sca": {
                        "s1_total_team_payoff": safe_payoff,
                        "s1_selected_post_warmup_arm": selected,
                        "s1_decision_reasons": [], "s1_coverage": {},
                    },
                }
        report = aggregate(data, self.protocol)
        self.assertTrue(report["safety_pass_all_anti_matrices"])
        self.assertTrue(report["routing_pass"])
        self.assertTrue(report["utility_gate"]["passed"])
        self.assertTrue(report["method_p3_pass"])


if __name__ == "__main__":
    unittest.main()
