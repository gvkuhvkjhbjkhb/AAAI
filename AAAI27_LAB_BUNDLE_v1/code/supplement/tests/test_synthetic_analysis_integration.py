import json
from pathlib import Path
import sys
import tempfile
import unittest


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from analyze_supplement_results import analyze_p0, analyze_p1, analyze_p2_domain
from supplement_protocol import load_protocol


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class SyntheticIntegrationTests(unittest.TestCase):
    def test_all_three_analyses_on_complete_synthetic_grid(self):
        protocol = load_protocol(CODE.parent / "protocols" / "supplement_frozen_protocol.json")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            reference = base / "reference"
            supplement = base / "supplement"
            for matrix in protocol["p3"]["matrix_ids"]:
                for seed in protocol["p3"]["seeds"]:
                    for policy, value in [
                        ("het_notom", 1.0), ("het_gated_atom_talk", 1.2),
                        ("het_safe_sca", 1.1),
                    ]:
                        write_json(reference / matrix / f"seed_{seed}" / policy / "metrics.json",
                                   {"team_mean_payoff": value})
                    write_json(reference / matrix / f"seed_{seed}" / "het_safe_sca" / "decision.json",
                               {"selected_post_warmup_arm": "NoAlign"})
                    for policy, value in [
                        ("het_notom", 1.0), ("het_gated_atom_talk", 1.2),
                        ("het_safe_sca", 1.1),
                        ("het_payoff_prompt", 1.3),
                    ]:
                        write_json(supplement / "p0_payoff_prompt" / matrix /
                                   f"seed_{seed}" / policy / "metrics.json",
                                   {"team_mean_payoff": value})
                    write_json(supplement / "p0_payoff_prompt" / matrix /
                               f"seed_{seed}" / "het_safe_sca" / "decision.json",
                               {"selected_post_warmup_arm": "NoAlign"})
                    for policy, value in [
                        ("het_notom", 1.0), ("het_gated_atom_talk", 1.2),
                        ("het_safe_sca", 1.1),
                    ]:
                        write_json(supplement / "p1_label_swap" / matrix /
                                   f"seed_{seed}" / policy / "metrics.json",
                                   {"team_mean_payoff": value})
                    write_json(supplement / "p1_label_swap" / matrix /
                               f"seed_{seed}" / "het_safe_sca" / "decision.json",
                               {"selected_post_warmup_arm": "NoAlign"})
                    write_json(supplement / "p2_teammean_bandit_p3" / matrix /
                               f"seed_{seed}" / "het_bandit_teammean" / "metrics.json",
                               {"team_mean_payoff": 1.2, "bandit_chosen_arm": "Gated",
                                "bandit_online_total_team_mean_payoff": 1.15})
            p0, rows0 = analyze_p0(protocol, supplement, reference, 100)
            p1, rows1 = analyze_p1(protocol, supplement, reference, 100)
            p2, rows2 = analyze_p2_domain(
                protocol, supplement / "p2_teammean_bandit_p3", reference,
                protocol["p3"]["matrix_ids"], protocol["p3"]["seeds"], 100, "p3")
            self.assertEqual(len(rows0), 80)
            self.assertEqual(len(rows1), 80)
            self.assertEqual(len(rows2), 80)
            self.assertAlmostEqual(p0["hierarchical_effect_vs_noalign"]["mean"], 0.3)
            self.assertTrue(p1["label_robustness_gate_pass"])
            self.assertEqual(p2["selection_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
