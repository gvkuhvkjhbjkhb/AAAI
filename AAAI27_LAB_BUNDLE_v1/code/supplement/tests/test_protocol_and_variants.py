from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import hettom_baseline as hb
import p3_label_variants as variants
import p3_matrices
from supplement_protocol import load_protocol


class ProtocolVariantTests(unittest.TestCase):
    def test_frozen_protocol_loads(self):
        protocol = load_protocol(CODE.parent / "protocols" / "supplement_frozen_protocol.json")
        self.assertEqual(protocol["analysis"]["primary_endpoint"], "team_mean_payoff")
        self.assertEqual(protocol["p0"]["new_cells"], 320)
        self.assertEqual(protocol["p1"]["new_cells"], 240)

    def test_label_swap_changes_names_not_payoffs(self):
        for matrix_id in p3_matrices.matrix_ids():
            original = p3_matrices.make_game(hb, matrix_id)
            swapped = variants.make_game(hb, matrix_id)
            self.assertEqual(original.action_names, list(reversed(swapped.action_names)))
            for a0 in range(2):
                for a1 in range(2):
                    self.assertEqual(original.payoff_vector((a0, a1)),
                                     swapped.payoff_vector((a0, a1)))

    def test_variant_hash_is_stable(self):
        self.assertEqual(
            variants.registry_sha256(),
            "614431f8cad56ba59ca8046d4dc351ff40e401e3723e9122ac12bc57804f87a9",
        )

    def test_payoff_prompt_is_agent_relative(self):
        game = hb.MatrixGame(
            name="test", n_agents=2, n_actions=2, action_names=["A", "B"],
            payoff=lambda actions: {
                (0, 0): (1, 1), (0, 1): (2, 9),
                (1, 0): (7, 3), (1, 1): (4, 4),
            }[tuple(actions)],
        )
        wrapped = SimpleNamespace(base=game)
        agent2 = hb.LLMAgent(2, "dummy", 0.5, "player2", payoff_in_prompt=True)
        prompt = agent2._build_action_prompt(wrapped, "none")
        # Agent 2 choosing B while teammate chooses A means joint (A,B):
        # rewards (2,9), which must be printed in (self,teammate) order.
        b_row = next(line for line in prompt.splitlines() if line.strip().startswith("B:"))
        self.assertIn("(9.0,2.0)", b_row)
        self.assertNotIn("(7.0,3.0)", b_row)


if __name__ == "__main__":
    unittest.main()
