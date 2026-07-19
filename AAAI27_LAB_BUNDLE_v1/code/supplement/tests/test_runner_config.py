from pathlib import Path
import sys
import unittest


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from run_supplement_task import episode_count, runner_namespace
from supplement_protocol import load_protocol


class RunnerConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(CODE.parent / "protocols" / "supplement_frozen_protocol.json")

    def test_runner_freeze(self):
        ns = runner_namespace(self.protocol, Path("results"), ["het_payoff_prompt"], 30)
        self.assertEqual(ns.top_p, 0.9)
        self.assertEqual(ns.safe_warmup, 15)
        self.assertTrue(ns.use_vllm)
        self.assertFalse(ns.force)

    def test_episode_accounting(self):
        self.assertEqual(episode_count(self.protocol, "p3", "p3_m01"), 30)
        self.assertEqual(episode_count(self.protocol, "source", "public_goods"), 20)
        self.assertEqual(episode_count(self.protocol, "source", "chicken"), 30)


if __name__ == "__main__":
    unittest.main()

