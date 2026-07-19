from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from p3_matrices import registry_sha256
from p3_protocol import load_protocol, sha256_json
from validate_p3_results import validate


class P3ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol(ROOT / "protocols" / "p3_frozen_protocol.json")

    def test_complete_minimal_fixture_is_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = {
                "protocol_sha256": sha256_json(self.protocol),
                "matrix_registry_sha256": registry_sha256(),
            }
            (root / "P3_CAMPAIGN_SNAPSHOT.json").write_text(json.dumps(snapshot), encoding="utf-8")
            for matrix in self.protocol["matrix_ids"]:
                for seed in self.protocol["seeds"]:
                    for policy in self.protocol["policies"]:
                        path = root / matrix / f"seed_{seed}" / policy / "metrics.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        metrics = {"team_mean_payoff": 1.0}
                        if policy == "het_safe_sca":
                            metrics.update({
                                "s1_policy": policy,
                                "s1_safe_config": self.protocol["safe_sca"],
                                "s1_selected_post_warmup_arm": "NoAlign",
                            })
                        path.write_text(json.dumps(metrics), encoding="utf-8")
            report = validate(root, self.protocol)
            self.assertTrue(report["ready_for_analysis"])
            self.assertEqual(report["checked_metrics"], 320)


if __name__ == "__main__":
    unittest.main()
