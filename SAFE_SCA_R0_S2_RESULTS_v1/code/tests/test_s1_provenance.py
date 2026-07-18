"""Fixture-level checks for S1's fail-closed provenance gates."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from run_s1_safe_sca import require_successful_preflight  # noqa: E402
from preflight_s1 import EXPECTED, evaluate  # noqa: E402
from validate_s1_results import validate  # noqa: E402


class S1ProvenanceTest(unittest.TestCase):
    frozen = {
        "warmup_episodes": 5,
        "tau": 0.1,
        "confidence": 0.95,
        "bootstrap_samples": 2000,
        "min_profile_coverage": 0.25,
        "min_stratum_observations": 2,
    }

    def write_manifest(self, root: Path, *, allowed_mismatch: bool = False) -> None:
        (root / "ENVIRONMENT_MANIFEST_S1.json").write_text(json.dumps({
            "preflight_passed": True,
            "allow_version_mismatch": allowed_mismatch,
        }), encoding="utf-8")

    def write_safe_cell(self, root: Path, *, oracle_used: bool = False) -> None:
        seed_dir = root / "chicken" / "seed_62"
        seed_dir.mkdir(parents=True)
        (seed_dir / "arm_order.json").write_text(json.dumps({
            "arm_order": ["het_safe_sca"],
        }), encoding="utf-8")
        cell_dir = seed_dir / "het_safe_sca"
        cell_dir.mkdir()
        (cell_dir / "metrics.json").write_text(json.dumps({
            "n_episodes": 3,
            "s1_safe_config": self.frozen,
            "s1_total_episode_team_payoffs": [2.0, 2.1, 2.2],
            "s1_oracle_label_used": oracle_used,
        }), encoding="utf-8")
        (cell_dir / "decision.json").write_text("{}", encoding="utf-8")

    def valid_preflight_manifest(self) -> dict:
        return {
            "package_versions": EXPECTED.copy(),
            "gpu": {
                "available": True,
                "gpus": [
                    "NVIDIA GeForce RTX 5090, 32607 MiB, 580.00, 12.0",
                    "NVIDIA GeForce RTX 5090, 32607 MiB, 580.00, 12.0",
                ],
                "error": None,
            },
            "vllm_endpoints": {
                "qwen": {"reachable": True, "models": ["Qwen/Qwen2.5-7B-Instruct"], "error": None},
                "glm": {"reachable": True, "models": ["THUDM/GLM-4-9B-0414"], "error": None},
            },
        }

    def test_strict_preflight_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "manifest missing"):
                require_successful_preflight(root)
            self.write_manifest(root)
            require_successful_preflight(root)
            self.write_manifest(root, allowed_mismatch=True)
            with self.assertRaisesRegex(RuntimeError, "allow-version-mismatch"):
                require_successful_preflight(root)

    def test_preflight_override_never_masks_a_wrong_deployment(self):
        manifest = self.valid_preflight_manifest()
        self.assertEqual(evaluate(manifest, allow_version_mismatch=False), [])
        manifest["package_versions"]["torch"] = "other"
        self.assertEqual(evaluate(manifest, allow_version_mismatch=True), [])
        manifest["vllm_endpoints"]["glm"]["models"] = ["wrong-model"]
        failures = evaluate(manifest, allow_version_mismatch=True)
        self.assertIn("glm endpoint does not advertise", failures[0])

    def test_validator_accepts_complete_label_free_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_safe_cell(root)
            report = validate(root, self.frozen, games=["chicken"], seeds=[62],
                              cells=["het_safe_sca"], expected_episodes=3)
            self.assertTrue(report["ready_for_analysis"])

    def test_validator_rejects_oracle_contamination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_manifest(root)
            self.write_safe_cell(root, oracle_used=True)
            report = validate(root, self.frozen, games=["chicken"], seeds=[62],
                              cells=["het_safe_sca"], expected_episodes=3)
            self.assertFalse(report["ready_for_analysis"])
            self.assertIn("deployable policy used oracle label", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
