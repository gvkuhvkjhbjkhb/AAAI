"""CPU-only checks for immutable R0/S2 campaign setup."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from run_safe_sca_campaign import immutable_json, require_strict_preflight, safe_flags  # noqa: E402


FROZEN = {
    "warmup_episodes": 15,
    "tau": 0.1,
    "confidence": 0.95,
    "bootstrap_samples": 2000,
    "min_profile_coverage": 0.125,
    "min_stratum_observations": 3,
}


class CampaignSetupTest(unittest.TestCase):
    def test_safe_flags_are_frozen_values(self):
        flags = safe_flags(FROZEN)
        self.assertIn("15", flags)
        self.assertIn("0.125", flags)
        self.assertIn("--safe_tau", flags)

    def test_preflight_and_snapshot_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "Missing preflight"):
                require_strict_preflight(root)
            manifest = root / "ENVIRONMENT_MANIFEST_S1.json"
            manifest.write_text(json.dumps({"preflight_passed": True, "allow_version_mismatch": False}),
                                encoding="utf-8")
            require_strict_preflight(root)
            snapshot = root / "CAMPAIGN_SNAPSHOT.json"
            immutable_json(snapshot, {"campaign": "r0", "seeds": [62, 63]})
            immutable_json(snapshot, {"campaign": "r0", "seeds": [62, 63]})
            with self.assertRaisesRegex(RuntimeError, "snapshot differs"):
                immutable_json(snapshot, {"campaign": "s2", "seeds": [82]})


if __name__ == "__main__":
    unittest.main()
