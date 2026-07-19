from pathlib import Path
import sys
import unittest


CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from analyze_supplement_results import bootstrap_mean_ci, holm_adjust, wilson


class AnalysisTests(unittest.TestCase):
    def test_bootstrap_constant(self):
        mean, lo, hi, p = bootstrap_mean_ci([0.2] * 10, 1000, 7)
        self.assertAlmostEqual(mean, 0.2)
        self.assertAlmostEqual(lo, 0.2)
        self.assertAlmostEqual(hi, 0.2)
        self.assertEqual(p, 0.0)

    def test_holm_monotonic_in_sorted_order(self):
        raw = [0.01, 0.04, 0.03]
        adjusted = holm_adjust(raw)
        ordered = [adjusted[i] for i in sorted(range(3), key=lambda i: raw[i])]
        self.assertEqual(ordered, sorted(ordered))
        self.assertTrue(all(a >= p for a, p in zip(adjusted, raw)))

    def test_wilson_bounds(self):
        lo, hi = wilson(8, 10)
        self.assertLess(lo, 0.8)
        self.assertGreater(hi, 0.8)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)


if __name__ == "__main__":
    unittest.main()

