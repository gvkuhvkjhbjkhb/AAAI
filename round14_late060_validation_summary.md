# Round 14 Late-Phase Validation Summary

Round 14 tested whether `adaptive_0.0003_late060` could rescue the main claim after Round 13 showed that the original `adaptive_0.0003_late045` result was seed-sensitive. It did not. Seeds 9-16 for late060 nearly duplicate late045 behavior, so the late-phase weight change does not produce a robust improvement under this implementation and evaluation budget.

## Completion

- Full job: `job_1783496563291_vo5wnw`
- Result directory: `results/round14_late060_validation_round14_full_20260708_074243/`
- Artifact: `artifacts/AAAI_round14_late060_validation_round14_full_20260708_074243.tar.gz`
- Runs: 40 completed, 0 failed

## Main 10x10 Comparison, 16 Seeds

| method | n | final return | 95% CI | train AUC | best train | stability gap | penalty total | bonus total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 16 | 0.2460 | [0.2056, 0.2908] | 0.1899 | 0.4204 | 0.1744 | 0.0000 | 0.0000 |
| adaptive_0.0003_late045 | 16 | 0.2341 | [0.1933, 0.2790] | 0.1838 | 0.4143 | 0.1802 | 29.5294 | 0.0000 |
| adaptive_0.0003_late060 | 16 | 0.2325 | [0.1919, 0.2772] | 0.1831 | 0.4061 | 0.1737 | 29.6398 | 0.0000 |
| uniform_budget_matched_0.0003_late045 | 16 | 0.2378 | [0.2067, 0.2694] | 0.1850 | 0.4260 | 0.1882 | 36.5010 | 0.0000 |
| random_type_budget_matched_0.0003_late045 | 16 | 0.2270 | [0.1995, 0.2559] | 0.1799 | 0.4153 | 0.1884 | 32.2528 | 0.8889 |
| uniform_budget_matched_0.0003_late060 | 16 | 0.2346 | [0.2060, 0.2637] | 0.1847 | 0.4284 | 0.1938 | 36.8388 | 0.0000 |
| random_type_budget_matched_0.0003_late060 | 16 | 0.2259 | [0.1989, 0.2548] | 0.1799 | 0.4159 | 0.1900 | 32.5470 | 0.8885 |

## Paired Comparisons Against Late060 Adaptive

| comparison | metric | n | mean delta | 95% CI | conclusion |
|---|---|---:|---:|---:|---|
| adaptive_0.0003_late060 - baseline | last_test_return | 16 | -0.0136 | [-0.0712, 0.0425] | inconclusive |
| adaptive_0.0003_late060 - baseline | train_auc | 16 | -0.0068 | [-0.0381, 0.0244] | inconclusive |
| adaptive_0.0003_late060 - baseline | best_train_return | 16 | -0.0143 | [-0.1464, 0.1208] | inconclusive |
| adaptive_0.0003_late060 - uniform_budget_matched_0.0003_late060 | last_test_return | 16 | -0.0022 | [-0.0475, 0.0449] | inconclusive |
| adaptive_0.0003_late060 - uniform_budget_matched_0.0003_late060 | train_auc | 16 | -0.0016 | [-0.0289, 0.0254] | inconclusive |
| adaptive_0.0003_late060 - uniform_budget_matched_0.0003_late060 | best_train_return | 16 | -0.0223 | [-0.1194, 0.0780] | inconclusive |
| adaptive_0.0003_late060 - random_type_budget_matched_0.0003_late060 | last_test_return | 16 | 0.0065 | [-0.0488, 0.0670] | inconclusive |
| adaptive_0.0003_late060 - random_type_budget_matched_0.0003_late060 | train_auc | 16 | 0.0033 | [-0.0313, 0.0406] | inconclusive |
| adaptive_0.0003_late060 - random_type_budget_matched_0.0003_late060 | best_train_return | 16 | -0.0098 | [-0.1350, 0.1211] | inconclusive |
| adaptive_0.0003_late060 - adaptive_0.0003_late045 | last_test_return | 16 | -0.0017 | [-0.0049, 0.0016] | inconclusive |
| adaptive_0.0003_late060 - adaptive_0.0003_late045 | train_auc | 16 | -0.0007 | [-0.0014, -0.0001] | negative |
| adaptive_0.0003_late060 - adaptive_0.0003_late045 | best_train_return | 16 | -0.0081 | [-0.0203, 0.0005] | inconclusive |

## Seed-Block Diagnostic

| method | seeds | n | final return | train AUC |
|---|---|---:|---:|---:|
| adaptive_0.0003_late045 | 1-8 | 8 | 0.3042 | 0.2263 |
| adaptive_0.0003_late045 | 9-16 | 8 | 0.1641 | 0.1414 |
| adaptive_0.0003_late060 | 1-8 | 8 | 0.3009 | 0.2251 |
| adaptive_0.0003_late060 | 9-16 | 8 | 0.1640 | 0.1412 |
| baseline | 1-8 | 8 | 0.2258 | 0.1827 |
| baseline | 9-16 | 8 | 0.2663 | 0.1971 |

## Evidence Interpretation

The 16-seed late060 adaptive result is `0.2325`, slightly below the 16-seed baseline `0.2460` and phase-uniform late060 `0.2346`, and only slightly above random-type late060 `0.2259`. The paired final-return delta against baseline is `-0.0136` with a 95% CI of `[-0.0719, 0.0426]`, so there is no positive main effect. The seed-block diagnostic shows the same problem as Round 13: the apparent sensitivity success on seeds 1-8 disappears on seeds 9-16.

## Final Experimental Conclusion

After Rounds 13 and 14, the current project should not be submitted to AAAI as a positive method paper claiming robust failure-triggered adaptive reward shaping. The defensible conclusion is mixed/negative: reward shaping sometimes helps, but the adaptive mechanism is seed-sensitive and not reliably better than carefully budget-matched controls. To reach a credible AAAI-level positive submission, the method needs redesign followed by a fresh preregistered 16-seed evaluation. If submitting now, the safest framing is a rigorous negative/mixed empirical study of adaptive reward-shaping pitfalls in cooperative MARL.

