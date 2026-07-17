# Round 13 AAAI Main-Claim Hardening Summary

Round 13 completed the planned hardening package: 60/60 full 500k runs completed with 0 failures. The new evidence does not support the original strong positive main claim; seeds 9-16 reverse the Round 8 advantage, so the paper must either be reframed as a mixed/negative study or the method must be redesigned before an AAAI-level submission.

## Completion

- Full job: `job_1783451969212_mg9ad2`
- Result directory: `results/round13_hardening_round13_full_20260707_191929/`
- Artifact: `artifacts/AAAI_round13_hardening_round13_full_20260707_191929.tar.gz`
- Runs: 60 completed, 0 failed

## Main 10x10, Round 8 + Round 13 Merged Seeds 1-16

| method | n | final return | 95% CI | train AUC | best train | penalty total | bonus total |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 16 | 0.2460 | [0.2060, 0.2907] | 0.1899 | 0.4204 | 0.0000 | 0.0000 |
| adaptive_0.0003_late045 | 16 | 0.2341 | [0.1923, 0.2792] | 0.1838 | 0.4143 | 29.5294 | 0.0000 |
| uniform_budget_matched_0.0003_late045 | 16 | 0.2378 | [0.2069, 0.2697] | 0.1850 | 0.4260 | 36.5010 | 0.0000 |
| random_type_budget_matched_0.0003_late045 | 16 | 0.2270 | [0.1996, 0.2569] | 0.1799 | 0.4153 | 32.2528 | 0.8889 |
| diagnosis_only | 8 | 0.2258 | [0.1826, 0.2784] | 0.1827 | 0.3532 | NA | NA |
| semantic_shuffled_budget_matched_0.0003_late045 | 8 | 0.2580 | [0.2158, 0.3091] | 0.2012 | 0.3935 | NA | NA |

## Paired Main Comparisons, Merged Seeds 1-16

| comparison | metric | n | mean delta | 95% CI | interpretation |
|---|---|---:|---:|---:|---|
| adaptive - baseline | last_test_return | 16 | -0.0119 | [-0.0726, 0.0460] | not positive |
| adaptive - baseline | train_auc | 16 | -0.0061 | [-0.0390, 0.0248] | not positive |
| adaptive - baseline | best_train_return | 16 | -0.0061 | [-0.1448, 0.1279] | not positive |
| adaptive - uniform_budget_matched_0.0003_late045 | last_test_return | 16 | -0.0037 | [-0.0542, 0.0474] | not positive |
| adaptive - uniform_budget_matched_0.0003_late045 | train_auc | 16 | -0.0012 | [-0.0289, 0.0261] | not positive |
| adaptive - uniform_budget_matched_0.0003_late045 | best_train_return | 16 | -0.0117 | [-0.1030, 0.0829] | not positive |
| adaptive - random_type_budget_matched_0.0003_late045 | last_test_return | 16 | 0.0072 | [-0.0476, 0.0671] | not positive |
| adaptive - random_type_budget_matched_0.0003_late045 | train_auc | 16 | 0.0040 | [-0.0307, 0.0417] | not positive |
| adaptive - random_type_budget_matched_0.0003_late045 | best_train_return | 16 | -0.0010 | [-0.1248, 0.1304] | not positive |

## Round 13 Seed Extension Alone, Seeds 9-16

Seeds 9-16 are the critical stress test: adaptive is below baseline and both budget-matched controls on final return and train AUC. This invalidates a robust 10x10 improvement claim.

| method | n | final return | 95% CI | train AUC |
|---|---:|---:|---:|---:|
| baseline | 8 | 0.2663 | [0.2015, 0.3335] | 0.1971 |
| adaptive_0.0003_late045 | 8 | 0.1641 | [0.1442, 0.1822] | 0.1414 |
| uniform_budget_matched_0.0003_late045 | 8 | 0.2409 | [0.2028, 0.2816] | 0.1825 |
| random_type_budget_matched_0.0003_late045 | 8 | 0.2347 | [0.1987, 0.2747] | 0.1838 |

## Actual-Budget Controls, Seeds 1-8

The actual-budget controls match adaptive's realized shaping budget closely, but the random actual-budget control outperforms adaptive Round 8 on the same seeds. This weakens a mechanism claim based on adaptive targeting.

| method | n | final return | 95% CI | train AUC | penalty total | bonus total |
|---|---:|---:|---:|---:|---:|---:|
| random_actual_budget_matched | 8 | 0.2636 | [0.2311, 0.2992] | 0.2055 | 29.4139 | 0.8025 |
| uniform_actual_budget_matched | 8 | 0.2190 | [0.1823, 0.2617] | 0.1765 | 29.3866 | 0.0000 |

| paired comparison | n | final delta | 95% CI | train AUC delta | 95% CI |
|---|---:|---:|---:|---:|---:|
| adaptive_round8 - random_actual_budget_matched | 8 | 0.0406 | [-0.0229, 0.0974] | 0.0207 | [-0.0150, 0.0521] |
| adaptive_round8 - uniform_actual_budget_matched | 8 | 0.0851 | [0.0121, 0.1560] | 0.0497 | [0.0114, 0.0920] |

## Sensitivity, Merged Seeds 1-8

| method | n | final return | 95% CI | train AUC | penalty total |
|---|---:|---:|---:|---:|---:|
| adaptive_0.0002_late045 | 8 | 0.2435 | [0.2017, 0.2850] | 0.1900 | 19.6067 |
| adaptive_0.0003_late060 | 8 | 0.3009 | [0.2535, 0.3519] | 0.2251 | 29.7316 |
| adaptive_0.0005_late045 | 8 | 0.2451 | [0.1907, 0.3084] | 0.1889 | 49.3037 |
| random_type_budget_matched_0.0003_late045 | 4 | 0.1986 | [0.1487, 0.2374] | 0.1603 | 31.9625 |
| uniform_budget_matched_0.0003_late045 | 4 | 0.2537 | [0.1892, 0.3223] | 0.1964 | 37.3217 |

## Paper-Level Conclusion

The completed hardening package reduces, rather than increases, the AAAI readiness of the current method. The honest conclusion is that the original Round 8 improvement was seed-sensitive. The current paper should not claim robust improvement on the main LBF benchmark. The best academically defensible path is either to submit a transparent negative/mixed empirical analysis, or to redesign the adaptive intervention and run a new preregistered 16-seed evaluation.
