# Round 13 AAAI Main-Claim Hardening Report

Round 13 strengthens the main 10x10 claim by adding seeds 9-16, actual-budget matched controls, and sensitivity extensions.

## 10x10 Seed Extension

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 8 | 0.1641 | [0.1441, 0.1816] | 0.1414 | 0.2966 | 0.1325 |
| baseline | 8 | 0.2663 | [0.2020, 0.3337] | 0.1971 | 0.4876 | 0.2213 |
| random_type_budget_matched_0.0003_late045 | 8 | 0.2347 | [0.1992, 0.2760] | 0.1838 | 0.4570 | 0.2223 |
| uniform_budget_matched_0.0003_late045 | 8 | 0.2409 | [0.2022, 0.2827] | 0.1825 | 0.4437 | 0.2028 |

## 10x10 Seed Extension Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | seeds |
|---|---|---:|---:|---:|---|
| adaptive_0.0003_late045 - baseline | last_test_return | 8 | -0.1022 | [-0.1658, -0.0436] | 9,10,11,12,13,14,15,16 |
| adaptive_0.0003_late045 - baseline | train_auc | 8 | -0.0557 | [-0.0897, -0.0236] | 9,10,11,12,13,14,15,16 |
| adaptive_0.0003_late045 - baseline | best_train_return | 8 | -0.1910 | [-0.3478, -0.0340] | 9,10,11,12,13,14,15,16 |
| adaptive_0.0003_late045 - baseline | stability_gap | 8 | -0.0888 | [-0.1979, 0.0300] | 9,10,11,12,13,14,15,16 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | last_test_return | 8 | -0.0769 | [-0.1207, -0.0339] | 9,10,11,12,13,14,15,16 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | train_auc | 8 | -0.0411 | [-0.0686, -0.0174] | 9,10,11,12,13,14,15,16 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | best_train_return | 8 | -0.1471 | [-0.2189, -0.0690] | 9,10,11,12,13,14,15,16 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | stability_gap | 8 | -0.0703 | [-0.1159, -0.0182] | 9,10,11,12,13,14,15,16 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | last_test_return | 8 | -0.0707 | [-0.1137, -0.0312] | 9,10,11,12,13,14,15,16 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | train_auc | 8 | -0.0424 | [-0.0710, -0.0222] | 9,10,11,12,13,14,15,16 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | best_train_return | 8 | -0.1604 | [-0.2552, -0.0628] | 9,10,11,12,13,14,15,16 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | stability_gap | 8 | -0.0898 | [-0.1726, 0.0002] | 9,10,11,12,13,14,15,16 |

## 10x10 Seed Extension Budget Accounting

| method | records | triggers | penalty total | bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 3000.0000 | 3000.0000 | 29.5294 | 0.0000 | 153000.0000 | 0.0098 | 51.0000 |
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_type_budget_matched_0.0003_late045 | 2994.7000 | 3000.0000 | 32.2528 | 0.8889 | 153000.0000 | 0.0108 | 51.0000 |
| uniform_budget_matched_0.0003_late045 | 2990.1000 | 3000.0000 | 36.5010 | 0.0000 | 153000.0000 | 0.0122 | 51.0000 |

## 10x10 Actual-Budget Controls

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| random_actual_budget_matched | 8 | 0.2636 | [0.2307, 0.2999] | 0.2055 | 0.4806 | 0.2171 |
| uniform_actual_budget_matched | 8 | 0.2190 | [0.1821, 0.2613] | 0.1765 | 0.3584 | 0.1394 |

## 10x10 Actual-Budget Controls Budget Accounting

| method | records | triggers | penalty total | bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_actual_budget_matched | 3000.0000 | 3000.0000 | 29.4139 | 0.8025 | 153000.0000 | 0.0098 | 51.0000 |
| uniform_actual_budget_matched | 2995.9000 | 3000.0000 | 29.3866 | 0.0000 | 153000.0000 | 0.0098 | 51.0000 |

## 10x10 Sensitivity Extension

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0002_late045 | 4 | 0.2452 | [0.1660, 0.3245] | 0.1928 | 0.4266 | 0.1814 |
| adaptive_0.0003_late060 | 4 | 0.2979 | [0.2434, 0.3551] | 0.2213 | 0.4798 | 0.1820 |
| adaptive_0.0005_late045 | 4 | 0.2526 | [0.1812, 0.3589] | 0.1848 | 0.4218 | 0.1692 |

## 10x10 Sensitivity Extension Budget Accounting

| method | records | triggers | penalty total | bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_0.0002_late045 | 2997.9500 | 3000.0000 | 19.5606 | 0.0000 | 153000.0000 | 0.0065 | 51.0000 |
| adaptive_0.0003_late060 | 3000.0000 | 3000.0000 | 29.8045 | 0.0000 | 153000.0000 | 0.0099 | 51.0000 |
| adaptive_0.0005_late045 | 3000.0000 | 3000.0000 | 49.4775 | 0.0000 | 153000.0000 | 0.0165 | 51.0000 |

## Decision Rule

Use the main claim as AAAI-ready only if adaptive remains positive against baseline, phase-uniform, random-type, and actual-budget controls when merged with prior 10x10 evidence. Sensitivity variants should show that the adaptive family is not a single fragile point.
