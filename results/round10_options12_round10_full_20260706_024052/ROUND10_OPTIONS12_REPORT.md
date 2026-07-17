# Round 10 Options 1+2 Report

Round 10 implements the conservative AAAI package: mechanism defense through budget accounting and sensitivity, plus one LBF-family generalization task.

## New LBF Family Task

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 8 | 0.3100 | [0.2666, 0.3623] | 0.2439 | 0.4947 | 0.1847 |
| baseline | 8 | 0.2945 | [0.2439, 0.3510] | 0.2368 | 0.5033 | 0.2088 |
| random_type_budget_matched_0.0003_late045 | 8 | 0.2704 | [0.2472, 0.2949] | 0.2251 | 0.4255 | 0.1550 |
| uniform_budget_matched_0.0003_late045 | 8 | 0.2853 | [0.2551, 0.3283] | 0.2415 | 0.4486 | 0.1633 |

## New LBF Family Task Budget Accounting

| method | records | triggers | penalty total | terminal bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 2785.6250 | 3000.0000 | 27.8313 | 0.0000 | 152841.5000 | 0.0093 | 50.9472 |
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_type_budget_matched_0.0003_late045 | 2774.5750 | 3000.0000 | 30.4363 | 0.8805 | 153000.0000 | 0.0101 | 51.0000 |
| uniform_budget_matched_0.0003_late045 | 2745.8000 | 3000.0000 | 34.4002 | 0.0000 | 152988.1250 | 0.0115 | 50.9960 |

## New LBF Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | seeds |
|---|---|---:|---:|---:|---|
| adaptive_0.0003_late045 - baseline | last_test_return | 8 | 0.0155 | [-0.0678, 0.1017] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | train_auc | 8 | 0.0070 | [-0.0494, 0.0631] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | best_train_return | 8 | -0.0086 | [-0.1611, 0.1607] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | stability_gap | 8 | -0.0241 | [-0.0969, 0.0613] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | last_test_return | 8 | 0.0248 | [-0.0435, 0.0952] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | train_auc | 8 | 0.0023 | [-0.0399, 0.0462] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | best_train_return | 8 | 0.0462 | [-0.1222, 0.1905] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | stability_gap | 8 | 0.0214 | [-0.0906, 0.1178] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | last_test_return | 8 | 0.0396 | [-0.0082, 0.0883] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | train_auc | 8 | 0.0188 | [-0.0056, 0.0444] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | best_train_return | 8 | 0.0693 | [-0.0139, 0.1570] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | stability_gap | 8 | 0.0296 | [-0.0164, 0.0879] | 1,2,3,4,5,6,7,8 |

## 10x10 Sensitivity and Budget Controls

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0002_late045 | 4 | 0.2418 | [0.2138, 0.2693] | 0.1873 | 0.4618 | 0.2201 |
| adaptive_0.0003_late060 | 4 | 0.3039 | [0.2260, 0.3944] | 0.2288 | 0.5604 | 0.2565 |
| adaptive_0.0005_late045 | 4 | 0.2376 | [0.1654, 0.3236] | 0.1931 | 0.4034 | 0.1658 |
| random_type_budget_matched_0.0003_late045 | 4 | 0.1986 | [0.1487, 0.2374] | 0.1603 | 0.3294 | 0.1308 |
| uniform_budget_matched_0.0003_late045 | 4 | 0.2537 | [0.1892, 0.3223] | 0.1964 | 0.4602 | 0.2064 |

## 10x10 Sensitivity and Budget Controls Budget Accounting

| method | records | triggers | penalty total | terminal bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_0.0002_late045 | 2999.6000 | 3000.0000 | 19.6528 | 0.0000 | 153000.0000 | 0.0066 | 51.0000 |
| adaptive_0.0003_late060 | 3000.0000 | 3000.0000 | 29.6587 | 0.0000 | 152941.5000 | 0.0099 | 50.9805 |
| adaptive_0.0005_late045 | 2981.3000 | 3000.0000 | 49.1299 | 0.0000 | 153000.0000 | 0.0164 | 51.0000 |
| random_type_budget_matched_0.0003_late045 | 2992.8000 | 3000.0000 | 31.9625 | 0.8904 | 153000.0000 | 0.0107 | 51.0000 |
| uniform_budget_matched_0.0003_late045 | 3000.0000 | 3000.0000 | 37.3217 | 0.0000 | 153000.0000 | 0.0124 | 51.0000 |

## Interpretation Guide

Use the new LBF family task as main-text generalization evidence only if adaptive beats baseline and remains competitive with phase-uniform/random controls. Use the sensitivity panel to show that Round 8's 0.0003/late0.45 setting is not an isolated cherry-pick and to quantify actual shaping budget.
