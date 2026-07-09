# Round 13 AAAI Main-Claim Hardening Report

Round 13 strengthens the main 10x10 claim by adding seeds 9-16, actual-budget matched controls, and sensitivity extensions.

## 10x10 Seed Extension

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 1 | 0.0668 | [0.0668, 0.0668] | 0.0617 | 0.0708 | 0.0040 |
| baseline | 1 | 0.0651 | [0.0651, 0.0651] | 0.0581 | 0.0708 | 0.0057 |

## 10x10 Seed Extension Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | seeds |
|---|---|---:|---:|---:|---|
| adaptive_0.0003_late045 - baseline | last_test_return | 1 | 0.0017 | [0.0017, 0.0017] | 9 |
| adaptive_0.0003_late045 - baseline | train_auc | 1 | 0.0036 | [0.0036, 0.0036] | 9 |
| adaptive_0.0003_late045 - baseline | best_train_return | 1 | 0.0000 | [0.0000, 0.0000] | 9 |
| adaptive_0.0003_late045 - baseline | stability_gap | 1 | -0.0017 | [-0.0017, -0.0017] | 9 |

## 10x10 Seed Extension Budget Accounting

| method | records | triggers | penalty total | bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 12.3333 | 19.0000 | 0.1492 | 0.0000 | 969.0000 | 0.0079 | 51.0000 |
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 10x10 Actual-Budget Controls

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| random_actual_budget_matched | 1 | 0.0632 | [0.0632, 0.0632] | 0.0849 | 0.1351 | 0.0719 |
| uniform_actual_budget_matched | 1 | 0.0715 | [0.0715, 0.0715] | 0.0863 | 0.1351 | 0.0636 |

## 10x10 Actual-Budget Controls Budget Accounting

| method | records | triggers | penalty total | bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_actual_budget_matched | 12.3333 | 20.0000 | 0.1390 | 0.0040 | 1020.0000 | 0.0069 | 51.0000 |
| uniform_actual_budget_matched | 12.3333 | 20.0000 | 0.1371 | 0.0000 | 1020.0000 | 0.0069 | 51.0000 |

## 10x10 Sensitivity Extension

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0002_late045 | 1 | 0.0893 | [0.0893, 0.0893] | 0.0861 | 0.1052 | 0.0159 |

## 10x10 Sensitivity Extension Budget Accounting

| method | records | triggers | penalty total | bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_0.0002_late045 | 11.6667 | 16.0000 | 0.0933 | 0.0000 | 816.0000 | 0.0058 | 51.0000 |

## Decision Rule

Use the main claim as AAAI-ready only if adaptive remains positive against baseline, phase-uniform, random-type, and actual-budget controls when merged with prior 10x10 evidence. Sensitivity variants should show that the adaptive family is not a single fragile point.
