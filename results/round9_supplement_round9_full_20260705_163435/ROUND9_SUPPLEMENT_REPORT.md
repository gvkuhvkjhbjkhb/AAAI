# Round 9 Supplemental Stabilization Report

Round 9 targets the remaining AAAI risk: cross-domain generalization and stronger 12x12 seed coverage. It keeps the conservative failure-triggered adaptive shaping claim and treats semantic diagnosis as exploratory.

## LBF 12x12 Seed Extension

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 4 | 0.1069 | [0.0897, 0.1283] | 0.0856 | 0.1322 | 0.0253 |
| baseline | 4 | 0.1059 | [0.0881, 0.1202] | 0.0823 | 0.1502 | 0.0442 |
| diagnosis_only | 4 | 0.1059 | [0.0881, 0.1202] | 0.0823 | 0.1502 | 0.0442 |
| random_type_budget_matched_0.0003_late045 | 4 | 0.1233 | [0.1113, 0.1407] | 0.0976 | 0.1629 | 0.0396 |
| semantic_shuffled_budget_matched_0.0003_late045 | 4 | 0.0938 | [0.0811, 0.1031] | 0.0836 | 0.1498 | 0.0560 |
| uniform_budget_matched_0.0003_late045 | 4 | 0.0848 | [0.0555, 0.1279] | 0.0743 | 0.1347 | 0.0498 |

## LBF 12x12 Seed Extension Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | Cliff delta | shared seeds |
|---|---|---:|---:|---:|---:|---|
| adaptive_0.0003_late045 - baseline | last_test_return | 4 | 0.0009 | [-0.0306, 0.0394] | 0.0000 | 9,10,11,12 |
| adaptive_0.0003_late045 - baseline | train_auc | 4 | 0.0033 | [-0.0093, 0.0171] | 0.3750 | 9,10,11,12 |
| adaptive_0.0003_late045 - baseline | best_train_return | 4 | -0.0180 | [-0.0438, 0.0078] | -0.3750 | 9,10,11,12 |
| adaptive_0.0003_late045 - baseline | stability_gap | 4 | -0.0189 | [-0.0341, -0.0056] | -0.7500 | 9,10,11,12 |
| adaptive_0.0003_late045 - diagnosis_only | last_test_return | 4 | 0.0009 | [-0.0306, 0.0394] | 0.0000 | 9,10,11,12 |
| adaptive_0.0003_late045 - diagnosis_only | train_auc | 4 | 0.0033 | [-0.0093, 0.0171] | 0.3750 | 9,10,11,12 |
| adaptive_0.0003_late045 - diagnosis_only | best_train_return | 4 | -0.0180 | [-0.0438, 0.0078] | -0.3750 | 9,10,11,12 |
| adaptive_0.0003_late045 - diagnosis_only | stability_gap | 4 | -0.0189 | [-0.0341, -0.0056] | -0.7500 | 9,10,11,12 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | last_test_return | 4 | 0.0220 | [-0.0324, 0.0718] | 0.5000 | 9,10,11,12 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | train_auc | 4 | 0.0114 | [-0.0181, 0.0356] | 0.5000 | 9,10,11,12 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | best_train_return | 4 | -0.0025 | [-0.0670, 0.0620] | 0.2500 | 9,10,11,12 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | stability_gap | 4 | -0.0245 | [-0.0473, 0.0075] | -0.7500 | 9,10,11,12 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | last_test_return | 4 | -0.0164 | [-0.0351, 0.0091] | -0.6250 | 9,10,11,12 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | train_auc | 4 | -0.0120 | [-0.0222, 0.0054] | -0.6250 | 9,10,11,12 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | best_train_return | 4 | -0.0307 | [-0.0743, 0.0129] | -0.5000 | 9,10,11,12 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | stability_gap | 4 | -0.0143 | [-0.0566, 0.0269] | 0.1250 | 9,10,11,12 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | last_test_return | 4 | 0.0130 | [-0.0037, 0.0295] | 0.2500 | 9,10,11,12 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | train_auc | 4 | 0.0020 | [-0.0086, 0.0094] | 0.2500 | 9,10,11,12 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | best_train_return | 4 | -0.0177 | [-0.0459, 0.0106] | 0.0000 | 9,10,11,12 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | stability_gap | 4 | -0.0307 | [-0.0536, -0.0046] | -0.6250 | 9,10,11,12 |

## RWARE Tiny Cross-Domain

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 5 | 0.0000 | [0.0000, 0.0000] | 0.0003 | 0.0012 | 0.0012 |
| baseline | 5 | 0.0000 | [0.0000, 0.0000] | 0.0002 | 0.0010 | 0.0010 |
| diagnosis_only | 5 | 0.0000 | [0.0000, 0.0000] | 0.0002 | 0.0010 | 0.0010 |
| random_type_budget_matched_0.0003_late045 | 5 | 0.0000 | [0.0000, 0.0000] | 0.0001 | 0.0005 | 0.0005 |
| semantic_shuffled_budget_matched_0.0003_late045 | 5 | 0.0000 | [0.0000, 0.0000] | 0.0003 | 0.0012 | 0.0012 |
| uniform_budget_matched_0.0003_late045 | 5 | 0.0000 | [0.0000, 0.0000] | 0.0002 | 0.0006 | 0.0006 |

## RWARE Tiny Cross-Domain Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | Cliff delta | shared seeds |
|---|---|---:|---:|---:|---:|---|
| adaptive_0.0003_late045 - baseline | last_test_return | 5 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - baseline | train_auc | 5 | 0.0000 | [-0.0001, 0.0002] | 0.1200 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - baseline | best_train_return | 5 | 0.0002 | [-0.0007, 0.0011] | 0.2400 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - baseline | stability_gap | 5 | 0.0002 | [-0.0007, 0.0011] | 0.2400 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - diagnosis_only | last_test_return | 5 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - diagnosis_only | train_auc | 5 | 0.0000 | [-0.0001, 0.0002] | 0.1200 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - diagnosis_only | best_train_return | 5 | 0.0002 | [-0.0007, 0.0011] | 0.2400 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - diagnosis_only | stability_gap | 5 | 0.0002 | [-0.0007, 0.0011] | 0.2400 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | last_test_return | 5 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | train_auc | 5 | 0.0001 | [0.0000, 0.0003] | 0.2800 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | best_train_return | 5 | 0.0006 | [0.0000, 0.0012] | 0.3200 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | stability_gap | 5 | 0.0006 | [0.0000, 0.0012] | 0.3200 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | last_test_return | 5 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | train_auc | 5 | 0.0001 | [0.0000, 0.0004] | 0.4800 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | best_train_return | 5 | 0.0007 | [0.0001, 0.0014] | 0.5200 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | stability_gap | 5 | 0.0007 | [0.0001, 0.0014] | 0.5200 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | last_test_return | 5 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | train_auc | 5 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | best_train_return | 5 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | stability_gap | 5 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 1,2,3,4,5 |

## VMAS Navigation Cross-Domain

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 5 | 6.8498 | [6.3596, 7.2903] | 4.7582 | 11.5938 | 4.7440 |
| baseline | 5 | 6.8142 | [5.9482, 7.3773] | 4.9329 | 11.7876 | 4.9735 |
| diagnosis_only | 5 | 6.8142 | [5.9482, 7.3773] | 4.9329 | 11.7876 | 4.9735 |
| random_type_budget_matched_0.0003_late045 | 5 | 7.7545 | [7.4996, 8.0330] | 5.1954 | 11.7480 | 3.9935 |
| uniform_budget_matched_0.0003_late045 | 5 | 7.0581 | [6.3185, 7.8587] | 4.8923 | 11.6076 | 4.5495 |

## VMAS Navigation Cross-Domain Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | Cliff delta | shared seeds |
|---|---|---:|---:|---:|---:|---|
| adaptive_0.0003_late045 - baseline | last_test_return | 5 | 0.0356 | [-0.3994, 0.5284] | -0.2000 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - baseline | train_auc | 5 | -0.1747 | [-0.3251, -0.0401] | -0.2800 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - baseline | best_train_return | 5 | -0.1938 | [-0.4473, 0.0000] | -0.3600 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - baseline | stability_gap | 5 | -0.2295 | [-0.9048, 0.3069] | 0.2000 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - diagnosis_only | last_test_return | 5 | 0.0356 | [-0.3994, 0.5284] | -0.2000 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - diagnosis_only | train_auc | 5 | -0.1747 | [-0.3251, -0.0401] | -0.2800 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - diagnosis_only | best_train_return | 5 | -0.1938 | [-0.4473, 0.0000] | -0.3600 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - diagnosis_only | stability_gap | 5 | -0.2295 | [-0.9048, 0.3069] | 0.2000 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | last_test_return | 5 | -0.2083 | [-0.7569, 0.3403] | -0.0400 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | train_auc | 5 | -0.1341 | [-0.3826, 0.1307] | -0.2000 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | best_train_return | 5 | -0.0138 | [-0.0984, 0.0625] | 0.1200 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | stability_gap | 5 | 0.1945 | [-0.3179, 0.7618] | -0.1200 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | last_test_return | 5 | -0.9047 | [-1.2720, -0.4207] | -0.7600 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | train_auc | 5 | -0.4373 | [-0.5822, -0.3251] | -0.5200 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | best_train_return | 5 | -0.1542 | [-0.4013, 0.0741] | -0.3600 | 1,2,3,4,5 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | stability_gap | 5 | 0.7505 | [0.2086, 1.1955] | 0.7600 | 1,2,3,4,5 |

## Decision Rule

Use RWARE/VMAS as cross-domain evidence only if adaptive is directionally positive against baseline and not worse than budget-matched controls. If cross-domain results are mixed, report them transparently as limits and rely on the completed LBF 10x10/12x12 package.
