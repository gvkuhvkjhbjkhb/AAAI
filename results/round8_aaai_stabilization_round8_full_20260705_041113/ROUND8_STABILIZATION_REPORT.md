# Round 8 AAAI Stabilization Report

This report is organized around the conservative AAAI claim: failure-triggered adaptive reward shaping, not LLM semantic causality, is the main mechanism.

## 10x10 Grouped Results

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0002_late045 | 8 | 0.2435 | [0.2016, 0.2845] | 0.1900 | 0.4442 | 0.2007 |
| adaptive_0.0002_late060 | 8 | 0.2432 | [0.2023, 0.2831] | 0.1899 | 0.4433 | 0.2001 |
| adaptive_0.0003_late045 | 16 | 0.3042 | [0.2701, 0.3391] | 0.2263 | 0.5320 | 0.2278 |
| adaptive_0.0005_late045 | 8 | 0.2451 | [0.1906, 0.3085] | 0.1889 | 0.4126 | 0.1675 |
| baseline | 16 | 0.2258 | [0.1937, 0.2639] | 0.1827 | 0.3532 | 0.1274 |
| diagnosis_only | 16 | 0.2258 | [0.1937, 0.2639] | 0.1827 | 0.3532 | 0.1274 |
| random_type_0.0002 | 8 | 0.3048 | [0.2328, 0.3732] | 0.2308 | 0.5270 | 0.2222 |
| random_type_0.0003_late045 | 8 | 0.2192 | [0.1801, 0.2638] | 0.1759 | 0.3736 | 0.1544 |
| random_type_0.0003_late060 | 8 | 0.2154 | [0.1772, 0.2599] | 0.1756 | 0.3746 | 0.1591 |
| random_type_0.0005_late045 | 8 | 0.2643 | [0.2117, 0.3195] | 0.2087 | 0.4349 | 0.1706 |
| random_type_budget_matched_0.0003_late045 | 8 | 0.2192 | [0.1801, 0.2638] | 0.1759 | 0.3736 | 0.1544 |
| random_type_matched_0.0003_late045 | 8 | 0.2580 | [0.2159, 0.3091] | 0.2012 | 0.3935 | 0.1355 |
| semantic_gate_0.0003_late045 | 8 | 0.2382 | [0.1997, 0.2808] | 0.1900 | 0.3987 | 0.1605 |
| semantic_shuffled_budget_matched_0.0003_late045 | 8 | 0.2580 | [0.2159, 0.3091] | 0.2012 | 0.3935 | 0.1355 |
| type_specific_0.0002 | 8 | 0.2514 | [0.2193, 0.2860] | 0.1981 | 0.3778 | 0.1264 |
| type_specific_0.0003 | 8 | 0.2755 | [0.2275, 0.3266] | 0.2072 | 0.4878 | 0.2123 |
| uniform_0.0002 | 8 | 0.2680 | [0.2112, 0.3281] | 0.2095 | 0.4513 | 0.1833 |
| uniform_0.0003 | 8 | 0.2892 | [0.2407, 0.3421] | 0.2092 | 0.5658 | 0.2767 |
| uniform_0.0005 | 8 | 0.2533 | [0.2303, 0.2761] | 0.1934 | 0.4432 | 0.1899 |
| uniform_budget_matched_0.0003_late045 | 8 | 0.2346 | [0.1865, 0.2832] | 0.1876 | 0.4082 | 0.1736 |

## 10x10 Paired Strong-Control Comparisons

| comparison | metric | n | mean delta | 95% CI | Cliff delta | shared seeds |
|---|---|---:|---:|---:|---:|---|
| adaptive_0.0003_late045 - baseline | last_test_return | 8 | 0.0784 | [0.0267, 0.1229] | 0.6562 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | train_auc | 8 | 0.0436 | [0.0181, 0.0659] | 0.5312 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | best_train_return | 8 | 0.1788 | [0.0518, 0.3045] | 0.6562 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | stability_gap | 8 | 0.1004 | [0.0091, 0.1954] | 0.4688 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | last_test_return | 8 | 0.0695 | [0.0074, 0.1237] | 0.5000 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | train_auc | 8 | 0.0387 | [0.0089, 0.0677] | 0.4062 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | best_train_return | 8 | 0.1238 | [0.0144, 0.2331] | 0.5000 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | stability_gap | 8 | 0.0542 | [-0.0139, 0.1215] | 0.2500 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | last_test_return | 8 | 0.0850 | [0.0091, 0.1653] | 0.6250 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | train_auc | 8 | 0.0504 | [-0.0009, 0.1036] | 0.6250 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | best_train_return | 8 | 0.1584 | [-0.0297, 0.3317] | 0.5000 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | stability_gap | 8 | 0.0734 | [-0.0481, 0.1741] | 0.3750 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | last_test_return | 8 | 0.0461 | [-0.0101, 0.0926] | 0.3750 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | train_auc | 8 | 0.0251 | [-0.0146, 0.0580] | 0.3438 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | best_train_return | 8 | 0.1385 | [-0.0048, 0.2666] | 0.5000 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | stability_gap | 8 | 0.0924 | [0.0023, 0.1776] | 0.5312 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - diagnosis_only | last_test_return | 8 | 0.0784 | [0.0267, 0.1229] | 0.6562 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - diagnosis_only | train_auc | 8 | 0.0436 | [0.0181, 0.0659] | 0.5312 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - diagnosis_only | best_train_return | 8 | 0.1788 | [0.0518, 0.3045] | 0.6562 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - diagnosis_only | stability_gap | 8 | 0.1004 | [0.0091, 0.1954] | 0.4688 | 1,2,3,4,5,6,7,8 |

## 12x12 Grouped Results

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0002_late045 | 5 | 0.1034 | [0.0829, 0.1239] | 0.0893 | 0.1469 | 0.0435 |
| adaptive_0.0002_late060 | 5 | 0.1034 | [0.0829, 0.1239] | 0.0893 | 0.1469 | 0.0435 |
| adaptive_0.0003_late045 | 11 | 0.1168 | [0.1039, 0.1289] | 0.0988 | 0.1531 | 0.0363 |
| baseline | 16 | 0.1021 | [0.0965, 0.1078] | 0.0917 | 0.1408 | 0.0387 |
| diagnosis_only | 8 | 0.1021 | [0.0939, 0.1099] | 0.0917 | 0.1408 | 0.0387 |
| random_type_budget_matched_0.0003_late045 | 8 | 0.1056 | [0.0964, 0.1167] | 0.0952 | 0.1584 | 0.0528 |
| semantic_gate_0.0003_late045 | 3 | 0.0995 | [0.0820, 0.1332] | 0.0854 | 0.1426 | 0.0431 |
| semantic_shuffled_budget_matched_0.0003_late045 | 8 | 0.1036 | [0.0914, 0.1153] | 0.0884 | 0.1439 | 0.0403 |
| uniform_0.0002 | 8 | 0.1143 | [0.0988, 0.1300] | 0.0928 | 0.1470 | 0.0328 |
| uniform_0.0003 | 5 | 0.1073 | [0.0917, 0.1221] | 0.0887 | 0.1360 | 0.0287 |
| uniform_budget_matched_0.0003_late045 | 8 | 0.1125 | [0.1028, 0.1226] | 0.0945 | 0.1534 | 0.0409 |

## 12x12 Paired Strong-Control Comparisons

| comparison | metric | n | mean delta | 95% CI | Cliff delta | shared seeds |
|---|---|---:|---:|---:|---:|---|
| adaptive_0.0003_late045 - baseline | last_test_return | 8 | 0.0143 | [-0.0009, 0.0283] | 0.4062 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | train_auc | 8 | 0.0058 | [-0.0048, 0.0140] | 0.3125 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | best_train_return | 8 | 0.0067 | [-0.0145, 0.0238] | 0.3750 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | stability_gap | 8 | -0.0076 | [-0.0214, 0.0075] | -0.3125 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | last_test_return | 8 | 0.0040 | [-0.0075, 0.0165] | 0.1875 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | train_auc | 8 | 0.0030 | [-0.0034, 0.0096] | 0.2188 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | best_train_return | 8 | -0.0059 | [-0.0203, 0.0100] | 0.0625 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_budget_matched_0.0003_late045 | stability_gap | 8 | -0.0098 | [-0.0208, 0.0014] | -0.3750 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | last_test_return | 8 | 0.0108 | [-0.0074, 0.0297] | 0.3438 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | train_auc | 8 | 0.0023 | [-0.0088, 0.0134] | 0.2188 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | best_train_return | 8 | -0.0109 | [-0.0348, 0.0070] | -0.1562 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_budget_matched_0.0003_late045 | stability_gap | 8 | -0.0217 | [-0.0397, -0.0038] | -0.6250 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | last_test_return | 8 | 0.0128 | [-0.0057, 0.0312] | 0.3750 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | train_auc | 8 | 0.0091 | [-0.0031, 0.0226] | 0.3125 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | best_train_return | 8 | 0.0036 | [-0.0196, 0.0268] | 0.1875 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - semantic_shuffled_budget_matched_0.0003_late045 | stability_gap | 8 | -0.0092 | [-0.0244, 0.0064] | -0.4375 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - diagnosis_only | last_test_return | 8 | 0.0143 | [-0.0009, 0.0283] | 0.4062 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - diagnosis_only | train_auc | 8 | 0.0058 | [-0.0048, 0.0140] | 0.3125 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - diagnosis_only | best_train_return | 8 | 0.0067 | [-0.0145, 0.0238] | 0.3750 | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - diagnosis_only | stability_gap | 8 | -0.0076 | [-0.0214, 0.0075] | -0.3125 | 1,2,3,4,5,6,7,8 |

## Decision Use

Use 10x10 as the established main-task evidence and 12x12 as the required generalization/stress-test panel. A submission-grade result requires adaptive shaping to beat baseline on both environments and to remain competitive with budget-matched uniform and shuffled/random controls.
