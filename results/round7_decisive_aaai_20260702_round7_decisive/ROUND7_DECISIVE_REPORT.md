# Round 7 Decisive AAAI Report

This report merges Round 6 controls with the completed Round 7 decisive controls while keeping 10x10 and 12x12 results separate.

## main10x10 Grouped Results

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0002_late045 | 8 | 0.2435 | [0.2010, 0.2858] | 0.1900 | 0.4442 | 0.2007 |
| adaptive_0.0002_late060 | 8 | 0.2432 | [0.2013, 0.2849] | 0.1899 | 0.4433 | 0.2001 |
| adaptive_0.0003_late045 | 8 | 0.3042 | [0.2568, 0.3527] | 0.2263 | 0.5320 | 0.2278 |
| baseline | 8 | 0.2258 | [0.1827, 0.2796] | 0.1827 | 0.3532 | 0.1274 |
| diagnosis_only | 8 | 0.2258 | [0.1827, 0.2796] | 0.1827 | 0.3532 | 0.1274 |
| random_type_0.0002 | 8 | 0.3048 | [0.2327, 0.3747] | 0.2308 | 0.5270 | 0.2222 |
| random_type_0.0003_late045 | 8 | 0.2192 | [0.1806, 0.2619] | 0.1759 | 0.3736 | 0.1544 |
| random_type_0.0003_late060 | 8 | 0.2154 | [0.1776, 0.2578] | 0.1756 | 0.3746 | 0.1591 |
| random_type_matched_0.0003_late045 | 8 | 0.2580 | [0.2171, 0.3090] | 0.2012 | 0.3935 | 0.1355 |
| semantic_gate_0.0003_late045 | 8 | 0.2382 | [0.1996, 0.2794] | 0.1900 | 0.3987 | 0.1605 |
| type_specific_0.0002 | 8 | 0.2514 | [0.2181, 0.2860] | 0.1981 | 0.3778 | 0.1264 |
| type_specific_0.0003 | 8 | 0.2755 | [0.2268, 0.3272] | 0.2072 | 0.4878 | 0.2123 |
| uniform_0.0002 | 8 | 0.2680 | [0.2123, 0.3278] | 0.2095 | 0.4513 | 0.1833 |
| uniform_0.0003 | 8 | 0.2892 | [0.2419, 0.3415] | 0.2092 | 0.5658 | 0.2767 |

## generalization Grouped Results

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0002_late045 | 5 | 0.1034 | [0.0829, 0.1239] | 0.0893 | 0.1469 | 0.0435 |
| adaptive_0.0002_late060 | 5 | 0.1034 | [0.0829, 0.1239] | 0.0893 | 0.1469 | 0.0435 |
| adaptive_0.0003_late045 | 3 | 0.1179 | [0.0997, 0.1407] | 0.1023 | 0.1681 | 0.0502 |
| baseline | 8 | 0.1021 | [0.0939, 0.1097] | 0.0917 | 0.1408 | 0.0387 |
| semantic_gate_0.0003_late045 | 3 | 0.0995 | [0.0820, 0.1332] | 0.0854 | 0.1426 | 0.0431 |
| uniform_0.0002 | 8 | 0.1143 | [0.0984, 0.1302] | 0.0928 | 0.1470 | 0.0328 |
| uniform_0.0003 | 5 | 0.1073 | [0.0917, 0.1222] | 0.0887 | 0.1360 | 0.0287 |

## main10x10 Paired Decision Comparisons

| comparison | metric | n | mean delta | 95% CI | shared seeds |
|---|---|---:|---:|---:|---|
| semantic_gate_0.0003_late045 - random_type_matched_0.0003_late045 | last_test_return | 8 | -0.0199 | [-0.0716, 0.0348] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - random_type_matched_0.0003_late045 | train_auc | 8 | -0.0111 | [-0.0455, 0.0198] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - random_type_matched_0.0003_late045 | best_train_return | 8 | 0.0052 | [-0.1301, 0.1372] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - random_type_matched_0.0003_late045 | stability_gap | 8 | 0.0251 | [-0.0646, 0.1110] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - random_type_0.0003_late045 | last_test_return | 8 | 0.0190 | [-0.0512, 0.0808] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - random_type_0.0003_late045 | train_auc | 8 | 0.0141 | [-0.0339, 0.0537] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - random_type_0.0003_late045 | best_train_return | 8 | 0.0251 | [-0.1174, 0.1663] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - random_type_0.0003_late045 | stability_gap | 8 | 0.0061 | [-0.0880, 0.1045] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - random_type_0.0003_late060 | last_test_return | 8 | 0.0227 | [-0.0491, 0.0854] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - random_type_0.0003_late060 | train_auc | 8 | 0.0144 | [-0.0336, 0.0542] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - random_type_0.0003_late060 | best_train_return | 8 | 0.0241 | [-0.1157, 0.1624] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - random_type_0.0003_late060 | stability_gap | 8 | 0.0014 | [-0.0885, 0.0912] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - adaptive_0.0003_late045 | last_test_return | 8 | -0.0660 | [-0.1237, 0.0034] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - adaptive_0.0003_late045 | train_auc | 8 | -0.0362 | [-0.0717, 0.0018] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - adaptive_0.0003_late045 | best_train_return | 8 | -0.1333 | [-0.2892, 0.0348] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - adaptive_0.0003_late045 | stability_gap | 8 | -0.0673 | [-0.1747, 0.0384] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - uniform_0.0003 | last_test_return | 8 | -0.0510 | [-0.1246, 0.0220] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - uniform_0.0003 | train_auc | 8 | -0.0191 | [-0.0602, 0.0176] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - uniform_0.0003 | best_train_return | 8 | -0.1671 | [-0.3135, -0.0096] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - uniform_0.0003 | stability_gap | 8 | -0.1161 | [-0.1933, -0.0287] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - baseline | last_test_return | 8 | 0.0124 | [-0.0332, 0.0553] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - baseline | train_auc | 8 | 0.0073 | [-0.0216, 0.0330] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - baseline | best_train_return | 8 | 0.0455 | [-0.0558, 0.1487] | 1,2,3,4,5,6,7,8 |
| semantic_gate_0.0003_late045 - baseline | stability_gap | 8 | 0.0332 | [-0.0314, 0.0984] | 1,2,3,4,5,6,7,8 |
| random_type_matched_0.0003_late045 - adaptive_0.0003_late045 | last_test_return | 8 | -0.0461 | [-0.0917, 0.0098] | 1,2,3,4,5,6,7,8 |
| random_type_matched_0.0003_late045 - adaptive_0.0003_late045 | train_auc | 8 | -0.0251 | [-0.0575, 0.0142] | 1,2,3,4,5,6,7,8 |
| random_type_matched_0.0003_late045 - adaptive_0.0003_late045 | best_train_return | 8 | -0.1385 | [-0.2651, 0.0029] | 1,2,3,4,5,6,7,8 |
| random_type_matched_0.0003_late045 - adaptive_0.0003_late045 | stability_gap | 8 | -0.0924 | [-0.1769, -0.0038] | 1,2,3,4,5,6,7,8 |
| random_type_matched_0.0003_late045 - uniform_0.0003 | last_test_return | 8 | -0.0311 | [-0.0863, 0.0217] | 1,2,3,4,5,6,7,8 |
| random_type_matched_0.0003_late045 - uniform_0.0003 | train_auc | 8 | -0.0080 | [-0.0416, 0.0266] | 1,2,3,4,5,6,7,8 |
| random_type_matched_0.0003_late045 - uniform_0.0003 | best_train_return | 8 | -0.1723 | [-0.2647, -0.0677] | 1,2,3,4,5,6,7,8 |
| random_type_matched_0.0003_late045 - uniform_0.0003 | stability_gap | 8 | -0.1412 | [-0.2086, -0.0744] | 1,2,3,4,5,6,7,8 |

## generalization Paired Stress-Test Comparisons

| comparison | metric | n | mean delta | 95% CI | shared seeds |
|---|---|---:|---:|---:|---|
| semantic_gate_0.0003_late045 - baseline | last_test_return | 3 | -0.0012 | [-0.0290, 0.0247] | 6,7,8 |
| semantic_gate_0.0003_late045 - baseline | train_auc | 3 | -0.0112 | [-0.0321, 0.0010] | 6,7,8 |
| semantic_gate_0.0003_late045 - uniform_0.0002 | last_test_return | 3 | -0.0230 | [-0.0562, 0.0108] | 6,7,8 |
| semantic_gate_0.0003_late045 - uniform_0.0002 | train_auc | 3 | -0.0103 | [-0.0235, 0.0048] | 6,7,8 |
| semantic_gate_0.0003_late045 - adaptive_0.0003_late045 | last_test_return | 3 | -0.0184 | [-0.0300, -0.0075] | 6,7,8 |
| semantic_gate_0.0003_late045 - adaptive_0.0003_late045 | train_auc | 3 | -0.0169 | [-0.0176, -0.0162] | 6,7,8 |
| adaptive_0.0003_late045 - baseline | last_test_return | 3 | 0.0172 | [-0.0113, 0.0322] | 6,7,8 |
| adaptive_0.0003_late045 - baseline | train_auc | 3 | 0.0058 | [-0.0151, 0.0171] | 6,7,8 |
| uniform_0.0002 - baseline | last_test_return | 8 | 0.0121 | [-0.0026, 0.0284] | 1,2,3,4,5,6,7,8 |
| uniform_0.0002 - baseline | train_auc | 8 | 0.0011 | [-0.0049, 0.0078] | 1,2,3,4,5,6,7,8 |

## Submission Rule

The strong semantic-causality claim is not supported unless semantic_gate beats matched random-type controls. When matched random remains tied or stronger, the safest AAAI framing is calibrated failure-triggered adaptive reward shaping, with semantic diagnosis used for interpretable gating and analysis rather than claimed as the sole causal mechanism.
