# Phase 1 Live Merged Report

This report merges Round 6, Round 7, and the current Phase 1 robustness run for the 10x10 task.

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0002_late045 | 8 | 0.2435 | [0.2015, 0.2863] | 0.1900 | 0.4442 | 0.2007 |
| adaptive_0.0002_late060 | 8 | 0.2432 | [0.2014, 0.2847] | 0.1899 | 0.4433 | 0.2001 |
| adaptive_0.0003_late045 | 8 | 0.3042 | [0.2550, 0.3545] | 0.2263 | 0.5320 | 0.2278 |
| adaptive_0.0005_late045 | 8 | 0.2451 | [0.1909, 0.3096] | 0.1889 | 0.4126 | 0.1675 |
| baseline | 8 | 0.2258 | [0.1827, 0.2813] | 0.1827 | 0.3532 | 0.1274 |
| diagnosis_only | 8 | 0.2258 | [0.1827, 0.2813] | 0.1827 | 0.3532 | 0.1274 |
| random_type_0.0002 | 8 | 0.3048 | [0.2333, 0.3733] | 0.2308 | 0.5270 | 0.2222 |
| random_type_0.0003_late045 | 8 | 0.2192 | [0.1807, 0.2639] | 0.1759 | 0.3736 | 0.1544 |
| random_type_0.0003_late060 | 8 | 0.2154 | [0.1786, 0.2601] | 0.1756 | 0.3746 | 0.1591 |
| random_type_0.0005_late045 | 8 | 0.2643 | [0.2117, 0.3215] | 0.2087 | 0.4349 | 0.1706 |
| random_type_matched_0.0003_late045 | 8 | 0.2580 | [0.2166, 0.3100] | 0.2012 | 0.3935 | 0.1355 |
| semantic_gate_0.0003_late045 | 8 | 0.2382 | [0.2005, 0.2808] | 0.1900 | 0.3987 | 0.1605 |
| type_specific_0.0002 | 8 | 0.2514 | [0.2181, 0.2873] | 0.1981 | 0.3778 | 0.1264 |
| type_specific_0.0003 | 8 | 0.2755 | [0.2273, 0.3287] | 0.2072 | 0.4878 | 0.2123 |
| uniform_0.0002 | 8 | 0.2680 | [0.2131, 0.3284] | 0.2095 | 0.4513 | 0.1833 |
| uniform_0.0003 | 8 | 0.2892 | [0.2405, 0.3432] | 0.2092 | 0.5658 | 0.2767 |
| uniform_0.0005 | 8 | 0.2533 | [0.2298, 0.2763] | 0.1934 | 0.4432 | 0.1899 |

## Paired Phase 1 Comparisons

| comparison | metric | n | mean delta | 95% CI | shared seeds |
|---|---|---:|---:|---:|---|
| adaptive_0.0005_late045 - baseline | last_test_return | 8 | 0.0193 | [-0.0553, 0.0907] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - baseline | train_auc | 8 | 0.0062 | [-0.0460, 0.0571] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - baseline | best_train_return | 8 | 0.0594 | [-0.0735, 0.1973] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - baseline | stability_gap | 8 | 0.0401 | [-0.0249, 0.1108] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - uniform_0.0005 | last_test_return | 8 | -0.0082 | [-0.0678, 0.0651] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - uniform_0.0005 | train_auc | 8 | -0.0044 | [-0.0476, 0.0457] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - uniform_0.0005 | best_train_return | 8 | -0.0306 | [-0.1298, 0.0827] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - uniform_0.0005 | stability_gap | 8 | -0.0224 | [-0.0700, 0.0257] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - random_type_0.0005_late045 | last_test_return | 8 | -0.0192 | [-0.0933, 0.0664] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - random_type_0.0005_late045 | train_auc | 8 | -0.0197 | [-0.0749, 0.0473] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - random_type_0.0005_late045 | best_train_return | 8 | -0.0223 | [-0.1467, 0.1191] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - random_type_0.0005_late045 | stability_gap | 8 | -0.0031 | [-0.0723, 0.0690] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - adaptive_0.0003_late045 | last_test_return | 8 | -0.0591 | [-0.1485, 0.0452] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - adaptive_0.0003_late045 | train_auc | 8 | -0.0373 | [-0.0972, 0.0320] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - adaptive_0.0003_late045 | best_train_return | 8 | -0.1194 | [-0.3228, 0.0946] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0005_late045 - adaptive_0.0003_late045 | stability_gap | 8 | -0.0603 | [-0.1813, 0.0566] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | last_test_return | 8 | 0.0784 | [0.0261, 0.1232] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | train_auc | 8 | 0.0436 | [0.0181, 0.0655] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | best_train_return | 8 | 0.1788 | [0.0500, 0.3058] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - baseline | stability_gap | 8 | 0.1004 | [0.0102, 0.1984] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_0.0003 | last_test_return | 8 | 0.0150 | [-0.0208, 0.0525] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_0.0003 | train_auc | 8 | 0.0171 | [-0.0018, 0.0386] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_0.0003 | best_train_return | 8 | -0.0338 | [-0.1021, 0.0261] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - uniform_0.0003 | stability_gap | 8 | -0.0488 | [-0.1179, 0.0108] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_matched_0.0003_late045 | last_test_return | 8 | 0.0461 | [-0.0111, 0.0933] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_matched_0.0003_late045 | train_auc | 8 | 0.0251 | [-0.0145, 0.0592] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_matched_0.0003_late045 | best_train_return | 8 | 0.1385 | [-0.0050, 0.2697] | 1,2,3,4,5,6,7,8 |
| adaptive_0.0003_late045 - random_type_matched_0.0003_late045 | stability_gap | 8 | 0.0924 | [0.0017, 0.1809] | 1,2,3,4,5,6,7,8 |
