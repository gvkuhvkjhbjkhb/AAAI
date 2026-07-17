# Round 6 Full Raw-Result Analysis

This file was generated after the Lab connection was restored. It is based on the complete `summary.csv` produced by the Round 6 two-GPU run, which contains 105 rows: 80 main 10x10 runs and 25 12x12 generalization runs. All planned runs completed successfully and no `.fail` marker was present.

## main10x10 grouped metrics

| method | n | last_test mean | last_test sd | last_test 95% bootstrap CI | train_auc mean | stability_gap mean | best_train mean |
|---|---:|---:|---:|---|---:|---:|---:|
| adaptive_0.0002_late045 | 8 | 0.2435 | 0.0645 | [0.2013, 0.2858] | 0.1900 | 0.2007 | 0.4442 |
| adaptive_0.0002_late060 | 8 | 0.2432 | 0.0633 | [0.2015, 0.2847] | 0.1899 | 0.2001 | 0.4433 |
| adaptive_0.0003_late045 | 8 | 0.3042 | 0.0750 | [0.2569, 0.3530] | 0.2263 | 0.2278 | 0.5320 |
| baseline | 8 | 0.2258 | 0.0769 | [0.1828, 0.2798] | 0.1827 | 0.1274 | 0.3532 |
| diagnosis_only | 8 | 0.2258 | 0.0769 | [0.1828, 0.2798] | 0.1827 | 0.1274 | 0.3532 |
| random_type_0.0002 | 8 | 0.3048 | 0.1094 | [0.2322, 0.3742] | 0.2308 | 0.2222 | 0.5270 |
| type_specific_0.0002 | 8 | 0.2514 | 0.0523 | [0.2187, 0.2862] | 0.1981 | 0.1264 | 0.3778 |
| type_specific_0.0003 | 8 | 0.2755 | 0.0774 | [0.2265, 0.3277] | 0.2072 | 0.2123 | 0.4878 |
| uniform_0.0002 | 8 | 0.2680 | 0.0905 | [0.2122, 0.3280] | 0.2095 | 0.1833 | 0.4513 |
| uniform_0.0003 | 8 | 0.2892 | 0.0780 | [0.2422, 0.3415] | 0.2092 | 0.2767 | 0.5658 |

## main10x10 paired differences versus baseline

| method | n | delta last_test | delta last_test 95% CI | delta train_auc | delta stability_gap |
|---|---:|---:|---|---:|---:|
| adaptive_0.0002_late045 | 8 | 0.0177 | [-0.0393, 0.0779] | 0.0073 | 0.0733 |
| adaptive_0.0002_late060 | 8 | 0.0174 | [-0.0397, 0.0784] | 0.0072 | 0.0727 |
| adaptive_0.0003_late045 | 8 | 0.0784 | [0.0266, 0.1223] | 0.0436 | 0.1004 |
| diagnosis_only | 8 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 |
| random_type_0.0002 | 8 | 0.0791 | [-0.0087, 0.1688] | 0.0481 | 0.0948 |
| type_specific_0.0002 | 8 | 0.0256 | [-0.0151, 0.0690] | 0.0154 | -0.0010 |
| type_specific_0.0003 | 8 | 0.0497 | [0.0092, 0.0916] | 0.0245 | 0.0849 |
| uniform_0.0002 | 8 | 0.0423 | [-0.0292, 0.1123] | 0.0268 | 0.0559 |
| uniform_0.0003 | 8 | 0.0634 | [0.0023, 0.1235] | 0.0265 | 0.1493 |

## generalization grouped metrics

| method | n | last_test mean | last_test sd | last_test 95% bootstrap CI | train_auc mean | stability_gap mean | best_train mean |
|---|---:|---:|---:|---|---:|---:|---:|
| adaptive_0.0002_late045 | 5 | 0.1034 | 0.0259 | [0.0829, 0.1239] | 0.0893 | 0.0435 | 0.1469 |
| adaptive_0.0002_late060 | 5 | 0.1034 | 0.0259 | [0.0829, 0.1239] | 0.0893 | 0.0435 | 0.1469 |
| baseline | 5 | 0.1030 | 0.0118 | [0.0939, 0.1120] | 0.0888 | 0.0379 | 0.1409 |
| uniform_0.0002 | 5 | 0.1093 | 0.0288 | [0.0873, 0.1323] | 0.0911 | 0.0405 | 0.1498 |
| uniform_0.0003 | 5 | 0.1073 | 0.0199 | [0.0929, 0.1222] | 0.0887 | 0.0287 | 0.1360 |

## generalization paired differences versus baseline

| method | n | delta last_test | delta last_test 95% CI | delta train_auc | delta stability_gap |
|---|---:|---:|---|---:|---:|
| adaptive_0.0002_late045 | 5 | 0.0005 | [-0.0199, 0.0246] | 0.0005 | 0.0056 |
| adaptive_0.0002_late060 | 5 | 0.0005 | [-0.0199, 0.0246] | 0.0005 | 0.0056 |
| uniform_0.0002 | 5 | 0.0064 | [-0.0084, 0.0215] | 0.0023 | 0.0026 |
| uniform_0.0003 | 5 | 0.0044 | [-0.0152, 0.0274] | -0.0001 | -0.0093 |

## Interpretation

The complete raw-result analysis confirms the console summary. The best diagnosis-conditioned method on the primary 10x10 task is `adaptive_0.0003_late045`, with mean final test return 0.3042 and train AUC 0.2263. It improves over baseline and over calibrated uniform shaping at 0.0003. The diagnosis-only condition matches baseline, which indicates that performance changes are caused by reward intervention rather than by passive logging or diagnosis overhead.

The central limitation is also confirmed. `random_type_0.0002` achieves mean final test return 0.3048 and train AUC 0.2308, slightly above the best adaptive semantic method. This prevents a strong causal statement that semantic labels are already responsible for the full improvement. The AAAI paper should frame the contribution as calibrated failure-triggered adaptive shaping and then use Round 7 to test whether confidence-gated semantic labels can beat matched random controls.

The 12x12 generalization task remains weak. `uniform_0.0002` is the best generalization method by mean final test return, while both adaptive 0.0002 variants are approximately tied with baseline. The current safe claim is transfer without catastrophic degradation, not strong cross-map improvement.
