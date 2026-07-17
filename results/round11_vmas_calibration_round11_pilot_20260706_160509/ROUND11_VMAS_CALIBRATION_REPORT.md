# Round 11 VMAS Reward-Scale Calibration Report

Round 11 tests whether Round 9's VMAS weakness was caused by transferring the LBF-tuned 0.0003 reward-shaping scale into a dense-navigation reward landscape.

## Penalty 0.00001

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3 | 7.3106 | [7.1555, 7.5732] | 5.1314 | 11.7326 | 4.4220 |
| adaptive | 3 | 7.6344 | [7.0200, 8.1794] | 5.1066 | 11.5606 | 3.9262 |
| uniform | 3 | 6.8880 | [6.4624, 7.6104] | 5.0356 | 11.8362 | 4.9482 |
| random | 3 | 7.8143 | [6.8330, 8.6781] | 5.1999 | 11.7660 | 3.9517 |

## Penalty 0.00001 Budget Accounting

| method | records | triggers | penalty total | terminal bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adaptive | 1392.6000 | 1879.3333 | 0.5333 | 0.0002 | 91651.6667 | 0.0003 | 48.7713 |
| uniform | 1398.0667 | 1881.6667 | 0.6714 | 0.0002 | 91296.3333 | 0.0004 | 48.5242 |
| random | 1398.2667 | 1879.0000 | 0.5879 | 0.0185 | 91051.3333 | 0.0003 | 48.4613 |

## Penalty 0.00001 Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | shared seeds |
|---|---|---:|---:|---:|---|
| adaptive - baseline | last_test_return | 3 | 0.3238 | [-0.5532, 0.9763] | 1,2,3 |
| adaptive - baseline | train_auc | 3 | -0.0248 | [-0.4221, 0.3606] | 1,2,3 |
| adaptive - baseline | best_train_return | 3 | -0.1720 | [-0.3389, 0.1558] | 1,2,3 |
| adaptive - baseline | stability_gap | 3 | -0.4958 | [-1.3093, 0.2143] | 1,2,3 |
| adaptive - uniform | last_test_return | 3 | 0.7464 | [-0.5904, 1.5882] | 1,2,3 |
| adaptive - uniform | train_auc | 3 | 0.0711 | [-0.2713, 0.5676] | 1,2,3 |
| adaptive - uniform | best_train_return | 3 | -0.2756 | [-0.3857, -0.1934] | 1,2,3 |
| adaptive - uniform | stability_gap | 3 | -1.0220 | [-1.9739, 0.3427] | 1,2,3 |
| adaptive - random | last_test_return | 3 | -0.1799 | [-0.4987, 0.1870] | 1,2,3 |
| adaptive - random | train_auc | 3 | -0.0932 | [-0.1508, -0.0110] | 1,2,3 |
| adaptive - random | best_train_return | 3 | -0.2054 | [-0.2556, -0.1563] | 1,2,3 |
| adaptive - random | stability_gap | 3 | -0.0255 | [-0.3433, 0.2431] | 1,2,3 |

## Penalty 0.00003

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3 | 7.3106 | [7.1555, 7.5732] | 5.1314 | 11.7326 | 4.4220 |
| adaptive | 3 | 7.7474 | [7.3245, 8.3202] | 5.1499 | 11.5803 | 3.8329 |
| uniform | 3 | 7.0454 | [6.3927, 7.7733] | 5.0326 | 11.7406 | 4.6952 |
| random | 3 | 6.9320 | [6.6333, 7.2638] | 5.0343 | 11.5617 | 4.6297 |

## Penalty 0.00003 Budget Accounting

| method | records | triggers | penalty total | terminal bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adaptive | 1392.0000 | 1880.0000 | 1.6016 | 0.0007 | 91615.3333 | 0.0009 | 48.7316 |
| uniform | 1402.4000 | 1889.6667 | 2.0105 | 0.0009 | 91288.6667 | 0.0011 | 48.3121 |
| random | 1381.4000 | 1855.3333 | 1.7560 | 0.0550 | 91205.6667 | 0.0009 | 49.1593 |

## Penalty 0.00003 Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | shared seeds |
|---|---|---:|---:|---:|---|
| adaptive - baseline | last_test_return | 3 | 0.4368 | [-0.2487, 1.1647] | 1,2,3 |
| adaptive - baseline | train_auc | 3 | 0.0185 | [-0.3058, 0.2715] | 1,2,3 |
| adaptive - baseline | best_train_return | 3 | -0.1523 | [-0.4107, 0.2046] | 1,2,3 |
| adaptive - baseline | stability_gap | 3 | -0.5891 | [-0.9601, -0.0021] | 1,2,3 |
| adaptive - uniform | last_test_return | 3 | 0.7020 | [-0.1758, 1.3499] | 1,2,3 |
| adaptive - uniform | train_auc | 3 | 0.1173 | [-0.1973, 0.2941] | 1,2,3 |
| adaptive - uniform | best_train_return | 3 | -0.1603 | [-0.4833, 0.0366] | 1,2,3 |
| adaptive - uniform | stability_gap | 3 | -0.8623 | [-1.3133, -0.3075] | 1,2,3 |
| adaptive - random | last_test_return | 3 | 0.8154 | [0.6912, 1.0564] | 1,2,3 |
| adaptive - random | train_auc | 3 | 0.1156 | [-0.0136, 0.3314] | 1,2,3 |
| adaptive - random | best_train_return | 3 | 0.0186 | [-0.0974, 0.2332] | 1,2,3 |
| adaptive - random | stability_gap | 3 | -0.7968 | [-0.8232, -0.7786] | 1,2,3 |

## Penalty 0.0001

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3 | 7.3106 | [7.1555, 7.5732] | 5.1314 | 11.7326 | 4.4220 |
| adaptive | 3 | 7.4340 | [6.8282, 8.2573] | 5.2100 | 11.7663 | 4.3323 |
| uniform | 3 | 7.1313 | [6.5898, 7.8770] | 5.0608 | 11.7095 | 4.5782 |
| random | 3 | 7.8222 | [7.4671, 8.0578] | 5.4304 | 11.7894 | 3.9672 |

## Penalty 0.0001 Budget Accounting

| method | records | triggers | penalty total | terminal bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adaptive | 1398.0000 | 1888.6667 | 5.3006 | 0.0030 | 91359.3333 | 0.0028 | 48.3747 |
| uniform | 1389.9333 | 1868.0000 | 6.7131 | 0.0026 | 91012.6667 | 0.0036 | 48.7267 |
| random | 1398.5333 | 1892.0000 | 5.9819 | 0.1860 | 91280.3333 | 0.0032 | 48.2460 |

## Penalty 0.0001 Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | shared seeds |
|---|---|---:|---:|---:|---|
| adaptive - baseline | last_test_return | 3 | 0.1234 | [-0.7450, 1.0542] | 1,2,3 |
| adaptive - baseline | train_auc | 3 | 0.0786 | [-0.4119, 0.5849] | 1,2,3 |
| adaptive - baseline | best_train_return | 3 | 0.0337 | [0.0150, 0.0670] | 1,2,3 |
| adaptive - baseline | stability_gap | 3 | -0.0897 | [-1.0392, 0.7641] | 1,2,3 |
| adaptive - uniform | last_test_return | 3 | 0.3027 | [-1.0488, 1.3302] | 1,2,3 |
| adaptive - uniform | train_auc | 3 | 0.1492 | [-0.3793, 0.4411] | 1,2,3 |
| adaptive - uniform | best_train_return | 3 | 0.0568 | [-0.1035, 0.2014] | 1,2,3 |
| adaptive - uniform | stability_gap | 3 | -0.2459 | [-1.1288, 1.1212] | 1,2,3 |
| adaptive - random | last_test_return | 3 | -0.3883 | [-1.2296, 0.7902] | 1,2,3 |
| adaptive - random | train_auc | 3 | -0.2204 | [-0.6002, 0.2370] | 1,2,3 |
| adaptive - random | best_train_return | 3 | -0.0231 | [-0.3386, 0.3298] | 1,2,3 |
| adaptive - random | stability_gap | 3 | 0.3652 | [-0.4604, 1.1691] | 1,2,3 |

## Penalty 0.0003

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 3 | 7.3106 | [7.1555, 7.5732] | 5.1314 | 11.7326 | 4.4220 |
| adaptive | 3 | 7.1639 | [6.8194, 7.6043] | 5.0032 | 11.6997 | 4.5358 |
| uniform | 3 | 7.5385 | [6.3824, 8.3390] | 5.1655 | 11.7036 | 4.1651 |
| random | 3 | 7.8243 | [7.5874, 8.2885] | 5.3716 | 11.7998 | 3.9755 |

## Penalty 0.0003 Budget Accounting

| method | records | triggers | penalty total | terminal bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adaptive | 1402.5333 | 1894.0000 | 16.0263 | 0.0097 | 91971.3333 | 0.0085 | 48.5623 |
| uniform | 1398.2667 | 1887.0000 | 20.1883 | 0.0080 | 91336.0000 | 0.0107 | 48.4060 |
| random | 1404.6667 | 1891.3333 | 17.5786 | 0.5582 | 91469.3333 | 0.0093 | 48.3662 |

## Penalty 0.0003 Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | shared seeds |
|---|---|---:|---:|---:|---|
| adaptive - baseline | last_test_return | 3 | -0.1467 | [-0.5052, 0.4488] | 1,2,3 |
| adaptive - baseline | train_auc | 3 | -0.1283 | [-0.2394, 0.0149] | 1,2,3 |
| adaptive - baseline | best_train_return | 3 | -0.0329 | [-0.1699, 0.0539] | 1,2,3 |
| adaptive - baseline | stability_gap | 3 | 0.1138 | [-0.3949, 0.5225] | 1,2,3 |
| adaptive - uniform | last_test_return | 3 | -0.3746 | [-1.0746, 0.6856] | 1,2,3 |
| adaptive - uniform | train_auc | 3 | -0.1623 | [-0.4450, 0.2790] | 1,2,3 |
| adaptive - uniform | best_train_return | 3 | -0.0039 | [-0.0481, 0.0265] | 1,2,3 |
| adaptive - uniform | stability_gap | 3 | 0.3707 | [-0.6591, 1.0846] | 1,2,3 |
| adaptive - random | last_test_return | 3 | -0.6604 | [-1.2205, 0.0169] | 1,2,3 |
| adaptive - random | train_auc | 3 | -0.3685 | [-0.4405, -0.2609] | 1,2,3 |
| adaptive - random | best_train_return | 3 | -0.1001 | [-0.3021, 0.0768] | 1,2,3 |
| adaptive - random | stability_gap | 3 | 0.5603 | [-0.3190, 1.2973] | 1,2,3 |

## Penalty Selection

| rank | penalty | heuristic score |
|---:|---:|---:|
| 1 | 0.00003 | 2.0798 |
| 2 | 0.00001 | 0.8668 |
| 3 | 0.0001 | 0.0414 |
| 4 | 0.0003 | -1.5112 |

Recommended full-run candidate: penalty `0.00003` if adaptive is not clearly worse than random-type and is positive versus baseline. If all scores are negative or random-type dominates, keep VMAS as a transparent reward-scale limitation instead of expanding.
