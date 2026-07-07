# Round 11 VMAS Reward-Scale Calibration Report

Round 11 tests whether Round 9's VMAS weakness was caused by transferring the LBF-tuned 0.0003 reward-shaping scale into a dense-navigation reward landscape.

## Penalty 0.00003

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 5 | 7.3667 | [6.1701, 8.3017] | 4.9946 | 11.8016 | 4.4349 |
| adaptive | 5 | 6.8461 | [6.3784, 7.3155] | 4.8046 | 11.7337 | 4.8876 |
| uniform | 5 | 7.2629 | [6.7640, 7.7807] | 4.9847 | 11.9009 | 4.6380 |
| random | 5 | 7.3854 | [7.0041, 7.6521] | 4.9845 | 11.7569 | 4.3715 |

## Penalty 0.00003 Budget Accounting

| method | records | triggers | penalty total | terminal bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adaptive | 1387.8000 | 1867.2000 | 1.5890 | 0.0006 | 90979.6000 | 0.0009 | 48.7275 |
| uniform | 1391.8400 | 1874.4000 | 2.0111 | 0.0008 | 91053.2000 | 0.0011 | 48.5796 |
| random | 1383.9600 | 1869.2000 | 1.7535 | 0.0554 | 90784.4000 | 0.0009 | 48.5693 |

## Penalty 0.00003 Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | shared seeds |
|---|---|---:|---:|---:|---|
| adaptive - baseline | last_test_return | 5 | -0.5206 | [-1.4960, 0.4595] | 4,5,6,7,8 |
| adaptive - baseline | train_auc | 5 | -0.1900 | [-0.5511, 0.0777] | 4,5,6,7,8 |
| adaptive - baseline | best_train_return | 5 | -0.0679 | [-0.2096, 0.0746] | 4,5,6,7,8 |
| adaptive - baseline | stability_gap | 5 | 0.4526 | [-0.6465, 1.5484] | 4,5,6,7,8 |
| adaptive - uniform | last_test_return | 5 | -0.4168 | [-0.8518, 0.3779] | 4,5,6,7,8 |
| adaptive - uniform | train_auc | 5 | -0.1800 | [-0.4163, 0.1168] | 4,5,6,7,8 |
| adaptive - uniform | best_train_return | 5 | -0.1672 | [-0.3526, 0.0166] | 4,5,6,7,8 |
| adaptive - uniform | stability_gap | 5 | 0.2495 | [-0.3978, 0.6575] | 4,5,6,7,8 |
| adaptive - random | last_test_return | 5 | -0.5393 | [-0.9477, -0.1309] | 4,5,6,7,8 |
| adaptive - random | train_auc | 5 | -0.1798 | [-0.4294, 0.0697] | 4,5,6,7,8 |
| adaptive - random | best_train_return | 5 | -0.0232 | [-0.0745, 0.0298] | 4,5,6,7,8 |
| adaptive - random | stability_gap | 5 | 0.5161 | [0.0960, 0.9362] | 4,5,6,7,8 |

## Penalty Selection

| rank | penalty | heuristic score |
|---:|---:|---:|
| 1 | 0.00003 | -1.7516 |

Recommended full-run candidate: penalty `0.00003` if adaptive is not clearly worse than random-type and is positive versus baseline. If all scores are negative or random-type dominates, keep VMAS as a transparent reward-scale limitation instead of expanding.
