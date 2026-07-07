# Round 12 Simple Cooperative Domain Report

Round 12 searches for simple sparse cooperative domains that can provide positive family-level evidence without relying on VMAS/RWARE/Qwen.

## LBF 8x8 2p-2f Coop

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 8 | 0.1677 | [0.0698, 0.2781] | 0.1031 | 0.3866 | 0.2189 |
| baseline | 8 | 0.1958 | [0.0917, 0.3010] | 0.1052 | 0.4083 | 0.2124 |
| random_type_budget_matched_0.0003_late045 | 8 | 0.2500 | [0.1781, 0.3260] | 0.1424 | 0.5620 | 0.3120 |
| uniform_budget_matched_0.0003_late045 | 8 | 0.1625 | [0.0677, 0.2677] | 0.0977 | 0.3972 | 0.2347 |

## LBF 8x8 2p-2f Coop Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | seeds |
|---|---|---:|---:|---:|---|
| adaptive - baseline | last_test_return | 8 | -0.0281 | [-0.1021, 0.0375] | 1,2,3,4,5,6,7,8 |
| adaptive - baseline | train_auc | 8 | -0.0021 | [-0.0434, 0.0389] | 1,2,3,4,5,6,7,8 |
| adaptive - baseline | best_train_return | 8 | -0.0216 | [-0.1501, 0.1096] | 1,2,3,4,5,6,7,8 |
| adaptive - baseline | stability_gap | 8 | 0.0065 | [-0.0760, 0.0937] | 1,2,3,4,5,6,7,8 |
| adaptive - uniform_budget_matched_0.0003_late045 | last_test_return | 8 | 0.0052 | [-0.0708, 0.0719] | 1,2,3,4,5,6,7,8 |
| adaptive - uniform_budget_matched_0.0003_late045 | train_auc | 8 | 0.0054 | [-0.0327, 0.0426] | 1,2,3,4,5,6,7,8 |
| adaptive - uniform_budget_matched_0.0003_late045 | best_train_return | 8 | -0.0106 | [-0.1960, 0.1487] | 1,2,3,4,5,6,7,8 |
| adaptive - uniform_budget_matched_0.0003_late045 | stability_gap | 8 | -0.0158 | [-0.1244, 0.0776] | 1,2,3,4,5,6,7,8 |
| adaptive - random_type_budget_matched_0.0003_late045 | last_test_return | 8 | -0.0823 | [-0.1906, 0.0240] | 1,2,3,4,5,6,7,8 |
| adaptive - random_type_budget_matched_0.0003_late045 | train_auc | 8 | -0.0393 | [-0.0953, 0.0200] | 1,2,3,4,5,6,7,8 |
| adaptive - random_type_budget_matched_0.0003_late045 | best_train_return | 8 | -0.1753 | [-0.3976, 0.0435] | 1,2,3,4,5,6,7,8 |
| adaptive - random_type_budget_matched_0.0003_late045 | stability_gap | 8 | -0.0930 | [-0.2079, 0.0232] | 1,2,3,4,5,6,7,8 |

## LBF 8x8 2p-2f Coop Budget Accounting

| method | records | triggers | penalty total | bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 2988.3000 | 3000.0000 | 28.6910 | 0.0004 | 147960.5000 | 0.0096 | 49.3202 |
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_type_budget_matched_0.0003_late045 | 2976.6750 | 3000.0000 | 32.5337 | 0.8871 | 149685.1250 | 0.0108 | 49.8950 |
| uniform_budget_matched_0.0003_late045 | 2991.4250 | 3000.0000 | 36.5402 | 0.0003 | 151323.1250 | 0.0122 | 50.4410 |

## LBF 10x10 3p-3f Coop

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 8 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0003 | 0.0003 |
| baseline | 8 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0001 | 0.0001 |
| random_type_budget_matched_0.0003_late045 | 8 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 |
| uniform_budget_matched_0.0003_late045 | 8 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0001 | 0.0001 |

## LBF 10x10 3p-3f Coop Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | seeds |
|---|---|---:|---:|---:|---|
| adaptive - baseline | last_test_return | 8 | 0.0000 | [0.0000, 0.0000] | 1,2,3,4,5,6,7,8 |
| adaptive - baseline | train_auc | 8 | 0.0000 | [0.0000, 0.0001] | 1,2,3,4,5,6,7,8 |
| adaptive - baseline | best_train_return | 8 | 0.0003 | [0.0000, 0.0005] | 1,2,3,4,5,6,7,8 |
| adaptive - baseline | stability_gap | 8 | 0.0003 | [0.0000, 0.0005] | 1,2,3,4,5,6,7,8 |
| adaptive - uniform_budget_matched_0.0003_late045 | last_test_return | 8 | 0.0000 | [0.0000, 0.0000] | 1,2,3,4,5,6,7,8 |
| adaptive - uniform_budget_matched_0.0003_late045 | train_auc | 8 | 0.0000 | [0.0000, 0.0001] | 1,2,3,4,5,6,7,8 |
| adaptive - uniform_budget_matched_0.0003_late045 | best_train_return | 8 | 0.0003 | [0.0001, 0.0004] | 1,2,3,4,5,6,7,8 |
| adaptive - uniform_budget_matched_0.0003_late045 | stability_gap | 8 | 0.0003 | [0.0001, 0.0004] | 1,2,3,4,5,6,7,8 |
| adaptive - random_type_budget_matched_0.0003_late045 | last_test_return | 8 | 0.0000 | [0.0000, 0.0000] | 1,2,3,4,5,6,7,8 |
| adaptive - random_type_budget_matched_0.0003_late045 | train_auc | 8 | 0.0000 | [0.0000, 0.0001] | 1,2,3,4,5,6,7,8 |
| adaptive - random_type_budget_matched_0.0003_late045 | best_train_return | 8 | 0.0003 | [0.0001, 0.0005] | 1,2,3,4,5,6,7,8 |
| adaptive - random_type_budget_matched_0.0003_late045 | stability_gap | 8 | 0.0003 | [0.0001, 0.0005] | 1,2,3,4,5,6,7,8 |

## LBF 10x10 3p-3f Coop Budget Accounting

| method | records | triggers | penalty total | bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 3000.0000 | 3000.0000 | 29.6970 | 0.0000 | 153000.0000 | 0.0099 | 51.0000 |
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| random_type_budget_matched_0.0003_late045 | 3000.0000 | 3000.0000 | 33.1453 | 0.4830 | 153000.0000 | 0.0110 | 51.0000 |
| uniform_budget_matched_0.0003_late045 | 3000.0000 | 3000.0000 | 36.7984 | 0.0000 | 153000.0000 | 0.0123 | 51.0000 |

## Decision Rule

Use a domain as positive evidence only if adaptive beats baseline and remains competitive with or better than phase-uniform/random controls under paired seeds. Otherwise report it as a stress test.
