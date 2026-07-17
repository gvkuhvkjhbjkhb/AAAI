# Round 12 Simple Cooperative Domain Report

Round 12 searches for simple sparse cooperative domains that can provide positive family-level evidence without relying on VMAS/RWARE/Qwen.

## LBF 8x8 2p-2f Coop

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 1 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 |
| baseline | 1 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 |

## LBF 8x8 2p-2f Coop Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | seeds |
|---|---|---:|---:|---:|---|
| adaptive - baseline | last_test_return | 1 | 0.0000 | [0.0000, 0.0000] | 1 |
| adaptive - baseline | train_auc | 1 | 0.0000 | [0.0000, 0.0000] | 1 |
| adaptive - baseline | best_train_return | 1 | 0.0000 | [0.0000, 0.0000] | 1 |
| adaptive - baseline | stability_gap | 1 | 0.0000 | [0.0000, 0.0000] | 1 |

## LBF 8x8 2p-2f Coop Budget Accounting

| method | records | triggers | penalty total | bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 20.0000 | 30.0000 | 0.2326 | 0.0000 | 1530.0000 | 0.0078 | 51.0000 |
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## LBF 10x10 3p-3f Coop

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 1 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 |
| baseline | 1 | 0.0000 | [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 |

## LBF 10x10 3p-3f Coop Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | seeds |
|---|---|---:|---:|---:|---|
| adaptive - baseline | last_test_return | 1 | 0.0000 | [0.0000, 0.0000] | 1 |
| adaptive - baseline | train_auc | 1 | 0.0000 | [0.0000, 0.0000] | 1 |
| adaptive - baseline | best_train_return | 1 | 0.0000 | [0.0000, 0.0000] | 1 |
| adaptive - baseline | stability_gap | 1 | 0.0000 | [0.0000, 0.0000] | 1 |

## LBF 10x10 3p-3f Coop Budget Accounting

| method | records | triggers | penalty total | bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 20.0000 | 30.0000 | 0.2460 | 0.0000 | 1530.0000 | 0.0082 | 51.0000 |
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Decision Rule

Use a domain as positive evidence only if adaptive beats baseline and remains competitive with or better than phase-uniform/random controls under paired seeds. Otherwise report it as a stress test.
