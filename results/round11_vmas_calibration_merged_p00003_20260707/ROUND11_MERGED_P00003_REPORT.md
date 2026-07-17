# Round 11 Merged VMAS p=0.00003 Report

This report merges the pilot seeds 1-3 and full-extension seeds 4-8 for the calibrated VMAS penalty p=0.00003.

| method | n | last test | last test 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 8 | 7.3457 | [6.6104, 7.9562] | 5.0459 | 11.7757 | 4.4301 |
| adaptive | 8 | 7.1841 | [6.7281, 7.6407] | 4.9341 | 11.6762 | 4.4921 |
| uniform | 8 | 7.1814 | [6.7755, 7.5876] | 5.0026 | 11.8408 | 4.6595 |
| random | 8 | 7.2154 | [6.9240, 7.4902] | 5.0032 | 11.6837 | 4.4683 |

## Budget Accounting

| method | records | triggers | penalty total | terminal bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | NA | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adaptive | 1389.3750 | 1872.0000 | 1.5937 | 0.0006 | 91218.0000 | 0.0009 | 48.7291 |
| uniform | 1395.8000 | 1880.1250 | 2.0109 | 0.0008 | 91141.5000 | 0.0011 | 48.4793 |
| random | 1383.0000 | 1864.0000 | 1.7545 | 0.0553 | 90942.3750 | 0.0009 | 48.7905 |

## Paired Comparisons

| comparison | metric | n | mean delta | 95% CI | seeds |
|---|---|---:|---:|---:|---|
| adaptive - baseline | last_test_return | 8 | -0.1616 | [-0.9309, 0.5444] | 1,2,3,4,5,6,7,8 |
| adaptive - baseline | train_auc | 8 | -0.1118 | [-0.3625, 0.0940] | 1,2,3,4,5,6,7,8 |
| adaptive - baseline | best_train_return | 8 | -0.0996 | [-0.2434, 0.0408] | 1,2,3,4,5,6,7,8 |
| adaptive - baseline | stability_gap | 8 | 0.0620 | [-0.6870, 0.8963] | 1,2,3,4,5,6,7,8 |
| adaptive - uniform | last_test_return | 8 | 0.0028 | [-0.5877, 0.6492] | 1,2,3,4,5,6,7,8 |
| adaptive - uniform | train_auc | 8 | -0.0685 | [-0.2820, 0.1471] | 1,2,3,4,5,6,7,8 |
| adaptive - uniform | best_train_return | 8 | -0.1647 | [-0.3177, -0.0266] | 1,2,3,4,5,6,7,8 |
| adaptive - uniform | stability_gap | 8 | -0.1674 | [-0.7084, 0.3607] | 1,2,3,4,5,6,7,8 |
| adaptive - random | last_test_return | 8 | -0.0313 | [-0.5688, 0.4873] | 1,2,3,4,5,6,7,8 |
| adaptive - random | train_auc | 8 | -0.0690 | [-0.2721, 0.1208] | 1,2,3,4,5,6,7,8 |
| adaptive - random | best_train_return | 8 | -0.0076 | [-0.0728, 0.0728] | 1,2,3,4,5,6,7,8 |
| adaptive - random | stability_gap | 8 | 0.0238 | [-0.4756, 0.5504] | 1,2,3,4,5,6,7,8 |
