# Exp B — 20-Seed Paired Statistical Analysis (Wilcoxon Signed-Rank)

> Generated: Wed Jul 15 17:07:07 UTC 2026 UTC
> Data: `/data/lab/results/v2/exp_b_20seed`
> Paired by seed across cells. Stats: Wilcoxon signed-rank (two-sided),
> Cohen's d (paired), 10k-resample percentile bootstrap 95% CI, paired win-rate.
> Cells: **NoToM**=het_notom (baseline), **Gated**=het_gated_atom_talk (forced alignment),
> **CGA**=het_dp_gated_atom_talk (conditional gated arbitration), **GSACA**=het_gsaca (structure-adaptive).

## 0. Data completeness
- battle_of_the_sexes / NoToM: 20/20 seeds
- battle_of_the_sexes / Gated: 20/20 seeds
- battle_of_the_sexes / CGA: 20/20 seeds
- battle_of_the_sexes / GSACA: 20/20 seeds
- public_goods / NoToM: 19/19 seeds
- public_goods / Gated: 19/19 seeds
- public_goods / CGA: 19/19 seeds
- public_goods / GSACA: 14/19 seeds

---
## BATTLE OF THE SEXES

### Descriptive statistics (cooperation payoff + secondary metrics)
| Cell | n | cooperation_payoff | perspective_diversity | equilibrium_conv | tom_pred_acc |
|---|---|---|---|---|---|
| NoToM | 20 | 1.550±0.194 [1.468,1.633] (n=20) | 0.267±0.099 [0.226,0.311] (n=20) | 0.965±0.023 [0.955,0.975] (n=20) | n/a |
| Gated | 20 | 2.216±0.132 [2.162,2.275] (n=20) | 0.004±0.006 [0.002,0.007] (n=20) | 0.906±0.052 [0.885,0.929] (n=20) | 0.714±0.026 [0.703,0.725] (n=20) |
| CGA | 20 | 1.660±0.125 [1.610,1.717] (n=20) | 0.230±0.085 [0.194,0.267] (n=20) | 0.969±0.030 [0.956,0.981] (n=20) | 0.606±0.031 [0.593,0.620] (n=20) |
| GSACA | 20 | 2.318±0.109 [2.270,2.363] (n=20) | 0.005±0.007 [0.002,0.008] (n=20) | 0.938±0.044 [0.919,0.955] (n=20) | 0.773±0.032 [0.759,0.786] (n=20) |

- **GSACA structure detection**: 20/20 = 100.0% correct
- Oracle structure (per seed): Counter({'coord': 20})
- Detected structure: Counter({'coord': 20})

### Pairwise paired comparisons (cooperation_payoff, Wilcoxon signed-rank)

| Comparison | n | mean Δ | 95% CI | Cohen's d | win/lose/tie | W | p | sig |
|---|---|---|---|---|---|---|---|---|
| Gated vs NoToM | 20 | +0.666 | [0.561,0.767] | 2.752 | 20/0/0 | 0.0 | 0.0000 | *** |
| CGA vs NoToM | 20 | +0.110 | [0.034,0.186] | 0.613 | 15/5/0 | 40.5 | 0.0153 | *   |
| CGA vs Gated | 20 | -0.556 | [-0.635,-0.471] | -2.902 | 0/20/0 | 0.0 | 0.0000 | *** |
| GSACA vs NoToM | 20 | +0.768 | [0.666,0.870] | 3.204 | 20/0/0 | 0.0 | 0.0000 | *** |
| GSACA vs Gated | 20 | +0.102 | [0.018,0.174] | 0.560 | 14/6/0 | 40.0 | 0.0136 | *   |
| GSACA vs CGA | 20 | +0.658 | [0.583,0.730] | 3.828 | 20/0/0 | 0.0 | 0.0000 | *** |

### Mechanism diagnostics (GSACA cell, mean across seeds)

- gate_trust_rate: 0.9070±0.0647 [0.8790,0.9330] (n=20)
- gated_prediction_accuracy: 0.7250±0.1214 [0.6710,0.7730] (n=20)
- signal_accuracy: 0.7640±0.0845 [0.7290,0.8000] (n=20)
- dp_conflict_rate: 0.3410±0.0989 [0.3020,0.3860] (n=20)
- dp_intervention_rate: 0.3410±0.0989 [0.3020,0.3860] (n=20)
- gsaca_split_score: -2.5000±0.0000 [-2.5000,-2.5000] (n=20)

---
## PUBLIC GOODS

### Descriptive statistics (cooperation payoff + secondary metrics)
| Cell | n | cooperation_payoff | perspective_diversity | equilibrium_conv | tom_pred_acc |
|---|---|---|---|---|---|
| NoToM | 19 | 2.573±0.022 [2.563,2.582] (n=19) | 0.119±0.034 [0.105,0.134] (n=19) | 0.928±0.049 [0.905,0.947] (n=19) | n/a |
| Gated | 19 | 2.630±0.023 [2.620,2.640] (n=19) | 0.176±0.211 [0.108,0.284] (n=19) | 0.944±0.044 [0.924,0.962] (n=19) | 0.700±0.039 [0.684,0.717] (n=19) |
| CGA | 19 | 2.615±0.022 [2.606,2.625] (n=19) | 0.101±0.034 [0.086,0.116] (n=19) | 0.944±0.034 [0.929,0.959] (n=19) | 0.640±0.027 [0.628,0.651] (n=19) |
| GSACA | 14 | 2.559±0.018 [2.550,2.568] (n=14) | 0.184±0.284 [0.047,0.340] (n=14) | 0.930±0.024 [0.919,0.943] (n=14) | 0.811±0.020 [0.801,0.821] (n=14) |

- **GSACA structure detection**: 14/14 = 100.0% correct
- Oracle structure (per seed): Counter({'coord': 14})
- Detected structure: Counter({'coord': 14})

### Pairwise paired comparisons (cooperation_payoff, Wilcoxon signed-rank)

| Comparison | n | mean Δ | 95% CI | Cohen's d | win/lose/tie | W | p | sig |
|---|---|---|---|---|---|---|---|---|
| Gated vs NoToM | 19 | +0.058 | [0.042,0.073] | 1.613 | 17/2/0 | 4.0 | 0.0000 | *** |
| CGA vs NoToM | 19 | +0.042 | [0.029,0.056] | 1.362 | 17/2/0 | 3.0 | 0.0000 | *** |
| CGA vs Gated | 19 | -0.016 | [-0.029,-0.003] | -0.528 | 7/12/0 | 47.0 | 0.0546 | .   |
| GSACA vs NoToM | 14 | -0.018 | [-0.033,-0.002] | -0.606 | 2/12/0 | 15.0 | 0.0166 | *   |
| GSACA vs Gated | 14 | -0.069 | [-0.083,-0.056] | -2.633 | 0/14/0 | 0.0 | 0.0001 | *** |
| GSACA vs CGA | 14 | -0.057 | [-0.072,-0.041] | -1.869 | 0/14/0 | 0.0 | 0.0001 | *** |

### Mechanism diagnostics (GSACA cell, mean across seeds)

- gate_trust_rate: 0.9135±0.0444 [0.8905,0.9361] (n=14)
- gated_prediction_accuracy: 0.7790±0.0875 [0.7345,0.8250] (n=14)
- signal_accuracy: 0.8345±0.0618 [0.8036,0.8679] (n=14)
- dp_conflict_rate: 0.3270±0.0678 [0.2929,0.3603] (n=14)
- dp_intervention_rate: 0.3270±0.0678 [0.2929,0.3603] (n=14)
- gsaca_split_score: -0.1980±0.0277 [-0.2117,-0.1836] (n=14)

---
## Cross-game summary

### GSACA vs baselines — does structure-adaptive alignment win? (Δ = GSACA − baseline, payoff)

| Game | vs NoToM Δ (p) | vs Gated Δ (p) | vs CGA Δ (p) |
|---|---|---|---|
| battle_of_the_sexes | +0.768 (***) | +0.102 (*) | +0.658 (***) |
| public_goods | -0.018 (*) | -0.069 (***) | -0.057 (***) |

### Interpretation key
- These two games are **coordination** games (symmetric/coordination NE).
- Expected pattern: forced alignment (Gated) helps coordination; CGA (conditional)
  should not hurt; **GSACA** detects 'coord' structure and selects the Gated arm,
  recovering Gated-level payoff while keeping the safety of abstaining when CGA would harm.
- sig codes: `***` p<0.001, `**` p<0.01, `*` p<0.05, `.` p<0.10, `ns` otherwise.
