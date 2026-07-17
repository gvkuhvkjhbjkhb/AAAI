# V2 Pilot Results — DG-PBS v2 + FME (500k, 4 seeds)

## Overview

This pilot tests two new methods against baseline and the original FA-PBS: **DG-PBS v2** (fixed bipolar gate: amplify when positive, suppress when negative) and **FME** (failure-mode-aware exploration: inject targeted intrinsic bonuses on failed episodes). Both aim to address ARMS (2026) oscillatory failure and Akella (2025) misalignment barrier.

## Results — Aggregate (4 seeds, mean ± std)

| Method | Final Test Return | Best Train Return | Train AUC | Stability Gap |
|---|---:|---:|---:|---:|
| baseline | 0.3735 ± 0.134 | 0.4331 ± 0.144 | 0.1973 ± 0.045 | 0.0595 ± 0.057 |
| **fa_pbrs_l002** | **0.4860** ± 0.102 | **0.5746** ± 0.070 | **0.2453** ± 0.050 | 0.0886 ± 0.081 |
| dg_pbs_v2 | 0.3100 ± 0.181 | 0.3677 ± 0.212 | 0.1772 ± 0.032 | 0.0577 ± 0.045 |
| fme_l01 | 0.3291 ± 0.044 | 0.3929 ± 0.107 | 0.1837 ± 0.022 | 0.0638 ± 0.076 |

## Paired Differences vs Baseline (bootstrap 95% CI, n=4)

### Final Test Return
| Method | Δ | 95% CI | Wins | Conclusion |
|---|---:|---:|---:|---|
| fa_pbrs_l002 | +0.1124 | [−0.0000, +0.2719] | 3/4 | Not significant (boundary) |
| dg_pbs_v2 | −0.0635 | [−0.2604, +0.1333] | 2/4 | Not significant (worse) |
| fme_l01 | −0.0444 | [−0.1785, +0.0897] | 2/4 | Not significant (worse) |

### Best Train Return
| Method | Δ | 95% CI | Wins | Conclusion |
|---|---:|---:|---:|---|
| fa_pbrs_l002 | +0.1416 | [+0.0586, +0.2926] | 4/4 | **SIGNIFICANT** |
| dg_pbs_v2 | −0.0654 | [−0.2903, +0.1595] | 2/4 | Not significant |
| fme_l01 | −0.0402 | [−0.1641, +0.0827] | 2/4 | Not significant |

## Per-Seed Detail

| Seed | baseline | fa_pbrs | dg_v2 | fme | fa−b | dg−b | fme−b |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5180 | 0.5867 | 0.2785 | 0.2742 | +0.069 | −0.240 | −0.244 |
| 2 | 0.3175 | 0.3798 | 0.5725 | 0.3819 | +0.062 | +0.255 | +0.064 |
| 3 | 0.4429 | 0.4200 | 0.1616 | 0.3297 | −0.023 | −0.281 | −0.113 |
| 4 | 0.2158 | 0.5575 | 0.2274 | 0.3307 | +0.342 | +0.012 | +0.115 |

## DG-PBS v1 vs v2 Comparison

| Method | v1 mean | v2 mean | v2−v1 |
|---|---:|---:|---:|
| dg_pbs | 0.3630 | 0.3100 | −0.0530 |

**v2 is worse than v1.** The bipolar gate (amplify when positive) made things worse, not better. The amplification of lambda when correlation is positive destabilizes training — seeds 1 and 3 crashed dramatically (0.28, 0.16 vs v1's 0.54, 0.36).

## Conclusion

### Both new methods failed

1. **DG-PBS v2** (bipolar gate): Δ=−0.0635, worse than baseline AND worse than v1 (−0.053). The amplification logic destabilizes training — 2 of 4 seeds crashed. The diagnosis-gating approach is a dead end at this configuration.

2. **FME** (failure-mode exploration): Δ=−0.0444, worse than baseline. Injecting exploration bonuses on failed episodes does not help — the bonuses may be redirecting agents away from productive cooperation patterns.

3. **fa_pbrs remains the only positive method**: +0.1124 final return (3/4 wins), +0.1416 best train return (4/4 wins, significant). This is the same result as the previous pilot — fa_pbrs at 500k is robustly positive on sample efficiency.

### Root cause analysis

The diagnosis-gating and exploration-trigger approaches share a common failure: **they intervene based on a noisy signal (episode-level return correlation) that is too coarse to guide per-step interventions.** The shaping-return correlation at 50-episode window is near zero (corr≈0.005-0.08), so the gate/trigger operates on noise, and amplification of a noise-driven signal destabilizes training.

fa_pbrs works precisely because it applies a **fixed, stable** shaping signal — no gating, no amplification, no noise-driven switching. The consistency of the signal matters more than adaptivity.

### Final recommendation

**Abandon DG-PBS and FME.** The diagnosis-driven intervention approach has now failed in 3 variants (v1 conservative gate, v2 bipolar gate, FME exploration trigger). The evidence consistently shows that **adaptive intervention based on noisy episode-level diagnostics hurts more than it helps** in cooperative MARL.

The viable path remains:
1. **fa_pbrs as sample-efficiency method** (significant at 500k on best-train-return, consistent across two independent 4-seed pilots)
2. **Evaluation methodology paper** (方案1) using the rich negative results from all intervention variants

## Files

- Implementation: `epymarl/src/llm_diagnosis/diagnosis_gated_pbrs.py` (v2), `failure_mode_exploration.py`
- Launcher: `run_v2_pilot.sh`
- Results: `results/v2_pilot_v2_gpu0_*`, `results/v2_pilot_v2_gpu1_*`
- This report: `v2_pilot_results.md`
