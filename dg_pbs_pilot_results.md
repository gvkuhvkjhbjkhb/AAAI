# DG-PBS Pilot Results — Diagnosis-Gated Reward Shaping (500k, 4 seeds)

## Overview

This pilot tests the **Diagnosis-Gated PBRS (DG-PBS)** — a new method that detects whether the shaping signal is aligned with learning progress and auto-gates the effective lambda. This is a direct response to ARMS (2026)'s oscillatory failure mode and Akella (2025)'s reward-global misalignment barrier. The pilot compares 3 methods at 500k steps, 4 seeds, on LBF 10x10-3p-3f.

## Experiment Configuration

| Parameter | Value |
|---|---|
| Environment | `lbforaging:Foraging-10x10-3p-3f-v3` |
| Timesteps | 500,000 |
| Seeds | 1, 2, 3, 4 |
| Methods | `baseline`, `fa_pbrs_l002`, `dg_pbs_l002` |
| Total runs | 12 (all completed, 0 failures) |
| GPUs | 2× RTX 5090 (6 runs each, parallel) |
| Wall time | ~20 minutes (15:56 → 16:16 UTC) |

## DG-PBS Method

DG-PBS extends FA-PBS with a **divergence gate**: it tracks the correlation between shaping magnitude and episode return over a sliding window (50 episodes). When the correlation is positive (shaping helps), the gate opens (→1.0). When negative (shaping hurts), the gate closes (→0.05), reducing effective lambda. The gate uses a sigmoid with temperature=5.0 and smoothing=0.8.

## Results — Aggregate (4 seeds, mean ± std)

| Method | Final Test Return | Best Train Return | Train AUC | Stability Gap |
|---|---:|---:|---:|---:|
| baseline | 0.3735 ± 0.134 | 0.4331 ± 0.144 | 0.1973 ± 0.045 | 0.0595 ± 0.057 |
| fa_pbrs_l002 | **0.4860** ± 0.102 | **0.5746** ± 0.070 | **0.2453** ± 0.050 | 0.0886 ± 0.081 |
| dg_pbs_l002 | 0.3630 ± 0.126 | 0.5044 ± 0.172 | 0.2077 ± 0.043 | 0.1413 ± 0.088 |

## Paired Differences (bootstrap 95% CI, n=4)

### Final Test Return
| Comparison | Δ | 95% CI | Conclusion |
|---|---:|---:|---|
| fa_pbrs − baseline | +0.1124 | [−0.0000, +0.2719] | Not significant (boundary) |
| dg_pbs − baseline | −0.0105 | [−0.0557, +0.0280] | Not significant |
| **dg_pbs − fa_pbrs** | **−0.1230** | **[−0.2451, −0.0528]** | **SIGNIFICANT (DG-PBS worse)** |

### Best Train Return
| Comparison | Δ | 95% CI | Conclusion |
|---|---:|---:|---|
| fa_pbrs − baseline | +0.1416 | [+0.0586, +0.2926] | **SIGNIFICANT** |
| dg_pbs − baseline | +0.0713 | [+0.0403, +0.1235] | **SIGNIFICANT** |
| dg_pbs − fa_pbrs | −0.0703 | [−0.2486, +0.0498] | Not significant |

### Train AUC
| Comparison | Δ | 95% CI | Conclusion |
|---|---:|---:|---|
| fa_pbrs − baseline | +0.0480 | [−0.0020, +0.0979] | Not significant (boundary) |
| dg_pbs − baseline | +0.0104 | [−0.0208, +0.0416] | Not significant |
| **dg_pbs − fa_pbrs** | **−0.0376** | **[−0.0563, −0.0189]** | **SIGNIFICANT (DG-PBS worse)** |

### Stability Gap (lower = better)
| Comparison | Δ | 95% CI | Conclusion |
|---|---:|---:|---|
| fa_pbrs − baseline | +0.0291 | [+0.0050, +0.0615] | SIG (fa_pbrs worse) |
| dg_pbs − baseline | +0.0818 | [+0.0351, +0.1253] | SIG (dg_pbs worse) |
| dg_pbs − fa_pbrs | +0.0527 | [+0.0016, +0.0997] | SIG (dg_pbs worse) |

## Per-Seed Detail

| Seed | baseline | fa_pbrs | dg_pbs | fa−b | dg−b | dg−fa |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5180 | 0.5867 | 0.5397 | +0.069 | +0.022 | −0.047 |
| 2 | 0.3175 | 0.3798 | 0.3009 | +0.062 | −0.017 | −0.079 |
| 3 | 0.4429 | 0.4200 | 0.3614 | −0.023 | −0.082 | −0.059 |
| 4 | 0.2158 | 0.5575 | 0.2502 | +0.342 | +0.034 | −0.307 |

DG-PBS loses to fa_pbrs on **all 4 seeds** (0/4 wins) and ties/loses to baseline (2/4 wins).

## Gate Behavior Analysis

| Seed | Final Gate | Effective Lambda | Gate Min |
|---:|---:|---:|---:|
| 1 | 0.925 | 0.0185 | 0.925 |
| 2 | 0.832 | 0.0166 | 0.832 |
| 3 | 0.801 | 0.0160 | 0.801 |
| 4 | 0.834 | 0.0167 | 0.834 |

The gate consistently reduces lambda from 0.02 to ~0.016-0.019 (8-20% reduction). However, this **conservative gating actually hurts**: it weakens the shaping signal that fa_pbrs applies at full strength, and the resulting reduced shaping underperforms both full-strength fa_pbrs AND baseline on final return.

## Key Findings

### 1. DG-PBS gate is counterproductive at 500k
The gate detects positive correlation between shaping and returns (gate stays 0.80-0.92), then *reduces* the effective lambda. But at 500k steps the method needs MORE shaping signal, not less. The gate's conservative behavior weakens the intervention precisely when it's helping.

### 2. fa_pbrs is actually the strongest method at 500k
This is a surprising reversal from the 8-seed 1M results. At 500k, fa_pbrs shows:
- Final Return +0.1124 vs baseline (CI nearly significant, [−0.0000, +0.2719])
- Best Train Return +0.1416 vs baseline (CI significant, [+0.0586, +0.2926])
- Wins 3/4 seeds on final return

This aligns with the horizon-effect finding: at 500k fa_pbrs helps (sample efficiency), but the advantage erodes by 1M as baseline catches up.

### 3. The stability gap result is concerning for both shaping methods
Both fa_pbrs (+0.029) and dg_pbs (+0.082) have **significantly worse stability gap** than baseline. This means shaping methods cause more late-training regression — the method's best return is reached earlier but then degrades. This is the opposite of the 3-seed 1M finding (where fa_pbrs had zero gap on seeds 2,3).

## Conclusion: Direction Assessment

**DG-PBS (diagnosis gating) is a negative result at this configuration.** The gate mechanism, while functioning as designed (detecting positive correlation and reducing lambda conservatively), is counterproductive: it weakens the shaping signal that fa_pbrs needs, producing worse results than both fa_pbrs and baseline.

The core issue: the gate's design assumes "less shaping when uncertain is safer," but the data shows that at 500k steps, the full-strength shaping (fa_pbrs) is actually beneficial and the gate's reduction removes exactly the signal that helps.

### What this tells us for the paper strategy

1. **fa_pbrs at 500k is genuinely promising** (+0.11 final return, 3/4 wins, best-train significant) — this is the "sample efficiency" story, consistent with the 8-seed AUC finding.
2. **The diagnosis-gating mechanism needs redesign**: instead of gating DOWN when correlation is positive, the gate should gate UP (amplify) when correlation is positive, and gate DOWN only when correlation turns negative. The current sigmoid design is backwards for the positive-correlation regime.
3. **Alternatively**, the gate should use a **hard cutoff** (full lambda when corr > 0, zero when corr < 0) rather than the smooth sigmoid that always reduces lambda.

### Recommended next step

Two options:
- **Option A**: Fix the gate logic (amplify when positive, suppress when negative) and re-run the 4-seed pilot. ~20 min on dual GPU.
- **Option B**: Pivot to the **evaluation methodology paper** (方案1), since we now have strong data: fa_pbrs helps at 500k but not 1M, DG-PBS gate fails, seed sensitivity across all methods. This is a rich empirical story about MARL evaluation pitfalls.

## Files

- Launcher: `run_dg_pbs_pilot.sh`
- Implementation: `epymarl/src/llm_diagnosis/diagnosis_gated_pbrs.py`
- Results GPU0: `results/dg_pbs_pilot_dg_pbs_gpu0_20260709_155627/`
- Results GPU1: `results/dg_pbs_pilot_dg_pbs_gpu1_20260709_155627/`
- This report: `dg_pbs_pilot_results.md`
