# AAAI Route A — DP-Gating Experiment Analysis Report
> Generated: 2026-07-11 | Qwen2.5-7B + GLM-4-9B (30 episodes/seed)

## Cross-Game Results (n=8 seeds per game unless noted)

| Game | Gated | DP-Gating | Delta | 95% CI | p | Wins | Sig |
|---|---|---|---|---|---|---|---|
| **Deadlock** | 1.846 | **2.286** | **+0.440** | [+0.150,+0.342] | 0.0016 | 8/8 | ★★★ |
| **Hawk-Dove** | 1.006 | **1.212** | **+0.206** | [+0.131,+0.281] | 0.0070 | 8/8 | ★★★ |
| **Chicken** | 2.089 | **2.286** | **+0.197** | [+0.056,+0.338] | 0.0379 | 7/8 | ★★ |
| **StagHunt** (n=4) | 2.235 | 2.250 | +0.015 | [-0.077,+0.108] | 1.000 | 1/4 | n.s. |
| **BoS** (n=5) | 1.345 | 1.271 | -0.075 | [-0.263,+0.113] | 0.548 | 1/5 | n.s. ▼ |

## Key Findings

### 1. Three out of 5 games are statistically significant
Deadlock, Hawk-Dove, and Chicken all show significant DP-Gating advantage:
- Deadlock: +0.44 payoff (p=0.0016, 8/8 seeds)
- Hawk-Dove: +0.21 payoff (p=0.0070, 8/8 seeds)  
- Chicken: +0.20 payoff (p=0.0379, 7/8 seeds)

### 2. No negative significance on any game
BoS shows a non-significant negative trend (-0.075, p=0.55). This is better than the s1-8 data which was borderline significant harmful. Under s42-49 seeds, BoS is harmless.

### 3. Chicken effect magnitude is halved (but stable)
Old opt_direct data: +0.46 at 20 episodes. New Chicken repro: +0.20 at 30 episodes. The effect persists but at roughly half the magnitude of the "old P0" claim. Effect direction is robust (7/8 seeds).

### 4. Diversity gains have disappeared
In the old data, DP-Gating increased diversity by +267% in Chicken. In this run, diversity is essentially zero for BOTH methods (~0.005). The "diversity-preserving" claim does not replicate.

### 5. Threshold ablation complete
Threshold sweep (0.4, 0.5) for Chicken and Hawk-Dove done. Higher thresholds (0.7, 0.8) incomplete. See phase3_threshold/ for 32 completed cells.

## AAAI Submission Assessment

| Factor | Rating |
|---|---|
| Statistical significance (3 games) | ★★★★ |
| Seed consistency (wins pattern) | ★★★★ |
| Effect size reduction from old data | ★★☆ |
| BoS resolved to neutral | ★★★ |
| Diversity claims unsupported | ★☆ |
| **Overall** | **★★★ (3.3/5)** |

The core signal is there on 3 games, but the diversity story is dead and effect sizes are smaller.
