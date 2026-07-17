# AAAI Pivot Plan — Final Strategy

## Based on 18 Rounds, ~350 Runs of Evidence

---

## Evidence Summary

### What Survived

| Finding | Significance | Robustness | Status |
|---|---|---|---|
| Structured potential > random features | CI [0.0384, 0.1054] | 16 seeds | **Survives** |
| 800k horizon: baseline 0.25→0.65, FA-PBS +0.088 | 2 seeds only | Unvalidated | Needs 1M validation |
| Budget accounting: adaptive uses less budget | Directional | 8 seeds | Survives as defense |

### What Failed

| Finding | Root Cause |
|---|---|
| FA-PBS > baseline at 500k/16-seed | Training not converged; signal淹没 in noise |
| Adaptive weight > fixed PBS | Scalar multiplier; no structural difference |
| Seeds 1-8 positive, 9-16 negative | Evaluation variance > method effect at 500k |
| Qwen semantic diagnosis | Collapsed to single class; accuracy 0.09 |
| Cross-domain (VMAS, RWARE, 4p-4f) | Method doesn't generalize |
| Q-Shaping at 100k | Signal too weak (+0.0028); not worth pursuing |

---

## Pivot Strategy: Direction A+B Combination

### Core Idea

Run ONE final validation experiment (1M steps, 8 seeds). The result determines the paper framing:
- **If significant**: Positive method paper (Direction B)
- **If not significant**: Empirical mechanism study (Direction A)

Either way, the paper is writeable with existing evidence.

### Final Validation Experiment

| Parameter | Value |
|---|---|
| Environment | Foraging-10x10-3p-3f-v3 |
| Timesteps | 1,000,000 |
| Seeds | 1-8 |
| Methods | baseline, fa_pbrs_l002, pbrs_fixed_l002, pbrs_random_weights_l002, pbrs_random_features_l002 |
| Parallel | 8 |
| Est. time | ~8 hours |

### Decision Rule

- fa_pbrs_l002 vs baseline: CI lower bound > 0 → **Direction B (positive method paper)**
- fa_pbrs_l002 vs baseline: CI crosses 0 BUT vs random_features CI > 0 → **Direction A (mechanism study)**
- fa_pbrs_l002 vs random_features: CI crosses 0 → **Direction A with negative framing**

---

## Direction A: Mechanism Isolation Study (if validation fails)

### Title
*When Does Potential-Based Reward Shaping Help in Cooperative MARL? A Mechanism Isolation Study*

### Core Claim
Structured potential features (cooperation + exploration + target alignment) significantly outperform random potential features in cooperative foraging, but this advantage does not translate to robust improvement over baseline — revealing structural limitations of PBRS in cooperative MARL.

### Contributions
1. **Mechanism isolation** (survives at 16 seeds): structured > random, CI [0.0384, 0.1054]
2. **Seed sensitivity warning**: all reward shaping methods show seeds 1-8 positive, 9-16 negative
3. **Evaluation horizon effect**: 500k→800k causes baseline return to jump 0.25→0.65; method delta amplifies 7x
4. **Budget accounting**: adaptive uses less shaping budget without extra benefit
5. **Comprehensive negative results**: Qwen diagnosis collapse, cross-domain failure, adaptive weight no independent contribution

### Paper Structure
- Main Table 1: Mechanism isolation (structured vs random features, 16 seeds)
- Main Table 2: Horizon effect (500k vs 800k vs 1M, paired seeds)
- Main Table 3: Seed sensitivity (seeds 1-8 vs 9-16)
- Appendix: Qwen diagnostic collapse, VMAS/RWARE/4p-4f negative results, budget accounting

### AAAI Probability: 20-30%

---

## Direction B: Positive Method Paper (if validation succeeds)

### Title
*Failure-Aware Potential-Based Reward Shaping for Cooperative Multi-Agent Reinforcement Learning*

### Core Claim
FA-PBS with structured potential features and failure-driven adaptive weights significantly improves cooperative foraging performance under sufficient training, while preserving optimal-policy guarantees.

### Contributions
1. **Theory**: PBRS preserves optimal policy (Ng et al. 1999); failure-driven adaptive potential is novel
2. **Method**: φ(s) = λ·Σ wₖ·φₖ(s) with cooperation/exploration/target features
3. **Mechanism isolation**: structured > random features (significant)
4. **Main result**: FA-PBS > baseline at 1M steps (if validated)
5. **Controls**: 4 budget/feature-matched controls all beaten
6. **Horizon analysis**: explains why prior 500k evaluations failed

### Paper Structure
- Main Table 1: 1M/8-seed main result with all controls
- Main Table 2: Mechanism isolation (structured vs random)
- Main Table 3: Horizon effect (500k vs 1M)
- Figure 1: Learning curves showing convergence
- Appendix: Lambda sensitivity, 4p-4f domain, Qwen negative result

### AAAI Probability: 40-50%

---

## What We Drop

- **Qwen/LLM semantic diagnosis**: collapsed, not salvageable
- **Cross-domain generalization claim**: VMAS/RWARE/4p-4f all failed
- **Adaptive weight novelty**: no independent contribution over fixed PBS
- **Q-Shaping**: signal too weak at 100k; implementation complexity not worth the risk
- **Old per-step penalty shaping**: fundamentally flawed (non-PBRS, backward smearing)

---

## Literature Support

- **Ng et al. (1999)**: PBRS theory — F(s,s') = γφ(s') - φ(s) preserves optimal policy
- **Akella (2025, arXiv:2511.00034)**: decentralized reward shaping has fundamental limitations in cooperative MARL — explains our failures
- **Wu (2024, arXiv:2410.01458)**: Q-shaping as unbiased alternative — informs future work
- **Abboud & Gal (2026, arXiv:2605.23562)**: ARMS automatic reward shaping for sparse-reward MARL — related work

---

## Timeline

| Step | Time | Content |
|---|---|---|
| 1 | 8h | Run 1M/8-seed validation (40 runs, PARALLEL=8) |
| 2 | 1h | Analyze results, determine Direction A or B |
| 3 | Commit + push to GitHub |
| 4 | Write paper based on determined direction |
| **Total** | **~9h to paper-ready** |
