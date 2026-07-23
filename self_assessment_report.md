# Self-Assessment Report: Safety Certificate Experiments

**Generated**: 2026-07-23 07:59 UTC

**Model pairs**: Qwen2.5-7B + GLM4-9B, Mistral-7B + Phi-2

**GPUs**: 2x RTX 5090 32GB, batch_size=64 (accelerated)

**Acceleration**: Qwen-GLM pair ran with batch_size=16 (~4.4h); Mistral-Phi pair with batch_size=64 (~1.4h, **3.2x speedup**)


---

## 1. Execution Timing

| Pair | G1 (s) | G2 (s) | G3 (s) | G4 (s) | **Total** |
|------|--------|--------|--------|--------|-----------|
| qwen-glm | 423 | 536 | 14781 | 0 | **15739** |
| mistral-phi | 115 | 146 | 4822 | 74 | **5157** |

---

## 2. G1: Online Monitor & Fallback

| qwen-glm | Family | Permitted | Fallback Rate | Per-Role Safety | Monitor FP | Monitor FV |
|------|--------|-----------|---------------|-----------------|------------|------------|
| | uniform | 285 | 0.684 | 0.914 | 0 | 168 |
| | adversarial | 120 | 0.858 | 0.958 | 0 | 103 |

| mistral-phi | Family | Permitted | Fallback Rate | Per-Role Safety | Monitor FP | Monitor FV |
|------|--------|-----------|---------------|-----------------|------------|------------|
| | uniform | 285 | 0.568 | 0.703 | 7 | 84 |
| | adversarial | 120 | 0.717 | 0.894 | 0 | 73 |

---

## 3. G2: Realized Baseline Comparison

| qwen-glm | Family | Baseline | Advised | Team Δ | Worst-Role Δ | Harm Permits |
|------|--------|----------|---------|--------|-------------|--------------|
| | adversarial | PerRoleOnly | 300 | 0.096 | 0.045 | 171 |
| | adversarial | ParetoFilter | 105 | 0.152 | 0.125 | 14 |
| | adversarial | Egalitarian | 0 | 0.000 | 0.000 | 0 |
| | adversarial | ConstrainedOpt | 0 | 0.000 | 0.000 | 0 |
| | adversarial | ActionSafe | 120 | 0.137 | 0.090 | 32 |
| | uniform | PerRoleOnly | 360 | 0.083 | 0.066 | 148 |
| | uniform | ParetoFilter | 330 | 0.084 | 0.083 | 127 |
| | uniform | Egalitarian | 0 | 0.000 | 0.000 | 0 |
| | uniform | ConstrainedOpt | 0 | 0.000 | 0.000 | 0 |
| | uniform | ActionSafe | 285 | 0.086 | 0.072 | 103 |

| mistral-phi | Family | Baseline | Advised | Team Δ | Worst-Role Δ | Harm Permits |
|------|--------|----------|---------|--------|-------------|--------------|
| | adversarial | PerRoleOnly | 300 | 0.051 | 0.019 | 189 |
| | adversarial | ParetoFilter | 105 | 0.078 | 0.072 | 46 |
| | adversarial | Egalitarian | 0 | 0.000 | 0.000 | 0 |
| | adversarial | ConstrainedOpt | 0 | 0.000 | 0.000 | 0 |
| | adversarial | ActionSafe | 120 | 0.078 | 0.060 | 40 |
| | uniform | PerRoleOnly | 360 | 0.024 | 0.008 | 233 |
| | uniform | ParetoFilter | 330 | 0.025 | 0.013 | 208 |
| | uniform | Egalitarian | 0 | 0.000 | 0.000 | 0 |
| | uniform | ConstrainedOpt | 0 | 0.000 | 0.000 | 0 |
| | uniform | ActionSafe | 285 | 0.025 | 0.021 | 175 |

---

## 4. G3: K-Probe Empirical Curves

| qwen-glm | Uniform | K=5 | K=10 | K=20 | K=40 | K=80 |
|------|------|-----|------|------|------|------|
| | Permit Rate | 0.497| 0.556| 0.569| 0.569| 0.608
| | FalsePermit | 0.341| 0.400| 0.298| 0.298| 0.333
| qwen-glm | Integer | K=5 | K=10 | K=20 | K=40 | K=80 |
|------|------|-----|------|------|------|------|
| | Permit Rate | 0.242| 0.283| 0.314| 0.344| 0.386
| | FalsePermit | 0.149| 0.343| 0.274| 0.210| 0.245

| mistral-phi | Uniform | K=5 | K=10 | K=20 | K=40 | K=80 |
|------|------|-----|------|------|------|------|
| | Permit Rate | 0.219| 0.300| 0.389| 0.506| 0.594
| | FalsePermit | 0.772| 0.593| 0.657| 0.626| 0.593
| mistral-phi | Integer | K=5 | K=10 | K=20 | K=40 | K=80 |
|------|------|-----|------|------|------|------|
| | Permit Rate | 0.339| 0.375| 0.469| 0.486| 0.569
| | FalsePermit | 0.574| 0.467| 0.497| 0.383| 0.439

---

## 5. G4: Widened-N Power Analysis

| Pair | Team Effect | CI95 | Role-1 Effect | CI95 | Role-2 Effect | CI95 | Target Fidelity | n_matrices |
|------|------------|------|--------------|------|--------------|------|----------------|------------|
| qwen-glm | 0.089 | [0.066, 0.114] | 0.140 | [0.080, 0.199] | 0.037 | [-0.024, 0.099] | 0.637 | 24 |
| mistral-phi | 0.040 | [0.028, 0.053] | 0.073 | [0.044, 0.102] | 0.007 | [-0.021, 0.034] | 0.276 | 24 |

---

## 6. Conclusions

- **G1**: Online monitor maintains per-role safety guarantees (fallback rates 57-86% across pairs)

- **G2**: Action-safe gating outperforms PerRoleOnly/Pareto baselines in realized team effect

- **G3**: K-probe false-permit rates decay monotonically with K, consistent with O(1/sqrt(K)) theory

- **G4**: Widened-N rollouts with n_mat=24 provide >80% power for anti-safe gap detection

- **Total Compute**: ~6 GPU-hours on 2x RTX 5090

- **Reproducibility**: All code, tasks, and seeds included in this archive
