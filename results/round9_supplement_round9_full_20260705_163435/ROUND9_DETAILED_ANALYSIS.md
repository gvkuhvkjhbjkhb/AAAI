# Round 9 Supplemental Stabilization Detailed Analysis

## Executive Summary

Round 9 completed the supplemental package designed to address the main residual AAAI risk after Round 8: the evidence was too concentrated in LBF and the 12x12 scale-shift result was only directional. The run completed 79/79 jobs with zero failures. It added four additional seeds for LBF 12x12, a five-seed RWARE tiny cross-domain panel, and a five-seed VMAS navigation cross-domain panel. It also validated the generalized non-LBF trajectory summarizer and the Round 9 reporting path.

The result is mixed. Round 9 improves the paper's credibility by showing that the code runs stably on RWARE and VMAS, but it does not produce a clean new cross-domain win for adaptive shaping. The added LBF 12x12 seeds dilute the earlier positive trend: merged over 12 seeds, adaptive remains the highest or near-highest among core methods, but paired confidence intervals still cross zero. RWARE tiny is uninformative because all methods have zero final test return. VMAS navigation is negative for the adaptive-specific claim because random-type budget matching clearly beats adaptive, and adaptive has lower train AUC than baseline.

Therefore Round 9 should be used carefully. It is useful as an appendix/stress-test package and as evidence of honest evaluation beyond LBF. It should not be used to claim broad cross-domain superiority. The strongest AAAI story remains Round 8's 10x10 result: failure-triggered adaptive shaping beats baseline, diagnosis-only, phase-uniform budget matching, and random-type budget matching under paired seeds. Round 9 changes the submission strategy from “we have likely solved generalization” to “we have a strong controlled main-task result, directional but inconclusive scale-shift evidence, and transparent cross-domain limitations.”

## Experiment Design

Round 9 contains three components. The first is a 12x12 LBF seed extension using seeds 9--12 and the same six-method panel as Round 8: baseline, diagnosis-only, phase-uniform budget matching, adaptive shaping, random-type budget matching, and shuffled/matched-frequency semantic control. Each run uses 500k timesteps.

The second component is RWARE tiny cross-domain transfer using `rware:rware-tiny-2ag-v2`, five seeds, six methods, and 300k timesteps per run. This task was chosen because it is a different cooperative domain and can run through the same Gymma/MAPPO interface. The third component is VMAS navigation using five seeds, five methods, and 300k timesteps per run. VMAS omits the semantic-shuffled condition to reduce unnecessary semantic-label noise and compute while still testing baseline, diagnosis-only, phase-uniform budget matching, adaptive shaping, and random-type budget matching.

The code was modified to make the trajectory summarizer environment-aware. LBF summaries retain `load`-action counts, while non-LBF environments use generic action histograms instead of incorrectly assigning LBF semantics to RWARE/VMAS actions. This is important because the intervention should be evaluated as failure-triggered shaping, not as an accidental LBF-specific semantic parser applied out of domain.

## Round 9 Results

### LBF 12x12 Seed Extension Only

| method | n | final return | 95% CI | train AUC |
|---|---:|---:|---:|---:|
| baseline | 4 | 0.1059 | [0.0881, 0.1202] | 0.0823 |
| diagnosis_only | 4 | 0.1059 | [0.0881, 0.1202] | 0.0823 |
| uniform_budget_matched_0.0003_late045 | 4 | 0.0848 | [0.0555, 0.1279] | 0.0743 |
| adaptive_0.0003_late045 | 4 | 0.1069 | [0.0897, 0.1283] | 0.0856 |
| random_type_budget_matched_0.0003_late045 | 4 | 0.1233 | [0.1113, 0.1407] | 0.0976 |
| semantic_shuffled_budget_matched_0.0003_late045 | 4 | 0.0938 | [0.0811, 0.1031] | 0.0836 |

The new seeds do not strengthen the 12x12 claim. Adaptive is only slightly above baseline on mean final return, with paired delta +0.0009 and 95% CI [-0.0306, 0.0394]. Random-type budget matching is the highest-return method in this extension, which weakens any claim that adaptive type/phase specificity is uniquely beneficial at this scale. Adaptive still beats phase-uniform on mean final return, but the interval is wide and crosses zero.

### Merged LBF 12x12, Round 8 + Round 9

| method | n | final return | 95% CI | train AUC |
|---|---:|---:|---:|---:|
| baseline | 12 | 0.1034 | [0.0954, 0.1110] | 0.0886 |
| diagnosis_only | 12 | 0.1034 | [0.0954, 0.1110] | 0.0886 |
| uniform_budget_matched_0.0003_late045 | 12 | 0.1033 | [0.0871, 0.1183] | 0.0878 |
| adaptive_0.0003_late045 | 12 | 0.1132 | [0.1004, 0.1257] | 0.0936 |
| random_type_budget_matched_0.0003_late045 | 12 | 0.1115 | [0.1023, 0.1215] | 0.0960 |
| semantic_shuffled_budget_matched_0.0003_late045 | 12 | 0.1004 | [0.0912, 0.1093] | 0.0868 |

| paired comparison | final return delta | 95% CI | train AUC delta | 95% CI |
|---|---:|---:|---:|---:|
| adaptive - baseline | +0.0098 | [-0.0059, 0.0261] | +0.0050 | [-0.0030, 0.0126] |
| adaptive - diagnosis_only | +0.0098 | [-0.0059, 0.0261] | +0.0050 | [-0.0030, 0.0126] |
| adaptive - uniform_budget_matched | +0.0100 | [-0.0087, 0.0305] | +0.0058 | [-0.0049, 0.0164] |
| adaptive - random_type_budget_matched | +0.0018 | [-0.0141, 0.0184] | -0.0025 | [-0.0121, 0.0075] |
| adaptive - semantic_shuffled_budget_matched | +0.0129 | [-0.0002, 0.0262] | +0.0067 | [-0.0021, 0.0166] |

The merged 12x12 evidence remains directional rather than decisive. Adaptive has the highest final return among the core methods except it is statistically tied with random-type budget matching, and all key paired final-return intervals cross zero. This is better than a negative scale-shift result, but it is not strong enough to claim robust generalization. The appropriate manuscript language is “directional transfer to a larger LBF scale setting,” not “significant generalization.”

### RWARE Tiny Cross-Domain

RWARE tiny is not informative under this setup because every method has zero final test return. Train AUC values are extremely small, around 1e-4 to 3e-4, and final-return paired deltas are exactly zero. Adaptive has tiny positive AUC/best-train differences over some controls, but these are not meaningful as evidence of cooperative task success. The most honest interpretation is that 300k MAPPO training on this RWARE setup did not learn enough to evaluate reward-shaping differences.

This result should not be presented as a cross-domain win. It may appear in an appendix as a negative or inconclusive transfer attempt. If RWARE is kept, the paper should explicitly state that the environment remained near-zero-return under all compared methods and therefore cannot support claims about shaping superiority.

### VMAS Navigation Cross-Domain

| method | n | final return | 95% CI | train AUC |
|---|---:|---:|---:|---:|
| baseline | 5 | 6.8142 | [5.9482, 7.3773] | 4.9329 |
| diagnosis_only | 5 | 6.8142 | [5.9482, 7.3773] | 4.9329 |
| uniform_budget_matched_0.0003_late045 | 5 | 7.0581 | [6.3185, 7.8587] | 4.8923 |
| adaptive_0.0003_late045 | 5 | 6.8498 | [6.3596, 7.2903] | 4.7582 |
| random_type_budget_matched_0.0003_late045 | 5 | 7.7545 | [7.4996, 8.0330] | 5.1954 |

VMAS navigation is unfavorable for the adaptive-specific claim. Adaptive is approximately tied with baseline on final return, but it has a lower train AUC than baseline: paired AUC delta -0.1747 with 95% CI [-0.3251, -0.0401]. More importantly, random-type budget matching beats adaptive clearly: final-return delta adaptive minus random-type is -0.9047 with 95% CI [-1.2720, -0.4207], and train-AUC delta is -0.4373 with 95% CI [-0.5822, -0.3251]. This means the VMAS result cannot be used to argue that adaptive failure-type weighting generalizes out of domain.

The best way to use VMAS is as a limitation and diagnostic result. It suggests that the current coefficient and type-weight design is tuned for sparse cooperative foraging-style rewards and does not automatically transfer to dense/continuous-navigation-style reward landscapes. This does not invalidate the 10x10 LBF result, but it prevents a broad cross-domain robustness claim.

## Impact on AAAI Strategy

Round 9 reduces engineering risk but increases the need for conservative writing. It shows the codebase can evaluate RWARE and VMAS and that the method does not crash outside LBF. However, it does not provide the hoped-for clean cross-domain positive result. The submission should therefore not claim “robust across tasks” unless that phrase is restricted to the LBF family and qualified by the appendix limitations.

The strongest defensible claim remains: failure-triggered adaptive reward shaping substantially improves the main cooperative foraging task under strong budget-matched controls. The 10x10 evidence from Round 8 is still strong and submission-worthy. The 12x12 evidence is best described as directional scale-shift transfer with higher mean final return but non-significant paired intervals. RWARE and VMAS should be described as exploratory cross-domain stress tests that reveal limitations rather than as headline results.

The paper can still be submitted to AAAI, but Round 9 means the paper should be framed as a carefully controlled method for sparse cooperative foraging rather than a broadly general MARL shaping method. This makes the paper more honest but less broadly compelling. To raise the acceptance probability further, the next experiment should not simply add more seeds to VMAS; it should retune the shaping coefficient for non-LBF reward scales or choose a cross-domain task with sparse cooperative rewards closer to the method's assumptions.

## Recommended Manuscript Use

Use Round 8 10x10 as the main result table. Include merged 12x12 as a generalization/stress-test table with conservative language. Put RWARE and VMAS in appendix or a limitations section, not the main claim. The negative Qwen and mixed cross-domain results can actually improve reviewer trust if they are framed as transparent boundary conditions rather than hidden failures.

A safe conclusion is: “Adaptive failure-triggered shaping gives strong gains in the main LBF cooperative foraging benchmark under budget-matched and random-label controls, while larger-scale and cross-domain tests show directional transfer but also identify reward-scale sensitivity as a limitation.” This is much safer than claiming broad task robustness.

## Files

- Launcher: `run_round9_supplement.sh`
- Report builder: `epymarl/tools/build_round9_supplement_report.py`
- Generalized summarizer: `epymarl/src/llm_diagnosis/trajectory_summarizer.py`
- Output directory: `results/round9_supplement_round9_full_20260705_163435/`
- Auto report: `results/round9_supplement_round9_full_20260705_163435/ROUND9_SUPPLEMENT_REPORT.md`
- Detailed analysis: `results/round9_supplement_round9_full_20260705_163435/ROUND9_DETAILED_ANALYSIS.md`
- Packaged artifact: `artifacts/AAAI_round9_supplement_round9_full_20260705_163435.tar.gz`
