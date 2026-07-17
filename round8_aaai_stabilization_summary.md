# Round 8 AAAI Stabilization Detailed Analysis

## Executive Summary

Round 8 completed the stabilization package recommended for a safer AAAI submission. The run finished 96/96 reinforcement-learning jobs with zero failures, covering two LBF environments, six methods, eight paired seeds, and 500k environment timesteps per run. It also completed a cached offline Qwen/Qwen3.5-4B diagnostic validation on 120 sampled failure records. The total wall time was about 10.04 hours from launch to artifact packaging, with the Qwen offline phase taking about 7.5 minutes and each RL run taking roughly 5.75--7.15 minutes.

The main conclusion is that the conservative framing is strongly supported on 10x10 and directionally supported on 12x12. On `Foraging-10x10-3p-3f-v3`, `adaptive_0.0003_late045` improves final test return over baseline by +0.0784 with a paired 95% bootstrap CI [0.0272, 0.1239], and improves train AUC by +0.0436 with CI [0.0184, 0.0659]. It also beats the newly introduced budget-matched phase-uniform control on final return by +0.0695 with CI [0.0064, 0.1247] and on AUC by +0.0387 with CI [0.0088, 0.0673]. This is the strongest AAAI-ready result because it demonstrates that the gain is not merely caused by adding any phase-matched penalty budget.

On `Foraging-12x12-3p-4f-v3`, adaptive shaping is the best mean final-return method among the Round 8 methods, but the paired intervals are not cleanly above zero. Adaptive improves over baseline by +0.0143 final return with CI [-0.0009, 0.0283] and +0.0058 AUC with CI [-0.0048, 0.0142]. It remains directionally above uniform, random-type, and shuffled controls, but all 12x12 strong-control CIs cross zero. Therefore 12x12 should be written as a scale-shift generalization/stress-test result showing consistent direction and no catastrophic transfer failure, not as a decisive second-environment win.

The Qwen result is negative and should not be used as the paper's main claim. On the cached 120-record validation sample, Qwen/Qwen3.5-4B predicted `insufficient_cooperation` for all 120 examples. Against the adjudicated labels from Phase 2, this gives 0.0917 accuracy and 0.0420 macro-F1. This confirms the Phase 2 diagnosis collapse and supports the decision to present semantic diagnosis as an interpretability/diagnostic appendix rather than as the causal mechanism behind the RL improvement.

## Experimental Package

The Round 8 experiment tests the main claim `Failure-Triggered Adaptive Reward Shaping for Cooperative Multi-Agent Reinforcement Learning`. The method uses failed episodes as triggers, applies calibrated reward shaping only after failure diagnosis, and uses a late-phase attenuation schedule to avoid destabilizing convergence. The semantic labels are produced by the existing enhanced heuristic for the RL shaping conditions, while Qwen/Qwen3.5-4B is evaluated only offline through a cached API path.

The reinforcement-learning design uses two environments: `lbforaging:Foraging-10x10-3p-3f-v3` and `lbforaging:Foraging-12x12-3p-4f-v3`. Each environment uses eight paired seeds, 500k timesteps per run, 100k test intervals, and 20 test episodes. The method set consists of `baseline`, `diagnosis_only`, `uniform_budget_matched_0.0003_late045`, `adaptive_0.0003_late045`, `random_type_budget_matched_0.0003_late045`, and `semantic_shuffled_budget_matched_0.0003_late045`. The design therefore directly separates failure-triggering, reward-budget effects, phase scheduling, random label effects, shuffled/matched-frequency label effects, and the adaptive intervention itself.

The code changes add three important pieces. First, `phase_uniform` in `reward_intervention.py` creates a label-independent control that receives the same phase schedule as adaptive shaping. Second, `failure_classifier.py` now supports cached OpenAI-compatible Qwen calls with retries and fail-soft error handling, eliminating the online timeout failure mode from Phase 2 for offline validation. Third, `build_round8_stabilization_report.py` creates grouped means, bootstrap confidence intervals, paired comparisons, and Cliff's delta for both environments.

## Current-Only Round 8 Results

### 10x10 Main Task

| method | n | final return | 95% CI | train AUC |
|---|---:|---:|---:|---:|
| baseline | 8 | 0.2258 | [0.1825, 0.2807] | 0.1827 |
| diagnosis_only | 8 | 0.2258 | [0.1825, 0.2807] | 0.1827 |
| uniform_budget_matched_0.0003_late045 | 8 | 0.2346 | [0.1862, 0.2839] | 0.1876 |
| adaptive_0.0003_late045 | 8 | 0.3042 | [0.2562, 0.3538] | 0.2263 |
| random_type_budget_matched_0.0003_late045 | 8 | 0.2192 | [0.1807, 0.2640] | 0.1759 |
| semantic_shuffled_budget_matched_0.0003_late045 | 8 | 0.2580 | [0.2171, 0.3099] | 0.2012 |

| paired comparison | final return delta | 95% CI | train AUC delta | 95% CI |
|---|---:|---:|---:|---:|
| adaptive - baseline | +0.0784 | [0.0272, 0.1239] | +0.0436 | [0.0184, 0.0659] |
| adaptive - diagnosis_only | +0.0784 | [0.0272, 0.1239] | +0.0436 | [0.0184, 0.0659] |
| adaptive - uniform_budget_matched | +0.0695 | [0.0064, 0.1247] | +0.0387 | [0.0088, 0.0673] |
| adaptive - random_type_budget_matched | +0.0850 | [0.0095, 0.1629] | +0.0504 | [-0.0004, 0.1019] |
| adaptive - semantic_shuffled_budget_matched | +0.0461 | [-0.0108, 0.0928] | +0.0251 | [-0.0153, 0.0586] |

The 10x10 results are the core AAAI evidence. Adaptive shaping gives a large and statistically clean gain over baseline, diagnosis-only, and the new phase-uniform budget-matched control. The comparison against random-type budget matching is also clean for final return and nearly clean for AUC. The shuffled/matched-frequency control remains the hardest control: adaptive is directionally better, and the stability-gap comparison is positive in the merged report, but the final-return and AUC intervals cross zero. This should be handled transparently by saying that semantic label specificity is not the main contribution; the adaptive failure-triggered schedule is.

### 12x12 Generalization / Stress Test

| method | n | final return | 95% CI | train AUC |
|---|---:|---:|---:|---:|
| baseline | 8 | 0.1021 | [0.0938, 0.1098] | 0.0917 |
| diagnosis_only | 8 | 0.1021 | [0.0938, 0.1098] | 0.0917 |
| uniform_budget_matched_0.0003_late045 | 8 | 0.1125 | [0.1025, 0.1225] | 0.0945 |
| adaptive_0.0003_late045 | 8 | 0.1164 | [0.1003, 0.1313] | 0.0975 |
| random_type_budget_matched_0.0003_late045 | 8 | 0.1056 | [0.0962, 0.1167] | 0.0952 |
| semantic_shuffled_budget_matched_0.0003_late045 | 8 | 0.1036 | [0.0910, 0.1153] | 0.0884 |

| paired comparison | final return delta | 95% CI | train AUC delta | 95% CI |
|---|---:|---:|---:|---:|
| adaptive - baseline | +0.0143 | [-0.0009, 0.0283] | +0.0058 | [-0.0048, 0.0142] |
| adaptive - diagnosis_only | +0.0143 | [-0.0009, 0.0283] | +0.0058 | [-0.0048, 0.0142] |
| adaptive - uniform_budget_matched | +0.0040 | [-0.0073, 0.0166] | +0.0030 | [-0.0035, 0.0097] |
| adaptive - random_type_budget_matched | +0.0108 | [-0.0073, 0.0296] | +0.0023 | [-0.0088, 0.0133] |
| adaptive - semantic_shuffled_budget_matched | +0.0128 | [-0.0059, 0.0309] | +0.0091 | [-0.0032, 0.0222] |

The 12x12 results are useful but not decisive. Adaptive has the highest mean final return among the Round 8 methods and is directionally above every paired control, but no paired final-return or AUC interval is strictly positive. The correct manuscript language is that 12x12 shows directional transfer under scale shift and no collapse, while the robust claim remains strongest on 10x10. This is substantially better than the earlier Round 6/7 situation because adaptive 0.0003 now has a complete eight-seed 12x12 panel and is no longer represented only by partial seeds.

## Qwen/Qwen3.5-4B Diagnostic Validation

The Qwen validation used cached API calls, `enable_thinking=false`, `max_tokens=256`, retries, and the fixed Phase 2 validation records. It completed 120 predictions without the online timeout problem. However, the label distribution collapsed completely: all 120 predictions were `insufficient_cooperation`. Against adjudicated labels, the sample contains 54 `low_value_overcommitment`, 48 `inefficient_exploration`, 7 `target_miscoordination`, and 11 `insufficient_cooperation`. Therefore Qwen obtains 0.0917 accuracy and 0.0420 macro-F1 on this sample.

This result is a strong negative finding for an LLM semantic diagnosis claim. The model is operationally usable in offline cached mode, but the diagnosis quality is not reliable enough to justify using Qwen labels as the causal intervention in the main RL story. The paper should say that current automatic semantic labels provide an exploratory diagnostic interface, while the core method does not rely on Qwen semantic correctness. This protects the submission from the obvious reviewer objection that the LLM classifier is not validated.

## AAAI Framing Recommendation

The strongest manuscript title direction remains `Failure-Triggered Adaptive Reward Shaping for Cooperative Multi-Agent Reinforcement Learning`. The abstract should claim that failed episodes can be converted into calibrated shaping signals and that an adaptive phase-aware schedule improves cooperative learning under strong controls. It should not claim that Qwen semantic reasoning improves MARL.

The main table should use the 10x10 Round 8 panel because it contains clean wins over baseline, diagnosis-only, phase-uniform budget matching, and random-type budget matching. The 12x12 panel should be a generalization/stress-test table showing the highest mean final return for adaptive but with conservative language because confidence intervals cross zero. The Qwen diagnostic validation should be placed in a diagnostics or appendix subsection as a negative result that motivates future work on reliable semantic failure classification.

The final submission claim should be: failure-triggered adaptive shaping is a robust and controlled intervention for cooperative foraging, with strong evidence on the main hard 10x10 task and directional transfer on a larger 12x12 task. The result is more credible than a semantic-causality claim because it survives budget-matched and random/shuffled controls and openly reports the limitations of automatic semantic diagnosis.

## Files

- Launcher: `run_round8_aaai_stabilization.sh`
- Report builder: `epymarl/tools/build_round8_stabilization_report.py`
- Full output directory: `results/round8_aaai_stabilization_round8_full_20260705_041113/`
- Raw summary: `results/round8_aaai_stabilization_round8_full_20260705_041113/summary.csv`
- Auto report: `results/round8_aaai_stabilization_round8_full_20260705_041113/ROUND8_STABILIZATION_REPORT.md`
- Detailed analysis: `results/round8_aaai_stabilization_round8_full_20260705_041113/ROUND8_DETAILED_ANALYSIS.md`
- Packaged artifact: `artifacts/AAAI_round8_stabilization_round8_full_20260705_041113.tar.gz`
