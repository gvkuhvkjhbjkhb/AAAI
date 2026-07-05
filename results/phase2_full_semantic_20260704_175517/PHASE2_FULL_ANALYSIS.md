# Phase 2 Full Semantic Diagnosis Analysis

## Scope

This run completed the requested Phase 2 program for semantic failure diagnosis and semantic-causality testing on the existing AAAI repository. The experiment used the saved 300-record diagnosis-validation sample, the fixed six-class taxonomy, Qwen/Qwen3.5-4B through an OpenAI-compatible endpoint, and the 10x10 LBF task `lbforaging:Foraging-10x10-3p-3f-v3`. The run started at Sat Jul 4 17:55:17 UTC 2026 and finished at Sat Jul 4 23:02:17 UTC 2026, for about 5.12 hours of wall time on one GPU.

## Code Changes

The run adds `run_phase2_full_semantic.sh`, an end-to-end script for Phase 2A/2B/2C/2D. It creates a second blind rule-based annotator, creates adjudicated labels, runs original heuristic, enhanced heuristic, and Qwen/Qwen3.5-4B diagnosis evaluation, then runs paired 10x10 RL controls across seeds 1--8. It also adds `epymarl/tools/phase2_diagnosis_validation.py`, which computes raw human agreement, Cohen kappa, accuracy, macro-F1, per-class F1, prediction distributions, and confusion matrices. Finally, `epymarl/src/llm_diagnosis/failure_classifier.py` now supports the optional OpenAI-compatible `enable_thinking` payload flag and treats timeout/OSError API failures as classifier errors rather than uncaught training crashes in future runs.

## Phase 2A/2B Diagnosis Validation

Human-human agreement is moderate but not strong: raw agreement is 0.6367 and Cohen kappa is 0.4322 over 300 samples. This indicates the taxonomy is partly learnable but still ambiguous, especially between `inefficient_exploration`, `low_value_overcommitment`, and broad coordination failures. The corrected validation report should be used, because the first generated report mapped JSONL line numbers to sample IDs incorrectly.

Against adjudicated labels, the original heuristic reaches accuracy 0.4867, macro-F1 0.2497, and kappa 0.2062. The enhanced heuristic reaches accuracy 0.3433, macro-F1 0.2038, and kappa 0.1060. Qwen/Qwen3.5-4B completed 138 of 300 records before API timeout; on those completed records it reaches accuracy 0.0580, macro-F1 0.0224, and kappa 0.0051. Counting the missing timed-out records as unknown over the full 300 gives accuracy 0.0267, macro-F1 0.0158, and kappa -0.0363.

The substantive diagnostic finding is negative for the LLM claim. Qwen collapses almost all completed predictions into `insufficient_cooperation`, producing 137 `insufficient_cooperation` predictions and only 1 `low_value_overcommitment` prediction among 138 completed examples. The original heuristic is weak but still materially better than both enhanced heuristic and Qwen under this label protocol. Therefore Phase 2B does not validate Qwen/Qwen3.5-4B as a reliable semantic failure classifier for this task.

## Phase 2C/2D RL Causality Results

The non-LLM RL part completed 40 successful runs: five methods times eight paired seeds. The online LLM semantic condition failed in all eight seeds due to API read timeouts during training, so it cannot be used as a valid RL result. Its partial log metrics are not submission-grade because the runs terminated early.

The strongest Phase 2 RL method is `semantic_adaptive_0.0003_late045`, with mean final return 0.3042 and train AUC 0.2263 over eight seeds. It improves over `matched_random_0.0003_late045`, which reaches final return 0.2192 and train AUC 0.1759. The paired final-return difference is +0.0850 with 95% bootstrap CI [0.0108, 0.1623], and the paired train-AUC difference is +0.0504 with CI [0.0013, 0.1016]. This is the cleanest positive Phase 2 RL result.

However, the semantic result is less decisive against the stricter shuffled/matched-frequency control. `semantic_shuffled_0.0003_late045` reaches final return 0.2580 and train AUC 0.2012. The paired difference for semantic adaptive over shuffled is +0.0461 final return with CI [-0.0098, 0.0918] and +0.0251 train AUC with CI [-0.0142, 0.0575], both crossing zero. Against `adaptive_penalty_0.0003_late045`, semantic adaptive is also only directionally positive: +0.0529 final return with CI [-0.0153, 0.1204] and +0.0291 train AUC with CI [-0.0048, 0.0667].

The mechanism-specific condition does not improve the story. `mechanism_specific_0.0003_late045` reaches final return 0.2382 and train AUC 0.1900. It is not clearly above matched random: +0.0190 final return with CI [-0.0512, 0.0808] and +0.0141 train AUC with CI [-0.0339, 0.0537]. It is also below semantic adaptive, suggesting that the current type-specific gate and weights are not a better mechanism than the simpler adaptive schedule.

## Integrated Interpretation

Phase 2 produces a mixed result. The RL sweep gives a positive signal for semantic adaptive shaping relative to one random-type control, but the diagnostic validation fails to establish that the semantic labels are reliable, and the stronger shuffled/matched-frequency comparison is only directional rather than statistically clean. Because the LLM classifier is weak and online LLM RL fails under API timeout, the paper cannot honestly claim that Qwen/Qwen3.5-4B semantic diagnosis causally improves cooperative MARL.

The results are more consistent with the Phase 1 framing: failed episodes provide useful online training signals, and calibrated failure-triggered shaping can improve MAPPO on 10x10 LBF. The strongest publishable method remains `adaptive_0.0003_late045` / `semantic_adaptive_0.0003_late045`, but the mechanism should be described as calibrated adaptive failure-triggered shaping rather than validated LLM reasoning. Semantic labels can be presented as an exploratory analysis and as a partially successful but not yet reliable diagnostic interface.

## AAAI Decision

This evidence does not support a strong AAAI main-track submission centered on `LLM semantic diagnosis improves MARL`. The diagnosis module is not validated, Qwen/Qwen3.5-4B underperforms simple heuristics on macro-F1, online LLM training is not robust, and the semantic causality tests do not consistently beat the strongest shuffled/matched controls with narrow confidence intervals.

A conservative AAAI submission may still be possible if the claim is narrowed to `Failure-Triggered Adaptive Reward Shaping for Cooperative Multi-Agent Reinforcement Learning`. The strongest support is the repeated 10x10 improvement of calibrated adaptive shaping over baseline and the penalty-calibration story from Phase 1. The submission would remain borderline because the evidence is concentrated in one main environment family, 12x12 transfer is weak, uniform/random controls are competitive, and the semantic component must be framed as negative or exploratory rather than as the headline contribution.

## Main Problems

1. Semantic diagnosis reliability is insufficient. Human kappa is only 0.4322, and automatic macro-F1 is low even for the best heuristic.
2. Qwen/Qwen3.5-4B is not usable as validated diagnosis evidence in the current prompt/protocol, because it collapses into `insufficient_cooperation` and times out.
3. Semantic causal evidence is incomplete: semantic adaptive beats ordinary matched random, but does not cleanly beat shuffled/matched-frequency controls or adaptive penalty-only controls under all metrics.
4. Mechanism-specific intervention is weak and does not add a stronger contribution than the simpler adaptive schedule.
5. Generalization remains limited; previous 12x12 results are stress-test-level rather than broad transfer evidence.
6. The paper currently has enough evidence for a careful workshop or borderline main-track story, but not for a robust AAAI semantic-diagnosis claim.

## Recommended Manuscript Position

Use `Failure-Triggered Adaptive Reward Shaping for Cooperative Multi-Agent Reinforcement Learning` as the title direction. The abstract and introduction should emphasize failure-triggered adaptive shaping, calibration, and rigorous negative controls. The semantic/LLM diagnosis module should be described as an interpretability and future-work component whose current validation is mixed or negative. Any AAAI submission should explicitly report the random, shuffled, matched, and uniform controls rather than hiding them, because reviewers are likely to ask for exactly those comparisons.
