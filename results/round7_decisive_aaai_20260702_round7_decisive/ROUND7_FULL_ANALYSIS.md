# Round 7 Full Analysis and AAAI Decision

## Completion

Round 7 decisive completed all 44 planned runs on one RTX 5090. There were 44 `.done` markers, 44 run logs, and 0 `.fail` markers. The run started at Thu Jul 2 20:26:48 UTC 2026 and finished at Fri Jul 3 01:00:39 UTC 2026. The total wall time was about 4.56 hours, matching the expected 4.5--5.5 hour range.

## Why This Round Was Needed

Round 6 established that adaptive reward intervention can substantially improve the hard 10x10 cooperative foraging task, but it also exposed a central reviewer risk: `random_type_0.0002` matched or slightly exceeded the best semantic adaptive method. Round 7 therefore added a stricter matched random-type control and a confidence-gated semantic method. The matched random control preserves the observed empirical failure-type frequency but shuffles the type assignment across failure records, which directly tests whether semantic assignment itself is the causal source of improvement.

## 10x10 Main Results

The best Round 6 semantic/adaptive configuration remains `adaptive_0.0003_late045`, with mean final test return 0.3042 and train AUC 0.2263 over eight seeds. It is clearly above baseline, which reached 0.2258 final test return and 0.1827 train AUC. It is also slightly above calibrated `uniform_0.0003` on final return, although uniform remains a very strong shaping baseline with final return 0.2892 and the highest best-train metric.

Round 7's new confidence-gated method did not solve the semantic causality problem. `semantic_gate_0.0003_late045` reached mean final test return 0.2382 and train AUC 0.1900. It was only slightly above baseline in paired comparisons and below `adaptive_0.0003_late045`, `uniform_0.0003`, and the new matched random-type control. The matched random-type control, `random_type_matched_0.0003_late045`, reached 0.2580 final test return and 0.2012 train AUC, exceeding semantic_gate by 0.0199 final return and 0.0111 train AUC on average.

The ordinary random-type controls at penalty 0.0003 were weaker than semantic_gate: `random_type_0.0003_late045` reached 0.2192 final return and `random_type_0.0003_late060` reached 0.2154. This shows that not every random control is competitive. However, because the stricter matched-frequency random control remains stronger than semantic_gate, the paper still cannot claim that semantic failure labels are decisively causal. The likely interpretation is that empirical label-frequency structure and calibrated failure-triggered shaping matter, while the current semantic gate and heuristic label assignment do not yet provide a robust additional gain.

## Paired Decision Tests

The decisive paired comparison is `semantic_gate_0.0003_late045 - random_type_matched_0.0003_late045`. The final-return mean delta is -0.0199 with bootstrap 95% CI [-0.0716, 0.0348], and the train-AUC delta is -0.0111 with CI [-0.0455, 0.0198]. These intervals cross zero, so the result is not a statistically decisive loss, but it is directionally unfavorable for the semantic-gate claim.

Against the best Round 6 adaptive method, semantic_gate is also weaker. The paired final-return delta against `adaptive_0.0003_late045` is -0.0660 with CI [-0.1237, 0.0034], and train-AUC delta is -0.0362 with CI [-0.0717, 0.0018]. Against `uniform_0.0003`, semantic_gate has final-return delta -0.0510 and train-AUC delta -0.0191. Against baseline, semantic_gate is only mildly positive: final-return delta 0.0124 and train-AUC delta 0.0073, both with confidence intervals crossing zero.

## 12x12 Stress Test

The 12x12 results remain a stress test, not a generalization win. Across all available baseline and uniform seeds, `uniform_0.0002` is the strongest conservative method with mean final test return 0.1143 over eight seeds. Baseline reaches 0.1021 over eight seeds. The newly tested `adaptive_0.0003_late045` reaches 0.1179 over seeds 6--8, while `semantic_gate_0.0003_late045` reaches only 0.0995 over seeds 6--8. On shared seeds 6--8, semantic_gate is below adaptive by -0.0184 final return and -0.0169 train AUC. Therefore, 12x12 should be reported as a scale-shift stress test with mixed or weak transfer, not as evidence of strong generalization.

## AAAI-Safe Interpretation

The safest AAAI framing is not a strong semantic-causality story. The evidence supports a more conservative but defensible claim: calibrated failure-triggered reward intervention can improve MAPPO on a difficult cooperative foraging task, and a late-preserving adaptive schedule is the strongest current method. Semantic diagnosis remains useful for logging, interpretability, and future gating, but the current heuristic semantic labels do not robustly outperform matched random-type controls.

The paper should state that semantic labels require further validation rather than claiming they are decisively causal. If the paper title or abstract currently emphasizes LLM semantic diagnosis as the central mechanism, it should be softened toward failure-triggered adaptive reward shaping with diagnosis-assisted analysis. A strong claim about LLM diagnosis quality should wait until a separate human-label/Qwen3.5-4B macro-F1 validation is completed.

## Recommended Manuscript Claim

A defensible claim is: LLM-FDCR shows that failure-triggered, calibrated reward intervention can improve MAPPO in hard cooperative foraging, with `adaptive_0.0003_late045` improving final return by about 34.7% over baseline and improving train AUC by about 23.9%. Calibrated uniform shaping and matched random-type controls are strong baselines, so current results support adaptive failure-triggered shaping more strongly than semantic label causality. The 12x12 environment should be presented as a stress test that reveals limits of transfer.

## Claims to Avoid

Do not claim that semantic diagnosis is decisively better than random labels. Do not claim that confidence-gated semantic adaptive shaping is the best method. Do not claim strong 12x12 generalization. Do not claim LLM diagnosis quality is validated without a separate human-label and Qwen3.5-4B macro-F1 experiment.

## Next Scientific Step

If time remains before submission, the only experiment that can revive the semantic claim is diagnosis-quality validation. Sample 300 failure records, label at least 200 manually, run original heuristic, enhanced heuristic, and Qwen3.5-4B, then report macro-F1, per-class F1, confusion matrix, and Cohen's kappa. If Qwen3.5-4B materially improves macro-F1 and a new semantic gate using those labels beats matched random-type control, the semantic claim can be restored. Otherwise, the submission should use the calibrated adaptive-shaping framing.
