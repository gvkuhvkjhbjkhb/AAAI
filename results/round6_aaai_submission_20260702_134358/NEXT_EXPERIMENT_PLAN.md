# Round 7 Targeted AAAI Optimization Plan

## Objective

Round 7 should be narrow and decisive. Round 6 already shows a strong 10x10 improvement for `adaptive_0.0003_late045`, but it also shows that `random_type_0.0002` is as strong as the best adaptive semantic method and that 12x12 transfer remains weak. The next experiments must separate semantic diagnosis from random regularization, validate diagnosis quality, and improve adaptive transfer without expanding into an unfocused sweep.

## Experiment 7A: Matched Random-Type Controls

Run `lbforaging:Foraging-10x10-3p-3f-v3` for 500000 timesteps with seeds 1-8. Compare `adaptive_0.0003_late045`, `random_type_0.0003_late045`, `random_type_0.0003_late060`, and `random_type_0.0002_late060`. Round 6 compared the strongest adaptive method against random type at a different base penalty, so the semantic claim remains under-controlled. The claim that semantic diagnosis matters is viable only if adaptive remains ahead of matched random controls in paired final test return or train AUC.

## Experiment 7B: Confidence-Gated Semantic Adaptive Shaping

Add a confidence gate so type-specific scaling is used only when diagnosis confidence is at least 0.55; otherwise the intervention falls back to uniform shaping. Test `semantic_gate_0.0003_late045` with `inefficient_exploration=0.65`, `target_miscoordination=1.45`, `insufficient_cooperation=1.35`, `low_value_overcommitment=1.20`, `timeout_near_success=0.00`, and `unknown=0.75`. This targets the Round 6 failure mode in which the dominant inefficient-exploration class may collapse semantic shaping into generic regularization. Run seeds 1-8 on 10x10 and compare against baseline, uniform 0.0003, adaptive 0.0003 late 0.45, and matched random type.

## Experiment 7C: Diagnosis Quality Validation

Sample 300 failure records from Round 6. Human-label at least 200 records using the taxonomy in `epymarl/src/llm_diagnosis/prompts.py`. Compare original heuristic, enhanced heuristic, and Qwen3.5-4B using macro-F1, per-class F1, confusion matrix, Cohen's kappa, and pairwise agreement. The paper should make a strong LLM diagnosis claim only if Qwen3.5-4B or the enhanced diagnostic pipeline materially improves macro-F1 over the original heuristic and obtains acceptable agreement with human labels.

## Experiment 7D: Conservative 12x12 Follow-Up

Run `lbforaging:Foraging-12x12-3p-4f-v3` for 500000 timesteps with seeds 1-8. Compare baseline, uniform 0.0002, adaptive 0.0003 late 0.45, and the best confidence-gated semantic method. If adaptive does not beat baseline or uniform, report 12x12 as a stress test and keep the main paper claim on 10x10 sample efficiency and calibrated intervention. If the confidence-gated method improves 12x12 final return or AUC, it becomes the preferred submission method.

## Submission Decision Rule

The AAAI main claim is safe if the confidence-gated semantic method beats baseline and matched random-type controls on 10x10 across eight seeds, and if diagnosis validation shows higher macro-F1 than the original heuristic. If random controls remain equal or stronger, pivot the paper to calibrated failure-triggered adaptive shaping and present semantic diagnosis as an analysis module rather than the central causal mechanism. If 12x12 remains weak, present it transparently as a hard-transfer stress test rather than overclaiming generalization.
