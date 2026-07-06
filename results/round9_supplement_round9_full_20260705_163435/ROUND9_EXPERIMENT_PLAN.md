# Round 9 Supplemental Stabilization

Goal: address the remaining AAAI risk by adding a cross-domain task package and extending 12x12 LBF seeds. The main claim remains failure-triggered adaptive reward shaping; Qwen/semantic causality is not used as the main causal claim.

Experiments:
- LBF 12x12 seed extension: seeds 9 10 11 12, t_max=500000, methods=baseline uniform_budget_matched_0.0003_late045 adaptive_0.0003_late045 random_type_budget_matched_0.0003_late045 semantic_shuffled_budget_matched_0.0003_late045 diagnosis_only.
- RWARE tiny cross-domain: seeds 1 2 3 4 5, t_max=300000, methods=baseline uniform_budget_matched_0.0003_late045 adaptive_0.0003_late045 random_type_budget_matched_0.0003_late045 semantic_shuffled_budget_matched_0.0003_late045 diagnosis_only.
- VMAS navigation cross-domain: seeds 1 2 3 4 5, t_max=300000, methods=baseline uniform_budget_matched_0.0003_late045 adaptive_0.0003_late045 random_type_budget_matched_0.0003_late045 diagnosis_only.

Decision use:
- A strong AAAI boost requires adaptive to remain positive against baseline and not lose to phase-uniform/random controls in RWARE or VMAS.
- If cross-domain results are mixed, report them transparently and use them as limitation evidence rather than overclaiming.
