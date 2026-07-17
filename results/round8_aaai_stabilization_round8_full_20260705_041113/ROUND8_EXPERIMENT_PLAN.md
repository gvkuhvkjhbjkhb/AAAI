# Round 8 AAAI Stabilization

Goal: make the submission defensible as **Failure-Triggered Adaptive Reward Shaping for Cooperative Multi-Agent Reinforcement Learning** rather than as an LLM semantic-causality paper.

Main package:
- Environments: lbforaging:Foraging-10x10-3p-3f-v3 and lbforaging:Foraging-12x12-3p-4f-v3.
- Methods: baseline uniform_budget_matched_0.0003_late045 adaptive_0.0003_late045 random_type_budget_matched_0.0003_late045 semantic_shuffled_budget_matched_0.0003_late045 diagnosis_only.
- Seeds: 1 2 3 4 5 6 7 8.
- Budget: 500000 timesteps per run, test interval 100000, 20 test episodes.

Controls:
- uniform_budget_matched uses a label-independent phase schedule with the same nominal penalty and late-phase attenuation as adaptive shaping.
- random_type_budget_matched uses random failure labels with the same trigger stream, confidence scaling, phase schedule, and type-weight budget.
- semantic_shuffled_budget_matched uses matched-frequency shuffled labels from the observed label pool.
- diagnosis_only records failures with no reward intervention.

Optional Qwen validation:
- RUN_QWEN_VALIDATION=1 evaluates Qwen/Qwen3.5-4B offline with cache/retry enabled on 120 validation records. It is not part of the online RL main claim.
