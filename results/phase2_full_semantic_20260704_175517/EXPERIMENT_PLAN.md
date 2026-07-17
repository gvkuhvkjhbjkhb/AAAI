# Phase 2 Full Semantic Diagnosis Experiment

Phase 2A validates human agreement using PaperGuru initial labels as annotator A and a deterministic independent rule-based second annotator as annotator B, then adjudicates disagreements. Phase 2B compares original heuristic, enhanced heuristic, and Qwen/Qwen3.5-4B on the 300-record validation set. Phase 2C/2D runs full 10x10 semantic-causality RL using 8 paired seeds.

Estimated wall time on one GPU: diagnosis validation 10-35 minutes depending on API latency; RL 40-48 runs * 6-8 minutes = 4.0-6.4 hours; summarization and packaging <10 minutes; total 4.5-7.0 hours. If API latency is high for online LLM RL, total can extend to 8+ hours.
