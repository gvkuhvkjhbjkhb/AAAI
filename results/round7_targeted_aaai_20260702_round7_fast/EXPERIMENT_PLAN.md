# Round 7 Targeted AAAI Optimization Experiment

This one-GPU run directly addresses the Round 6 reviewer risk that random-type controls matched the best semantic adaptive method. The queue prioritizes matched random controls and confidence-gated semantic adaptive shaping on 10x10, then fills missing 12x12 seeds for a conservative stress-test panel.

## Main 10x10 Jobs

- Environment: lbforaging:Foraging-10x10-3p-3f-v3
- Seeds: 1 2 3 4 5 6 7 8
- Budget: 500000 timesteps
- Methods: random_type_0.0003_late045, random_type_0.0003_late060, random_type_0.0002_late060, semantic_gate_0.0003_late045

## Generalization Completion Jobs

- Environment: lbforaging:Foraging-12x12-3p-4f-v3
- Seeds: 6 7 8
- Budget: 500000 timesteps
- Methods: baseline, uniform_0.0002, adaptive_0.0003_late045, semantic_gate_0.0003_late045

## Runtime Estimate

Round 6 logs show about 5.5-7.0 minutes per 500k run on RTX 5090. This 44-run queue should take roughly 4.5-6.0 hours on one GPU, plus summarization and packaging.
