# Round 7 Decisive AAAI One-GPU Experiment

This run is designed to close the Round 6 reviewer gap in one pass on a single RTX 5090. It reuses completed Round 6 baseline, uniform, and adaptive seeds, then adds only the decisive missing controls: matched-frequency random type, ordinary random type at matched penalty/late schedule, confidence-gated semantic adaptive, and conservative 12x12 completion seeds.

## Main 10x10 Jobs

- Environment: lbforaging:Foraging-10x10-3p-3f-v3
- Seeds: 1 2 3 4 5 6 7 8
- Budget: 500000 timesteps
- Methods: random_type_matched_0.0003_late045, random_type_0.0003_late045, random_type_0.0003_late060, semantic_gate_0.0003_late045
- Primary decision: semantic_gate_0.0003_late045 must beat matched random-type and calibrated uniform/adaptive baselines in paired seed comparisons for a strong semantic AAAI claim.

## Conservative 12x12 Completion Jobs

- Environment: lbforaging:Foraging-12x12-3p-4f-v3
- Seeds: 6 7 8
- Methods: baseline, uniform_0.0002, adaptive_0.0003_late045, semantic_gate_0.0003_late045
- Interpretation: stress test unless semantic_gate clearly wins.

## Expected Runtime

Typical Round 6/7 runtime is 6-7 minutes per 500k run. With 44 queued jobs and per-run summarization, expected wall time is 4.5-5.5 hours. STATUS.txt is updated after every START, DONE, FAIL, and summary refresh.
