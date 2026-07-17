# Phase 1 Adaptive Robustness Experiment

This run continues Phase 1 of the roadmap: failure-triggered adaptive reward shaping on the 10x10 LBF task. Round 6 already contains eight seeds for adaptive_0.0002_late045 and adaptive_0.0003_late045. This continuation adds the missing upper penalty point adaptive_0.0005_late045 and two calibrated controls at the same nominal intervention magnitude: uniform_0.0005 and random_type_0.0005_late045.

The decision question is whether adaptive shaping remains competitive across penalty values rather than being a single lucky 0.0003 setting. The primary comparisons after merging Round 6, Round 7, and this run are adaptive_0.0005_late045 against baseline, uniform_0.0005, random_type_0.0005_late045, adaptive_0.0003_late045, and uniform_0.0003 on final test return and train AUC.

Environment: lbforaging:Foraging-10x10-3p-3f-v3
Seeds: 1 2 3 4 5 6 7 8
Budget: 500000 timesteps per run
Methods: adaptive_0.0005_late045 uniform_0.0005 random_type_0.0005_late045
