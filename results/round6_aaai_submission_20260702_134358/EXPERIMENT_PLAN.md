# Round 6 AAAI Submission-Grade Two-GPU Experiment

This run prioritizes the most defensible AAAI claim: diagnosis-conditioned intervention should beat MAPPO and random-type controls, while calibrated uniform shaping is treated as a strong baseline rather than a weak foil.

## Main 10x10 Experiment

- Environment: lbforaging:Foraging-10x10-3p-3f-v3
- Seeds: 1 2 3 4 5 6 7 8
- Budget: 500000 timesteps
- Methods: baseline, diagnosis_only, uniform_0.0002, uniform_0.0003, type_specific_0.0002, type_specific_0.0003, adaptive_0.0002_late045, adaptive_0.0002_late060, adaptive_0.0003_late045, random_type_0.0002
- Primary metrics: last_test_return, train_auc, stability_gap, best_train_return
- Main acceptance criterion: adaptive or type-specific intervention improves baseline and random_type, and is competitive with or better than the best uniform penalty.

## Generalization Experiment

- Environment: lbforaging:Foraging-12x12-3p-4f-v3, with fallback to lbforaging:Foraging-10x10-3p-4f-v3 if unavailable
- Seeds: 1 2 3 4 5
- Budget: 500000 timesteps
- Methods: baseline, uniform_0.0002, uniform_0.0003, adaptive_0.0002_late045, adaptive_0.0002_late060

## Compute

- Worker 0 uses CUDA_VISIBLE_DEVICES=0
- Worker 1 uses CUDA_VISIBLE_DEVICES=1
- Each worker has a private source copy to avoid Sacred result-directory races.
