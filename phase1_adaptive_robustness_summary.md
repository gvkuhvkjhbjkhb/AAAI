# Phase 1 Adaptive Robustness Summary

This experiment continues the Phase 1 roadmap for failure-triggered adaptive reward shaping on `lbforaging:Foraging-10x10-3p-3f-v3`. It adds the missing upper penalty setting and matched controls at `0.0005` across eight seeds, then merges those results with Round 6 and Round 7 summaries for paired comparisons.

## Run Configuration

- Output directory: `results/phase1_adaptive_robustness_20260704_phase1_robustness`
- Artifact: `artifacts/AAAI_phase1_adaptive_robustness_20260704_phase1_robustness.tar.gz`
- Seeds: `1 2 3 4 5 6 7 8`
- Budget: `500000` timesteps per run
- Methods added: `adaptive_0.0005_late045`, `uniform_0.0005`, `random_type_0.0005_late045`
- Completion: 24/24 runs completed, 0 failures

## Main Findings

The new upper-penalty adaptive condition does not improve over the previously selected `adaptive_0.0003_late045`. `adaptive_0.0005_late045` reaches mean final test return `0.2451` with 95% bootstrap CI `[0.1909, 0.3096]`, whereas `adaptive_0.0003_late045` remains stronger at `0.3042` with CI `[0.2550, 0.3545]`. The paired difference `adaptive_0.0005_late045 - adaptive_0.0003_late045` is `-0.0591` on final test return and `-0.0373` on train AUC, so the higher penalty appears over-regularized rather than beneficial.

At the same `0.0005` intervention magnitude, adaptive shaping is statistically tied with uniform shaping and random-type shaping. The paired final-return difference versus `uniform_0.0005` is `-0.0082` with CI `[-0.0678, 0.0651]`, and the paired final-return difference versus `random_type_0.0005_late045` is `-0.0192` with CI `[-0.0933, 0.0664]`. These intervals cross zero, so this setting should not be used as the headline method.

The strongest Phase 1 result remains `adaptive_0.0003_late045`. It improves over baseline by `+0.0784` paired final test return with CI `[0.0261, 0.1232]` and improves train AUC by `+0.0436` with CI `[0.0181, 0.0655]`. It is also competitive with `uniform_0.0003`, with paired differences of `+0.0150` final test return and `+0.0171` train AUC, although the final-return CI crosses zero.

## Interpretation

The robustness sweep supports a calibrated-shaping story rather than a monotonic penalty-strength story. The `0.0003` adaptive setting is the best current Phase 1 default; increasing the penalty to `0.0005` reduces performance and does not separate adaptive shaping from non-semantic controls. For the paper, the safe claim is that failure-triggered adaptive reward shaping can improve MAPPO when calibrated, but overly strong interventions degrade the benefit.

## Recommendation

Use `adaptive_0.0003_late045` as the main Phase 1 method. Present `adaptive_0.0005_late045` as a robustness/penalty-sensitivity ablation showing that the method requires calibration. Avoid claiming that semantic labels are causal, because random-type and uniform controls remain competitive in several comparisons.
