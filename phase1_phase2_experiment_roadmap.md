# Phase 1 and Phase 2 Experiment Roadmap

This roadmap records the recommended pivot after the completed Round 6 and Round 7 experiments. The current evidence supports failure-triggered adaptive reward shaping more strongly than a causal claim that semantic LLM diagnosis alone improves MARL. The plan therefore separates a lower-risk Phase 1 paper direction from a higher-risk Phase 2 semantic diagnosis program.

## Current Evidence

The strongest positive evidence is on `lbforaging:Foraging-10x10-3p-3f-v3`. In Round 6, `adaptive_0.0003_late045` improved mean final test return and train AUC over MAPPO and the calibrated uniform baseline. The same round also showed that `random_type_0.0002` matched or slightly exceeded the best adaptive semantic method, which prevents a strong causal claim that semantic labels are the source of the gain. Round 7 further weakened the semantic-causality story because `semantic_gate_0.0003_late045` did not beat matched random-type controls, uniform shaping, or the strongest adaptive schedule. The 12x12 generalization evidence remains weak and should be treated as a stress test rather than as proof of broad transfer.

## Phase 1: Failure-Triggered Adaptive Reward Shaping

### Objective

Phase 1 should produce the most defensible near-term paper. The central claim is that failed episodes provide actionable training signals for cooperative MARL, and that calibrated failure-triggered adaptive reward shaping can improve MAPPO on the 10x10 LBF task. Semantic diagnosis should be presented as an interpretability and analysis module, not as the main causal mechanism.

### Paper Framing

Recommended title direction: `Failure-Triggered Adaptive Reward Shaping for Cooperative Multi-Agent Reinforcement Learning`. The method should be described as an online failure-detection and intervention framework that records low-return trajectories, summarizes failures, and applies calibrated reward interventions. The paper should explicitly report that current semantic labels do not yet explain all performance gains, because random-type and uniform controls are strong.

### Required Analyses

1. Repair the paired-seed summary logic so that full method names are matched correctly against the baseline. The current Round 6 summary contains paired-difference sections with `n=0`, which should not be used in a paper table.
2. Regenerate Round 6 and Round 7 tables with mean, standard deviation, bootstrap confidence intervals, and paired seed differences for final test return, train AUC, best train return, and stability gap.
3. Use `adaptive_0.0003_late045` as the main Phase 1 method and compare it with baseline, diagnosis-only, `uniform_0.0002`, `uniform_0.0003`, type-specific variants, and random-type controls.
4. Present Round 7 semantic gate as a transparent negative or inconclusive ablation rather than omitting it.
5. Present 12x12 as a hard-transfer stress test. Do not claim strong adaptive transfer unless new results support it.

### Optional Follow-Up Experiments

If compute is available, run a small robustness experiment on 10x10 with 5-8 seeds for `adaptive_0.0002_late045`, `adaptive_0.0003_late045`, and `adaptive_0.0005_late045`. This verifies that the observed improvement is not a single lucky penalty value. A budget-matched control should also be added if possible, matching average reward modification magnitude across uniform, adaptive, and random-type interventions.

### Phase 1 Decision Rule

Phase 1 is viable if the repaired paired analysis confirms that adaptive failure-triggered shaping improves MAPPO and remains competitive with calibrated uniform shaping on 10x10. It does not require semantic labels to beat random controls, because semantic diagnosis is not the primary claim. If adaptive shaping fails under repaired analysis or budget-matched controls, the paper should be downgraded to a negative/diagnostic study.

## Phase 2: Validated Semantic Diagnosis and Mechanism-Specific Intervention

### Objective

Phase 2 should test whether semantic failure diagnosis can become a causal performance mechanism. This phase should start with diagnosis validation rather than another large RL sweep. The goal is to determine whether humans, heuristics, and LLMs can reliably distinguish the proposed failure types from trajectory summaries.

### Experiment 2A: Human Diagnosis Validation

Use the saved annotation set in `analysis/diagnosis_validation`. The blind annotation set contains 300 sampled summaries from existing failure records. A second annotator should independently label `annotation_sample_blind.csv` or `annotation_sample_blind.md` without seeing automatic labels or PaperGuru's initial labels. After independent labeling, compute raw agreement and Cohen's kappa between annotators, adjudicate disagreements, and create `human_labels_adjudicated.csv`.

The labels are fixed to six classes: `target_miscoordination`, `insufficient_cooperation`, `inefficient_exploration`, `low_value_overcommitment`, `timeout_near_success`, and `unknown`. Each sample should receive exactly one primary label and a confidence score from 1 to 3. If human-human kappa is below 0.3, the taxonomy or summaries are not reliable enough for a semantic diagnosis paper. If kappa is between 0.3 and 0.5, semantic claims should be cautious. If kappa exceeds 0.5 and automatic diagnosis improves macro-F1 over the original heuristic, semantic diagnosis is worth pursuing further.

### Experiment 2B: Automatic Diagnosis Comparison

Evaluate original heuristic, enhanced heuristic, and an LLM classifier against adjudicated human labels. Report accuracy, macro-F1, per-class F1, confusion matrix, Cohen's kappa, predicted label distribution, and human label distribution. Macro-F1 should be treated as the main metric because the data are class-imbalanced. Accuracy alone is insufficient because the current automatic labels collapse heavily into `inefficient_exploration`.

### Experiment 2C: Semantic Causality in RL

Run the 10x10 semantic-causality experiment only if diagnosis validation is positive. Compare baseline, calibrated uniform shaping, adaptive penalty-only shaping, semantic adaptive shaping, shuffled semantic labels, matched random-type labels, and LLM semantic adaptive shaping if LLM labels validate well. Use at least 8 seeds and preferably 10 seeds. The semantic method must beat shuffled and matched random controls in paired seed comparisons before the paper can claim that semantic labels causally improve learning.

### Experiment 2D: Mechanism-Specific Intervention

If semantic labels validate but penalty-only semantic shaping remains weak, replace penalty multipliers with type-specific mechanisms. `inefficient_exploration` can trigger exploration-oriented shaping, `insufficient_cooperation` can trigger cooperation/proximity shaping, `target_miscoordination` can trigger target-alignment shaping, `timeout_near_success` can trigger near-success curriculum or terminal shaping, and `low_value_overcommitment` can penalize repeated unproductive local actions. Compare mechanism-specific semantic intervention against mechanism-specific shuffled and matched-random controls.

### Phase 2 Decision Rule

Phase 2 can revive a strong AAAI-style semantic diagnosis claim only if three conditions hold: human labels show acceptable agreement, LLM or enhanced diagnosis improves macro-F1 over the original heuristic, and semantic or LLM-conditioned intervention beats matched random and shuffled controls in paired RL results. If any of these fail, semantic diagnosis should remain an analysis module while the main paper stays with calibrated failure-triggered adaptive shaping.

## Saved Annotation Files

The current repository includes the following diagnosis-validation files:

- `analysis/diagnosis_validation/annotation_guide.md`: label definitions and confidence scale.
- `analysis/diagnosis_validation/annotation_sample_blind.md`: blind markdown document for human annotation.
- `analysis/diagnosis_validation/annotation_sample_blind.csv`: blind CSV version for a second annotator.
- `analysis/diagnosis_validation/annotation_sample_full.csv`: full metadata and automatic labels for auditing.
- `analysis/diagnosis_validation/human_labels_paperguru_initial.csv`: PaperGuru's initial human-style labels.
- `analysis/diagnosis_validation/annotation_sample_paperguru_labeled.csv`: full sample with PaperGuru labels attached.
- `analysis/diagnosis_validation/sample_manifest.txt`: sampling statistics.
- `analysis/diagnosis_validation/paperguru_label_summary.txt`: PaperGuru label distribution and cross-tab against existing automatic labels.

PaperGuru's initial labels should not be treated as final gold labels. They are useful as a first annotator or audit reference. For a publishable diagnosis validation, a second independent human annotator and an adjudicated label file are still required.
