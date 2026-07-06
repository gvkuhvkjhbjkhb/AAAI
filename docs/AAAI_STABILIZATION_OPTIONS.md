# AAAI Stabilization Options After Round 8/9

## Overall Assessment

The current project is suitable for an AAAI submission only under a conservative claim. The safest claim is not that Qwen or semantic diagnosis causally improves MARL, and not that the method robustly generalizes across all cooperative MARL tasks. The safest claim is that failure-triggered adaptive reward shaping improves sparse cooperative foraging under strong budget-matched controls, with directional but inconclusive scale-shift transfer and transparent cross-domain limitations.

Round 8 provides the strongest evidence: on the LBF 10x10 main task, `adaptive_0.0003_late045` clearly improves over baseline, diagnosis-only, phase-uniform budget matching, and random-type budget matching. Round 9 adds useful stress tests but does not provide a clean cross-domain win. The merged 12x12 result remains directional but not significant, RWARE tiny is uninformative because all methods obtain zero final test return, and VMAS navigation is unfavorable to the adaptive-specific claim because random-type budget matching outperforms adaptive.

Therefore the next steps should avoid overclaiming broad cross-domain robustness. The most reliable strategy is to strengthen the LBF-family evidence, add sensitivity and budget-accounting analyses, and write the paper with transparent limitations.

## Option 1: Conservative Mainline and Mechanism Defense

### Goal

Avoid making broad generalization claims and instead make the existing 10x10 result difficult to dismiss. The paper should frame failure-triggered adaptive shaping as a controlled method for sparse cooperative foraging, with the key evidence coming from strong controls and paired statistics.

### Experiments and Analyses

- Keep the Round 8 10x10 main table as the primary result.
- Keep merged Round 8+9 12x12 as a scale-shift stress test.
- Add or整理 sensitivity results for `penalty in {0.0002, 0.0003, 0.0005}` and `late_weight in {0.45, 0.60}`.
- Add budget accounting: actual shaping reward total, failure-trigger count, average penalty, terminal bonus total, and triggered-episode fraction.
- Add learning curves and AUC for all main controls.

### Why It Is Safe

This directly addresses the most likely reviewer objections: reward-budget artifact, cherry-picking, and missing control comparisons. It does not expose the negative Round 9 cross-domain result as a main claim.

### Risk

The task family remains narrow. Reviewers may still ask whether this is only an LBF-specific method.

### Estimated Time

- If mostly computed from existing logs: 0.5--1 day.
- If missing sensitivity seeds must be filled: 1--2 days.

### Recommendation

Highest priority. This is the safest and most submission-aligned path.

## Option 2: LBF-Family Generalization

### Goal

Concede that the method targets sparse cooperative foraging, but strengthen the claim beyond a single map. This is more reliable than forcing RWARE/VMAS because the reward structure and failure semantics remain aligned with the method.

### Experiments

Add one new LBF-family task, preferably one of:

- `lbforaging:Foraging-10x10-4p-4f-v3`
- `lbforaging:Foraging-15x15-3p-5f-v3`
- `lbforaging:Foraging-8x8-2p-2f-coop-v3`

Run the core four-method panel:

- `baseline`
- `uniform_budget_matched_0.0003_late045`
- `adaptive_0.0003_late045`
- `random_type_budget_matched_0.0003_late045`

Use 8 seeds and 300k--500k timesteps. If compute is sufficient, include `diagnosis_only` and `semantic_shuffled_budget_matched_0.0003_late045` as appendix controls.

### Why It Is Safe

This directly addresses the “only one LBF map” criticism while avoiding the reward-scale mismatch observed in VMAS and the learning failure observed in RWARE.

### Risk

It remains within the LBF family, so it supports family-level generalization rather than broad MARL cross-domain generalization.

### Estimated Time

- 4 methods x 8 seeds x 500k: about 32 runs.
- Expected wall time on the current single-GPU setup: about 3.5--5 hours plus analysis.

### Recommendation

Very high priority. This is the best cost-benefit experiment to improve the AAAI story.

## Option 3: VMAS Reward-Scale Calibration

### Goal

Test whether Round 9's VMAS negative result is due to reward-scale mismatch rather than a fundamental failure of failure-triggered shaping.

### Experiments

Use `vmas-navigation` and run a penalty sweep:

- `1e-5`
- `3e-5`
- `1e-4`
- `3e-4`

Pilot setup:

- methods: `baseline`, `adaptive`, `uniform_budget_matched`, `random_type_budget_matched`
- 3 seeds
- 300k timesteps

If one coefficient shows adaptive improving over baseline and not losing to random-type controls, expand the best coefficient to 8 seeds.

### Why It May Help

Round 9 shows that the LBF-tuned `0.0003` coefficient is not appropriate for VMAS. A calibrated coefficient may recover useful behavior and support a limited cross-domain statement.

### Risk

This can be seen as post-hoc tuning unless written transparently. It may still fail, and Round 9's original VMAS result must not be hidden.

### Estimated Time

- Pilot: 4 coefficients x 4 methods x 3 seeds = 48 runs, about 5--6 hours.
- Full best-coefficient expansion: 4 methods x 8 seeds = 32 runs, about 4--5 hours.
- Total: about 1--1.5 days with analysis.

### Recommendation

Medium-high priority if deadline allows. Do this after Option 2 and budget accounting.

## Option 4: RWARE Repair

### Goal

Turn the uninformative RWARE result into a usable cross-domain evaluation by making the environment learnable under the current algorithm.

### Experiments

- First run a minimal learnability pilot with only `baseline` and `adaptive`.
- Increase training to 1M or 2M timesteps.
- Consider easier RWARE settings if available.
- Stop early if baseline remains near-zero.

### Why It Is Not Preferred

Round 9 shows all methods have zero final test return at 300k. This means the current RWARE setting is not an evaluation environment for the shaping method; it is a learnability failure.

### Risk

High compute cost and likely no usable result. Even if a result emerges, it may require substantial environment-specific tuning.

### Estimated Time

- 1M pilot: 2 methods x 3 seeds, about 2--4 hours.
- Full control panel: 1--2 days.

### Recommendation

Medium-low priority. Only run after confirming baseline can learn.

## Option 5: Manuscript Restructuring and Transparent Limitations

### Goal

Make the paper's claim exactly match the evidence. This is mandatory regardless of whether additional experiments are run.

### Writing Plan

- Title should not mention LLM, Qwen, or semantic causality.
- Abstract should emphasize failure-triggered adaptive reward shaping, paired seeds, strong budget-matched controls, and sparse cooperative foraging.
- Main result table should use Round 8 10x10.
- Generalization/stress-test table should use merged 12x12 and any new LBF-family environment from Option 2.
- Qwen diagnosis should be appendix or limitations, clearly reported as noisy/collapsed.
- RWARE and VMAS should be appendix stress tests or limitations, not headline evidence.

### Why It Is Safe

This prevents reviewers from using the project's own controls and negative results to reject an overclaimed story.

### Risk

The paper becomes narrower. The novelty and motivation must be written strongly.

### Estimated Time

- Manuscript restructuring: 2--4 days.
- Figure/table cleanup: 0.5--1 day.

### Recommendation

Mandatory. This should be done even if no new experiments are added.

## Option 6: Full Strongest Package

### Goal

Maximize the chance of a credible AAAI submission by combining the safest experiments and writing strategy.

### Package

- Round 8 10x10 as the main result.
- Merged Round 8+9 12x12 as scale-shift stress test.
- One additional LBF-family task with 8 seeds and core controls.
- Budget accounting and coefficient/late-weight sensitivity.
- Optional VMAS reward-scale calibration pilot.
- Qwen/RWARE/VMAS original results as transparent limitations.

### Why It Is Strong

This package has a clear strong result, family-level generalization, strong controls, sensitivity, and honest negative results. It does not depend on Qwen quality or broad cross-domain success.

### Risk

Requires more time. VMAS calibration may still fail and should remain optional.

### Estimated Time

- New LBF task: about 0.5 day.
- Budget accounting and sensitivity: about 0.5--1 day.
- Optional VMAS calibration: about 1--1.5 days.
- Writing and figures: about 3--5 days.
- Total: about 5--7 days if VMAS calibration is included.

### Recommendation

Best overall plan if the deadline allows.

## Recommended Execution Order

1. Execute Option 2: add one LBF-family task, preferably `Foraging-10x10-4p-4f-v3` or `Foraging-15x15-3p-5f-v3`.
2. Execute Option 1: add budget accounting and sensitivity analysis.
3. Execute Option 5: restructure the manuscript around the conservative claim.
4. If time remains, execute Option 3: VMAS reward-scale calibration pilot.
5. Do not continue investing in Qwen as the main claim.
6. Do not use RWARE as a main result unless a learnability pilot first shows nonzero returns.

## Recommended Final AAAI Experiment Structure

- Main Table 1: LBF 10x10, six methods, eight paired seeds, bootstrap CI.
- Main Table 2: LBF family generalization, including merged 12x12 and one new LBF task.
- Main Table 3: Mechanism and budget controls, including diagnosis-only, phase-uniform budget, random-type budget, shuffled/matched-frequency, and sensitivity.
- Appendix Table 1: Qwen diagnostic negative result.
- Appendix Table 2: RWARE/VMAS cross-domain stress tests and limitations.
- Figure 1: method diagram.
- Figure 2: 10x10 learning curves.
- Figure 3: 12x12 and new LBF learning curves.
- Figure 4: budget accounting and failure-trigger counts.

## Final Recommendation

The best route is not to prove broad MARL generalization. The best route is to make a careful, statistically rigorous AAAI paper about failure-triggered adaptive reward shaping in sparse cooperative foraging. The evidence should be presented as strong on the main LBF 10x10 task, directional on 12x12, strengthened by a new LBF-family task, and bounded by transparent Qwen/RWARE/VMAS limitations.
