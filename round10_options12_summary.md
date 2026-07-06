# Round 10 Options 1+2 Detailed Analysis

## Executive Summary

Round 10 completed the two safest post-Round-9 improvements: mechanism defense through exact budget accounting and LBF-family generalization through a new structural variant. The experiment finished 52/52 jobs with zero failures. It added exact shaping-budget logs to the training code, ran a new `Foraging-10x10-4p-4f-v3` task with four core methods and eight paired seeds, and ran a 10x10 sensitivity/budget panel with four paired seeds.

The result strengthens the paper's engineering and analysis layer, but it does not create a new statistically decisive generalization win. On the new 4-agent LBF task, adaptive has the highest mean final return among the four methods: 0.3100 versus 0.2945 for baseline, 0.2853 for phase-uniform budget matching, and 0.2704 for random-type budget matching. However, paired confidence intervals cross zero: adaptive minus baseline is +0.0155 with 95% CI [-0.0678, 0.1017], adaptive minus uniform is +0.0248 with CI [-0.0435, 0.0952], and adaptive minus random-type is +0.0396 with CI [-0.0082, 0.0883]. This is useful as directional LBF-family stress-test evidence, but not as a strong second-task victory.

The budget accounting is more important for the AAAI argument. On the new LBF task, adaptive uses less total penalty budget than both phase-uniform and random-type controls while still obtaining the highest mean final return. Adaptive's mean penalty total is 27.8313, compared with 34.4002 for phase-uniform and 30.4363 plus 0.8805 terminal-bonus budget for random-type. On the 10x10 sensitivity panel, adaptive 0.0003 with late weight 0.45 remains much stronger than the lower and higher penalty variants and is essentially tied with late weight 0.60 on shared seeds. This supports the claim that the Round 8 result is not merely caused by a larger reward-shaping budget.

The final AAAI interpretation should remain conservative. Round 10 improves the story from “single LBF map” to “main LBF result plus a directionally positive LBF-family structural variant and exact budget accounting.” It still does not justify broad cross-domain generalization claims. The best manuscript framing is: failure-triggered adaptive shaping is strongly validated on the main sparse cooperative foraging task under strong controls; related LBF-family stress tests are directionally consistent; cross-domain tests reveal reward-scale limitations.

## Experiment Design

Round 10 implements Option 1 and Option 2 from `docs/AAAI_STABILIZATION_OPTIONS.md`. Option 1 is the mechanism-defense package: add sensitivity analysis and exact budget accounting so reviewers cannot attribute the 10x10 result to hidden reward-budget differences or cherry-picked coefficients. Option 2 is LBF-family generalization: add a new cooperative foraging structure rather than forcing a cross-domain task whose reward scale and failure semantics differ from the method's assumptions.

The new LBF family task is `lbforaging:Foraging-10x10-4p-4f-v3`. It changes the original 10x10 task from three agents/three foods to four agents/four foods while preserving the sparse cooperative foraging structure. The method panel contains `baseline`, `uniform_budget_matched_0.0003_late045`, `adaptive_0.0003_late045`, and `random_type_budget_matched_0.0003_late045`, with eight paired seeds and 500k timesteps per run.

The 10x10 sensitivity panel uses `lbforaging:Foraging-10x10-3p-3f-v3`, four seeds, and five methods: `adaptive_0.0002_late045`, `adaptive_0.0005_late045`, `adaptive_0.0003_late060`, `uniform_budget_matched_0.0003_late045`, and `random_type_budget_matched_0.0003_late045`. These runs supplement Round 8 rather than replacing it. Round 8 remains the main eight-seed 10x10 evidence for `adaptive_0.0003_late045`.

## Code Changes

Round 10 adds exact budget accounting to the reward-intervention path. `FailureRewardIntervention` now tracks the cumulative number of shaped failure triggers, total applied penalty, terminal-bonus total, and shaped episode steps. At each log interval it writes `llm_fd_shaping_*` stats; at the end of each run it emits an exact `LLM_FD_ACCOUNTING_FINAL` line so that the summary is not distorted by the logger's recent-window averaging.

The log summarizer now parses exact final accounting fields and writes them to `summary.csv`. The new report builder groups performance metrics and budget metrics by experiment phase. This creates a direct table showing that adaptive shaping can outperform budget-matched controls while using less shaping penalty budget, which is central to defending against the reward-budget-artifact objection.

## New LBF Family Task Results

| method | n | final return | 95% CI | train AUC | penalty total | terminal bonus total |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 8 | 0.2945 | [0.2439, 0.3510] | 0.2368 | 0.0000 | 0.0000 |
| uniform_budget_matched_0.0003_late045 | 8 | 0.2853 | [0.2551, 0.3283] | 0.2415 | 34.4002 | 0.0000 |
| adaptive_0.0003_late045 | 8 | 0.3100 | [0.2666, 0.3623] | 0.2439 | 27.8313 | 0.0000 |
| random_type_budget_matched_0.0003_late045 | 8 | 0.2704 | [0.2472, 0.2949] | 0.2251 | 30.4363 | 0.8805 |

| paired comparison | final return delta | 95% CI | train AUC delta | 95% CI |
|---|---:|---:|---:|---:|
| adaptive - baseline | +0.0155 | [-0.0678, 0.1017] | +0.0070 | [-0.0494, 0.0631] |
| adaptive - uniform budget | +0.0248 | [-0.0435, 0.0952] | +0.0023 | [-0.0399, 0.0462] |
| adaptive - random type | +0.0396 | [-0.0082, 0.0883] | +0.0188 | [-0.0056, 0.0444] |

The new LBF task supports a cautious generalization statement but not a strong one. Adaptive has the best mean final return and train AUC, and it uses less penalty budget than both active controls. This is a favorable pattern for the paper because it shows that the adaptive method is not buying its return with a larger shaping budget. However, the paired intervals are wide because seed-level variance is large. The paper should say that the new LBF task is directionally consistent with the main 10x10 result, not that it proves significant generalization.

The seed-level deltas show why caution is required. Adaptive beats baseline strongly on seeds 3, 4, 6, and 7, but loses on seeds 1, 2, 5, and 8. Against random-type controls it is positive on most seeds but still not cleanly significant. This makes the new LBF task useful for reviewer defense against “only one map,” but it should not be the headline result.

## 10x10 Sensitivity and Budget Results

Round 8 remains the main 10x10 evidence. On eight seeds, adaptive 0.0003 late0.45 has mean final return 0.3042, compared with 0.2258 for baseline, 0.2346 for phase-uniform budget matching, and 0.2192 for random-type budget matching. The paired deltas remain clean: +0.0784 versus baseline with CI [0.0272, 0.1232], +0.0695 versus phase-uniform with CI [0.0067, 0.1244], and +0.0850 versus random-type with CI [0.0109, 0.1630].

The new sensitivity variants show that the coefficient choice is meaningful rather than arbitrary. On the four shared seeds, adaptive 0.0003 late0.45 is essentially tied with adaptive 0.0003 late0.60: final-return delta +0.0002 with CI [-0.0115, 0.0120]. This means the late-phase setting is not a fragile single value. Lower penalty 0.0002 underperforms the main setting by +0.0623 in favor of 0.0003 late0.45, though the four-seed CI crosses zero. Higher penalty 0.0005 also underperforms by +0.0664 in favor of the main setting, with wide CI. The practical conclusion is that 0.0003 is a reasonable calibrated midpoint, while late weight has some tolerance around 0.45--0.60.

The exact budget table is particularly useful. On the 10x10 sensitivity panel, adaptive 0.0003 late0.60 uses penalty total 29.6587, while uniform budget matching uses 37.3217 and random-type uses 31.9625 plus terminal bonuses. Despite using less penalty budget, adaptive 0.0003 late0.60 reaches final return 0.3039, well above uniform 0.2537 and random-type 0.1986 on the four-seed panel. This directly supports the claim that the performance gain is not reducible to “more shaping reward.”

## Integrated Interpretation Across Rounds 8--10

The stable evidence hierarchy is now clear. Round 8 10x10 is the main result and remains the strongest statistically. Round 10 new LBF is a favorable but non-significant family-level stress test. Round 8+9 12x12 is also favorable but non-significant. Round 9 VMAS and RWARE should remain appendix limitations, not main results. Qwen diagnosis remains a negative diagnostic finding, not a causal mechanism.

This means the paper can now be written at a more AAAI-credible level if it is framed correctly. The claim should be “failure-triggered adaptive reward shaping improves sparse cooperative foraging under strong controls and exact budget accounting.” The claim should not be “robust across MARL tasks” or “LLM semantic diagnosis causally improves MARL.” Round 10 strengthens the budget/artifact defense and gives an additional LBF-family setting, but it does not eliminate the need for conservative language.

## Recommended Manuscript Use

Use the Round 8 10x10 table as Main Table 1. Use the Round 10 new LBF result and merged 12x12 result as Main Table 2 or a generalization/stress-test table, explicitly saying that the effect is directionally consistent but not always significant. Use the Round 10 budget accounting table as Main Table 3 because it directly answers a major reviewer concern. Put Qwen, RWARE, and VMAS in appendix/limitations.

A safe conclusion is: “Adaptive failure-triggered shaping substantially improves the main sparse cooperative foraging benchmark under paired seeds, strong budget-matched controls, and exact shaping-budget accounting. Additional LBF-family and scale-shift tests show directionally consistent but higher-variance effects, while cross-domain experiments reveal reward-scale limitations.”

## Files

- Launcher: `run_round10_options12.sh`
- Report builder: `epymarl/tools/build_round10_options12_report.py`
- Accounting implementation: `epymarl/src/llm_diagnosis/reward_intervention.py`
- Final accounting logging: `epymarl/src/run.py`
- Summary parser: `epymarl/tools/summarize_llm_fdcr_logs.py`
- Output directory: `results/round10_options12_round10_full_20260706_024052/`
- Auto report: `results/round10_options12_round10_full_20260706_024052/ROUND10_OPTIONS12_REPORT.md`
- Detailed analysis: `results/round10_options12_round10_full_20260706_024052/ROUND10_DETAILED_ANALYSIS.md`
- Packaged artifact: `artifacts/AAAI_round10_options12_round10_full_20260706_024052.tar.gz`
