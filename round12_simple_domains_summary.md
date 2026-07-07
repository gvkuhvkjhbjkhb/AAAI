# Round 12 Simple Cooperative Domain Detailed Analysis

## Executive Summary

Round 12 tested two simple sparse cooperative LBF domains as candidate positive family-level evidence: `Foraging-8x8-2p-2f-coop-v3` and `Foraging-10x10-3p-3f-coop-v3`. The experiment completed 64/64 runs with zero failures. Each environment used four methods, eight paired seeds, and 300k timesteps per run. The intended goal was to find a simple domain where adaptive failure-triggered shaping clearly beats baseline and budget-matched controls.

The experiment did not produce a usable positive LBF-family result. On `Foraging-8x8-2p-2f-coop-v3`, adaptive achieved mean final return 0.1677, below baseline at 0.1958 and below random-type budget matching at 0.2500. Adaptive was only slightly above phase-uniform budget matching at 0.1625, with a paired final-return delta of +0.0052 and a confidence interval crossing zero. On `Foraging-10x10-3p-3f-coop-v3`, all methods had zero final return over eight seeds, so the task is not useful as positive evidence under the current 300k setup.

Therefore no Round 12 domain should be used as main-text positive generalization evidence. The 8x8 cooperative task can be kept as an appendix stress test showing that adaptive shaping is not universally beneficial even within LBF; the 10x10 cooperative-only task is a learnability failure at this training budget. The result reinforces the current conservative AAAI strategy: use Round 8 10x10 as the main significant result, Round 10 budget accounting as the main mechanism defense, and LBF 12x12/10x10-4p-4f as directional stress tests. Do not claim broad LBF-family robustness.

## Experiment Design

The Round 12 package targeted simple sparse cooperative domains where the method was most likely to help. The first domain, `lbforaging:Foraging-8x8-2p-2f-coop-v3`, is a small cooperative task with two agents and two foods. The second domain, `lbforaging:Foraging-10x10-3p-3f-coop-v3`, is a cooperative-only version of the original 10x10 setup. Both preserve the foraging reward structure and avoid the reward-scale mismatch observed in VMAS.

The methods were `baseline`, `uniform_budget_matched_0.0003_late045`, `adaptive_0.0003_late045`, and `random_type_budget_matched_0.0003_late045`. Each domain used seeds 1--8 and 300k timesteps. The decision rule was strict: a domain could be treated as positive evidence only if adaptive improved over baseline and remained competitive with or better than phase-uniform and random-type budget controls.

## LBF 8x8 2p-2f Coop Results

| method | n | final return | 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_0.0003_late045 | 8 | 0.1677 | [0.0698, 0.2781] | 0.1031 | 0.3866 | 0.2189 |
| baseline | 8 | 0.1958 | [0.0917, 0.3010] | 0.1052 | 0.4083 | 0.2124 |
| random_type_budget_matched_0.0003_late045 | 8 | 0.2500 | [0.1781, 0.3260] | 0.1424 | 0.5620 | 0.3120 |
| uniform_budget_matched_0.0003_late045 | 8 | 0.1625 | [0.0677, 0.2677] | 0.0977 | 0.3972 | 0.2347 |

Paired comparisons do not support adaptive as a positive result. Adaptive minus baseline final-return delta is -0.0281 with 95% CI [-0.1021, 0.0375]. Adaptive minus random-type final-return delta is -0.0823 with CI [-0.1906, 0.0240]. Adaptive is only +0.0052 over phase-uniform, with CI [-0.0708, 0.0719]. The random-type control has the best mean final return and train AUC. This suggests that, in this small cooperative environment, generic budget-matched random shaping or exploration noise is at least as helpful as adaptive failure-type weighting.

The budget accounting strengthens the conclusion. Adaptive used less penalty budget than random and uniform: 28.6910 versus 32.5337 and 36.5402. However, the lower budget did not translate into better performance. This means the negative result is not because adaptive used too much reward shaping; rather, the adaptive intervention is not the best inductive bias for this small task at the tested coefficient.

## LBF 10x10 3p-3f Coop Results

The 10x10 cooperative-only environment is uninformative at 300k timesteps. All methods have zero final return across all eight seeds. Adaptive shows tiny nonzero best-train/AUC values, but final test return is exactly zero and therefore cannot support a claim about reward-shaping superiority. The correct interpretation is learnability failure or insufficient training horizon, not a meaningful method comparison.

This environment should not be placed in the main paper. It can be mentioned internally or excluded. If it is included anywhere, it should be framed as a hard cooperative-only variant where no method learned reliably under the current budget.

## Positive Evidence Decision

No Round 12 domain satisfies the positive-evidence decision rule. The 8x8 cooperative task fails because adaptive is below baseline and random-type budget matching. The 10x10 cooperative-only task fails because final returns are zero for all methods. Therefore neither domain should be used as a main-text positive LBF-family generalization table.

The closest usable statement is negative/limitation-oriented: adaptive failure-triggered shaping is not universally beneficial on every LBF variant, particularly when the task is very small or when cooperative-only dynamics create hard all-zero learning. This is useful for honest boundary conditions, but it does not improve the AAAI main claim.

## Impact on AAAI Strategy

Round 12 does not change the evidence hierarchy. The strongest result remains Round 8's `Foraging-10x10-3p-3f-v3` panel, where adaptive clearly beats baseline, diagnosis-only, phase-uniform budget matching, and random-type controls. Round 10 remains the strongest mechanism defense because it adds exact budget accounting and shows adaptive can use less shaping budget than controls. Round 10's `Foraging-10x10-4p-4f-v3` and merged 12x12 remain directional stress tests, not significant victories. Round 11 VMAS remains a cross-domain limitation.

For AAAI, the manuscript should not claim broad LBF-family generalization. The best claim is narrower but defensible: adaptive failure-triggered shaping is effective on the main sparse cooperative foraging benchmark under strong controls and exact budget accounting, with related stress tests exposing where the method is less reliable.

## Recommendation

Do not use Round 12 as main-text positive evidence. If space allows, mention it in appendix as an additional stress test and boundary condition. If more experiments are allowed, the next attempt should not reuse `0.0003` blindly. A better search would be a short coefficient sweep on `8x8-2p-2f-coop` because random-type and uniform budgets suggest this small task is sensitive to shaping scale and exploration. However, for the current AAAI submission, additional fishing for a positive domain risks weakening the story. It is safer to proceed with the conservative Round 8/10-centered narrative.

## Files

- Launcher: `run_round12_simple_domains.sh`
- Report builder: `epymarl/tools/build_round12_simple_domain_report.py`
- Output directory: `results/round12_simple_domains_round12_full_20260707_121253/`
- Auto report: `results/round12_simple_domains_round12_full_20260707_121253/ROUND12_SIMPLE_DOMAIN_REPORT.md`
- Detailed analysis: `results/round12_simple_domains_round12_full_20260707_121253/ROUND12_DETAILED_ANALYSIS.md`
- Packaged artifact: `artifacts/AAAI_round12_simple_domains_round12_full_20260707_121253.tar.gz`
