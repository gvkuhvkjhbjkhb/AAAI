# Round 11 VMAS Reward-Scale Calibration Detailed Analysis

## Executive Summary

Round 11 implemented Option 3: VMAS reward-scale calibration. The goal was to test whether the negative Round 9 VMAS result was caused by transferring the LBF-tuned shaping penalty `0.0003` into a dense-navigation reward landscape. The pilot swept four penalties, `0.00001`, `0.00003`, `0.0001`, and `0.0003`, using `baseline`, `adaptive`, `uniform`, and `random` on three seeds. The pilot was initially promising: `0.00003` produced the best heuristic score, adaptive had the highest mean final return among the four methods, and adaptive beat random-type clearly on the three pilot seeds.

Following the predefined decision rule, the best penalty `0.00003` was expanded from three seeds to eight seeds by adding seeds 4--8. The full eight-seed result did not confirm the pilot win. In the merged eight-seed panel, baseline has the highest mean final return, `7.3457`, while adaptive reaches `7.1841`, uniform reaches `7.1814`, and random reaches `7.2154`. Paired adaptive-minus-baseline final-return delta is `-0.1616` with 95% CI `[-0.9309, 0.5444]`. Adaptive is essentially tied with uniform and random, but it is not better than baseline. Therefore VMAS should not be used as main-text evidence for cross-domain transfer.

The scientific conclusion is still useful. The pilot shows that reward scale matters: `0.0003` remains poor on VMAS, while smaller penalties, especially `0.00003`, avoid the strong degradation observed in Round 9. However, after expansion the calibrated adaptive method does not yield a robust VMAS improvement. The correct AAAI use is a transparent appendix/limitation: failure-triggered shaping is sensitive to dense-reward domain calibration, and the current method is strongest in sparse cooperative foraging rather than broadly cross-domain MARL.

## Experiment Design

The experiment used `vmas-navigation` with 300k timesteps per run. The pilot swept four penalty scales: `0.00001`, `0.00003`, `0.0001`, and `0.0003`. For each penalty, four methods were run: baseline, adaptive, phase-uniform budget matching, and random-type budget matching. The pilot used seeds 1--3. The full expansion used penalty `0.00003` with seeds 4--8, so the merged final panel contains eight paired seeds for this calibrated penalty.

The experiment retains the same model and method family; only the reward-shaping scale is varied. No Qwen online RL route is used. This is important because the goal is to test reward-scale calibration, not to revive the semantic/LLM causal claim.

## Pilot Results

The pilot selected `0.00003` as the best candidate. At `0.00003`, adaptive achieved final return `7.7474`, compared with baseline `7.3106`, uniform `7.0454`, and random `6.9320`. Adaptive-minus-baseline final-return delta was `+0.4368`, although the 95% CI crossed zero. Adaptive-minus-random final-return delta was `+0.8154` with CI `[0.6912, 1.0564]`, which justified full expansion under the predefined rule.

Other scales were less attractive. At `0.00001`, adaptive improved over baseline on final return but was below random. At `0.0001`, random was strongest. At `0.0003`, adaptive was again weak, matching the Round 9 pattern. This pattern supports the reward-scale mismatch diagnosis: VMAS is not compatible with directly reusing the LBF-tuned `0.0003` coefficient.

## Merged Eight-Seed p=0.00003 Results

| method | n | final return | 95% CI | train AUC | best train | stability gap |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 8 | 7.3457 | [6.6104, 7.9562] | 5.0459 | 11.7757 | 4.4301 |
| adaptive | 8 | 7.1841 | [6.7281, 7.6407] | 4.9341 | 11.6762 | 4.4921 |
| uniform | 8 | 7.1814 | [6.7755, 7.5876] | 5.0026 | 11.8408 | 4.6595 |
| random | 8 | 7.2154 | [6.9240, 7.4902] | 5.0032 | 11.6837 | 4.4683 |

| comparison | final return delta | 95% CI | train AUC delta | 95% CI |
|---|---:|---:|---:|---:|
| adaptive - baseline | -0.1616 | [-0.9309, 0.5444] | -0.1118 | [-0.3625, 0.0940] |
| adaptive - uniform | +0.0028 | [-0.5877, 0.6492] | -0.0685 | [-0.2820, 0.1471] |
| adaptive - random | -0.0313 | [-0.5688, 0.4873] | -0.0690 | [-0.2721, 0.1208] |

The expansion reverses the pilot-level optimism. Adaptive remains competitive with the two shaping controls, but it does not beat baseline. The confidence intervals are broad, and the mean differences are small except for the adaptive-baseline trend, which is negative. This means VMAS calibration should not be treated as a successful cross-domain transfer result.

## Budget Accounting

At `0.00003`, adaptive uses less shaping budget than uniform and slightly less than random. Mean penalty total is `1.5937` for adaptive, `2.0109` for uniform, and `1.7545` for random. Random also receives terminal bonus total `0.0553`, while adaptive receives only `0.0006`. This confirms that the VMAS experiment is not confounded by adaptive using a larger shaping budget. The limitation is not budget inflation; it is that the calibrated adaptive intervention does not improve this dense navigation task over baseline.

## Interpretation for AAAI

Round 11 should be included only as appendix or limitations evidence. It supports the claim that reward-scale calibration matters, and it partially explains why Round 9 at `0.0003` was unfavorable. It does not support a broad cross-domain generalization claim. The main AAAI evidence should remain Round 8's 10x10 LBF result, Round 10's exact budget accounting, and the directionally positive but non-significant LBF-family stress tests.

The recommended manuscript language is: “In a dense VMAS navigation task, reward-scale calibration reduces the degradation observed with the LBF-tuned coefficient, but the calibrated adaptive intervention does not consistently outperform baseline over eight seeds. We therefore treat dense-reward cross-domain transfer as a limitation and leave normalized reward-scale adaptation for future work.”

## Final Recommendation

Do not run further VMAS expansion unless the method is modified to normalize shaping magnitude against environment reward statistics. More seeds at the same setting are unlikely to turn this into a strong positive result because the eight-seed mean is already below baseline. The best AAAI strategy remains conservative: sparse cooperative foraging main claim, strong controls, exact budget accounting, and transparent cross-domain limitations.

## Files

- Pilot output: `results/round11_vmas_calibration_round11_pilot_20260706_160509/`
- Expansion output: `results/round11_vmas_calibration_round11_full_p00003_ext_20260707_073445/`
- Merged output: `results/round11_vmas_calibration_merged_p00003_20260707/`
- Pilot artifact: `artifacts/AAAI_round11_vmas_calibration_round11_pilot_20260706_160509.tar.gz`
- Expansion artifact: `artifacts/AAAI_round11_vmas_calibration_round11_full_p00003_ext_20260707_073445.tar.gz`
- Launcher: `run_round11_vmas_calibration.sh`
- Report builder: `epymarl/tools/build_round11_vmas_calibration_report.py`
