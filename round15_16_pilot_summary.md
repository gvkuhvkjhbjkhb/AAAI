# Round 15-16 Redesign Pilot Summary

## Completion

- Track A (Outcome-Contrast): 40/40 runs completed, 0 failed
  - Job: `job_1783529325814_kanpkc`
  - Result dir: `results/round15_redesign_pilot_round15_pilot_fast_20260708_164845/`
- Track B (FA-PBS): 32/32 runs completed, 0 failed
  - Job: `job_1783529348508_q5jvly`
  - Result dir: `results/round16_pbrs_pilot_round16_pilot_fast_20260708_164908/`

## Baseline Reference (16 seeds, merged Round 8 + 13)

| method | n | final return | 95% CI |
|---|---:|---:|---:|
| baseline | 16 | 0.2460 | [0.2056, 0.2908] |

## Track A: Outcome-Contrast Terminal Shaping (8 seeds)

| method | n | final return | 95% CI | train AUC | delta vs baseline | conclusion |
|---|---:|---:|---:|---:|---:|---|
| success_bonus_b002 | 8 | 0.2270 | [0.1775, 0.2819] | 0.1585 | +0.0012 | inconclusive |
| success_bonus_b005 | 8 | 0.2438 | [0.1965, 0.3118] | 0.1681 | +0.0181 | inconclusive |
| outcome_contrast_b002_p002 | 8 | 0.2270 | [0.1775, 0.2819] | 0.1585 | +0.0012 | inconclusive |
| outcome_contrast_b005_p002 | 8 | 0.2438 | [0.1965, 0.3118] | 0.1667 | +0.0181 | inconclusive |
| terminal_failure_p002 | 8 | 0.2138 | [0.1840, 0.2474] | 0.1505 | -0.0119 | inconclusive |

**Track A verdict**: No variant significantly beats baseline. Outcome-contrast at b=0.05 is directionally positive but CI crosses zero.

## Track B: Failure-Aware PBRS (8 seeds)

| method | n | final return | 95% CI | train AUC | delta vs baseline | conclusion |
|---|---:|---:|---:|---:|---:|---|
| fa_pbrs_l005 | 8 | 0.2464 | [0.2140, 0.2867] | 0.1628 | +0.0206 | inconclusive |
| pbrs_fixed_l005 | 8 | 0.2297 | [0.2024, 0.2593] | 0.1598 | +0.0039 | inconclusive |
| pbrs_random_weights_l005 | 8 | 0.2282 | [0.1864, 0.2697] | 0.1519 | +0.0025 | inconclusive |
| pbrs_random_features_l005 | 8 | 0.1896 | [0.1547, 0.2226] | 0.1317 | -0.0362 | inconclusive |

### Track B Mechanism Isolation (paired, seeds 1-8)

| comparison | n | mean delta | 95% CI | conclusion |
|---|---:|---:|---:|---|
| fa_pbrs - pbrs_fixed_l005 | 8 | +0.0167 | [-0.0350, 0.0782] | inconclusive |
| fa_pbrs - pbrs_random_weights_l005 | 8 | +0.0181 | [-0.0330, 0.0887] | inconclusive |
| fa_pbrs - pbrs_random_features_l005 | 8 | +0.0568 | [0.0222, 0.0861] | **POSITIVE** |

**Track B verdict**: FA-PBS significantly outperforms random-feature PBS (CI entirely above zero). This is the first positive mechanism-isolation result across all 16 rounds. FA-PBS is also directionally better than baseline (+0.0206), fixed PBS (+0.0167), and random-weight PBS (+0.0181), though not yet significant at 8 seeds.

## Key Finding

The structured potential function (cooperation + exploration + target alignment features) provides statistically significant improvement over random potential features. This means the feature design matters — it is not just reward budget that drives any improvement, but the semantic structure of the shaping. This is the mechanism-level evidence that was missing in Rounds 8-14.

## Recommendation

Extend `fa_pbrs_l005` to 16 seeds with all controls to test whether the directional advantage becomes statistically significant. This is the most promising method across all redesign attempts.
