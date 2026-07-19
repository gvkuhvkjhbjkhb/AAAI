# P3 frozen transfer protocol

## Objective

P3 tests whether the S1/S2-frozen Safe-SCA rule transfers to **eight unseen
payoff matrices and opaque action-label surfaces**. It is a transfer test, not
a chance to repair S2 or select a favorable matrix.

## Fixed design

- Eight matrices: `p3_m01`–`p3_m08`; four coordination/boundary and four anti.
- Ten never-used seeds: 102–111.
- Four policies: NoAlign, Always-Gated, Point-SCA, Safe-SCA.
- `8 × 10 × 4 = 320` cells; 30 episodes, five rounds/episode, memory two.
- Safe-SCA configuration is byte-for-byte semantically identical to S1/S2:
  warm-up 15, tau .10, 95% confidence, 2,000 bootstrap samples, coverage .125,
  minimum stratum size three.
- Qwen and GLM use the revision pins recorded in
  `protocols/p3_frozen_protocol.json`; `top_p=.9`, request-seed base 1000, and
  Latin-square policy order are frozen.
- The P3 scheduler uses 32 workers, matching S2's documented cross-environment
  topology. This is an intentional P3 constant, not a same-topology R0 replay.

## Information barrier

Matrix category, matrix ID, and payoff table are analysis metadata. They are
never passed to Safe-SCA. All LLM prompts use the same game name,
`Anonymous interaction`; they see only opaque labels and realized history.
`payoff_in_prompt=false` is enforced by the runner.

## Confirmatory P3 gates

After `320/320` integrity validation, P3 passes only if all conditions hold:

1. In every anti matrix, Safe-SCA's paired-bootstrap 95% lower CI versus
   NoAlign is at least `−0.10`.
2. Safe-SCA makes zero false-align routes in the 40 anti matrix-seed cells.
3. In at least two of the four coordination/boundary matrices with a positive
   Always-Gated gain, Safe-SCA recovers at least 30% of that gain.

Every matrix is reported separately. There are no post-hoc matrix exclusions,
no S1/S2/P3 p-value pooling, and no edits to the frozen configuration after
launch.

## Stop conditions

Stop and preserve artifacts if a metric is missing, a task exhausts two
retries, the preflight manifest is invalid, the matrix registry hash differs
from the campaign snapshot, or the protocol configuration does not validate.

P3 safety failure removes the unseen-matrix generalization claim; it does not
erase the already-supported S1/S2 in-distribution result.
