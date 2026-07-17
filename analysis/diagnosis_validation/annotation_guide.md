# Failure Diagnosis Annotation Guide

Label exactly one primary failure mode per sample.

- target_miscoordination: Agents appear active but pursue incompatible targets or fail to converge on the same cooperative target. Evidence may include strongly imbalanced load attempts or movement patterns suggesting split objectives.
- insufficient_cooperation: A task requires multiple agents, but one or more agents barely participate in cooperative actions. Evidence may include zero or near-zero load counts for an agent.
- inefficient_exploration: Agents fail to obtain reward and appear not to reach useful objectives. Evidence may include zero positive reward steps, high zero-reward count, and no more specific coordination signal.
- low_value_overcommitment: Agents repeatedly commit to low-value or unproductive behavior despite little payoff. Evidence may include repeated load/local behavior with low return when a more specific cooperation/target explanation is not supported.
- timeout_near_success: Agents obtain some positive reward or appear close to successful completion but reach the time limit.
- unknown: The summary lacks enough evidence or multiple labels are equally plausible.

Confidence scale: 1 = low confidence, 2 = medium confidence, 3 = high confidence.
