# S1 — Coverage-Certified Safe-SCA held-out analysis

Primary endpoint: `total_horizon_team_payoff_including_warmup`.
Safety margin: 0.100; all anti-coordination games pass: **True**.
Utility gate: recover at least 30% of the positive Always-Gated gain in 2 coordination/boundary games; qualified=['stag_hunt', 'battle_of_the_sexes']; pass: **True**.
Method-paper gate (safety AND utility): **True**.

## Paired payoff against NoAlign

| game | policy | n | NoAlign | policy | delta | 95% paired bootstrap CI | win rate |
|---|---|---:|---:|---:|---:|---|---:|
| chicken | NoAlign | 20 | 2.389 | 2.389 | +0.000 | [+0.000, +0.000] | 0% |
| chicken | Always Gated | 20 | 2.389 | 2.880 | +0.491 | [+0.440, +0.540] | 100% |
| chicken | Legacy GSACA | 20 | 2.389 | 2.453 | +0.064 | [+0.025, +0.102] | 70% |
| chicken | Point-SCA | 20 | 2.389 | 2.391 | +0.002 | [-0.005, +0.010] | 25% |
| chicken | Safe-SCA | 20 | 2.389 | 2.392 | +0.003 | [-0.009, +0.015] | 35% |
| chicken | Label-oracle SCA (diagnostic) | 20 | 2.389 | 2.397 | +0.008 | [-0.004, +0.019] | 45% |
| deadlock | NoAlign | 20 | 2.000 | 2.000 | +0.000 | [+0.000, +0.000] | 0% |
| deadlock | Always Gated | 20 | 2.000 | 1.724 | -0.276 | [-0.283, -0.268] | 0% |
| deadlock | Legacy GSACA | 20 | 2.000 | 1.976 | -0.024 | [-0.033, -0.015] | 0% |
| deadlock | Point-SCA | 20 | 2.000 | 1.864 | -0.136 | [-0.143, -0.128] | 0% |
| deadlock | Safe-SCA | 20 | 2.000 | 2.000 | +0.000 | [+0.000, +0.000] | 0% |
| deadlock | Label-oracle SCA (diagnostic) | 20 | 2.000 | 2.000 | +0.000 | [+0.000, +0.000] | 0% |
| hawk_dove | NoAlign | 20 | 1.204 | 1.204 | +0.000 | [+0.000, +0.000] | 0% |
| hawk_dove | Always Gated | 20 | 1.204 | 1.998 | +0.794 | [+0.765, +0.826] | 100% |
| hawk_dove | Legacy GSACA | 20 | 1.204 | 1.567 | +0.363 | [+0.251, +0.479] | 100% |
| hawk_dove | Point-SCA | 20 | 1.204 | 1.205 | +0.001 | [-0.008, +0.010] | 35% |
| hawk_dove | Safe-SCA | 20 | 1.204 | 1.202 | -0.002 | [-0.008, +0.003] | 25% |
| hawk_dove | Label-oracle SCA (diagnostic) | 20 | 1.204 | 1.202 | -0.002 | [-0.011, +0.005] | 35% |
| stag_hunt | NoAlign | 20 | 2.852 | 2.852 | +0.000 | [+0.000, +0.000] | 0% |
| stag_hunt | Always Gated | 20 | 2.852 | 2.999 | +0.147 | [+0.111, +0.183] | 90% |
| stag_hunt | Legacy GSACA | 20 | 2.852 | 2.919 | +0.067 | [+0.037, +0.100] | 85% |
| stag_hunt | Point-SCA | 20 | 2.852 | 2.933 | +0.081 | [+0.055, +0.108] | 80% |
| stag_hunt | Safe-SCA | 20 | 2.852 | 2.914 | +0.062 | [+0.031, +0.092] | 80% |
| stag_hunt | Label-oracle SCA (diagnostic) | 20 | 2.852 | 2.920 | +0.068 | [+0.043, +0.093] | 80% |
| battle_of_the_sexes | NoAlign | 20 | 1.123 | 1.123 | +0.000 | [+0.000, +0.000] | 0% |
| battle_of_the_sexes | Always Gated | 20 | 1.123 | 2.374 | +1.251 | [+1.195, +1.311] | 100% |
| battle_of_the_sexes | Legacy GSACA | 20 | 1.123 | 2.233 | +1.109 | [+1.058, +1.158] | 100% |
| battle_of_the_sexes | Point-SCA | 20 | 1.123 | 1.732 | +0.609 | [+0.578, +0.641] | 100% |
| battle_of_the_sexes | Safe-SCA | 20 | 1.123 | 1.723 | +0.599 | [+0.568, +0.633] | 100% |
| battle_of_the_sexes | Label-oracle SCA (diagnostic) | 20 | 1.123 | 1.732 | +0.608 | [+0.577, +0.642] | 100% |
| public_goods | NoAlign | 20 | 2.569 | 2.569 | +0.000 | [+0.000, +0.000] | 0% |
| public_goods | Always Gated | 20 | 2.569 | 2.530 | -0.039 | [-0.046, -0.032] | 5% |
| public_goods | Legacy GSACA | 20 | 2.569 | 2.584 | +0.014 | [+0.011, +0.018] | 95% |
| public_goods | Point-SCA | 20 | 2.569 | 2.548 | -0.021 | [-0.026, -0.016] | 0% |
| public_goods | Safe-SCA | 20 | 2.569 | 2.549 | -0.020 | [-0.025, -0.015] | 5% |
| public_goods | Label-oracle SCA (diagnostic) | 20 | 2.569 | 2.545 | -0.024 | [-0.028, -0.020] | 0% |

## Safe-SCA routing risk

- Anti-coordination false-align: **0/60** (0.0%).
- Coordination/boundary false-abstain: **3/60** (5.0%).

## Coordination gain recovery

| game | Gated-NoAlign | Safe-SCA-NoAlign | recovery |
|---|---:|---:|---:|
| chicken | +0.491 | +0.003 | 0.6% |
| deadlock | -0.276 | +0.000 | -0.0% |
| hawk_dove | +0.794 | -0.002 | -0.3% |
| stag_hunt | +0.147 | +0.062 | 42.3% |
| battle_of_the_sexes | +1.251 | +0.599 | 47.9% |
| public_goods | -0.039 | -0.020 | 51.6% |
