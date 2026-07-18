# S1 — Coverage-Certified Safe-SCA held-out analysis

Primary endpoint: `total_horizon_team_payoff_including_warmup`.
Safety margin: 0.100; all anti-coordination games pass: **True**.
Utility gate: recover at least 30% of the positive Always-Gated gain in 2 coordination/boundary games; qualified=['stag_hunt', 'battle_of_the_sexes']; pass: **True**.
Method-paper gate (safety AND utility): **True**.

## Paired payoff against NoAlign

| game | policy | n | NoAlign | policy | delta | 95% paired bootstrap CI | win rate |
|---|---|---:|---:|---:|---:|---|---:|
| chicken | NoAlign | 20 | 2.379 | 2.379 | +0.000 | [+0.000, +0.000] | 0% |
| chicken | Always Gated | 20 | 2.379 | 2.905 | +0.526 | [+0.472, +0.579] | 100% |
| chicken | Legacy GSACA | 20 | 2.379 | 2.425 | +0.046 | [-0.011, +0.100] | 70% |
| chicken | Point-SCA | 20 | 2.379 | 2.371 | -0.008 | [-0.030, +0.014] | 45% |
| chicken | Safe-SCA | 20 | 2.379 | 2.374 | -0.005 | [-0.022, +0.016] | 35% |
| chicken | Label-oracle SCA (diagnostic) | 20 | 2.379 | 2.367 | -0.012 | [-0.033, +0.010] | 25% |
| deadlock | NoAlign | 20 | 2.000 | 2.000 | +0.000 | [+0.000, +0.000] | 0% |
| deadlock | Always Gated | 20 | 2.000 | 1.794 | -0.206 | [-0.212, -0.200] | 0% |
| deadlock | Legacy GSACA | 20 | 2.000 | 1.985 | -0.015 | [-0.024, -0.006] | 0% |
| deadlock | Point-SCA | 20 | 2.000 | 1.898 | -0.102 | [-0.107, -0.097] | 0% |
| deadlock | Safe-SCA | 20 | 2.000 | 2.000 | +0.000 | [+0.000, +0.000] | 0% |
| deadlock | Label-oracle SCA (diagnostic) | 20 | 2.000 | 2.000 | +0.000 | [+0.000, +0.000] | 0% |
| hawk_dove | NoAlign | 20 | 1.247 | 1.247 | +0.000 | [+0.000, +0.000] | 0% |
| hawk_dove | Always Gated | 20 | 1.247 | 1.999 | +0.752 | [+0.724, +0.779] | 100% |
| hawk_dove | Legacy GSACA | 20 | 1.247 | 1.658 | +0.410 | [+0.315, +0.504] | 100% |
| hawk_dove | Point-SCA | 20 | 1.247 | 1.251 | +0.004 | [-0.007, +0.015] | 45% |
| hawk_dove | Safe-SCA | 20 | 1.247 | 1.244 | -0.004 | [-0.016, +0.009] | 35% |
| hawk_dove | Label-oracle SCA (diagnostic) | 20 | 1.247 | 1.239 | -0.008 | [-0.020, +0.005] | 25% |
| stag_hunt | NoAlign | 20 | 2.670 | 2.670 | +0.000 | [+0.000, +0.000] | 0% |
| stag_hunt | Always Gated | 20 | 2.670 | 3.000 | +0.330 | [+0.280, +0.384] | 100% |
| stag_hunt | Legacy GSACA | 20 | 2.670 | 2.866 | +0.197 | [+0.120, +0.268] | 85% |
| stag_hunt | Point-SCA | 20 | 2.670 | 2.825 | +0.156 | [+0.118, +0.195] | 100% |
| stag_hunt | Safe-SCA | 20 | 2.670 | 2.826 | +0.157 | [+0.120, +0.196] | 100% |
| stag_hunt | Label-oracle SCA (diagnostic) | 20 | 2.670 | 2.826 | +0.156 | [+0.120, +0.196] | 100% |
| battle_of_the_sexes | NoAlign | 20 | 1.173 | 1.173 | +0.000 | [+0.000, +0.000] | 0% |
| battle_of_the_sexes | Always Gated | 20 | 1.173 | 2.340 | +1.167 | [+1.091, +1.232] | 100% |
| battle_of_the_sexes | Legacy GSACA | 20 | 1.173 | 2.256 | +1.083 | [+1.022, +1.138] | 100% |
| battle_of_the_sexes | Point-SCA | 20 | 1.173 | 1.767 | +0.593 | [+0.552, +0.631] | 100% |
| battle_of_the_sexes | Safe-SCA | 20 | 1.173 | 1.778 | +0.605 | [+0.564, +0.642] | 100% |
| battle_of_the_sexes | Label-oracle SCA (diagnostic) | 20 | 1.173 | 1.770 | +0.597 | [+0.555, +0.634] | 100% |
| public_goods | NoAlign | 20 | 2.571 | 2.571 | +0.000 | [+0.000, +0.000] | 0% |
| public_goods | Always Gated | 20 | 2.571 | 2.534 | -0.036 | [-0.042, -0.031] | 0% |
| public_goods | Legacy GSACA | 20 | 2.571 | 2.582 | +0.011 | [+0.008, +0.015] | 90% |
| public_goods | Point-SCA | 20 | 2.571 | 2.553 | -0.017 | [-0.022, -0.014] | 0% |
| public_goods | Safe-SCA | 20 | 2.571 | 2.555 | -0.016 | [-0.019, -0.013] | 0% |
| public_goods | Label-oracle SCA (diagnostic) | 20 | 2.571 | 2.555 | -0.015 | [-0.020, -0.011] | 5% |

## Safe-SCA routing risk

- Anti-coordination false-align: **0/60** (0.0%).
- Coordination/boundary false-abstain: **0/60** (0.0%).

## Coordination gain recovery

| game | Gated-NoAlign | Safe-SCA-NoAlign | recovery |
|---|---:|---:|---:|
| chicken | +0.526 | -0.005 | -1.0% |
| deadlock | -0.206 | +0.000 | -0.0% |
| hawk_dove | +0.752 | -0.004 | -0.5% |
| stag_hunt | +0.330 | +0.157 | 47.4% |
| battle_of_the_sexes | +1.167 | +0.605 | 51.9% |
| public_goods | -0.036 | -0.016 | 43.5% |
