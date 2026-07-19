# P3 transfer analysis

Primary endpoint: `total_horizon_team_payoff_including_warmup`.
Anti-matrix safety pass: **True**.
Anti false-align routing pass: **True**.
Utility qualified matrices: ['p3_m03']; pass: **False**.
Overall P3 gate: **False**.

## Paired payoff against NoAlign

| matrix | category | policy | n | delta | 95% paired CI |
|---|---|---|---:|---:|---|
| p3_m01 | coord_or_boundary | NoAlign | 10 | +0.000 | [+0.000, +0.000] |
| p3_m01 | coord_or_boundary | Always Gated | 10 | -0.419 | [-0.560, -0.280] |
| p3_m01 | coord_or_boundary | Point-SCA | 10 | -0.220 | [-0.312, -0.140] |
| p3_m01 | coord_or_boundary | Safe-SCA | 10 | -0.159 | [-0.248, -0.069] |
| p3_m02 | coord_or_boundary | NoAlign | 10 | +0.000 | [+0.000, +0.000] |
| p3_m02 | coord_or_boundary | Always Gated | 10 | -0.147 | [-0.247, -0.035] |
| p3_m02 | coord_or_boundary | Point-SCA | 10 | -0.056 | [-0.149, +0.028] |
| p3_m02 | coord_or_boundary | Safe-SCA | 10 | -0.068 | [-0.161, +0.026] |
| p3_m03 | coord_or_boundary | NoAlign | 10 | +0.000 | [+0.000, +0.000] |
| p3_m03 | coord_or_boundary | Always Gated | 10 | +0.285 | [+0.153, +0.437] |
| p3_m03 | coord_or_boundary | Point-SCA | 10 | +0.209 | [+0.013, +0.404] |
| p3_m03 | coord_or_boundary | Safe-SCA | 10 | +0.099 | [-0.011, +0.203] |
| p3_m04 | coord_or_boundary | NoAlign | 10 | +0.000 | [+0.000, +0.000] |
| p3_m04 | coord_or_boundary | Always Gated | 10 | -0.183 | [-0.311, -0.065] |
| p3_m04 | coord_or_boundary | Point-SCA | 10 | -0.150 | [-0.240, -0.073] |
| p3_m04 | coord_or_boundary | Safe-SCA | 10 | -0.099 | [-0.149, -0.046] |
| p3_m05 | anti | NoAlign | 10 | +0.000 | [+0.000, +0.000] |
| p3_m05 | anti | Always Gated | 10 | +0.114 | [+0.036, +0.182] |
| p3_m05 | anti | Point-SCA | 10 | +0.001 | [-0.010, +0.012] |
| p3_m05 | anti | Safe-SCA | 10 | +0.003 | [-0.005, +0.012] |
| p3_m06 | anti | NoAlign | 10 | +0.000 | [+0.000, +0.000] |
| p3_m06 | anti | Always Gated | 10 | -0.358 | [-0.413, -0.306] |
| p3_m06 | anti | Point-SCA | 10 | -0.003 | [-0.012, +0.003] |
| p3_m06 | anti | Safe-SCA | 10 | +0.000 | [-0.007, +0.006] |
| p3_m07 | anti | NoAlign | 10 | +0.000 | [+0.000, +0.000] |
| p3_m07 | anti | Always Gated | 10 | -1.219 | [-1.341, -1.101] |
| p3_m07 | anti | Point-SCA | 10 | -0.005 | [-0.025, +0.015] |
| p3_m07 | anti | Safe-SCA | 10 | +0.000 | [-0.036, +0.039] |
| p3_m08 | anti | NoAlign | 10 | +0.000 | [+0.000, +0.000] |
| p3_m08 | anti | Always Gated | 10 | -0.909 | [-1.028, -0.789] |
| p3_m08 | anti | Point-SCA | 10 | -0.006 | [-0.016, +0.005] |
| p3_m08 | anti | Safe-SCA | 10 | -0.002 | [-0.020, +0.015] |

## Routing

- Anti false-align: **0/40**.
- Coordination/boundary false-abstain: **0/40**.

## Gain recovery

| matrix | Gated-NoAlign | Safe-SCA-NoAlign | recovery |
|---|---:|---:|---:|
| p3_m01 | -0.419 | -0.159 | 37.9% |
| p3_m02 | -0.147 | -0.068 | 46.0% |
| p3_m03 | +0.285 | +0.099 | 34.6% |
| p3_m04 | -0.183 | -0.099 | 53.8% |
| p3_m05 | +0.114 | +0.003 | 2.9% |
| p3_m06 | -0.358 | -0.000 | 0.0% |
| p3_m07 | -1.219 | +0.000 | -0.0% |
| p3_m08 | -0.909 | -0.002 | 0.2% |
