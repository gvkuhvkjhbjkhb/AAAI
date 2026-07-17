# G2 — top_p single-factor ablation (1.0 vs frozen 0.9)

3 anti-coordination games, seeds 42-51. Arms NoToM (NoAlign) & Gated.


## Per-game: NoToM vs Gated at each top_p (paired Wilcoxon)

| game | top_p | NoToM | Gated | Δ(Gated-NoToM) | p | dz | flip(NoToM>=Gated) |
|---|---|---|---|---|---|---|---|
| chicken | 0.9 | 2.389 | 2.176 | +0.213 | 0.0003 | +0.99 | YES |
| chicken | 1.0 | 2.387 | 2.099 | +0.289 | 0.0059 | +1.19 | YES |
| deadlock | 0.9 | 1.463 | 2.154 | -0.691 | 0.0000 | -6.25 | no |
| deadlock | 1.0 | 1.411 | 2.129 | -0.718 | 0.0020 | -4.35 | no |
| hawk_dove | 0.9 | 1.137 | 1.996 | -0.859 | 0.0000 | -4.41 | no |
| hawk_dove | 1.0 | 1.151 | 1.811 | -0.660 | 0.0020 | -3.63 | no |

## Attribution

- top_p=0.9 flip (NoToM>=Gated) in: ['chicken']

- top_p=1.0 flip (NoToM>=Gated) in: ['chicken']

- **verdict: SAMPLING-ROBUST: flip persists at top_p=1.0 -> attributable to precision/template/other factors**


## Per-arm shift top_p 0.9 -> 1.0 (same seeds)

| game | arm | 0.9 | 1.0 | shift |
|---|---|---|---|---|
| chicken | NoToM | 2.325 | 2.387 | +0.062 |
| chicken | Gated | 2.170 | 2.099 | -0.071 |
| deadlock | NoToM | 1.439 | 1.411 | -0.027 |
| deadlock | Gated | 2.152 | 2.129 | -0.023 |
| hawk_dove | NoToM | 1.051 | 1.151 | +0.100 |
| hawk_dove | Gated | 1.999 | 1.811 | -0.187 |
