# G2 — top_p single-factor ablation (1.0 vs frozen 0.9)

3 anti-coordination games, seeds 42-51. Arms NoToM (NoAlign) & Gated.


## Per-game: NoToM vs Gated at each top_p (paired Wilcoxon)

| game | top_p | NoToM | Gated | Δ(Gated-NoToM) | p | dz | flip(NoToM>=Gated) |
|---|---|---|---|---|---|---|---|
| chicken | 0.9 | 2.389 | 2.176 | +0.213 | 0.0009 | +0.99 | YES |
| chicken | 1.0 | 2.563 | 2.164 | +0.399 | 0.0039 | +1.08 | YES |
| deadlock | 0.9 | 1.463 | 2.154 | -0.691 | 0.0001 | -6.25 | no |
| deadlock | 1.0 | 1.399 | 2.100 | -0.701 | 0.0020 | -4.67 | no |
| hawk_dove | 0.9 | 1.137 | 1.996 | -0.859 | 0.0001 | -4.41 | no |
| hawk_dove | 1.0 | 1.211 | 1.792 | -0.581 | 0.0020 | -4.08 | no |

## Attribution

- top_p=0.9 flip (NoToM>=Gated) in: ['chicken']

- top_p=1.0 flip (NoToM>=Gated) in: ['chicken']

- **verdict: SAMPLING-ROBUST: flip persists at top_p=1.0 -> attributable to precision/template/other factors**


## Per-arm shift top_p 0.9 -> 1.0 (same seeds)

| game | arm | 0.9 | 1.0 | shift |
|---|---|---|---|---|
| chicken | NoToM | 2.325 | 2.563 | +0.238 |
| chicken | Gated | 2.170 | 2.164 | -0.006 |
| deadlock | NoToM | 1.439 | 1.399 | -0.039 |
| deadlock | Gated | 2.152 | 2.100 | -0.052 |
| hawk_dove | NoToM | 1.051 | 1.211 | +0.160 |
| hawk_dove | Gated | 1.999 | 1.792 | -0.207 |
