# G1 — End-to-end 2-arm attainment bandit

cells: 119/120  |  metric: cooperation_payoff (commit phase, 20 ep for 2-player / 10 ep for public_goods)


## Headline (pre-registered: accuracy>=85% AND bandit>=SCA)

- selection accuracy: **57/119 = 47.9%** (offline probe-argmax reconstruction was 116/120=96.7%)

- bandit commit mean: **2.3181** vs SCA 2.2467 (Δ=+0.0714, p=0.5383, dz=+0.24) vs oracle 2.5059

- **verdict: PARTIAL/FAIL** (acc>=85%: no; bandit>=SCA: yes)

- probe exploration payoff: 2.4660 (commit 2.3181; probe-commit gap = -0.1478)


## Per-game

| game | grp | n | acc | bandit | oracle | frac Gated |
|---|---|---|---|---|---|---|
| chicken | anti | 20 | 5% | 2.148 | 2.401 | 100% |
| deadlock | anti | 20 | 100% | 1.465 | 2.154 | nan% |
| hawk_dove | anti | 20 | 0% | 1.989 | 1.996 | 100% |
| stag_hunt | coord | 20 | 65% | 2.953 | 2.999 | 100% |
| battle_of_the_sexes | coord | 20 | 100% | 2.832 | 2.848 | 100% |
| public_goods | boundary | 19 | 16% | 2.532 | 2.645 | 100% |

## Decision log (misses)

62 misses: hawk_dove/s43: chose Gated (true NoAlign, probe N=1.22/G=2.00); hawk_dove/s57: chose Gated (true NoAlign, probe N=1.42/G=1.98); hawk_dove/s48: chose Gated (true NoAlign, probe N=1.30/G=2.00); hawk_dove/s51: chose Gated (true NoAlign, probe N=1.36/G=2.00); hawk_dove/s58: chose Gated (true NoAlign, probe N=1.30/G=2.00); hawk_dove/s56: chose Gated (true NoAlign, probe N=1.24/G=2.00); hawk_dove/s46: chose Gated (true NoAlign, probe N=1.08/G=1.92); hawk_dove/s53: chose Gated (true NoAlign, probe N=0.92/G=2.00); hawk_dove/s42: chose Gated (true NoAlign, probe N=1.10/G=1.98); hawk_dove/s52: chose Gated (true NoAlign, probe N=1.30/G=2.00); hawk_dove/s55: chose Gated (true NoAlign, probe N=1.12/G=2.00); hawk_dove/s47: chose Gated (true NoAlign, probe N=1.24/G=1.98); hawk_dove/s59: chose Gated (true NoAlign, probe N=1.04/G=2.00); hawk_dove/s54: chose Gated (true NoAlign, probe N=1.42/G=2.00); hawk_dove/s49: chose Gated (true NoAlign, probe N=1.20/G=1.96); hawk_dove/s45: chose Gated (true NoAlign, probe N=1.12/G=1.98); hawk_dove/s60: chose Gated (true NoAlign, probe N=1.48/G=2.00); hawk_dove/s61: chose Gated (true NoAlign, probe N=1.62/G=2.00); hawk_dove/s44: chose Gated (true NoAlign, probe N=1.36/G=1.96); hawk_dove/s50: chose Gated (true NoAlign, probe N=1.30/G=2.00); stag_hunt/s58: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s56: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s53: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s42: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s59: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s54: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s61: chose NoAlign (true Gated, probe N=3.00/G=3.00); chicken/s43: chose Gated (true NoAlign, probe N=2.16/G=3.00); chicken/s57: chose Gated (true NoAlign, probe N=2.28/G=2.64); chicken/s48: chose Gated (true NoAlign, probe N=2.04/G=3.00)
