# G1 — End-to-end 2-arm attainment bandit

cells: 120/120  |  metric: cooperation_payoff (commit phase, 20 ep for 2-player / 10 ep for public_goods)


## Headline (pre-registered: accuracy>=85% AND bandit>=SCA)

- selection accuracy: **58/120 = 48.3%** (offline probe-argmax reconstruction was 116/120=96.7%)

- bandit commit mean: **2.3333** vs SCA 2.2273 (Δ=+0.1060, p=0.4232, dz=+0.29) vs oracle 2.5069

- **verdict: PARTIAL/FAIL** (acc>=85%: no; bandit>=SCA: yes)

- probe exploration payoff: 2.4659 (commit 2.3333; probe-commit gap = -0.1325)


## Per-game

| game | grp | n | acc | bandit | oracle | frac Gated |
|---|---|---|---|---|---|---|
| chicken | anti | 20 | 25% | 2.185 | 2.401 | 100% |
| deadlock | anti | 20 | 0% | 1.443 | 2.154 | nan% |
| hawk_dove | anti | 20 | 100% | 1.990 | 1.996 | 100% |
| stag_hunt | coord | 20 | 55% | 2.966 | 2.999 | 100% |
| battle_of_the_sexes | coord | 20 | 100% | 2.875 | 2.848 | 100% |
| public_goods | boundary | 20 | 10% | 2.540 | 2.644 | 100% |

## Decision log (misses)

62 misses: stag_hunt/s51: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s56: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s46: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s55: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s54: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s45: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s60: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s44: chose NoAlign (true Gated, probe N=3.00/G=3.00); stag_hunt/s50: chose NoAlign (true Gated, probe N=3.00/G=3.00); deadlock/s43: chose NoAlign (true Gated, probe N=2.00/G=1.80); deadlock/s57: chose NoAlign (true Gated, probe N=2.00/G=1.72); deadlock/s48: chose NoAlign (true Gated, probe N=2.00/G=1.68); deadlock/s51: chose NoAlign (true Gated, probe N=2.00/G=1.80); deadlock/s58: chose NoAlign (true Gated, probe N=2.00/G=1.68); deadlock/s56: chose NoAlign (true Gated, probe N=2.00/G=1.80); deadlock/s46: chose NoAlign (true Gated, probe N=2.00/G=1.76); deadlock/s53: chose NoAlign (true Gated, probe N=2.00/G=1.76); deadlock/s42: chose NoAlign (true Gated, probe N=2.00/G=1.76); deadlock/s52: chose NoAlign (true Gated, probe N=2.00/G=1.80); deadlock/s55: chose NoAlign (true Gated, probe N=2.00/G=1.76); deadlock/s47: chose NoAlign (true Gated, probe N=2.00/G=1.80); deadlock/s59: chose NoAlign (true Gated, probe N=2.00/G=1.76); deadlock/s54: chose NoAlign (true Gated, probe N=2.00/G=1.76); deadlock/s49: chose NoAlign (true Gated, probe N=2.00/G=1.80); deadlock/s45: chose NoAlign (true Gated, probe N=2.00/G=1.76); deadlock/s60: chose NoAlign (true Gated, probe N=2.00/G=1.68); deadlock/s61: chose NoAlign (true Gated, probe N=2.00/G=1.68); deadlock/s44: chose NoAlign (true Gated, probe N=2.00/G=1.76); deadlock/s50: chose NoAlign (true Gated, probe N=2.00/G=1.80); chicken/s57: chose NoAlign (true NoToM, probe N=2.64/G=2.28)
