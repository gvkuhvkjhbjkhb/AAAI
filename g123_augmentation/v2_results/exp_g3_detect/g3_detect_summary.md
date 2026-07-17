# G3 — Stack-B online-detection validation (het_gsaca)

cells: 28/30  |  detection accuracy: **21/28**


## Per-cell (split sign + correctness)

| game | grp | seed | split | detected | oracle | correct |
|---|---|---|---|---|---|---|
| chicken | anti | 42 | +0.333 | anti_coord | anti_coord | OK |
| chicken | anti | 43 | +1.154 | anti_coord | anti_coord | OK |
| chicken | anti | 44 | +1.091 | anti_coord | anti_coord | OK |
| chicken | anti | 45 | +0.750 | anti_coord | anti_coord | OK |
| chicken | anti | 46 | +1.286 | anti_coord | anti_coord | OK |
| deadlock | anti | 42 | +0.050 | anti_coord | anti_coord | OK |
| deadlock | anti | 43 | +0.000 | coord | anti_coord | MISS |
| deadlock | anti | 44 | +0.000 | coord | anti_coord | MISS |
| deadlock | anti | 45 | +0.000 | coord | anti_coord | MISS |
| deadlock | anti | 46 | +0.111 | anti_coord | anti_coord | OK |
| hawk_dove | anti | 42 | -0.500 | coord | anti_coord | MISS |
| hawk_dove | anti | 43 | +0.300 | anti_coord | anti_coord | OK |
| hawk_dove | anti | 44 | -0.071 | coord | anti_coord | MISS |
| hawk_dove | anti | 45 | +0.318 | anti_coord | anti_coord | OK |
| hawk_dove | anti | 46 | -0.071 | coord | anti_coord | MISS |
| stag_hunt | coord | 42 | -1.944 | coord | coord | OK |
| stag_hunt | coord | 43 | -1.944 | coord | coord | OK |
| stag_hunt | coord | 44 | +0.000 | anti_coord | coord | MISS |
| stag_hunt | coord | 45 | -1.895 | coord | coord | OK |
| stag_hunt | coord | 46 | -1.957 | coord | coord | OK |
| battle_of_the_sexes | coord | 42 | -2.500 | coord | coord | OK |
| battle_of_the_sexes | coord | 43 | -2.500 | coord | coord | OK |
| battle_of_the_sexes | coord | 44 | -2.500 | coord | coord | OK |
| battle_of_the_sexes | coord | 45 | -2.500 | coord | coord | OK |
| battle_of_the_sexes | coord | 46 | -2.500 | coord | coord | OK |
| public_goods | boundary | 43 | -0.167 | coord | coord | OK |
| public_goods | boundary | 44 | -0.300 | coord | coord | OK |
| public_goods | boundary | 45 | -0.225 | coord | coord | OK |

## Per-game split (mean ± std) and accuracy

| game | grp | n | split mean | split std | acc |
|---|---|---|---|---|---|
| chicken | anti | 5 | +0.923 | 0.344 | 100% |
| deadlock | anti | 5 | +0.032 | 0.044 | 40% |
| hawk_dove | anti | 5 | -0.005 | 0.300 | 40% |
| stag_hunt | coord | 5 | -1.548 | 0.774 | 80% |
| battle_of_the_sexes | coord | 5 | -2.500 | 0.000 | 100% |
| public_goods | boundary | 3 | -0.231 | 0.055 | 100% |

## Verdict
- **21/28: misses present -> Prop 1 precondition (warm-up profile coverage) fails**

