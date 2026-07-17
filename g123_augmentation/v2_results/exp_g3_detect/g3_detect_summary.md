# G3 — Stack-B online-detection validation (het_gsaca)

cells: 30/30  |  detection accuracy: **22/30**


## Per-cell (split sign + correctness)

| game | grp | seed | split | detected | oracle | correct |
|---|---|---|---|---|---|---|
| chicken | anti | 42 | +1.600 | anti_coord | anti_coord | OK |
| chicken | anti | 43 | +1.400 | anti_coord | anti_coord | OK |
| chicken | anti | 44 | +0.375 | anti_coord | anti_coord | OK |
| chicken | anti | 45 | +1.500 | anti_coord | anti_coord | OK |
| chicken | anti | 46 | +1.000 | anti_coord | anti_coord | OK |
| deadlock | anti | 42 | +0.048 | anti_coord | anti_coord | OK |
| deadlock | anti | 43 | +0.067 | anti_coord | anti_coord | OK |
| deadlock | anti | 44 | +0.000 | coord | anti_coord | MISS |
| deadlock | anti | 45 | +0.000 | coord | anti_coord | MISS |
| deadlock | anti | 46 | +0.095 | anti_coord | anti_coord | OK |
| hawk_dove | anti | 42 | +0.100 | anti_coord | anti_coord | OK |
| hawk_dove | anti | 43 | +0.000 | coord | anti_coord | MISS |
| hawk_dove | anti | 44 | +0.100 | anti_coord | anti_coord | OK |
| hawk_dove | anti | 45 | -0.071 | coord | anti_coord | MISS |
| hawk_dove | anti | 46 | -0.038 | coord | anti_coord | MISS |
| stag_hunt | coord | 42 | +0.000 | anti_coord | coord | MISS |
| stag_hunt | coord | 43 | -2.000 | coord | coord | OK |
| stag_hunt | coord | 44 | +0.000 | anti_coord | coord | MISS |
| stag_hunt | coord | 45 | -2.000 | coord | coord | OK |
| stag_hunt | coord | 46 | +0.000 | anti_coord | coord | MISS |
| battle_of_the_sexes | coord | 42 | -2.500 | coord | coord | OK |
| battle_of_the_sexes | coord | 43 | -2.500 | coord | coord | OK |
| battle_of_the_sexes | coord | 44 | -2.500 | coord | coord | OK |
| battle_of_the_sexes | coord | 45 | -2.500 | coord | coord | OK |
| battle_of_the_sexes | coord | 46 | -2.500 | coord | coord | OK |
| public_goods | boundary | 42 | -0.250 | coord | coord | OK |
| public_goods | boundary | 43 | -0.150 | coord | coord | OK |
| public_goods | boundary | 44 | -0.150 | coord | coord | OK |
| public_goods | boundary | 45 | -0.150 | coord | coord | OK |
| public_goods | boundary | 46 | -0.150 | coord | coord | OK |

## Per-game split (mean ± std) and accuracy

| game | grp | n | split mean | split std | acc |
|---|---|---|---|---|---|
| chicken | anti | 5 | +1.175 | 0.449 | 100% |
| deadlock | anti | 5 | +0.042 | 0.037 | 60% |
| hawk_dove | anti | 5 | +0.018 | 0.071 | 40% |
| stag_hunt | coord | 5 | -0.800 | 0.980 | 40% |
| battle_of_the_sexes | coord | 5 | -2.500 | 0.000 | 100% |
| public_goods | boundary | 5 | -0.170 | 0.040 | 100% |

## Verdict
- **22/30: misses present -> Prop 1 precondition (warm-up profile coverage) fails**

