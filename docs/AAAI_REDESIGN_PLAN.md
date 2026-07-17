# AAAI Method Redesign Plan

## Root Cause of Prior Failure (Rounds 8-14)

| Problem | Old Method | Consequence |
|---|---|---|
| Non-potential shaping | `rewards[:, :terminal_t+1] -= penalty` applied per-step | Alters optimal policy (violates PBS, Ng et al. 1999), causing seed sensitivity |
| Backward smearing | Failure diagnosed post-episode, penalty smeared to all steps | Credit assignment error: good and bad steps equally penalized |
| Adaptive = scalar multiplier | type_weight × confidence × phase_weight scales penalty | No structural difference from uniform/random budget-matched controls |
| Controls too easy to match | Matching penalty total replicates effect | Adaptive mechanism has no independent contribution |

## Three Redesign Tracks

### Track A: Outcome-Contrast Terminal Shaping (implemented, pilot running)

Modifies only the terminal transition: success episodes get terminal bonus, failure episodes get terminal penalty, no per-step smearing.

- `success_bonus`: top-30% return episodes receive terminal bonus
- `terminal_failure`: bottom-30% return episodes receive terminal penalty
- `outcome_contrast`: both simultaneously

**Why potentially more robust**: old method applies 0.0003 per step × 50 steps = 0.015 total penalty density; new method applies only on 1 transition, far less value distortion.

**Limitation**: still not strictly potential-based, may still alter optimal policy.

**Status**: implemented, smoke-tested, 8-seed pilot running (Round 15).

### Track B: Failure-Aware Potential-Based Reward Shaping (main contribution)

**Theory**: Ng et al. (1999) proved F(s,s') = γφ(s') - φ(s) does not change optimal policy.

**Method**:
```
φ(s) = λ · Σ_k w_k · φ_k(s)

Features:
  φ_coop(s)    = -mean_pairwise_distance(agents)
  φ_explore(s) = state_novelty(visited_cells)
  φ_target(s)  = direction_agreement(agents)

Adaptive weight update:
  failure diagnosis type k → w_k ← w_k + α
  success episode          → w_k ← w_k · (1-β)

Per-step shaping:
  F_t = γ · φ(s_{t+1}) - φ(s_t)
  reward_t += F_t
```

**Why AAAI-level**:
1. Theoretical guarantee: PBS preserves optimal policy → eliminates seed sensitivity
2. Novel combination: failure diagnosis + PBRS in cooperative MARL
3. Strong control structure: budget-matched controls cannot replicate potential structure
4. Failure diagnosis is meaningful: drives potential feature weights, not scalar multiplier

**Control design (mechanism isolation)**:

| Control | Isolates |
|---|---|
| baseline | no shaping reference |
| FA-PBS | full method |
| PBS-fixed | fixed uniform weights, no diagnosis → isolates adaptation |
| PBS-random-weights | random weights same magnitude → isolates diagnosis |
| PBS-random-features | random potential function → isolates feature structure |
| old-penalty | non-PBS → proves PBS importance |

### Track C: Counterfactual Difference Rewards (backup if time allows)

COMA-style counterfactual baseline per agent, failure diagnosis identifies which agent's counterfactual to emphasize.

## Execution Strategy

```
Track A pilot (running, ~4.5h)
    ├── 8-seed positive → extend 16-seed + controls
    └── 8-seed negative → negative result in paper

Track B implementation (parallel, ~3h)
    ├── Implement φ_k feature extraction
    ├── Implement adaptive weight update
    ├── Integrate into reward_intervention.py
    └── smoke test

Track B pilot (after Track A, ~5h)
    ├── 8-seed FA-PBS vs controls
    └── positive → extend 16-seed + all controls (~8h)

Final decision:
    Track A or B positive at 16-seed → AAAI positive method paper
    Both fail → AAAI empirical study paper (full negative evidence)
```

## Timeline

| Phase | Content | Est. Time |
|---|---|---|
| Track A pilot | 40 runs, 8 seeds | ~4.5h (running) |
| Track B implementation | PBRS code + integration | ~3h |
| Track B smoke | validation | ~15min |
| Track B pilot | 40 runs, 8 seeds | ~5h |
| 16-seed extension | winner track + controls | ~8h |
| Analysis + packaging | report + GitHub | ~1h |
| **Total** | | **~22h (2 days)** |
