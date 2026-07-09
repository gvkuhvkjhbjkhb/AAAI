# HetToM Round-2 Results: Cheap-talk, A-ToM, and Extended Seeds

> Experiment date: 2026-07-09
> Game: Stag Hunt, horizon=5, memory=2, 20 episodes/cell for new seeds
> Models: Qwen2.5-3B-Instruct (homogeneous), Qwen2.5-3B + Qwen2.5-1.5B (heterogeneous)
> Seeds: base cells use 6 seeds (round-1 seeds 1-3 + new seeds 4-6); round-2 mechanism cells use 3 seeds (4-6).

## 1. Aggregate results

| Cell | Diversity | Payoff | ToM Acc | n |
|---|---:|---:|---:|---:|
| hom_notom | 0.132 | **2.346** | — | 6 |
| hom_tom | 0.085 | 2.188 | 0.673 | 6 |
| het_notom | **1.712** | 0.595 | — | 6 |
| het_tom | 0.541 | 1.259 | 0.538 | 6 |
| hom_notom_talk | 0.000 | **2.527** | — | 3 |
| hom_tom_talk | 0.103 | 2.413 | 0.710 | 3 |
| het_notom_talk | 0.005 | **1.760** | — | 3 |
| het_tom_talk | 1.028 | 0.890 | 0.518 | 3 |
| hom_atom | 0.029 | 2.197 | 0.652 | 3 |
| het_atom | 0.470 | 1.260 | 0.542 | 3 |
| het_atom_talk | 0.919 | 0.990 | 0.498 | 3 |

## 2. Main conclusions

### 2.1 The round-1 finding becomes statistically significant

Expanding the base cells from 3 to 6 seeds converts the previously directional result into a statistically significant pattern. Heterogeneity still breaks the diversity trap, with `het_notom` diversity 1.712 versus `hom_notom` 0.132 (p=0.002), but it still destroys cooperation: `het_notom` payoff 0.595 versus `hom_notom` 2.346 (p=0.002, r=-1.0). Fixed ToM significantly rescues heterogeneous cooperation: `het_tom` improves payoff from 0.595 to 1.259 (delta=+0.664, p=0.002, r=+1.0), while reducing diversity from 1.712 to 0.541. This confirms the round-1 story with stronger evidence: diversity without alignment is destructive, while ToM is an alignment bridge.

### 2.2 Cheap-talk is the strongest positive mechanism

Cheap-talk is the clear round-2 success. In heterogeneous teams without ToM, cheap-talk improves payoff from 0.595 to 1.760 (delta=+1.165, Mann-Whitney p=0.024, r=+1.0). This is the only new mechanism that directly and significantly rescues heterogeneous cooperation. It also collapses diversity from 1.712 to 0.005, showing that its mechanism is not "better diversity" but rapid convention formation: agents announce intentions and coordinate on a shared action.

### 2.3 Fixed ToM and cheap-talk do not compose naively

Adding cheap-talk on top of fixed ToM hurts rather than helps in this implementation: `het_tom_talk` payoff is 0.890 versus `het_tom` 1.259 (delta=-0.369). The likely mechanism is cognitive interference: agents first predict teammate actions through ToM and then condition on explicit signals; when those signals and predictions disagree, the current prompt does not specify which source should dominate. This produces high diversity (1.028) but lower coordination.

### 2.4 A-ToM did not outperform fixed ToM in this quick test

`het_atom` is statistically indistinguishable from `het_tom` (payoff 1.260 vs 1.259, p=1.000). The simple hit-rate based adaptation rule neither hurts nor helps. This does not falsify adaptive ToM as a concept, but it shows that our implementation is too weak: only 20 episodes are available per cell, and the adjustment rule is coarse (`<0.4` increase order, `>0.75` decrease order). The literature-motivated idea remains plausible, but it needs a richer order-estimation model.

### 2.5 The combined method is negative

The full `het_atom_talk` combination underperforms both the homogeneous baseline and fixed `het_tom` (payoff 0.990 vs 2.346 baseline; payoff 0.990 vs 1.259 fixed ToM). Diversity is high (0.919), so the combination reintroduces coordination instability. This indicates that "more mechanisms" is not better: talk and ToM must be integrated hierarchically, not simply concatenated in prompts.

## 3. Pre-registered decision

The strict trap-breaking rule is not met. Both `het_tom` and `het_atom_talk` have diversity above `hom_notom`, but neither has payoff above `hom_notom`. The correct paper framing remains PARTIAL/POSITIVE-MECHANISM rather than full method victory.

The publishable claim is now stronger and statistically supported:

> In heterogeneous LLM teams, diversity without alignment destroys cooperation; fixed ToM partially restores cooperation; cheap-talk is an even stronger alignment mechanism that significantly rescues heterogeneous cooperation, but ToM and communication interfere unless explicitly arbitrated.

## 4. Actionable next design change

The next method should not be `ToM + talk` by prompt concatenation. It should be a gated hierarchy:

1. Cheap-talk first establishes a public commitment / convention.
2. ToM is used only to predict whether the signal is credible or likely to be followed.
3. The final action follows signal if credibility is high; otherwise falls back to ToM prediction.

This yields a new `het_gated_talk_tom` cell that directly addresses the failure of `het_tom_talk` and `het_atom_talk`.

## 5. Files

- `results/hettom_layer1_round2/stag_hunt/aggregated.csv`
- `results/hettom_layer1_round2/stag_hunt/analysis_report.txt`
- `results/hettom_layer1_round2/stag_hunt/seed_<n>/<cell>/metrics.json`
- `hettom_experiments/run_round2_quick.sh`
- `hettom_experiments/run_missing.py`
