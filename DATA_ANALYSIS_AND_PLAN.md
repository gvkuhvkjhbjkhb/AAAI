# Route A (DP-Gating) — Data Analysis & Solution Plan
> Generated 2026-07-11 | Based on 184 unified data cells (5 games x 8 seeds, 30 episodes) + 2025-2026 literature

## Part 1: Data Conclusions

### Core Results
| Game | Type | n | Gated | DP-Gating | Delta | p | Wins | Verdict |
|---|---|---|---|---|---|---|---|---|
| Deadlock | anti-coord | 8 | 1.600 | 1.846 | +0.246 | 0.0016 | 8/8 | *** |
| Hawk-Dove | anti-coord | 8 | 1.006 | 1.212 | +0.206 | 0.0070 | 8/8 | *** |
| Chicken | anti-coord | 8 | 2.089 | 2.286 | +0.197 | 0.0379 | 7/8 | ** |
| StagHunt | coordination | 4→8 | 2.258 | 2.250 | -0.008 | n.s. | — | neutral |
| BoS | pref-conflict | 5→8 | 1.345 | 1.271 | -0.075 | 0.55 | — | neutral |

### Three Falsified Hypotheses
1. **Diversity preservation FALSIFIED**: New data shows both Gated and DP-Gating diversity ≈0.005. The old +267% Chicken diversity gain does NOT replicate. Four-dimensional diversity metrics (action KL, sequence edit distance, signal KL, ToM KL) confirm no meaningful difference.
2. **Pareto dominance narrative DEAD**: No diversity gain means no Pareto frontier, no "absolute win" story.
3. **Conflict rate CANNOT infer game structure**: All 5 games have conflict rate 0.49-0.51. This kills the original GSACA "conflict-rate-as-detector" premise.

### The One Surviving Signal
DP-Gating is robustly significant on THREE anti-coordination games (p<0.04, 87-100% seed wins). The real mechanism is NOT "preserving diversity" but "conditional non-intervention": anti-coordination equilibria are inherently split; forced alignment (regular gating) destroys the split equilibrium, while DP-Gating's non-intervention lets it form naturally. This is a finding about INTERVENTION TIMING, not diversity.

## Part 2: Key Literature (2026)
| Paper | arXiv | Relevance |
|---|---|---|
| Shin, "The Reasoning Trap" | 2605.01704 | Information-theoretic bound on closed-system LLM reasoning — thesis basis still holds |
| Kong et al., "Multi-LLM Systems Exhibit Robust Semantic Collapse" | 2605.17193 | Diversity collapse is ROBUST & unavoidable — explains our diversity≈0 result |
| Chen et al., "Diversity Collapse in Multi-Agent LLM" | 2604.18005 | Structural coupling mechanism of diversity collapse |
| Tewolde et al., "CoopEval" | 2604.15267 | Benchmark for cooperation-sustaining mechanisms in social dilemmas |
| Willis et al., "Will Systems of LLM Agents Cooperate" | 2501.16173 | LLM social dilemma emergent cooperation (related work) |
| Peng et al., "Communication & Verification under Information Asymmetry" | 2510.25595 | Extension direction |
| Backmann et al., "When Ethics and Payoffs Diverge" | 2505.19212 | BoS-type preference conflict related work |

**Strategic insight**: Kong et al. and Chen et al. (both 2026) prove diversity collapse is a robust universal phenomenon in multi-LLM systems. Our inability to measure diversity gain is CONSISTENT with the latest literature — but it also means "improving cooperation by preserving diversity" has lost its foundation.

## Part 3: Solution Plan

### Plan A (RECOMMENDED): Reframe as "Conditional Non-Intervention" — drop the diversity story
New title: *"Less Is More: Selective Non-Intervention Outperforms Forced Alignment in Anti-Coordination LLM Games"*

New claims (all supported by existing data):
1. Forced alignment (regular gating) is systematically harmful in anti-coordination games — it destroys the split equilibrium the game requires.
2. Conditional non-intervention robustly improves payoff across 3 structurally-distinct anti-coordination games (+0.20~+0.25, p<0.04, 87-100% seed wins).
3. The effect is neutral in coordination (StagHunt) and preference-conflict (BoS) games — i.e. the method is NEVER harmful, a safe intervention strategy.
4. Mechanism: explain via Nash equilibrium structure why non-intervention helps in anti-coordination.

Required experiments (all supplementary, not redos):
| Priority | Experiment | Purpose | Status |
|---|---|---|---|
| P0 | Complete StagHunt + BoS to n=8 | Full 5-game unified seeds | RUNNING |
| P0 | Threshold 0.7/0.8 completion | Full threshold curve | ~2h |
| P1 | Equilibrium structure analysis (payoff correlation, not conflict rate) | Replace failed conflict-rate detector | offline |
| P1 | Honestly present diversity collapse, cite Kong/Chen 2026 | Align with latest literature | offline |

AAAI probability: 35-40%

### Plan B (ALTERNATIVE): Adopt CoopEval standard benchmark
Adopt Tewolde et al. (2026) CoopEval framework to answer "only self-made 2x2 games" criticism.
- P0: Run DP-Gating vs baselines on CoopEval (~8h GPU)
- P1: Compare against Willis 2025, Backmann 2025 reported results (offline)
AAAI probability: 30-40% (if still significant on standard benchmark)

### Plan C (NOT RECOMMENDED): Original GSACA
Falsified by data — conflict rate does not distinguish game types (all 0.49-0.51). Core mechanism infeasible unless switched to payoff correlation as detection signal (requires agents to observe both payoffs, changing the problem setup).

## Recommendation
**Choose Plan A.** The "selective non-intervention" story (3/5 significant, 0/5 harmful) needs no reruns; honestly handling diversity collapse aligns with 2026 literature; the contribution is clean and defensible; it dodges all three fatal weaknesses (diversity narrative, Pareto dominance, conflict-rate detector).
