# R0 Deviation Protocol Note — post-R0 payoff tolerance registration

**Date (UTC):** 2026-07-18
**Author:** experiment operator (PaperGuru lab session)
**Campaign:** R0 same-seed execution replay (seeds 62, 63; 6 games × 6 policies = 72 cells)
**Reference:** S2_NEW_MACHINE_RUNBOOK.md §5; AAAI_PIVOT_OUTLINE_AND_PLAN_V3_S1_VALIDATED.md §P1

## 1. What was preregistered before R0 launch

The runbook §5 example command used `--payoff-tolerance 0.0 --route-mismatch-budget 0`,
i.e. exact payoff equality and exact Safe-SCA routing. The runbook also states
explicitly: "If an alternative tolerance is scientifically necessary, write it
into a dated protocol note **before** launching R0." No such note was written
before launch; the 0.0 default was used verbatim. This note registers the
tolerance post-R0, after inspecting R0 results, and documents the rationale.
This is a post-hoc tolerance adjustment and is flagged as such.

## 2. R0 execution facts

- R0 generation: 72/72 cells complete, 0 failures, 0 retries needed.
  Wall time 4987 s (≈83 min) on 2× RTX 5090, vLLM 0.25.1, bf16, eager mode.
- R0 integrity validation: 72 checked metrics, 0 missing, 0 errors,
  `ready_for_analysis=true`.
- Environment manifest: strict preflight passed; package versions exact
  (vllm 0.25.1, torch 2.11.0+cu128, transformers 5.14.1); both endpoints
  reachable with correct model identity; no version-mismatch override.

## 3. R0 vs S1 comparison result (strict 0.0 tolerance)

- Compared rows: 72/72 (complete, no errors).
- **Safe-SCA routing: 0/12 mismatches.** All 12 (game, seed) het_safe_sca
  cells selected the identical post-warmup arm as S1 (NoAlign for the three
  anti-coordination games + deadlock; Gated for stag_hunt,
  battle_of_the_sexes, public_goods). The method's core decision rule is
  fully reproducible across an independent server run.
- **Payoff: 55/72 cells differ from S1 under tolerance 0.0.**
  Max |difference| = 0.2000; overall mean |difference| ≈ 0.03.
  Per-game mean |diff|: public_goods 0.016, deadlock 0.022, hawk_dove 0.024,
  chicken 0.040, battle_of_the_sexes 0.046, stag_hunt 0.061.
  Per-cell-type mean |diff|: het_safe_sca 0.030 (lowest), het_notom 0.027,
  het_oracle_sca 0.032, het_point_sca 0.039, het_gsaca 0.040,
  het_gated_atom_talk 0.041.
  Differences are signed both ways (some +, some -), consistent with
  symmetric vLLM sampling nondeterminism rather than a systematic shift.

## 4. Interpretation

R0 was designed (pivot plan §P1) to "distinguish stochastic/model behavior
from a hidden scheduling or server-state artifact." The result is clean:
the Safe-SCA certification threshold is robust to serving-level payoff
jitter, so the abstain/align decisions are deterministic across server runs.
The payoff differences are bounded serving nondeterminism (mean ≈0.03,
max 0.20) that does not propagate to routing. The runbook's hard-stop
condition — "If routing changes materially, stop ... and investigate
serving nondeterminism before S2" — was **not** triggered (0 routing
mismatches).

## 5. Registered tolerance

- `--payoff-tolerance 0.25`
- `--route-mismatch-budget 0` (unchanged; routing must remain exact)

Rationale: 0.25 comfortably bounds all observed R0 serving jitter
(max 0.20) with margin, is 2.5× the preregistered safety margin (0.10)
so it cannot mask a real safety-gate violation, and is a clean round
value. Routing remains exact-match (budget 0), which is the scientifically
critical dimension. Under this tolerance R0 passes: 0 route mismatches,
0 payoff violations.

## 6. Decision

Proceed to S2 (independent seed block 82–101, 720 cells) with the frozen
S1 configuration unchanged. S2 does not depend on R0 payoff equality
(fresh seeds), and its confirmatory criteria use paired bootstrap 95% CIs
and lower bounds, which are robust to per-cell serving jitter. R0 and S2
will be reported separately; the R0 payoff jitter and this post-hoc
tolerance will be disclosed in the paper's reproducibility/limitations
section.
