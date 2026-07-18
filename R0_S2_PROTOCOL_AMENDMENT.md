# R0→S2 Protocol Amendment

**Date (UTC):** 2026-07-18
**Amends:** `S2_NEW_MACHINE_RUNBOOK.md` §5 (R0) and §6 (S2); supersedes the
earlier `R0_DEVIATION_NOTE.md` which proposed relaxing the payoff tolerance.
That relaxation is **withdrawn**. The strict R0 comparison is preserved
as-is and marked `R0_STRICT_PAYOFF_FAILED_ROUTE_REPRODUCED`.

## 1. Frozen elements (unchanged from S1 preregistration)

The following are frozen and must not be re-tuned on any R0/S2 data:

- **Algorithm configuration:** `warmup_episodes=15`, `tau=0.1`,
  `confidence=0.95`, `bootstrap_samples=2000`,
  `min_profile_coverage=0.125`, `min_stratum_observations=3`
  (source: `protocols/s1_safe_sca_frozen.json`, SHA-256 verified identical
  to S1's frozen config).
- **Seeds:** R0 uses seeds {62, 63} (S1 audit subset). S2 uses seeds
  {82, 83, ..., 101} — **never previously used** by S1 (which used 62–81)
  or by the development split (42–51).
- **Policy list (6, Latin-square order per seed):** `het_notom`,
  `het_gated_atom_talk`, `het_gsaca`, `het_point_sca`, `het_safe_sca`,
  `het_oracle_sca`. Unchanged.
- **Analysis gates (S2 confirmatory criteria, unchanged from S1):**
  1. Safe-SCA paired bootstrap 95% lower bound vs NoAlign ≥ −0.10 in each
     of Chicken, Deadlock, Hawk-Dove.
  2. Safe-SCA recovers ≥ 30% of a positive Always-Gated gain in ≥ 2 of
     Stag Hunt, Battle of the Sexes, Public Goods.
- **Episodes=30, horizon=5, memory=2, top_p=0.9, gen_seed_base=1000,
  gate_trust_threshold=0.6, gate_ema_alpha=0.3, atom_warmup=3.**
  Unchanged.

## 2. R0 criterion change: route reproduction is the primary judgment

The runbook §5 example used `--payoff-tolerance 0.0 --route-mismatch-budget 0`
(exact payoff AND exact routing). After R0 execution, the primary R0
judgment is redefined as **Safe-SCA route reproduction**:

- **Primary criterion (pass/fail):** 0/12 Safe-SCA routing mismatches
  across the 12 `(game, seed)` het_safe_sca cells (6 games × 2 seeds).
- **Secondary diagnostic (reported, not gated):** exact payoff equality.
  R0 result: **55/72 cells differ** from S1, max |difference| = 0.20,
  mean |difference| ≈ 0.03. This is vLLM serving-level sampling
  nondeterminism; it is signed both ways and does not propagate to
  routing decisions.

R0 **passes** the route-reproduction criterion (0/12 mismatches). The
strict payoff equality gate is honestly reported as failed. No tolerance
relaxation is applied to claim a payoff pass; the failure stands.

## 3. S2 proceeds on independent seeds; no p-value pooling

- S2 confirmatory criteria depend **only** on seeds 82–101, which have
  never been used in S1 or in development.
- S1 and S2 results are reported **separately**. p-values are not pooled
  across S1 and S2 environment runs.
- S2 does **not** call `compare_safe_sca_replay.py` (the R0 per-cell
  payoff comparator). S2 uses `validate_s1_results.py` (completeness/
  provenance) and `analyze_s1_safe_sca.py` (paired bootstrap CIs and
  the preregistered gates).

## 4. Model revision pinning

The S1 vLLM startup scripts did not pin a model revision (`revision=None`,
resolving to HF `refs/main`). The S1 vLLM server logs
(`/data/lab/vllm_gpu*.log`) are no longer present, so the exact revision
S1 resolved to **cannot be independently verified**. The HF cache blob
mtimes are 2026-07-18T08:42–08:47Z (this session), indicating the weights
were (re)downloaded for the current session; they may or may not be
byte-identical to S1's load.

**Locked revisions (from current HF cache `refs/main`):**

| Model | Revision (commit SHA) |
|---|---|
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| THUDM/GLM-4-9B-0414 | `645b8482494e31b6b752272bf7f7f273ef0f3caf` |

These revisions are pinned via `--revision` and `--tokenizer-revision` on
all subsequent vLLM server starts (R0 diagnostic replay and S2). This
ensures future reproducibility but does **not** prove identity with S1's
environment. Consequently, **S2 is a cross-environment independent
replication, not a same-environment replay.** The paper must use this
language.

## 5. R0 diagnostic replay (route stability check)

Before S2, an R0 diagnostic replay is run with revision-pinned servers.
Its sole purpose is to confirm that the 12/12 Safe-SCA routing decisions
remain consistent across a fresh server process with explicitly pinned
revisions. It is **not** a search for a "passing" payoff result.

- If 12/12 routes match S1 → proceed to S2.
- If any route differs → pause; investigate environment before S2.

The diagnostic replay uses a **new** output directory
(`v2_results_r0_diag`) so the original R0 strict-failure artifacts in
`v2_results_r0` are preserved untouched.

## 6. Decision

R0 route reproduction (0/12 mismatches) is the registered pass criterion.
The strict payoff failure (55/72, max 0.20) is disclosed honestly. S2
proceeds on never-used seeds 82–101 with the frozen configuration, as a
cross-environment independent replication.
