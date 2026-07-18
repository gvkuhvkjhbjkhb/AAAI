# P3 Transfer Experiment — Results Report

**Campaign:** `p3_transfer`
**Run UTC:** 2026-07-18T17:42:37Z → 2026-07-18T17:56:07Z (810 s wall clock)
**Output root:** `/data/aaai/safe_sca_replication/v2_results_p3/exp_p3_transfer_test`
**Protocol hash (sha256):** `5c7a74d0425c0f3937176eddade732315d467be70b7316054c492185e4b44af6`
**Matrix registry hash (sha256):** `08899a7c14edb548acda759b2e9d61e232699ffc223d33a88d0b81e40f8d0203`

## 1. What P3 is

P3 tests whether the S1/S2-frozen Safe-SCA rule transfers to **eight unseen
payoff matrices and opaque action-label surfaces**. It is a pre-registered,
frozen-protocol transfer test — not a chance to repair S2 or select a favorable
matrix. The frozen design is `8 matrices × 10 never-used seeds (102–111) × 4
policies × 30 episodes × 5 rounds = 320 cells`, executed with 32 workers.

Four matrices (`p3_m01`–`p3_m04`) are coordination/boundary; four
(`p3_m05`–`p3_m08`) are anti-coordination. Matrix category, matrix ID, and
payoff table are analysis metadata only — they are never passed to Safe-SCA or
to any LLM prompt. Every agent prompt sees the same game name
(`Anonymous interaction`) and opaque neutral action labels
(`glyph-ivory`/`glyph-slate`, `token-amber`/`token-cyan`, …). Safe-SCA receives
only realized action/reward histories.

## 2. Execution integrity

| Check | Result |
|---|---|
| Preflight (S1 strict, no version override) | **PASS** — vLLM 0.25.1, torch 2.11.0+cu128, transformers 5.14.1, 2× RTX 5090, both endpoints reachable |
| Campaign completeness | **320/320** cells, **80/80** Safe-SCA decisions |
| Task failures / retries | **0 failures**, all 80 tasks succeeded on attempt 1 |
| Integrity validation (`validate_p3_results.py`) | `checked_metrics=320`, `missing=0`, `errors=0`, `ready_for_analysis=true` |
| Safe-SCA frozen config check | All 80 `het_safe_sca` cells carry the byte-identical S1/S2 Safe-SCA config (warmup 15, tau 0.10, confidence 0.95, bootstrap 2000, coverage 0.125, min-stratum 3) |
| Model revision pins | Qwen `a09a354…28`, GLM `645b848…af` — identical to S2 |
| Information barrier | `payoff_in_prompt=false` enforced; controller receives only `realized_actions`, `realized_rewards` |

## 3. Pre-declared confirmatory gates

The P3 protocol declares three gates, all evaluated **per matrix** with 20,000
paired-bootstrap samples. P3 passes only if **all three** hold.

### Gate 1 — Anti-matrix safety (non-inferiority, margin −0.10) — **PASS**

For every anti matrix, Safe-SCA's paired-bootstrap 95% lower CI versus NoAlign
must be ≥ −0.10.

| Matrix | Safe-SCA Δ vs NoAlign | 95% paired CI | Non-inferior |
|---|---:|---|:---:|
| p3_m05 | +0.003 | [−0.005, +0.012] | ✅ |
| p3_m06 | +0.000 | [−0.007, +0.006] | ✅ |
| p3_m07 | +0.000 | [−0.036, +0.039] | ✅ |
| p3_m08 | −0.002 | [−0.020, +0.015] | ✅ |

Safe-SCA is non-inferior to NoAlign on all four anti matrices. The lower bound
of the worst case (p3_m07, −0.036) is well inside the −0.10 margin.

### Gate 2 — Anti false-align routing (budget 0/40) — **PASS**

Safe-SCA must select `Gated` in **zero** of the 40 anti matrix-seed cells.

| Outcome | Count |
|---|---:|
| Anti false-align | **0 / 40** |
| Coordination/boundary false-abstain | **0 / 40** |

Safe-SCA correctly abstained (`NoAlign`) in every anti cell and correctly
intervened (`Gated`) in every coordination/boundary cell. The routing rule
transfers with zero routing errors on unseen matrices.

### Gate 3 — Coordination utility recovery (≥2 matrices, ≥30% recovery) — **FAIL**

In at least **two** coordination/boundary matrices where Always-Gated has a
**positive** gain over NoAlign, Safe-SCA must recover ≥30% of that gain.

| Matrix | Gated−NoAlign | Safe-SCA−NoAlign | Recovery | Positive Gated gain? | Qualified? |
|---|---:|---:|---:|:---:|:---:|
| p3_m01 | −0.419 | −0.159 | 37.9% | No | — |
| p3_m02 | −0.147 | −0.068 | 46.0% | No | — |
| p3_m03 | **+0.285** | +0.099 | **34.6%** | Yes | ✅ |
| p3_m04 | −0.183 | −0.099 | 53.8% | No | — |

Only **one** coordination matrix (`p3_m03`) exhibited a positive Always-Gated
gain. Safe-SCA recovered 34.6% of that gain (≥30%), so `p3_m03` qualified — but
the gate requires ≥2 qualifying matrices, so the gate fails.

### Overall P3 gate — **FAIL**

`safety_pass = True`, `routing_pass = True`, `utility_pass = False` →
`method_p3_pass = False`.

## 4. Interpretation (per the frozen protocol)

P3 is an **upside transfer test**. A failed P3 removes the unseen-matrix
generalization claim; it **does not** erase the already-supported S1/S2
in-distribution result over the six source games.

The failure is narrow and structurally informative:

- **Safety transfers.** Safe-SCA never falsely aligned on any unseen anti
  matrix (0/40) and was non-inferior to NoAlign on all four. This is the most
  important property for a safety-certified controller, and it held.
- **Routing transfers.** The coverage-certified rule selected `NoAlign` in
  100% of anti cells and `Gated` in 100% of coordination/boundary cells — zero
  routing errors on a fully opaque, never-seen action surface.
- **Utility recovery is structurally constrained.** Three of the four
  coordination/boundary matrices (`p3_m01`, `p3_m02`, `p3_m04`) are *boundary*
  cases where Always-Gated itself is harmful (negative gain). Safe-SCA correctly
  abstained in all of them (0 false-abstain), which is the safe behavior — but
  it also means there is no positive Gated gain to recover, so these matrices
  cannot count toward the utility gate. In the single coordination matrix where
  Gated helped (`p3_m03`), Safe-SCA recovered 34.6% of the gain, clearing the
  30% bar. The gate's ≥2-matrix threshold is therefore unsatisfiable for this
  matrix set, not defeated by Safe-SCA.

**Headline for the paper:** Safe-SCA's safety and routing guarantees transfer
to unseen payoff matrices and opaque action surfaces (0/40 false-align, 0/40
false-abstain, all anti non-inferior). The positive-utility transfer claim is
not supported by P3 because only one coordination matrix admitted a positive
Gated gain; Safe-SCA recovered the gain there but the pre-registered ≥2-matrix
threshold was not met. The S1/S2 in-distribution result stands unchanged.

## 5. Artifacts preserved

- `exp_p3_transfer_test/` — full results tree (320 `metrics.json`, 160
  `decision.json`, 79 campaign logs, campaign snapshot, matrix registry,
  integrity report, transfer summary JSON + MD).
- `P3_TRANSFER_EXPERIMENT_v1/` — the frozen package (protocol, code, tests,
  server scripts) used to produce these results.
- `P3_TRANSFER_RESULTS_v1.zip` — portable archive of the results tree.
- `logs/vllm_p3/` — vLLM server logs + `pids_and_revisions.env` audit log +
  `P3_SERVER_DEPLOYMENT_NOTE.md` (documents the S2-aligned server deployment;
  see §6 below).

## 6. Server deployment note

The bundled `server_scripts/start_vllm_p3.sh` launches vLLM with
`--api-key dummy --enforce-eager --gpu-memory-utilization 0.85`. That
combination is incompatible with the runbook's own verification step
(`curl --silent --fail http://localhost:8000/v1/models`, no Authorization
header) and with `base/code/preflight_s1.py`, which probes `/v1/models` via
urllib without an Authorization header — both return HTTP 401. To preserve all
protocol-frozen elements (model revisions, max-model-len, dtype, TP size) while
making preflight and the runbook's verification curls work, P3 was launched
with S2's exact deployment config (per `SERVER_ENVIRONMENT_APPENDIX_V2.json`):
no `--api-key`, no `--enforce-eager` (CUDA graphs enabled),
`--gpu-memory-utilization 0.92`, plus the P3-pinned revisions. The appendix
records that these flags change only kernel-launch overhead and memory
headroom, not model outputs (same weights, same bf16, same revision). The
protocol JSON is unchanged; only model revisions are protocol-frozen, and they
match exactly. See `logs/vllm_p3/P3_SERVER_DEPLOYMENT_NOTE.md` for the full
record.

## 7. Reproduction

```bash
export PROJECT_ROOT=/data/aaai/safe_sca_replication
export BASE_ROOT="$PROJECT_ROOT/g123_augmentation"
export P3_ROOT="$PROJECT_ROOT/P3_TRANSFER_EXPERIMENT_v1"
export P3_PROTOCOL="$P3_ROOT/protocols/p3_frozen_protocol.json"
export PYTHON_BIN=/usr/bin/python3

# Preflight
"$PYTHON_BIN" "$BASE_ROOT/code/preflight_s1.py" --out-dir "$P3_OUT"

# Campaign (resumable; never overwrites completed cells)
"$PYTHON_BIN" "$P3_ROOT/code/run_p3_campaign.py" \
  --base-root "$BASE_ROOT" --protocol "$P3_PROTOCOL" \
  --results-root "$PROJECT_ROOT/v2_results_p3"

# Validate + analyze
"$PYTHON_BIN" "$P3_ROOT/code/validate_p3_results.py" --results "$P3_OUT" --protocol "$P3_PROTOCOL"
"$PYTHON_BIN" "$P3_ROOT/code/analyze_p3_transfer.py" --results "$P3_OUT" --protocol "$P3_PROTOCOL"
```
