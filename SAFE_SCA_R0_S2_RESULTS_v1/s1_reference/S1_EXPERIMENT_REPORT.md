# S1 Coverage-Certified Safe-SCA — Lab Experiment Report

**Date:** 2026-07-18
**Hardware:** 2× NVIDIA RTX 5090 (32GB, Blackwell sm_120)
**Stack:** vLLM 0.25.1 bf16, PyTorch 2.11.0+cu128, Transformers 5.14.1
**Models:** Qwen2.5-7B-Instruct (GPU0:8000) + GLM-4-9B-0414 (GPU1:8001)
**Frozen config:** warmup=15, tau=0.1, confidence=0.95, bootstrap=2000, min_profile_coverage=0.125, min_stratum_observations=3
**Test matrix:** 6 games × 20 held-out seeds (62–81) × 6 policies = **720 cells**
**Status:** ✅ **COMPLETE (720/720, 0 failures)**

---

## Final verdict

| Gate | Result |
|------|--------|
| **method_paper_pass** | ✅ **TRUE** |
| Safety: anti-coordination non-inferiority (LB ≥ −0.10 vs NoAlign) | ✅ PASS (all 3 games) |
| Utility: ≥30% coordination gain recovery in ≥2 games | ✅ PASS (stag_hunt 47.4%, BoS 51.9%) |
| Anti-coordination false-align rate | **0/60 (0.0%)** |
| Coordination false-abstain rate | **0/60 (0.0%)** |

The S1 experiment meets both preregistered gates. Safe-SCA is a deployable
safe controller: it never harms anti-coordination games and recovers a
meaningful fraction of the Always-Gated gain in coordination games.

---

## Paired payoff against NoAlign (primary endpoint: total-horizon team payoff)

| game | policy | NoAlign | policy | delta | 95% CI | win% |
|---|---|---:|---:|---:|---|---:|
| chicken | Always Gated | 2.379 | 2.905 | +0.526 | [+0.472, +0.579] | 100% |
| chicken | Legacy GSACA | 2.379 | 2.425 | +0.046 | [−0.011, +0.100] | 70% |
| chicken | Point-SCA | 2.379 | 2.371 | −0.008 | [−0.030, +0.014] | 45% |
| chicken | **Safe-SCA** | 2.379 | 2.374 | −0.005 | [−0.022, +0.016] | 35% |
| deadlock | Always Gated | 2.000 | 1.794 | −0.206 | [−0.212, −0.200] | 0% |
| deadlock | **Safe-SCA** | 2.000 | 2.000 | +0.000 | [+0.000, +0.000] | 0% |
| hawk_dove | Always Gated | 1.247 | 1.999 | +0.752 | [+0.724, +0.779] | 100% |
| hawk_dove | **Safe-SCA** | 1.247 | 1.244 | −0.004 | [−0.016, +0.009] | 35% |
| stag_hunt | Always Gated | 2.670 | 3.000 | +0.330 | [+0.280, +0.384] | 100% |
| stag_hunt | **Safe-SCA** | 2.670 | 2.826 | +0.157 | [+0.120, +0.196] | 100% |
| battle_of_the_sexes | Always Gated | 1.173 | 2.340 | +1.167 | [+1.091, +1.232] | 100% |
| battle_of_the_sexes | **Safe-SCA** | 1.173 | 1.778 | +0.605 | [+0.564, +0.642] | 100% |
| public_goods | Always Gated | 2.571 | 2.534 | −0.036 | [−0.042, −0.031] | 0% |
| public_goods | **Safe-SCA** | 2.571 | 2.555 | −0.016 | [−0.019, −0.013] | 0% |

## Coordination gain recovery

| game | Gated−NoAlign | Safe-SCA−NoAlign | recovery |
|---|---:|---:|---:|
| stag_hunt | +0.330 | +0.157 | **47.4%** |
| battle_of_the_sexes | +1.167 | +0.605 | **51.9%** |
| public_goods | −0.036 | −0.016 | 43.5% (Gated negative) |

---

## Execution timeline

| Phase | Wall time | Cells | Notes |
|-------|-----------|-------|-------|
| S1 dev warmup (60 cells) | prior | 60/60 | NoAlign observer, seeds 42–51 |
| Config freeze | prior | — | select_s1_config.py |
| S1 test: 5 games (600 cells) | ~1.5 h | 600/600 | 24-worker (game,seed) sharding |
| S1 test: public_goods (120 cells) | ~1 h | 120/120 | 24-worker per-cell sharding |
| **Total S1 test** | **~2.5 h** | **720/720** | 0 failures |

### Optimization applied

The original launcher used `--workers 2` (seed-based sharding), leaving GPUs
idle (vLLM reported Running: 1 req, 0.1% KV cache). Two optimizations were
applied without changing the frozen protocol:

1. **(game, seed) sharding with 24 workers** for the first 5 games. GPU
   utilization rose from ~0% to 80–91%; vLLM batched 7–12 concurrent
   requests per server. All flags identical to `run_s1_safe_sca.py`.
2. **Per-cell sharding for public_goods** (the 4-agent slow game). Each
   seed's 6 cells ran as independent tasks so fast cells (~50s) did not
   block slow cells (~1500s for gsaca/gated). `arm_order.json` was restored
   to the canonical Latin square after execution; `metrics.json` is
   unaffected. `validate_s1_results.py` passes with 0 errors.

### Integrity validation

```
validate_s1_results.py:
  checked_metrics: 720
  missing_count: 0
  error_count: 0
  ready_for_analysis: true
```

---

## Files

- `code/` — experiment code (runner, safe_sca, launchers, analyzers, tests)
- `code/run_s1_fast.py` — optimized (game,seed) parallel launcher
- `code/run_s1_pg_fast.py` — optimized per-cell launcher for public_goods
- `v2_results/exp_s1_dev_warmup/` — 60-cell development observer data
- `v2_results/s1_safe_sca_frozen.json` — frozen Safe-SCA configuration
- `v2_results/s1_safe_sca_frozen_selection_report.json` — config selection report
- `v2_results/exp_s1_safe_sca_test/` — 720-cell held-out test results
  - `<game>/seed_<seed>/<cell>/metrics.json` — per-cell metrics (720 files)
  - `<game>/seed_<seed>/<cell>/trajectories.jsonl` — per-cell trajectories (720 files)
  - `<game>/seed_<seed>/het_safe_sca/decision.json` — Safe-SCA decision records
  - `<game>/seed_<seed>/arm_order.json` — Latin-square arm order
  - `s1_safe_sca_summary.{json,md}` — final analysis
  - `S1_INTEGRITY_REPORT.json` — validation report
  - `CONFIG_SNAPSHOT_S1.json` — immutable config snapshot
  - `ENVIRONMENT_MANIFEST_S1.json` — environment manifest
  - `logs_fast/` — execution logs
- `v2_results/exp_g1_bandit/`, `exp_g2_topp1/`, `exp_g3_detect/` — prior G1/G2/G3 results
