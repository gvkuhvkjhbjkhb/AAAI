# Independent Same-Batch 3-Arm Paradox / Attainment Study (bf16 / vLLM)

**Status: COMPLETE — 360/360 cells, all validation checks PASS.**
Output directory: `v2_results/exp_vllm_paradox_attainment_v1/`

## 1. Purpose

A fresh, fully independent, single-batch experiment that runs all three arms
(**NoToM, Gated, CGA**) for the same `(game, seed)` back-to-back in one vLLM
batch, so every same-seed pairing is also same-batch. This removes the
cross-batch confound of comparing CGA against a separately-collected Gated run,
and simultaneously tests three things:

1. **Paradox** — does CGA relative to Gated still show the original stack's
   "anti-coordination favorable, coordination unfavorable" pattern under bf16/vLLM?
2. **Baseline attainment** — is the alignment gain (Gated − NoToM) explained by
   independent baseline attainment rather than the game label?
3. **Mild-intervention cost** — what does SCA abstention (staying at NoToM)
   forgo, measured as CGA − NoToM?

Design: **6 games × 20 seeds × 3 arms = 360 cells.** Arms map to runner cells:

| Arm   | Runner cell              | Role                                    |
|-------|--------------------------|-----------------------------------------|
| NoToM | `het_notom`              | independent baseline / abstain control  |
| Gated | `het_gated_atom_talk`    | forced alignment                        |
| CGA   | `het_dp_gated_atom_talk` | mild / diversity-preserving alignment   |

## 2. Frozen configuration (matches the vLLM main table)

- **Models:** Qwen2.5-7B-Instruct (`:8000`) + GLM-4-9B-0414 (`:8001`), heterogeneous.
- **Inference:** vLLM 0.11.0, **bf16**, `top_p = 0.9`, `--enforce-eager`.
- **Temperatures:** Qwen 0.5, GLM 0.8 (`temps_het = [0.5, 0.8]`).
- **Seeds:** 42–61 (n = 20). **Two-player:** 30 episodes; **public_goods:** 20 episodes.
  Both: horizon 5, memory 2, 4 agents for public_goods.
- **ToM:** order 1, θ = 0.6, EMA = 0.3, atom warmup 3.
- Full version/GPU/driver/launch snapshot: `v2_results/exp_vllm_paradox_attainment_v1/CONFIG_SNAPSHOT.txt`.

Frozen code changes made **before** the run (and used for all cells):

- **Reproducible generation seed.** The original code stored only the experiment
  seed and never passed a generation seed to vLLM. `hettom_baseline.LLMAgent`
  now derives a deterministic per-request seed
  `gen_seed_base + experiment_seed*100 + agent_id + call_index` and passes it as
  `seed=` on every `chat.completions.create`.
- **Latin-square arm ordering.** `run_experiment_local.py` gained `--latin_square`,
  which balances arm run-order within each `(game, seed)` by a seed-keyed rotation
  and writes an `arm_order.json` manifest per seed. Schedule (pre-registered):
  - seeds 42,45,48,… → NoToM → Gated → CGA
  - seeds 43,46,49,… → Gated → CGA → NoToM
  - seeds 44,47,50,… → CGA → NoToM → Gated
- The new runner **explicitly passes `--use_vllm`** (the old
  `run_samebatch_maintable.sh` did not, risking a silent fallback to the local
  4-bit path).

## 3. How to reproduce

```bash
# 1. start the two bf16 servers (one model per GPU)
bash code/start_vllm.sh                      # Qwen :8000 (GPU0), GLM :8001 (GPU1)

# 2a. sequential same-batch runner (2 client procs)
bash code/run_paradox_attainment.sh

# 2b. OR the parallel driver used here (30 concurrent seed-shard clients)
NSHARD=5 bash code/run_paradox_parallel.sh   # identical config/output, resumable

# 3. validate + analyze
python3 code/audit_paradox_attainment.py   --root v2_results/exp_vllm_paradox_attainment_v1
python3 code/analyze_paradox_attainment.py --root v2_results/exp_vllm_paradox_attainment_v1
```

The parallel driver only shards the 20 seeds across concurrent client processes;
all three arms of a given `(game, seed)` still run in the **same** worker, so
same-batch same-seed pairing and Latin-square order are preserved. The runner
skips cells whose `metrics.json` already exists, so a partial run resumes safely.

## 4. Validation (all PASS) — `AUDIT_REPORT.txt`

- **Completeness:** 360/360 `metrics.json` present.
- **Balance:** every `(game, arm)` has exactly its 20 seeds.
- **Arm-order manifests:** all 120 match the pre-registered Latin square.
- **Trajectory re-computation:** for a random 10% sample (36/360) cells,
  `cooperation_payoff` recomputed from raw `trajectories.jsonl` matches the stored
  value with **max |err| = 0.0e+00, 0 mismatches**.
- **No contamination:** the payoff-in-prompt arm (`het_payoff_prompt`) is absent
  from this batch (verified `payoff_in_prompt = None` in every cell during smoke).

## 5. Results (cooperation payoff; n = 20 paired per game)

Reported: mean difference, paired Wilcoxon p, Cohen's d_z, paired win rate, and
95% bootstrap CI. The six paradox tests (CGA − Gated) are Holm-corrected.
Full numbers in `paradox_attainment_summary.csv` / `.json`, LaTeX in
`paradox_attainment_table.tex`, console transcript in `paradox_attainment_audit.log`.

### (A) Paradox — CGA − Gated (Holm-corrected across 6 games)

| Game | Grp | CGA | Gated | Δ | 95% CI | p_Holm | d_z | win | expected |
|------|-----|-----|-------|---|--------|--------|-----|-----|----------|
| chicken | anti | 2.440 | 2.176 | **+0.264** | [+0.204, +0.323] | 0.0000*** | +1.90 | 95% | >0 ✓ |
| deadlock | anti | 1.560 | 2.154 | −0.594 | [−0.635, −0.551] | 0.0003*** | −6.01 | 0% | ≤0 (indep-baseline) ✓ |
| hawk_dove | anti | 1.116 | 1.996 | −0.880 | [−0.939, −0.824] | 0.0003*** | −6.51 | 0% | ≤0 (indep-baseline) ✓ |
| stag_hunt | coord | 2.921 | 2.999 | −0.078 | [−0.094, −0.062] | 0.0003*** | −2.07 | 0% | <0 ✓ |
| battle_of_the_sexes | coord | 1.281 | 2.848 | −1.567 | [−1.638, −1.495] | 0.0000*** | −9.33 | 0% | <0 ✓ |
| public_goods | boundary | 2.554 | 2.644 | −0.090 | [−0.101, −0.079] | 0.0000*** | −3.40 | 0% | exploratory |

**Headline:** CGA > Gated only in **chicken** (1/6, Holm-significant); CGA < Gated
in the other 5/6 (all Holm-significant).

### (B) Baseline attainment — Gated − NoToM

| Game | Grp | Gated | NoToM | Δ | 95% CI | p | d_z | win |
|------|-----|-------|-------|---|--------|---|-----|-----|
| chicken | anti | 2.176 | 2.389 | −0.213 | [−0.309, −0.123] | 0.0003*** | −0.99 | 25% |
| deadlock | anti | 2.154 | 1.463 | +0.691 | [+0.643, +0.737] | 0.0000*** | +6.25 | 100% |
| hawk_dove | anti | 1.996 | 1.137 | +0.859 | [+0.776, +0.943] | 0.0000*** | +4.41 | 100% |
| stag_hunt | coord | 2.999 | 2.917 | +0.082 | [+0.065, +0.098] | 0.0001*** | +2.15 | 100% |
| battle_of_the_sexes | coord | 2.848 | 1.288 | +1.560 | [+1.489, +1.628] | 0.0000*** | +9.56 | 100% |
| public_goods | boundary | 2.644 | 2.528 | +0.116 | [+0.105, +0.129] | 0.0001*** | +4.23 | 100% |

**Headline:** Gated ≥ NoToM in **5/6** games. Chicken is the lone exception —
forced alignment *hurts* the pure anti-coordination game, exactly where the
paradox predicts alignment should not be forced.

### (C) Mild-intervention cost — CGA − NoToM

| Game | Grp | CGA | NoToM | Δ | 95% CI | p | d_z | win |
|------|-----|-----|-------|---|--------|---|-----|-----|
| chicken | anti | 2.440 | 2.389 | +0.051 | [−0.063, +0.160] | 0.330 ns | +0.19 | 70% |
| deadlock | anti | 1.560 | 1.463 | +0.097 | [+0.030, +0.163] | 0.019 * | +0.62 | 70% |
| hawk_dove | anti | 1.116 | 1.137 | −0.021 | [−0.108, +0.070] | 0.701 ns | −0.10 | 40% |
| stag_hunt | coord | 2.921 | 2.917 | +0.003 | [−0.018, +0.026] | 0.904 ns | +0.06 | 40% |
| battle_of_the_sexes | coord | 1.281 | 1.288 | −0.008 | [−0.088, +0.079] | 0.546 ns | −0.04 | 35% |
| public_goods | boundary | 2.554 | 2.528 | +0.026 | [+0.013, +0.040] | 0.001 ** | +0.84 | 75% |

**Headline:** CGA ≈ NoToM everywhere (small, mostly non-significant differences).
CGA behaves like a near-neutral, low-cost intervention: it neither reliably helps
nor hurts relative to the independent baseline.

## 6. Interpretation (boundary rule, not a forced replication)

1. **The paradox reproduces under bf16/vLLM as a boundary rule.** CGA beats forced
   alignment only in **chicken** (pure anti-coordination); everywhere else forced
   alignment (Gated) is at least as good. Direction and CIs agree with the
   original stack's "anti-coordination favorable / coordination unfavorable"
   pattern for chicken, stag_hunt, and battle_of_the_sexes.
2. **The alignment gain is baseline-attainment-driven, not game-label-driven.**
   In deadlock and hawk_dove — nominally anti-coordination games — Gated still
   beats NoToM (Δ = +0.69, +0.86; d_z > 4) *and* CGA − Gated ≤ 0. This directly
   supports the "independent baseline attainment" account: what matters is whether
   the baseline is already high, not the game's anti/coord label.
3. **SCA abstention forgoes real gains where Gated wins.** CGA ≈ NoToM (mild cost
   analysis), so abstaining (staying at NoToM) leaves on the table the large
   Gated − NoToM gains observed in deadlock, hawk_dove, BoS, stag_hunt and
   public_goods. Abstention is only clearly the right call in **chicken**, where
   forced alignment actively hurts (Gated − NoToM = −0.21).

Per the pre-registration, effect **direction and CIs** are emphasized over any
single significance flag; with n = 20 and tight CIs the directions are
unambiguous.

## 7. Timing

- **Servers/setup:** clone + install vLLM/scipy/openai + pin transformers 4.56.2 +
  `python3-dev` + download both models (~33 GB) + load bf16 ≈ 25 min.
- **Full 360-cell run (parallel, 30 concurrent seed-shard clients):**
  **09:22:08 → 11:18:06 = 1 h 56 min** wall clock.
- Per-cell mean 223.5 s (median 203.5 s); serial-equivalent compute ≈ **22.3 h**,
  compressed to ~2 h wall by ~11× client parallelism against the two shared
  servers (GPU utilization 86–88%).
- **Validation + statistics:** < 2 min.
- **End-to-end (first launch → analyzed results):** ≈ **2 h 05 min** of run time
  (≈ 2.5 h including environment setup), well inside the 3–5 h budget.

## 8. Artifact index (`v2_results/exp_vllm_paradox_attainment_v1/`)

- `<game>/seed_<n>/<arm>/metrics.json` — 360 cell metrics.
- `<game>/seed_<n>/<arm>/trajectories.jsonl` — full per-episode trajectories (audit source).
- `<game>/seed_<n>/arm_order.json` — Latin-square manifest per seed.
- `CONFIG_SNAPSHOT.txt` — commit, vLLM/torch/transformers/CUDA/GPU/driver, launch cmds.
- `AUDIT_REPORT.txt` — completeness / balance / manifest / trajectory audit (ALL PASS).
- `paradox_attainment_summary.csv`, `.json` — all three contrasts, all games.
- `paradox_attainment_table.tex` — paper-ready LaTeX table (booktabs).
- `paradox_attainment_audit.log` — full analysis console transcript.
- `logs/` — runner stdout per shard.

Code: `code/run_experiment_local.py` (patched), `code/hettom_baseline.py` (patched),
`code/run_paradox_attainment.sh`, `code/run_paradox_parallel.sh`,
`code/start_vllm.sh`, `code/audit_paradox_attainment.py`,
`code/analyze_paradox_attainment.py`.
