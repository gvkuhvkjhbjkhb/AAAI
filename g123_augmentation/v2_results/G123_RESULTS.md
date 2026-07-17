# G1/G2/G3 Augmentation Experiments — Results

**Run completed:** 2026-07-17, 210/210 cells, wall time 3278s (54.6 min) on 2×RTX 5090.
**Hardware/stack:** vLLM 0.23.0 bf16 (enforce-eager), Qwen2.5-7B-Instruct + GLM-4-9B-0414,
VLLM_USE_FLASHINFER_SAMPLER=0 (5090 sm_120 rejects FlashInfer JIT), python3-dev (Python.h).
See `v2_results/CONFIG_SNAPSHOT_G123.txt` for the frozen protocol.

## G1 — End-to-end 2-arm attainment bandit (replaces v6 §6.5 offline reconstruction)
- **Design:** per (game,seed), probe NoAlign & Gated K=5 episodes each (order alternated by
  seed parity), commit remaining 20/10 episodes to the probe-mean winner (ties→NoAlign).
  No game-label/oracle arm assignment. 6 games × 20 seeds = 120 cells.
- **Selection accuracy: 58/120 = 48.3%** (offline probe-argmax reconstruction was 116/120 = 96.7%).
- **Bandit commit mean: 2.3333** vs SCA (two-arm-abstain) 2.2273 (Δ=+0.106, p=0.42, dz=+0.29)
  vs oracle upper bound 2.5069.
- **Verdict: PARTIAL/FAIL** on the accuracy criterion (≥85%); the mean criterion (≥SCA) holds.
- **Root cause (pre-registered "decoupling" outcome, now empirically characterized):**
  the Gated arm needs gate-trust EMA warmup, so 5-episode probes systematically *underestimate*
  Gated. Decoupling rate 62/120 (51.7%); 47/62 misses picked NoAlign where Gated was truly better.
  Deadlock is the clearest case: probe NoAlign=2.00 vs probe Gated=1.76 (probe favors abstain),
  but true NoToM=1.46 vs true Gated=2.15 (commit favors Gated). Probe payoff (2.466) even exceeds
  commit payoff (2.333) — exploration collected the easy early-episode gains that commit then forgoes.
- **Paper action (per pre-registration, both outcomes writable):** retain the offline reconstruction
  as an *upper bound*; §6.5 reports the end-to-end result honestly — "probe/commit behavioral
  coupling weakened the offline estimate; the offline 116/120 is an upper bound, not achievable
  without longer warmup." Discussion gains a regret-bound sentence: probe regret (≈0.13 payoff)
  quantifies the bandit's exploration cost and the EMA-warmup floor.

## G2 — top_p single-factor ablation (1.0 vs frozen 0.9)
- **Design:** stack-B config with only top_p flipped 0.9→1.0; NoToM + Gated × 3 anti-coord games
  × 10 seeds = 60 cells.
- **Flip (NoToM ≥ Gated, i.e. forced alignment hurts) persists at top_p=1.0:** chicken flips at
  both 0.9 (Δ=+0.213, p=0.0003) and 1.0 (Δ=+0.289, p=0.006); deadlock/hawk_dove do NOT flip at
  either (Gated wins, dz −4 to −6).
- **Verdict: SAMPLING-ROBUST** — the anti-coordination flip is *not* a top_p=0.9 truncation
  artifact; it is attributable to precision/template/other stack-B factors. §6.5 gains a precise
  attribution sentence; the multi-factor confound statement is retained.

## G3 — Stack-B online-detection validation (het_gsaca)
- **Design:** het_gsaca (full online detector) re-run on stack B; 6 games × 5 seeds = 30 cells.
- **Detection accuracy: 22/30 (73%)**, NOT the Prop-1-predicted 30/30.
- **Verdict: Prop 1 precondition fails on stack B.** Misses concentrate where the warm-up profile
  lacks diversity: deadlock (split≈0.04, 60%), hawk_dove (split≈0.02, 40%), stag_hunt (3/5 misses,
  split collapses to 0.0 when agents all-cooperate in warmup). Chicken/BoS/public_goods stay 100%.
- **Paper action (per pre-registration, valuable either way):** this is empirical evidence that
  Prop 1's precondition (sufficient warm-up behavioral diversity) can fail; §6.6 detection boundary
  is now *demonstrated*, not just asserted. Table 2 caption disclosure is revised to the honest
  "22/30 on stack B; misses arise where warm-up profiles are low-diversity."

## Timing
- Setup (deps + 33GB models + 5090 server fixes): ~50 min (one-time).
- Full G1+G2+G3 run: **54.6 min** wall (20 concurrent workers vs 2 shared vLLM servers, GPU util
  ~82-92%). vs the 7.5h serial estimate — ~8× speedup.
