#!/usr/bin/env bash
# =============================================================================
# INDEPENDENT SAME-BATCH 3-ARM PARADOX / ATTAINMENT STUDY  (bf16 / vLLM)
#
#   6 games x 20 seeds x 3 arms = 360 cells, all in ONE fresh out_dir.
#   Arms (NoToM, Gated, CGA):
#     het_notom              -> NoToM  (independent baseline / abstain control)
#     het_gated_atom_talk    -> Gated  (forced alignment)
#     het_dp_gated_atom_talk -> CGA    (mild / diversity-preserving alignment)
#
#   Fixed frozen config (matches vLLM main table):
#     models: Qwen2.5-7B-Instruct (:8000) + GLM-4-9B-0414 (:8001), bf16, top_p=0.9
#     temps:  Qwen 0.5, GLM 0.8   (temps_het defaults [0.5,0.8])
#     seeds 42..61 (n=20), horizon 5, memory 2, tom_order 1, theta 0.6,
#     EMA 0.3, atom warmup 3.  2-player: 30 episodes; public_goods: 20 episodes.
#
#   MUST pass --use_vllm explicitly (else falls back to local 4-bit).
#   Latin-square arm ordering per (game,seed) via --latin_square; each seed
#   writes arm_order.json manifest. Reproducible per-request generation seed
#   via --gen_seed_base.
# =============================================================================
set -uo pipefail

GSACA_ROOT="${GSACA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUNNER="$GSACA_ROOT/code/run_experiment_local.py"
OUT="${OUT:-$GSACA_ROOT/v2_results/exp_vllm_paradox_attainment_v1}"
LOG="${LOG:-$OUT/logs}"
mkdir -p "$OUT" "$LOG"

QWEN=Qwen/Qwen2.5-7B-Instruct
GLM=THUDM/GLM-4-9B-0414
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61}"
ARMS="het_notom het_gated_atom_talk het_dp_gated_atom_talk"
GEN_SEED_BASE="${GEN_SEED_BASE:-1000}"

echo "[$(date '+%H:%M:%S')] PARADOX/ATTAINMENT START -> $OUT (target 360 cells)" | tee "$LOG/run.log"
echo "  arms=[$ARMS]  seeds=n=$(echo $SEEDS|wc -w)  use_vllm=YES latin_square=YES" | tee -a "$LOG/run.log"

# Both runner processes talk to the SAME two vLLM servers (Qwen:8000, GLM:8001);
# CUDA_VISIBLE_DEVICES on the runner is irrelevant in --use_vllm mode (no local
# model load). Run 2-player and public_goods concurrently as two client procs.

# 5 two-player games x 20 seeds x 3 arms = 300 cells, 30 episodes
python3 "$RUNNER" \
    --use_vllm --latin_square --gen_seed_base $GEN_SEED_BASE \
    --games chicken deadlock hawk_dove stag_hunt battle_of_the_sexes \
    --seeds $SEEDS \
    --episodes 30 --horizon 5 --memory 2 \
    --cells $ARMS --out_dir "$OUT" \
    --log_every 50 --models_het $QWEN $GLM \
    --gate_trust_threshold 0.6 --gate_ema_alpha 0.3 --atom_warmup 3 \
    > "$LOG/2player.log" 2>&1 &
P0=$!
echo "[$(date '+%H:%M:%S')] 2-player client PID $P0 (5 games ep30)" | tee -a "$LOG/run.log"

# public_goods x 20 seeds x 3 arms = 60 cells, 20 episodes
python3 "$RUNNER" \
    --use_vllm --latin_square --gen_seed_base $GEN_SEED_BASE \
    --games public_goods \
    --seeds $SEEDS \
    --episodes 20 --horizon 5 --memory 2 \
    --cells $ARMS --out_dir "$OUT" \
    --log_every 50 --models_het $QWEN $GLM \
    --gate_trust_threshold 0.6 --gate_ema_alpha 0.3 --atom_warmup 3 \
    > "$LOG/public_goods.log" 2>&1 &
P1=$!
echo "[$(date '+%H:%M:%S')] public_goods client PID $P1 (ep20)" | tee -a "$LOG/run.log"

wait $P0 $P1
N=$(find "$OUT" -name metrics.json | wc -l)
echo "[$(date '+%H:%M:%S')] DONE. metrics: $N/360  out=$OUT" | tee -a "$LOG/run.log"
echo "Now run:  python3 code/analyze_paradox_attainment.py --root $OUT" | tee -a "$LOG/run.log"
