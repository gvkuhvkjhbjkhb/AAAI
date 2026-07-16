#!/usr/bin/env bash
# =============================================================================
# W1 fix: SAME-BATCH replication of the Table 2 main comparison.
#
# The submitted Table 2 pairs the "ours"/SCA column (read from the exp_b_20seed
# batch) against the payoff-in-prompt baseline (read from a SEPARATE
# exp_c_payoff_prompt batch), matched only by seed. Reviewer W1: this crosses
# experimental batches whose self-reported absolute-payoff drift (0.4-0.9)
# overlaps three of the six reported effects, so part of the "5/6 significant
# wins" may be a batch effect rather than a method effect.
#
# This script removes the confound by running BOTH the SCA arms and the
# payoff-in-prompt baseline for ALL SIX games x 20 seeds INTO A SINGLE FRESH
# out_dir, in the same run, on the same GPU, with identical hyperparameters,
# episode counts, models, and horizon. Because all cells for a given (game,seed)
# are produced back-to-back in one batch, same-seed pairing is now also
# same-batch: any residual batch effect is shared by ours and baseline and
# cancels in the paired difference.
#
# Arms produced per (game, seed):
#   het_notom            -> "ours"/SCA arm for anti-coord + boundary games
#                           (chicken, deadlock, hawk_dove, public_goods)
#   het_gated_atom_talk  -> "ours"/SCA arm for coord games (stag_hunt, BoS)
#   het_payoff_prompt    -> the payoff-in-prompt baseline (all 6 games)
# Running all three arms everywhere (rather than only the arm each game needs)
# costs a little extra but lets the analysis pick the correct "ours" arm per
# game from the SAME batch and also reports the drift diagnostics.
#
# Cell budget: 6 games x 20 seeds x 3 arms = 360 cells.
# (The reviewer's "~120 cells" refers to the payoff-in-prompt arm alone;
#  we add the two SCA arms so the whole comparison is same-batch, not just
#  the baseline.)
#
# Config mirrors run_exp_c.sh / EXPERIMENT_REPORT_V5 exactly:
#   episodes: 30 for 2-player games, 20 for public_goods
#   horizon 5, memory 2, tom_order 1, seeds 42..61 (20), Qwen2.5-7B + GLM-4-9B.
#
# Edit GSACA_ROOT / CUDA device split for your machine, then:
#   bash code/run_samebatch_maintable.sh
# =============================================================================
set -euo pipefail

# ---- paths (edit GSACA_ROOT to point at the repo root on the run machine) ----
GSACA_ROOT="${GSACA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUNNER="$GSACA_ROOT/code/run_experiment_local.py"
OUT="${OUT:-$GSACA_ROOT/v2_results/exp_samebatch_maintable}"
LOG="${LOG:-$GSACA_ROOT/results/v2/logs}"
mkdir -p "$OUT" "$LOG"

QWEN=Qwen/Qwen2.5-7B-Instruct
GLM=THUDM/GLM-4-9B-0414
SEEDS="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61"   # n=20
# All three arms in ONE invocation per (game) so they share the batch/process.
ARMS="het_notom het_gated_atom_talk het_payoff_prompt"

# GPU visibility: default to a single GPU. Set GPUS="0 1" to split games across
# two GPUs (2-player games on GPU0, public_goods on GPU1).
GPU0="${GPU0:-0}"
GPU1="${GPU1:-$GPU0}"

echo "[$(date '+%H:%M:%S')] SAME-BATCH main table START -> $OUT (360 cells)" | tee "$LOG/samebatch_run.log"
echo "  runner=$RUNNER  arms=[$ARMS]  seeds=n=20" | tee -a "$LOG/samebatch_run.log"

# --- 2-player games, 30 episodes (GPU0): 5 games x 20 seeds x 3 arms = 300 cells
CUDA_VISIBLE_DEVICES=$GPU0 python3 "$RUNNER" \
    --games chicken deadlock hawk_dove stag_hunt battle_of_the_sexes \
    --seeds $SEEDS \
    --episodes 30 --horizon 5 --memory 2 \
    --cells $ARMS --out_dir "$OUT" \
    --log_every 50 --models_het $QWEN $GLM \
    > "$LOG/samebatch_2player.log" 2>&1 &
P0=$!
echo "[$(date '+%H:%M:%S')] GPU$GPU0 PID $P0 (5 two-player games ep30)" | tee -a "$LOG/samebatch_run.log"

# --- public_goods, 20 episodes (GPU1): 1 game x 20 seeds x 3 arms = 60 cells ---
CUDA_VISIBLE_DEVICES=$GPU1 python3 "$RUNNER" \
    --games public_goods \
    --seeds $SEEDS \
    --episodes 20 --horizon 5 --memory 2 \
    --cells $ARMS --out_dir "$OUT" \
    --log_every 50 --models_het $QWEN $GLM \
    > "$LOG/samebatch_pg.log" 2>&1 &
P1=$!
echo "[$(date '+%H:%M:%S')] GPU$GPU1 PID $P1 (public_goods ep20)" | tee -a "$LOG/samebatch_run.log"

wait
N=$(find "$OUT" -name metrics.json | wc -l)
echo "[$(date '+%H:%M:%S')] SAME-BATCH DONE. metrics: $N/360  out=$OUT" | tee -a "$LOG/samebatch_run.log"
echo "Now run:  python3 code/analyze_samebatch_maintable.py --root $OUT" | tee -a "$LOG/samebatch_run.log"
