#!/usr/bin/env bash
# Layer 3 (PRIORITY): Exp C — payoff-in-prompt baseline n=5 -> n=20.
# Cell het_payoff_prompt: no ToM, no cheap-talk, full payoff matrix in prompt.
# Fills seeds 47-61 (existing 42-46 kept). 6 games x 15 new seeds = 90 cells.
# Split: GPU0 = chicken,deadlock,stag_hunt (ep30); GPU1 = hawk_dove,battle_of_the_sexes (ep30) then public_goods (ep20).
set -e
cd /data/lab/gsaca
GSACA=/data/lab/gsaca
OUT=$GSACA/v2_results/exp_c_payoff_prompt
LOG=$GSACA/results/v2/logs
mkdir -p "$OUT" "$LOG"
QWEN=Qwen/Qwen2.5-7B-Instruct
GLM=THUDM/GLM-4-9B-0414
SEEDS_NEW="47 48 49 50 51 52 53 54 55 56 57 58 59 60 61"

echo "[$(date '+%H:%M:%S')] Exp C payoff-prompt n=5->20 START (90 new cells)" | tee "$LOG/exp_c_run.log"

# GPU0: 3 two-player games, ep30 (45 cells)
CUDA_VISIBLE_DEVICES=0 python3 $GSACA/code/run_experiment_local.py \
    --games chicken deadlock stag_hunt --seeds $SEEDS_NEW \
    --episodes 30 --horizon 5 --memory 2 \
    --cells het_payoff_prompt --out_dir "$OUT" \
    --log_every 50 --models_het $QWEN $GLM \
    > "$LOG/exp_c_gpu0.log" 2>&1 &
P0=$!
echo "[$(date '+%H:%M:%S')] GPU0 PID $P0 (chicken,deadlock,stag_hunt ep30)" | tee -a "$LOG/exp_c_run.log"

# GPU1: 2 two-player games ep30 (30 cells) THEN public_goods ep20 (15 cells) -- sequential in one shell
( CUDA_VISIBLE_DEVICES=1 python3 $GSACA/code/run_experiment_local.py \
    --games hawk_dove battle_of_the_sexes --seeds $SEEDS_NEW \
    --episodes 30 --horizon 5 --memory 2 \
    --cells het_payoff_prompt --out_dir "$OUT" \
    --log_every 50 --models_het $QWEN $GLM \
    > "$LOG/exp_c_gpu1a.log" 2>&1
  echo "[$(date '+%H:%M:%S')] GPU1 two-player done, starting public_goods ep20" | tee -a "$LOG/exp_c_run.log"
  CUDA_VISIBLE_DEVICES=1 python3 $GSACA/code/run_experiment_local.py \
    --games public_goods --seeds $SEEDS_NEW \
    --episodes 20 --horizon 5 --memory 2 \
    --cells het_payoff_prompt --out_dir "$OUT" \
    --log_every 50 --models_het $QWEN $GLM \
    > "$LOG/exp_c_gpu1b.log" 2>&1
  echo "[$(date '+%H:%M:%S')] GPU1 public_goods done" | tee -a "$LOG/exp_c_run.log"
) &
P1=$!
echo "[$(date '+%H:%M:%S')] GPU1 PID $P1 (hawk_dove,BoS ep30 -> public_goods ep20)" | tee -a "$LOG/exp_c_run.log"

wait
echo "[$(date '+%H:%M:%S')] Exp C DONE. metrics: $(find $OUT -name metrics.json | wc -l)/120" | tee -a "$LOG/exp_c_run.log"
