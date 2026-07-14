#!/usr/bin/env bash
# Wait for fast run to free GPUs, then launch 3-arm GSACA experiments.
# Tests τ ∈ {0.0, 0.2, 0.4, 0.6} on 4 games × 5 seeds × 30 episodes.
# τ=0.0 = 2-arm baseline (validation), τ=0.4 = predicted optimal.
set -e
cd /data/lab/gsaca
GSACA=/data/lab/gsaca
V2=/data/lab/results/v2
OUT=$V2/exp_3arm
LOG=$V2/exp_3arm_wait.log
QWEN=Qwen/Qwen2.5-7B-Instruct
GLM=THUDM/GLM-4-9B-0414
SEEDS="42 43 44 45 46"
GAMES="chicken hawk_dove deadlock stag_hunt"

echo "[$(date '+%H:%M:%S')] Waiting for fast run to free GPUs..." | tee "$LOG"
while pgrep -f run_final_fast > /dev/null 2>&1; do
    echo "[$(date '+%H:%M:%S')] fast run still active. Sleeping 120s..." >> "$LOG"
    sleep 120
done
echo "[$(date '+%H:%M:%S')] Fast run done! Launching 3-arm experiments..." | tee -a "$LOG"
sleep 10
mkdir -p "$OUT"

# 4 τ values × 4 games × 5 seeds = 80 cells per τ = 320 total
# GPU0: τ=0.0 + τ=0.2, GPU1: τ=0.4 + τ=0.6
# Each worker: 4 games × 5 seeds × 1 cell = 20 cells

CUDA_VISIBLE_DEVICES=0 nohup python3 $GSACA/run_experiment_local.py \
    --games $GAMES --seeds $SEEDS --episodes 30 --horizon 5 --memory 2 \
    --cells het_3arm --out_dir $OUT/tau_0.0 \
    --log_every 50 --gsaca_warmup 5 --abstain_tau 0.0 --force \
    --models_het $QWEN $GLM > $OUT/tau_0.0.log 2>&1 &
echo "[$(date '+%H:%M:%S')] GPU0: τ=0.0 (PID $!)" | tee -a "$LOG"
sleep 90

CUDA_VISIBLE_DEVICES=1 nohup python3 $GSACA/run_experiment_local.py \
    --games $GAMES --seeds $SEEDS --episodes 30 --horizon 5 --memory 2 \
    --cells het_3arm --out_dir $OUT/tau_0.4 \
    --log_every 50 --gsaca_warmup 5 --abstain_tau 0.4 --force \
    --models_het $QWEN $GLM > $OUT/tau_0.4.log 2>&1 &
echo "[$(date '+%H:%M:%S')] GPU1: τ=0.4 (PID $!)" | tee -a "$LOG"
sleep 90

CUDA_VISIBLE_DEVICES=0 nohup python3 $GSACA/run_experiment_local.py \
    --games $GAMES --seeds $SEEDS --episodes 30 --horizon 5 --memory 2 \
    --cells het_3arm --out_dir $OUT/tau_0.2 \
    --log_every 50 --gsaca_warmup 5 --abstain_tau 0.2 --force \
    --models_het $QWEN $GLM > $OUT/tau_0.2.log 2>&1 &
echo "[$(date '+%H:%M:%S')] GPU0: τ=0.2 (PID $!)" | tee -a "$LOG"
sleep 90

CUDA_VISIBLE_DEVICES=1 nohup python3 $GSACA/run_experiment_local.py \
    --games $GAMES --seeds $SEEDS --episodes 30 --horizon 5 --memory 2 \
    --cells het_3arm --out_dir $OUT/tau_0.6 \
    --log_every 50 --gsaca_warmup 5 --abstain_tau 0.6 --force \
    --models_het $QWEN $GLM > $OUT/tau_0.6.log 2>&1 &
echo "[$(date '+%H:%M:%S')] GPU1: τ=0.6 (PID $!)" | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] All 4 τ workers launched. Waiting..." | tee -a "$LOG"
wait
echo "[$(date '+%H:%M:%S')] ALL 3-ARM EXPERIMENTS DONE" | tee -a "$LOG"
echo "Metrics: $(find $OUT -name metrics.json | wc -l)" | tee -a "$LOG"
