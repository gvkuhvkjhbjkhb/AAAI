#!/bin/bash
# Run after T1 on GPU0: anti_test deadlock (25 cells)
SEEDS_5="42 43 44 45 46"
QWEN="Qwen/Qwen2.5-7B-Instruct"
GLM="THUDM/GLM-4-9B-0414"
T4_ANTI="/data/lab/results/v2/exp_anti_test"
mkdir -p "$T4_ANTI"

echo "[$(date '+%H:%M:%S')] ANTI_TEST: deadlock on GPU0"
CUDA_VISIBLE_DEVICES=0 python3 run_experiment_local.py \
    --games deadlock \
    --seeds 42 43 44 45 46 \
    --cells het_gsaca het_role_asym het_hist_split het_adapt_interv het_combo_anti \
    --episodes 30 --horizon 5 --memory 2 \
    --out_dir "$T4_ANTI" \
    --models_het $QWEN $GLM \
    --gsaca_warmup 5 --log_every 10 --force 2>&1
echo "[$(date '+%H:%M:%S')] ANTI_TEST DONE on GPU0"
