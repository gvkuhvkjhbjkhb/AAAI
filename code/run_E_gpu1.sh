#!/bin/bash
# Run after T2 on GPU1: E hyperparam sweeps (140 cells)
SEEDS_5="42 43 44 45 46"
QWEN="Qwen/Qwen2.5-7B-Instruct"
GLM="THUDM/GLM-4-9B-0414"
T4_E="/data/lab/results/v2/exp_e_fix"
mkdir -p "$T4_E"

echo "[$(date '+%H:%M:%S')] E: theta sweep on GPU1"
for th in 0.3 0.45 0.6 0.75 0.9; do
    echo "[$(date '+%H:%M:%S')]   theta=$th"
    CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
        --games chicken battle_of_the_sexes --seeds 42 43 44 45 46 \
        --cells het_dp_gated_atom_talk --episodes 30 --horizon 5 --memory 2 \
        --out_dir "$T4_E/theta_$th" --models_het $QWEN $GLM \
        --gate_trust_threshold $th --gate_ema_alpha 0.3 \
        --gsaca_warmup 5 --log_every 10 --force 2>&1
done

echo "[$(date '+%H:%M:%S')] E: alpha sweep on GPU1"
for al in 0.1 0.2 0.3 0.5; do
    echo "[$(date '+%H:%M:%S')]   alpha=$al"
    CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
        --games chicken battle_of_the_sexes --seeds 42 43 44 45 46 \
        --cells het_dp_gated_atom_talk --episodes 30 --horizon 5 --memory 2 \
        --out_dir "$T4_E/alpha_$al" --models_het $QWEN $GLM \
        --gate_trust_threshold 0.6 --gate_ema_alpha $al \
        --gsaca_warmup 5 --log_every 10 --force 2>&1
done

echo "[$(date '+%H:%M:%S')] E: warmup sweep on GPU1"
for w in 2 3 5 8 10; do
    echo "[$(date '+%H:%M:%S')]   warmup=$w"
    CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
        --games chicken battle_of_the_sexes --seeds 42 43 44 45 46 \
        --cells het_gsaca --episodes 30 --horizon 5 --memory 2 \
        --out_dir "$T4_E/warmup_$w" --models_het $QWEN $GLM \
        --gate_trust_threshold 0.6 --gate_ema_alpha 0.3 \
        --gsaca_warmup $w --log_every 10 --force 2>&1
done
echo "[$(date '+%H:%M:%S')] E DONE on GPU1"
