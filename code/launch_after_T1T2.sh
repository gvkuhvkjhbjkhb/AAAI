#!/bin/bash
# Watch T1 (PID=3210) and T2 (PID=3211), launch anti/E when each finishes
cd /data/lab/gsaca/code
QWEN="Qwen/Qwen2.5-7B-Instruct"
GLM="THUDM/GLM-4-9B-0414"

# Anti on GPU0 after T1
(
    while ps -p 3210 >/dev/null 2>&1; do sleep 30; done
    sleep 5
    echo "[$(date '+%H:%M:%S')] T1 done, launching ANTI on GPU0"
    T4_ANTI="/data/lab/results/v2/exp_anti_test"
    mkdir -p "$T4_ANTI"
    CUDA_VISIBLE_DEVICES=0 python3 run_experiment_local.py \
        --games deadlock --seeds 42 43 44 45 46 \
        --cells het_gsaca het_role_asym het_hist_split het_adapt_interv het_combo_anti \
        --episodes 30 --horizon 5 --memory 2 \
        --out_dir "$T4_ANTI" --models_het $QWEN $GLM \
        --gsaca_warmup 5 --log_every 10 --force 2>&1
    echo "[$(date '+%H:%M:%S')] ANTI DONE on GPU0"
) >> /data/lab/results/v2/logs/type1_anti_append.log 2>&1 &

# E on GPU1 after T2
(
    while ps -p 3211 >/dev/null 2>&1; do sleep 30; done
    sleep 5
    echo "[$(date '+%H:%M:%S')] T2 done, launching E on GPU1"
    T4_E="/data/lab/results/v2/exp_e_fix"
    mkdir -p "$T4_E"
    
    for th in 0.3 0.45 0.6 0.75 0.9; do
        CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
            --games chicken battle_of_the_sexes --seeds 42 43 44 45 46 \
            --cells het_dp_gated_atom_talk --episodes 30 --horizon 5 --memory 2 \
            --out_dir "$T4_E/theta_$th" --models_het $QWEN $GLM \
            --gate_trust_threshold $th --gate_ema_alpha 0.3 \
            --gsaca_warmup 5 --log_every 10 --force 2>&1
    done
    for al in 0.1 0.2 0.3 0.5; do
        CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
            --games chicken battle_of_the_sexes --seeds 42 43 44 45 46 \
            --cells het_dp_gated_atom_talk --episodes 30 --horizon 5 --memory 2 \
            --out_dir "$T4_E/alpha_$al" --models_het $QWEN $GLM \
            --gate_trust_threshold 0.6 --gate_ema_alpha $al \
            --gsaca_warmup 5 --log_every 10 --force 2>&1
    done
    for w in 2 3 5 8 10; do
        CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
            --games chicken battle_of_the_sexes --seeds 42 43 44 45 46 \
            --cells het_gsaca --episodes 30 --horizon 5 --memory 2 \
            --out_dir "$T4_E/warmup_$w" --models_het $QWEN $GLM \
            --gate_trust_threshold 0.6 --gate_ema_alpha 0.3 \
            --gsaca_warmup $w --log_every 10 --force 2>&1
    done
    echo "[$(date '+%H:%M:%S')] E DONE on GPU1"
) >> /data/lab/results/v2/logs/type2_E_append.log 2>&1 &

echo "Watchers launched: anti→GPU0 after T1(3210), E→GPU1 after T2(3211)"
