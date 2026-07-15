#!/bin/bash
# TYPE 4: V2 recovery (GPU 3) — FIXED (no tr in subshell)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/code"
BASE="/data/lab/results/v2"
LOG_BASE="$BASE/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EPOCH_START=$(date +%s)
GSACA="$SCRIPT_DIR/code"
SEEDS_5="42 43 44 45 46"
QWEN="Qwen/Qwen2.5-7B-Instruct"
GLM="THUDM/GLM-4-9B-0414"

T4_D1="$BASE/exp_d_fix"
T4_E="$BASE/exp_e_fix"
T4_ANTI="$BASE/exp_anti_test"
mkdir -p "$T4_D1" "$T4_E" "$T4_ANTI"

run_local() {
    CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py "$@"
}

{
    echo "[$(date '+%H:%M:%S')] TYPE 4 START: V2 recovery (GPU 3)"
    
    # D1: noise sweep — 4 noise levels x 6 two-player games x 5 seeds
    echo "[$(date '+%H:%M:%S')] D1: noise sweep"
    for noise in 0.0 0.5 1.0 2.0; do
        case "$noise" in
            0.0) tag="n00";; 0.5) tag="n05";; 1.0) tag="n10";; 2.0) tag="n20";;
        esac
        echo "[$(date '+%H:%M:%S')] D1 noise=$noise (tag=$tag)"
        run_local \
            --games chicken hawk_dove deadlock stag_hunt battle_of_the_sexes \
            --seeds $SEEDS_5 --cells het_gsaca --episodes 30 --horizon 5 --memory 2 \
            --out_dir "$T4_D1/d1_$tag" \
            --models_het $QWEN $GLM --payoff_noise_std $noise \
            --gsaca_warmup 5 --log_every 10 --force 2>&1
        run_local \
            --games public_goods \
            --seeds $SEEDS_5 --cells het_gsaca --episodes 20 --horizon 5 --memory 2 \
            --out_dir "$T4_D1/d1_$tag" \
            --models_het $QWEN $GLM --payoff_noise_std $noise \
            --gsaca_warmup 3 --log_every 10 --force 2>&1
    done
    
    # E: theta sweep
    echo "[$(date '+%H:%M:%S')] E: theta sweep"
    for th in 0.3 0.45 0.6 0.75 0.9; do
        echo "[$(date '+%H:%M:%S')] E theta=$th"
        run_local \
            --games chicken battle_of_the_sexes --seeds $SEEDS_5 \
            --cells het_dp_gated_atom_talk --episodes 30 --horizon 5 --memory 2 \
            --out_dir "$T4_E/theta_$th" --models_het $QWEN $GLM \
            --gate_trust_threshold $th --gate_ema_alpha 0.3 \
            --gsaca_warmup 5 --log_every 10 --force 2>&1
    done
    
    # E: alpha sweep
    echo "[$(date '+%H:%M:%S')] E: alpha sweep"
    for al in 0.1 0.2 0.3 0.5; do
        echo "[$(date '+%H:%M:%S')] E alpha=$al"
        run_local \
            --games chicken battle_of_the_sexes --seeds $SEEDS_5 \
            --cells het_dp_gated_atom_talk --episodes 30 --horizon 5 --memory 2 \
            --out_dir "$T4_E/alpha_$al" --models_het $QWEN $GLM \
            --gate_trust_threshold 0.6 --gate_ema_alpha $al \
            --gsaca_warmup 5 --log_every 10 --force 2>&1
    done
    
    # E: warmup sweep
    echo "[$(date '+%H:%M:%S')] E: warmup sweep"
    for w in 2 3 5 8 10; do
        echo "[$(date '+%H:%M:%S')] E warmup=$w"
        run_local \
            --games chicken battle_of_the_sexes --seeds $SEEDS_5 \
            --cells het_gsaca --episodes 30 --horizon 5 --memory 2 \
            --out_dir "$T4_E/warmup_$w" --models_het $QWEN $GLM \
            --gate_trust_threshold 0.6 --gate_ema_alpha 0.3 \
            --gsaca_warmup $w --log_every 10 --force 2>&1
    done
    
    # anti_test: deadlock
    echo "[$(date '+%H:%M:%S')] anti_test: deadlock"
    run_local \
        --games deadlock --seeds $SEEDS_5 \
        --cells het_gsaca het_role_asym het_hist_split het_adapt_interv het_combo_anti \
        --episodes 30 --horizon 5 --memory 2 \
        --out_dir "$T4_ANTI" --models_het $QWEN $GLM \
        --gsaca_warmup 5 --log_every 10 --force 2>&1
    
    echo "[$(date '+%H:%M:%S')] TYPE 4 DONE: GPU 3 finished"
    echo "__T4_COMPLETE__"
} >> "$LOG_BASE/type4_v2recovery_${TIMESTAMP}.log" 2>&1

echo "TYPE 4 launched in background, log=$LOG_BASE/type4_v2recovery_${TIMESTAMP}.log"
