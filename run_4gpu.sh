#!/bin/bash
# 4-GPU parallel launcher: one experiment TYPE per GPU
# TYPE 1: 3-arm abstention GSACA (GPU 0)
# TYPE 2: silent-anti-coord (GPU 1)  
# TYPE 3: BoS + public_goods 20-seed (GPU 2)
# TYPE 4: V2 recovery (GPU 3)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/code"

BASE="/data/lab/results/v2"
LOG_BASE="$BASE/logs"
mkdir -p "$LOG_BASE" "$BASE"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EPOCH_START=$(date +%s)
GSACA="$SCRIPT_DIR/code"
ALL_GAMES="chicken hawk_dove deadlock stag_hunt battle_of_the_sexes public_goods"
SEEDS_5="42 43 44 45 46"
SEEDS_20="42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61"
QWEN="Qwen/Qwen2.5-7B-Instruct"
GLM="THUDM/GLM-4-9B-0414"
HET_CELLS="het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca"

# ===== TYPE 1: 3-arm abstention GSACA (GPU 0) =====
T1_OUT="$BASE/exp_3arm"
mkdir -p "$T1_OUT"
T1_LOG="$LOG_BASE/type1_3arm_${TIMESTAMP}.log"
{
    echo "[$(date '+%H:%M:%S')] TYPE 1 START: 3-arm GSACA (GPU 0)"
    for tau in 0.0 0.2 0.4 0.6; do
        echo "[$(date '+%H:%M:%S')] tau=$tau"
        CUDA_VISIBLE_DEVICES=0 python3 run_experiment_local.py \
            --games $ALL_GAMES \
            --seeds $SEEDS_5 \
            --cells het_3arm \
            --episodes 30 --horizon 5 --memory 2 \
            --out_dir "$T1_OUT/tau_$tau" \
            --models_het $QWEN $GLM \
            --abstain_tau $tau \
            --gsaca_warmup 5 --log_every 10 --force \
            2>&1
    done
    echo "[$(date '+%H:%M:%S')] TYPE 1 DONE: GPU 0 finished" 
    echo "__T1_COMPLETE__"
} > "$T1_LOG" 2>&1 &
PID1=$!
echo "TYPE 1 (GPU 0): PID=$PID1 log=$T1_LOG"

# ===== TYPE 2: silent-anti-coord (GPU 1) =====
T2_LOG="$LOG_BASE/type2_silent_${TIMESTAMP}.log"
{
    echo "[$(date '+%H:%M:%S')] TYPE 2 START: silent-anti-coord (GPU 1)"
    for game in chicken hawk_dove deadlock; do
        CUDA_VISIBLE_DEVICES=1 python3 scheme2_silent.py \
            --games $game \
            --seeds $SEEDS_20 \
            --episodes 30 --log_every 10 --force \
            2>&1
    done
    echo "[$(date '+%H:%M:%S')] TYPE 2 DONE: GPU 1 finished"
    echo "__T2_COMPLETE__"
} > "$T2_LOG" 2>&1 &
PID2=$!
echo "TYPE 2 (GPU 1): PID=$PID2 log=$T2_LOG"

# ===== TYPE 3: BoS + public_goods 20-seed (GPU 2) =====
T3_OUT="$BASE/exp_b_20seed"
mkdir -p "$T3_OUT"
T3_LOG="$LOG_BASE/type3_bospg_${TIMESTAMP}.log"
{
    echo "[$(date '+%H:%M:%S')] TYPE 3 START: BoS + public_goods 20-seed (GPU 2)"
    CUDA_VISIBLE_DEVICES=2 python3 run_experiment_local.py \
        --games battle_of_the_sexes public_goods \
        --seeds $SEEDS_20 \
        --cells $HET_CELLS \
        --episodes 30 --horizon 5 --memory 2 \
        --out_dir "$T3_OUT" \
        --models_het $QWEN $GLM \
        --gsaca_warmup 5 --log_every 10 --force \
        2>&1
    echo "[$(date '+%H:%M:%S')] TYPE 3 DONE: GPU 2 finished"
    echo "__T3_COMPLETE__"
} > "$T3_LOG" 2>&1 &
PID3=$!
echo "TYPE 3 (GPU 2): PID=$PID3 log=$T3_LOG"

# ===== TYPE 4: V2 recovery (GPU 3) =====
# D1: noise sweep + E: hyperparam + anti_test: deadlock
T4_D1="$BASE/exp_d_fix"
T4_E="$BASE/exp_e_fix"
T4_ANTI="$BASE/exp_anti_test"
mkdir -p "$T4_D1" "$T4_E" "$T4_ANTI"
T4_LOG="$LOG_BASE/type4_v2recovery_${TIMESTAMP}.log"
{
    echo "[$(date '+%H:%M:%S')] TYPE 4 START: V2 recovery (GPU 3)"
    
    # D1: noise sweep — 4 noise levels × 6 two-player games × 5 seeds
    echo "[$(date '+%H:%M:%S')] D1: noise sweep"
    for noise in 0.0 0.5 1.0 2.0; do
        tag="n$(echo $noise | tr '.' '')"
        echo "[$(date '+%H:%M:%S')] D1 noise=$noise"
        # two-player games
        CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
            --games chicken hawk_dove deadlock stag_hunt battle_of_the_sexes \
            --seeds $SEEDS_5 \
            --cells het_gsaca \
            --episodes 30 --horizon 5 --memory 2 \
            --out_dir "$T4_D1/d1_$tag" \
            --models_het $QWEN $GLM \
            --payoff_noise_std $noise \
            --gsaca_warmup 5 --log_every 10 --force \
            2>&1
        # public_goods
        CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
            --games public_goods \
            --seeds $SEEDS_5 \
            --cells het_gsaca \
            --episodes 20 --horizon 5 --memory 2 \
            --out_dir "$T4_D1/d1_$tag" \
            --models_het $QWEN $GLM \
            --payoff_noise_std $noise \
            --gsaca_warmup 3 --log_every 10 --force \
            2>&1
    done
    
    # E: hyperparam sweeps — θ sweep
    echo "[$(date '+%H:%M:%S')] E: theta sweep"
    for th in 0.3 0.45 0.6 0.75 0.9; do
        echo "[$(date '+%H:%M:%S')] E theta=$th"
        CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
            --games chicken battle_of_the_sexes \
            --seeds $SEEDS_5 \
            --cells het_dp_gated_atom_talk \
            --episodes 30 --horizon 5 --memory 2 \
            --out_dir "$T4_E/theta_$th" \
            --models_het $QWEN $GLM \
            --gate_trust_threshold $th --gate_ema_alpha 0.3 \
            --gsaca_warmup 5 --log_every 10 --force \
            2>&1
    done
    
    # E: alpha sweep
    echo "[$(date '+%H:%M:%S')] E: alpha sweep"
    for al in 0.1 0.2 0.3 0.5; do
        echo "[$(date '+%H:%M:%S')] E alpha=$al"
        CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
            --games chicken battle_of_the_sexes \
            --seeds $SEEDS_5 \
            --cells het_dp_gated_atom_talk \
            --episodes 30 --horizon 5 --memory 2 \
            --out_dir "$T4_E/alpha_$al" \
            --models_het $QWEN $GLM \
            --gate_trust_threshold 0.6 --gate_ema_alpha $al \
            --gsaca_warmup 5 --log_every 10 --force \
            2>&1
    done
    
    # E: warmup sweep
    echo "[$(date '+%H:%M:%S')] E: warmup sweep"
    for w in 2 3 5 8 10; do
        echo "[$(date '+%H:%M:%S')] E warmup=$w"
        CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
            --games chicken battle_of_the_sexes \
            --seeds $SEEDS_5 \
            --cells het_gsaca \
            --episodes 30 --horizon 5 --memory 2 \
            --out_dir "$T4_E/warmup_$w" \
            --models_het $QWEN $GLM \
            --gate_trust_threshold 0.6 --gate_ema_alpha 0.3 \
            --gsaca_warmup $w --log_every 10 --force \
            2>&1
    done
    
    # anti_test: deadlock
    echo "[$(date '+%H:%M:%S')] anti_test: deadlock"
    CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
        --games deadlock \
        --seeds $SEEDS_5 \
        --cells het_gsaca het_role_asym het_hist_split het_adapt_interv het_combo_anti \
        --episodes 30 --horizon 5 --memory 2 \
        --out_dir "$T4_ANTI" \
        --models_het $QWEN $GLM \
        --gsaca_warmup 5 --log_every 10 --force \
        2>&1
    
    echo "[$(date '+%H:%M:%S')] TYPE 4 DONE: GPU 3 finished"
    echo "__T4_COMPLETE__"
} > "$T4_LOG" 2>&1 &
PID4=$!
echo "TYPE 4 (GPU 3): PID=$PID4 log=$T4_LOG"

echo ""
echo "=== All 4 types launched ==="
echo "TYPE 1 (3-arm, GPU0):       PID=$PID1"
echo "TYPE 2 (silent, GPU1):      PID=$PID2"
echo "TYPE 3 (BoS+pg, GPU2):      PID=$PID3"
echo "TYPE 4 (V2 recovery, GPU3): PID=$PID4"
echo "Logs: $LOG_BASE/*_${TIMESTAMP}.log"

# Save PIDs for monitoring
echo "$PID1 $PID2 $PID3 $PID4" > "$LOG_BASE/pids_${TIMESTAMP}.txt"
echo "$T1_LOG" > "$LOG_BASE/last_T1_log.txt"
echo "$T2_LOG" > "$LOG_BASE/last_T2_log.txt"
echo "$T3_LOG" > "$LOG_BASE/last_T3_log.txt"
echo "$T4_LOG" > "$LOG_BASE/last_T4_log.txt"

# Wait for all
echo "Waiting for all 4 GPU processes to complete..."
wait $PID1 && echo "TYPE 1 (GPU0) DONE" || echo "TYPE 1 (GPU0) FAILED with $?"
wait $PID2 && echo "TYPE 2 (GPU1) DONE" || echo "TYPE 2 (GPU1) FAILED with $?"
wait $PID3 && echo "TYPE 3 (GPU2) DONE" || echo "TYPE 3 (GPU2) FAILED with $?"
wait $PID4 && echo "TYPE 4 (GPU3) DONE" || echo "TYPE 4 (GPU3) FAILED with $?"

EPOCH_END=$(date +%s)
ELAPSED=$((EPOCH_END - EPOCH_START))
echo ""
echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "Total wall time: ${ELAPSED}s ($((ELAPSED/60))m $((ELAPSED%60))s)"
echo "Results: $BASE/"
echo "Logs: $LOG_BASE/*_${TIMESTAMP}.log"
