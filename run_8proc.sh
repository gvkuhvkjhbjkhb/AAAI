#!/bin/bash
# 8-PROCESS OPTIMIZED LAUNCHER: 2 processes per GPU (26GB/32GB safe)
# GPU util was only 18-45% → 2x processes = ~2x throughput
# Stagger 15s between launches to avoid VRAM spike
cd /data/lab/gsaca/code
QWEN="Qwen/Qwen2.5-7B-Instruct"
GLM="THUDM/GLM-4-9B-0414"
V2="/data/lab/results/v2"
LOGS="$V2/logs"
mkdir -p "$LOGS"
TS=$(date +%H%M%S)

# ========== GPU0: T1 3-arm + anti (145 cells → 2 streams) ==========
# 0a: tau=0.0 + tau=0.2 (60 cells)
nohup bash -c "
for tau in 0.0 0.2; do
  CUDA_VISIBLE_DEVICES=0 python3 run_experiment_local.py \
    --games chicken hawk_dove deadlock stag_hunt battle_of_the_sexes public_goods \
    --seeds 42 43 44 45 46 --cells het_3arm --episodes 30 --horizon 5 --memory 2 \
    --out_dir $V2/exp_3arm/tau_$tau --models_het $QWEN $GLM \
    --abstain_tau $tau --gsaca_warmup 5 --log_every 10
done
echo DONE_0a
" > "$LOGS/gpu0a_${TS}.log" 2>&1 &
echo "GPU0a PID=$! (tau 0.0+0.2)"

sleep 15

# 0b: tau=0.4 + tau=0.6 + anti (85 cells)
nohup bash -c "
for tau in 0.4 0.6; do
  CUDA_VISIBLE_DEVICES=0 python3 run_experiment_local.py \
    --games chicken hawk_dove deadlock stag_hunt battle_of_the_sexes public_goods \
    --seeds 42 43 44 45 46 --cells het_3arm --episodes 30 --horizon 5 --memory 2 \
    --out_dir $V2/exp_3arm/tau_$tau --models_het $QWEN $GLM \
    --abstain_tau $tau --gsaca_warmup 5 --log_every 10
done
CUDA_VISIBLE_DEVICES=0 python3 run_experiment_local.py \
  --games deadlock --seeds 42 43 44 45 46 \
  --cells het_gsaca het_role_asym het_hist_split het_adapt_interv het_combo_anti \
  --episodes 30 --horizon 5 --memory 2 --out_dir $V2/exp_anti_test \
  --models_het $QWEN $GLM --gsaca_warmup 5 --log_every 10
echo DONE_0b
" > "$LOGS/gpu0b_${TS}.log" 2>&1 &
echo "GPU0b PID=$! (tau 0.4+0.6 + anti)"

sleep 15

# ========== GPU1: T2 silent + E sweep (200 cells → 2 streams) ==========
# 1a: T2 chicken+hawk_dove + E theta (90 cells)
nohup bash -c "
CUDA_VISIBLE_DEVICES=1 python3 scheme2_silent.py \
  --games chicken hawk_dove --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
  --episodes 30 --log_every 10
for th in 0.3 0.45 0.6 0.75 0.9; do
  CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
    --games chicken battle_of_the_sexes --seeds 42 43 44 45 46 \
    --cells het_dp_gated_atom_talk --episodes 30 --horizon 5 --memory 2 \
    --out_dir $V2/exp_e_fix/theta_$th --models_het $QWEN $GLM \
    --gate_trust_threshold $th --gate_ema_alpha 0.3 --gsaca_warmup 5 --log_every 10
done
echo DONE_1a
" > "$LOGS/gpu1a_${TS}.log" 2>&1 &
echo "GPU1a PID=$! (T2 chi+hd + E theta)"

sleep 15

# 1b: T2 deadlock + E alpha + E warmup (110 cells)
nohup bash -c "
CUDA_VISIBLE_DEVICES=1 python3 scheme2_silent.py \
  --games deadlock --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
  --episodes 30 --log_every 10
for al in 0.1 0.2 0.3 0.5; do
  CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
    --games chicken battle_of_the_sexes --seeds 42 43 44 45 46 \
    --cells het_dp_gated_atom_talk --episodes 30 --horizon 5 --memory 2 \
    --out_dir $V2/exp_e_fix/alpha_$al --models_het $QWEN $GLM \
    --gate_trust_threshold 0.6 --gate_ema_alpha $al --gsaca_warmup 5 --log_every 10
done
for w in 2 3 5 8 10; do
  CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
    --games chicken battle_of_the_sexes --seeds 42 43 44 45 46 \
    --cells het_gsaca --episodes 30 --horizon 5 --memory 2 \
    --out_dir $V2/exp_e_fix/warmup_$w --models_het $QWEN $GLM \
    --gate_trust_threshold 0.6 --gate_ema_alpha 0.3 --gsaca_warmup $w --log_every 10
done
echo DONE_1b
" > "$LOGS/gpu1b_${TS}.log" 2>&1 &
echo "GPU1b PID=$! (T2 deadlock + E alpha+warmup)"

sleep 15

# ========== GPU2: T3 BoS + pg (160 cells → 2 streams) ==========
# 2a: BoS 20 seeds × 4 cells (80 cells)
nohup bash -c "
CUDA_VISIBLE_DEVICES=2 python3 run_experiment_local.py \
  --games battle_of_the_sexes \
  --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
  --cells het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca \
  --episodes 30 --horizon 5 --memory 2 --out_dir $V2/exp_b_20seed \
  --models_het $QWEN $GLM --gsaca_warmup 5 --log_every 10
echo DONE_2a
" > "$LOGS/gpu2a_${TS}.log" 2>&1 &
echo "GPU2a PID=$! (BoS 20seed)"

sleep 15

# 2b: public_goods 20 seeds × 4 cells (80 cells)
nohup bash -c "
CUDA_VISIBLE_DEVICES=2 python3 run_experiment_local.py \
  --games public_goods \
  --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
  --cells het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca \
  --episodes 20 --horizon 5 --memory 2 --out_dir $V2/exp_b_20seed \
  --models_het $QWEN $GLM --gsaca_warmup 3 --log_every 10
echo DONE_2b
" > "$LOGS/gpu2b_${TS}.log" 2>&1 &
echo "GPU2b PID=$! (public_goods 20seed)"

sleep 15

# ========== GPU3: D1 noise sweep (120 cells → 2 streams) ==========
# 3a: noise=0.0 + 0.5 (60 cells)
nohup bash -c "
for noise in 0.0 0.5; do
  case \$noise in 0.0) tag=n00;; 0.5) tag=n05;; esac
  CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
    --games chicken hawk_dove deadlock stag_hunt battle_of_the_sexes \
    --seeds 42 43 44 45 46 --cells het_gsaca --episodes 30 --horizon 5 --memory 2 \
    --out_dir $V2/exp_d_fix/d1_\$tag --models_het $QWEN $GLM \
    --payoff_noise_std \$noise --gsaca_warmup 5 --log_every 10
  CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
    --games public_goods --seeds 42 43 44 45 46 --cells het_gsaca \
    --episodes 20 --horizon 5 --memory 2 --out_dir $V2/exp_d_fix/d1_\$tag \
    --models_het $QWEN $GLM --payoff_noise_std \$noise --gsaca_warmup 3 --log_every 10
done
echo DONE_3a
" > "$LOGS/gpu3a_${TS}.log" 2>&1 &
echo "GPU3a PID=$! (D1 noise 0.0+0.5)"

sleep 15

# 3b: noise=1.0 + 2.0 (60 cells)
nohup bash -c "
for noise in 1.0 2.0; do
  case \$noise in 1.0) tag=n10;; 2.0) tag=n20;; esac
  CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
    --games chicken hawk_dove deadlock stag_hunt battle_of_the_sexes \
    --seeds 42 43 44 45 46 --cells het_gsaca --episodes 30 --horizon 5 --memory 2 \
    --out_dir $V2/exp_d_fix/d1_\$tag --models_het $QWEN $GLM \
    --payoff_noise_std \$noise --gsaca_warmup 5 --log_every 10
  CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
    --games public_goods --seeds 42 43 44 45 46 --cells het_gsaca \
    --episodes 20 --horizon 5 --memory 2 --out_dir $V2/exp_d_fix/d1_\$tag \
    --models_het $QWEN $GLM --payoff_noise_std \$noise --gsaca_warmup 3 --log_every 10
done
echo DONE_3b
" > "$LOGS/gpu3b_${TS}.log" 2>&1 &
echo "GPU3b PID=$! (D1 noise 1.0+2.0)"

echo ""
echo "=== 8 processes launched (2 per GPU) ==="
echo "Expected: ~625 cells / 8 streams ≈ 4-5h (was ~17h)"
