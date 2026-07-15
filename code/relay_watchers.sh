#!/bin/bash
# 接力watcher系统 — 优化所有GPU利用率
cd /data/lab/gsaca/code
QWEN="Qwen/Qwen2.5-7B-Instruct"
GLM="THUDM/GLM-4-9B-0414"
V2="/data/lab/results/v2"
LOGS="$V2/logs"

# ===== Watcher A: GPU0 — 等tau=0.2完成(PID 5980) → 只跑tau=0.4 =====
(
    echo "[$(date +%H:%M:%S)] GPU0 watcher: waiting for PID 5980 (tau=0.2) to finish..."
    while kill -0 5980 2>/dev/null; do sleep 20; done
    sleep 3
    echo "[$(date +%H:%M:%S)] GPU0 watcher: tau=0.2 done, launching tau=0.4 on GPU0"
    CUDA_VISIBLE_DEVICES=0 python3 run_experiment_local.py \
        --games chicken hawk_dove deadlock stag_hunt battle_of_the_sexes public_goods \
        --seeds 42 43 44 45 46 --cells het_3arm --episodes 30 --horizon 5 --memory 2 \
        --out_dir $V2/exp_3arm/tau_0.4 --models_het $QWEN $GLM \
        --abstain_tau 0.4 --gsaca_warmup 5 --log_every 10 2>&1
    echo "[$(date +%H:%M:%S)] GPU0 watcher: tau=0.4 DONE"
    # tau=0.4完成后，兜底跑tau=0.6剩余
    echo "[$(date +%H:%M:%S)] GPU0 watcher: launching tau=0.6 remaining on GPU0"
    CUDA_VISIBLE_DEVICES=0 python3 run_experiment_local.py \
        --games chicken hawk_dove deadlock stag_hunt battle_of_the_sexes public_goods \
        --seeds 42 43 44 45 46 --cells het_3arm --episodes 30 --horizon 5 --memory 2 \
        --out_dir $V2/exp_3arm/tau_0.6 --models_het $QWEN $GLM \
        --abstain_tau 0.6 --gsaca_warmup 5 --log_every 10 2>&1
    echo "[$(date +%H:%M:%S)] GPU0 watcher: tau=0.6 remaining DONE"
) >> "$LOGS/relay_gpu0.log" 2>&1 &

# ===== Watcher B: GPU1 — 等E sweep完成(PID 6827) → 跑tau=0.6的2-player =====
(
    echo "[$(date +%H:%M:%S)] GPU1 watcher: waiting for PID 6827 (E sweep)..."
    while kill -0 6827 2>/dev/null; do sleep 20; done
    sleep 3
    echo "[$(date +%H:%M:%S)] GPU1 watcher: E sweep done, launching tau=0.6 2-player on GPU1"
    CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
        --games chicken hawk_dove deadlock stag_hunt battle_of_the_sexes \
        --seeds 42 43 44 45 46 --cells het_3arm --episodes 30 --horizon 5 --memory 2 \
        --out_dir $V2/exp_3arm/tau_0.6 --models_het $QWEN $GLM \
        --abstain_tau 0.6 --gsaca_warmup 5 --log_every 10 2>&1
    echo "[$(date +%H:%M:%S)] GPU1 watcher: tau=0.6 2-player DONE"
    # 完成后接手pg剩余
    echo "[$(date +%H:%M:%S)] GPU1 watcher: launching pg remaining on GPU1"
    CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
        --games public_goods --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
        --cells het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca \
        --episodes 20 --horizon 5 --memory 2 --out_dir $V2/exp_b_20seed \
        --models_het $QWEN $GLM --gsaca_warmup 3 --log_every 10 2>&1
    echo "[$(date +%H:%M:%S)] GPU1 watcher: pg remaining DONE"
) >> "$LOGS/relay_gpu1.log" 2>&1 &

# ===== Watcher C: GPU3 — 等D1完成(PID 5459+6956) → 跑tau=0.6的public_goods =====
(
    echo "[$(date +%H:%M:%S)] GPU3 watcher: waiting for PID 5459 AND 6956 (D1)..."
    while kill -0 5459 2>/dev/null || kill -0 6956 2>/dev/null; do sleep 20; done
    sleep 3
    echo "[$(date +%H:%M:%S)] GPU3 watcher: D1 done, launching tau=0.6 public_goods on GPU3"
    CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
        --games public_goods \
        --seeds 42 43 44 45 46 --cells het_3arm --episodes 30 --horizon 5 --memory 2 \
        --out_dir $V2/exp_3arm/tau_0.6 --models_het $QWEN $GLM \
        --abstain_tau 0.6 --gsaca_warmup 5 --log_every 10 2>&1
    echo "[$(date +%H:%M:%S)] GPU3 watcher: tau=0.6 public_goods DONE"
    # 完成后接手pg剩余
    echo "[$(date +%H:%M:%S)] GPU3 watcher: launching pg remaining on GPU3"
    CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
        --games public_goods --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
        --cells het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca \
        --episodes 20 --horizon 5 --memory 2 --out_dir $V2/exp_b_20seed \
        --models_het $QWEN $GLM --gsaca_warmup 3 --log_every 10 2>&1
    echo "[$(date +%H:%M:%S)] GPU3 watcher: pg remaining DONE"
) >> "$LOGS/relay_gpu3.log" 2>&1 &

# ===== Watcher D: GPU2 — 等pg完成(PID 5330) → 兜底跑tau=0.6剩余 =====
(
    echo "[$(date +%H:%M:%S)] GPU2 watcher: waiting for PID 5330 (pg)..."
    while kill -0 5330 2>/dev/null; do sleep 20; done
    sleep 3
    echo "[$(date +%H:%M:%S)] GPU2 watcher: pg done, launching tau=0.6 remaining on GPU2"
    CUDA_VISIBLE_DEVICES=2 python3 run_experiment_local.py \
        --games chicken hawk_dove deadlock stag_hunt battle_of_the_sexes public_goods \
        --seeds 42 43 44 45 46 --cells het_3arm --episodes 30 --horizon 5 --memory 2 \
        --out_dir $V2/exp_3arm/tau_0.6 --models_het $QWEN $GLM \
        --abstain_tau 0.6 --gsaca_warmup 5 --log_every 10 2>&1
    echo "[$(date +%H:%M:%S)] GPU2 watcher: tau=0.6 remaining DONE"
) >> "$LOGS/relay_gpu2.log" 2>&1 &

echo "4 relay watchers launched:"
echo "  GPU0: tau=0.2 done → tau=0.4 → tau=0.6兜底"
echo "  GPU1: E sweep done → tau=0.6 2-player → pg兜底"
echo "  GPU3: D1 done → tau=0.6 public_goods → pg兜底"
echo "  GPU2: pg done → tau=0.6 兜底"
