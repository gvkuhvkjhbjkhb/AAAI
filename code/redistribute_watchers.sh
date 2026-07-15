#!/bin/bash
# 3个watcher并行运行：
#   watcher-GPU1: 等 E sweep(PID 6827)结束 → GPU1 跑 public_goods seeds 48-54
#   watcher-GPU3: 等 D1(PID 5459 AND 6956)都结束 → GPU3 跑 public_goods seeds 55-61
#   watcher-GPU0: 等 T1(PID 5980)结束 → GPU0 跑 public_goods 任何剩余

cd /data/lab/gsaca/code
QWEN="Qwen/Qwen2.5-7B-Instruct"
GLM="THUDM/GLM-4-9B-0414"
V2="/data/lab/results/v2"
LOGS="$V2/logs"

# ===== Watcher 1: GPU1 — 等 E sweep 完成 → 跑 pg seeds 48-54 =====
(
    echo "[$(date '+%H:%M:%S')] GPU1 watcher: waiting for PID 6827 (E sweep)..."
    while kill -0 6827 2>/dev/null; do sleep 30; done
    sleep 5  # 等模型释放
    echo "[$(date '+%H:%M:%S')] GPU1 watcher: E sweep done, launching pg seeds 48-54"
    CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
        --games public_goods \
        --seeds 48 49 50 51 52 53 54 \
        --cells het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca \
        --episodes 20 --horizon 5 --memory 2 \
        --out_dir $V2/exp_b_20seed \
        --models_het $QWEN $GLM \
        --gsaca_warmup 3 --log_every 10 \
        2>&1
    echo "[$(date '+%H:%M:%S')] GPU1 watcher: pg seeds 48-54 DONE"
) >> "$LOGS/watcher_gpu1_pg.log" 2>&1 &

# ===== Watcher 2: GPU3 — 等 D1 两个进程都完成 → 跑 pg seeds 55-61 =====
(
    echo "[$(date '+%H:%M:%S')] GPU3 watcher: waiting for PID 5459 AND 6956 (D1)..."
    while kill -0 5459 2>/dev/null || kill -0 6956 2>/dev/null; do sleep 30; done
    sleep 5
    echo "[$(date '+%H:%M:%S')] GPU3 watcher: D1 done, launching pg seeds 55-61"
    CUDA_VISIBLE_DEVICES=3 python3 run_experiment_local.py \
        --games public_goods \
        --seeds 55 56 57 58 59 60 61 \
        --cells het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca \
        --episodes 20 --horizon 5 --memory 2 \
        --out_dir $V2/exp_b_20seed \
        --models_het $QWEN $GLM \
        --gsaca_warmup 3 --log_every 10 \
        2>&1
    echo "[$(date '+%H:%M:%S')] GPU3 watcher: pg seeds 55-61 DONE"
) >> "$LOGS/watcher_gpu3_pg.log" 2>&1 &

# ===== Watcher 3: GPU0 — 等 T1 完成 → 跑 pg 任何剩余 =====
(
    echo "[$(date '+%H:%M:%S')] GPU0 watcher: waiting for PID 5980 (T1 3arm)..."
    while kill -0 5980 2>/dev/null; do sleep 30; done
    sleep 5
    echo "[$(date '+%H:%M:%S')] GPU0 watcher: T1 done, launching pg all remaining seeds"
    # 跑所有seeds（skip机制会自动跳过已完成的）
    CUDA_VISIBLE_DEVICES=0 python3 run_experiment_local.py \
        --games public_goods \
        --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
        --cells het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca \
        --episodes 20 --horizon 5 --memory 2 \
        --out_dir $V2/exp_b_20seed \
        --models_het $QWEN $GLM \
        --gsaca_warmup 3 --log_every 10 \
        2>&1
    echo "[$(date '+%H:%M:%S')] GPU0 watcher: pg all remaining DONE"
) >> "$LOGS/watcher_gpu0_pg.log" 2>&1 &

echo "3 watchers launched:"
echo "  GPU1 watcher → pg seeds 48-54 (after E sweep PID 6827 ends)"
echo "  GPU3 watcher → pg seeds 55-61 (after D1 PID 5459+6956 end)"
echo "  GPU0 watcher → pg all remaining (after T1 PID 5980 ends)"
