#!/bin/bash
# Queued launcher: waits for GPU slots to free, then starts remaining games.
# GPU 0: after chicken finishes → start deadlock
# GPU 1: after stag_hunt finishes → start public_goods
set -e
cd /data/lab/gsaca

OUTDIR=/data/lab/results/gsaca_full_20260712_120138
CELLS="het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca"
SEEDS="42 43 44 45 46"

echo "[$(date '+%H:%M:%S')] Queued launcher started. Monitoring GPU slots..."

# --- GPU 0: wait for chicken to finish, then start deadlock ---
(
    echo "[$(date '+%H:%M:%S')] GPU0: waiting for chicken to finish..."
    while true; do
        if grep -q "Complete:" "$OUTDIR/chicken_local.log" 2>/dev/null; then
            break
        fi
        sleep 30
    done
    echo "[$(date '+%H:%M:%S')] GPU0: chicken done. Starting deadlock..."
    CUDA_VISIBLE_DEVICES=0 python3 run_experiment_local.py \
        --games deadlock --seeds $SEEDS --episodes 30 \
        --cells $CELLS --out_dir "$OUTDIR" \
        --log_every 10 --gsaca_warmup 5 \
        > "$OUTDIR/deadlock_local.log" 2>&1
    echo "[$(date '+%H:%M:%S')] GPU0: deadlock done."
) &

# --- GPU 1: wait for stag_hunt to finish, then start public_goods ---
(
    echo "[$(date '+%H:%M:%S')] GPU1: waiting for stag_hunt to finish..."
    while true; do
        if grep -q "Complete:" "$OUTDIR/stag_hunt_local.log" 2>/dev/null; then
            break
        fi
        sleep 30
    done
    echo "[$(date '+%H:%M:%S')] GPU1: stag_hunt done. Starting public_goods..."
    CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
        --games public_goods --seeds $SEEDS --episodes 20 \
        --cells $CELLS --out_dir "$OUTDIR" \
        --log_every 5 --gsaca_warmup 3 \
        > "$OUTDIR/public_goods_local.log" 2>&1
    echo "[$(date '+%H:%M:%S')] GPU1: public_goods done."
) &

echo "[$(date '+%H:%M:%S')] Queued jobs waiting. PID0=$!"
wait
echo "[$(date '+%H:%M:%S')] All queued jobs complete."
echo "Total metrics.json: $(find "$OUTDIR" -name metrics.json | wc -l)"
