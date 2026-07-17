#!/bin/bash
# Launch 6 local-HF-GPU workers: 3 per GPU, 4-bit quantization.
# GPU utilization is near 0% (Python/generate overhead bound, not compute bound),
# so 3 parallel processes per GPU give near-linear speedup.
#
# GPU 0: chicken, hawk_dove, deadlock (anti-coordination, 30 episodes)
# GPU 1: stag_hunt, battle_of_the_sexes (30 ep) + public_goods (20 ep)
#
# Each worker loads both models in 4-bit: Qwen ~4GB + GLM ~5GB = ~9GB.
# 3 processes per GPU: 3 × 9GB = 27GB < 32GB → fits.
# Starts are staggered 60s to avoid simultaneous model loading VRAM spikes.
set -e
cd /data/lab/gsaca

OUTDIR=/data/lab/results/gsaca_full_20260712_120138
CELLS="het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca"
SEEDS="42 43 44 45 46"

echo "[$(date '+%H:%M:%S')] Launching 6 local-HF-GPU workers (3 per GPU)..."

# --- GPU 0: 3 anti-coordination games ---
for game in chicken hawk_dove deadlock; do
    CUDA_VISIBLE_DEVICES=0 python3 run_experiment_local.py \
        --games "$game" --seeds $SEEDS --episodes 30 \
        --cells $CELLS --out_dir "$OUTDIR" \
        --log_every 10 --gsaca_warmup 5 \
        > "$OUTDIR/${game}_local.log" 2>&1 &
    echo "  GPU0: $game (PID $!)"
    sleep 60
done

# --- GPU 1: 2 coordination games + public_goods ---
for game in stag_hunt battle_of_the_sexes; do
    CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
        --games "$game" --seeds $SEEDS --episodes 30 \
        --cells $CELLS --out_dir "$OUTDIR" \
        --log_every 10 --gsaca_warmup 5 \
        > "$OUTDIR/${game}_local.log" 2>&1 &
    echo "  GPU1: $game (PID $!)"
    sleep 60
done

CUDA_VISIBLE_DEVICES=1 python3 run_experiment_local.py \
    --games public_goods --seeds $SEEDS --episodes 20 \
    --cells $CELLS --out_dir "$OUTDIR" \
    --log_every 5 --gsaca_warmup 3 \
    > "$OUTDIR/public_goods_local.log" 2>&1 &
echo "  GPU1: public_goods (PID $!)"

echo ""
echo "============================================"
echo "6 workers launched (3 per GPU). Waiting..."
echo "GPU0: chicken, hawk_dove, deadlock"
echo "GPU1: stag_hunt, battle_of_the_sexes, public_goods"
echo "============================================"
wait

echo ""
echo "============================================"
echo "ALL DONE at $(date '+%H:%M:%S')"
echo "Total metrics.json: $(find "$OUTDIR" -name metrics.json | wc -l)"
echo "============================================"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
