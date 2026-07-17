#!/bin/bash
# Launch 6 parallel experiment workers using local vLLM servers.
# Preserves completed cells (skips existing metrics.json).
set -e
cd /data/lab/gsaca

# Reuse the existing output dir (has 6 completed het_notom cells)
OUTDIR=/data/lab/results/gsaca_full_20260712_120138
CELLS="het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca"
SEEDS="42 43 44 45 46"

echo "[$(date '+%H:%M:%S')] Launching 6 vLLM-backed workers..."
echo "Output: $OUTDIR"

GAMES_2P="chicken hawk_dove deadlock stag_hunt battle_of_the_sexes"
for game in $GAMES_2P; do
  python3 run_experiment_vllm.py \
    --games "$game" \
    --seeds $SEEDS \
    --episodes 30 \
    --cells $CELLS \
    --out_dir "$OUTDIR" \
    --log_every 10 \
    --gsaca_warmup 5 \
    > "$OUTDIR/${game}_vllm.log" 2>&1 &
  echo "  launched $game (PID $!)"
  sleep 1
done

python3 run_experiment_vllm.py \
  --games public_goods \
  --seeds $SEEDS \
  --episodes 20 \
  --cells $CELLS \
  --out_dir "$OUTDIR" \
  --log_every 5 \
  --gsaca_warmup 3 \
  > "$OUTDIR/public_goods_vllm.log" 2>&1 &
echo "  launched public_goods (PID $!)"

echo ""
echo "============================================"
echo "6 workers launched. Waiting for completion..."
echo "============================================"
wait

echo ""
echo "============================================"
echo "ALL DONE at $(date '+%H:%M:%S')"
echo "Total metrics.json: $(find "$OUTDIR" -name metrics.json | wc -l)"
echo "============================================"
