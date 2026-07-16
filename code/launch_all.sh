#!/bin/bash
# Launch all 6 game workers in parallel (background).
# 2-player games: 5 seeds (42-46), 30 episodes.
# public_goods: 5 seeds (42-46), 20 episodes (faster, 4 agents).
set -e
cd /data/lab/gsaca

OUTDIR=/data/lab/results/gsaca_full_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTDIR"

GAMES_2P="chicken hawk_dove deadlock stag_hunt battle_of_the_sexes"
CELLS="het_notom het_gated_atom_talk het_dp_gated_atom_talk het_gsaca"
SEEDS="42 43 44 45 46"

PIDS=""
for game in $GAMES_2P; do
  echo "[$(date '+%H:%M:%S')] Launching $game (30ep, 5seeds)..."
  python3 run_experiment.py \
    --games "$game" \
    --seeds $SEEDS \
    --episodes 30 \
    --cells $CELLS \
    --out_dir "$OUTDIR" \
    --log_every 10 \
    --gsaca_warmup 5 \
    > "$OUTDIR/${game}.log" 2>&1 &
  PIDS="$PIDS $!"
  sleep 2
done

echo "[$(date '+%H:%M:%S')] Launching public_goods (20ep, 5seeds)..."
python3 run_experiment.py \
  --games public_goods \
  --seeds $SEEDS \
  --episodes 20 \
  --cells $CELLS \
  --out_dir "$OUTDIR" \
  --log_every 5 \
  --gsaca_warmup 3 \
  > "$OUTDIR/public_goods.log" 2>&1 &
PIDS="$PIDS $!"

echo "============================================"
echo "Launched 6 workers. PIDs:$PIDS"
echo "Output: $OUTDIR"
echo "Logs: $OUTDIR/*.log"
echo "============================================"
echo "Waiting for all to complete..."
wait

echo ""
echo "============================================"
echo "ALL WORKERS COMPLETE at $(date '+%H:%M:%S')"
echo "============================================"
echo "Output dir: $OUTDIR"
echo ""
echo "=== Quick summary ==="
find "$OUTDIR" -name metrics.json | wc -l
echo "metrics.json files found"
