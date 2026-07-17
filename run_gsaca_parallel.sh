#!/bin/bash
# GSACA Quick-Start: launch on 2 GPUs in parallel
# Usage: bash run_gsaca_parallel.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/hettom_experiments"

OUTDIR="../results/gsaca_$(date +%Y%m%d_%H%M%S)"
EPISODES=25
SEEDS="1 2 3 4 5 6 7 8"

# Priority order: anti-coordination games first (strongest signal)
GAMES_GPU0="chicken deadlock"           # strongest anti-coordination
GAMES_GPU1="hawk_dove stag_hunt battle_of_the_sexes"  # remaining + coordination + conflict

echo "============================================"
echo "GSACA Parallel Launch (2 GPUs)"
echo "Output: $OUTDIR"
echo "Episodes: $EPISODES"
echo "Seeds: $SEEDS"
echo "GPU 0: $GAMES_GPU0"
echo "GPU 1: $GAMES_GPU1"
echo "============================================"

# Kill any remaining processes on exit
cleanup() {
    echo "Cleaning up..."
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

# GPU 0: chicken + deadlock
(
    export CUDA_VISIBLE_DEVICES=0
    echo "[$(date '+%H:%M:%S')] GPU 0 starting: chicken deadlock"
    python3 run_gsaca.py \
        --games $GAMES_GPU0 \
        --seeds $SEEDS \
        --episodes $EPISODES \
        --out-dir "${OUTDIR}_gpu0" \
        2>&1 | tee "${OUTDIR}_gpu0.log"
    echo "[$(date '+%H:%M:%S')] GPU 0 done"
) &
PID0=$!

# GPU 1: hawk_dove + stag_hunt + battle_of_the_sexes
(
    export CUDA_VISIBLE_DEVICES=1
    echo "[$(date '+%H:%M:%S')] GPU 1 starting: $GAMES_GPU1"
    python3 run_gsaca.py \
        --games $GAMES_GPU1 \
        --seeds $SEEDS \
        --episodes $EPISODES \
        --out-dir "${OUTDIR}_gpu1" \
        2>&1 | tee "${OUTDIR}_gpu1.log"
    echo "[$(date '+%H:%M:%S')] GPU 1 done"
) &
PID1=$?

echo "GPU 0 PID: $PID0"
echo "GPU 1 PID: $PID1"
echo "Waiting for both to complete..."
wait $PID0
wait $PID1

echo ""
echo "============================================"
echo "GSACA complete!"
echo "Results: ${OUTDIR}_gpu0"
echo "         ${OUTDIR}_gpu1"
echo "============================================"
