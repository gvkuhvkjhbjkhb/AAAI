#!/usr/bin/env bash
# Wait for fix run to complete, then launch anti-coordination enhancement tests.
# Tests 3 schemes + combo on chicken+hawk_dove+deadlock (5 seeds, 30 episodes).
# Expected runtime: ~20-30 min (3 games × 5 cells × 5 seeds, 2 GPU).

set -e
GSACA=/data/lab/gsaca
OUT=/data/lab/results/v2/exp_anti_test
LOG=/data/lab/results/v2/anti_test_wait.log

echo "[$(date '+%H:%M:%S')] Waiting for fix run to free GPUs..." | tee "$LOG"

# Wait until fix_fast orchestrator exits (no run_v2_fix_fast process)
while pgrep -f run_v2_fix_fast > /dev/null 2>&1; do
    echo "[$(date '+%H:%M:%S')] fix run still active, GPU busy. Sleeping 120s..." >> "$LOG"
    sleep 120
done

echo "[$(date '+%H:%M:%S')] Fix run complete! GPU should be free. Launching anti-coord tests..." | tee -a "$LOG"

sleep 10  # let any final cleanup happen
mkdir -p "$OUT"

CELLS="het_gsaca het_role_asym het_hist_split het_adapt_interv het_combo_anti"
SEEDS="42 43 44 45 46"

# GPU0: chicken + hawk_dove (2 games × 5 seeds × 5 cells = 50 cells)
CUDA_VISIBLE_DEVICES=0 python3 $GSACA/run_experiment_local.py \
    --games chicken hawk_dove --seeds $SEEDS --episodes 30 --horizon 5 --memory 2 \
    --cells $CELLS --out_dir "$OUT" \
    --log_every 10 --gsaca_warmup 5 --force \
    > "$OUT/gpu0_chicken_hawkdove.log" 2>&1 &
PID0=$!
echo "[$(date '+%H:%M:%S')] GPU0 PID $PID0: chicken+hawk_dove" | tee -a "$LOG"

sleep 60  # stagger to avoid simultaneous model loading

# GPU1: deadlock (1 game × 5 seeds × 5 cells = 25 cells)
CUDA_VISIBLE_DEVICES=1 python3 $GSACA/run_experiment_local.py \
    --games deadlock --seeds $SEEDS --episodes 30 --horizon 5 --memory 2 \
    --cells $CELLS --out_dir "$OUT" \
    --log_every 10 --gsaca_warmup 5 --force \
    > "$OUT/gpu1_deadlock.log" 2>&1 &
PID1=$!
echo "[$(date '+%H:%M:%S')] GPU1 PID $PID1: deadlock" | tee -a "$LOG"

echo "[$(date '+%H:%M:%S')] Both workers launched. Waiting..." | tee -a "$LOG"
wait $PID0 $PID1
echo "[$(date '+%H:%M:%S')] ALL ANTI-COORD TESTS DONE" | tee -a "$LOG"
echo "Metrics: $(find $OUT -name metrics.json | wc -l)" | tee -a "$LOG"
