#!/bin/bash
# ==============================================================================
# Queue Runner: waits for current GSACA experiments to finish, then recovers
# the lost phase1_5_unified data (409 metrics.json) by running 4 jobs.
#
# Pipeline:
#   1. Wait for PIDs 7021 7189 7518 7718 (current local-GPU experiments) to exit
#   2. Batch 1: job1 (deadlock+stag_hunt) + job2 (hawk_dove+BoS)  — parallel
#   3. Batch 2: job3 (chicken_repro+public_goods) + job4 (threshold ablation) — parallel
#   4. Run analyze_results.py + nash_equilibrium_analysis.py
#   5. Report final metrics count
#
# Each job uses multiprocessing.Pool(6) hitting SiliconFlow API.
# 2 jobs in parallel = 12 API workers (safe for rate limits).
# ==============================================================================
set -u
cd /data/lab/gsaca

OUT=/data/lab/results/phase1_5_unified
LOGDIR=/data/lab/results/phase1_5_logs
mkdir -p "$LOGDIR"

CURRENT_PIDS=(7021 7189 7518 7718)
QUEUE_LOG="$LOGDIR/queue_runner.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$QUEUE_LOG"; }

# ── Phase 0: Wait for current experiments ──────────────────────────────────
log "============================================"
log "QUEUE RUNNER STARTED"
log "Waiting for current GSACA experiments (PIDs: ${CURRENT_PIDS[*]})..."
log "============================================"

while true; do
    RUNNING=0
    for PID in "${CURRENT_PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done
    if [ "$RUNNING" -eq 0 ]; then
        log "All current experiments finished."
        break
    fi
    log "  Still $RUNNING/4 processes running. Sleeping 60s..."
    sleep 60
done

# Kill any lingering vLLM / python processes that might hold GPU memory
log "Cleaning up any lingering GPU processes..."
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 5

log "GPU status after cleanup:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | tee -a "$QUEUE_LOG"

# ── Phase 1: Batch 1 — job1 + job2 in parallel ────────────────────────────
log "============================================"
log "BATCH 1: job1 (deadlock+stag_hunt, 64 runs) + job2 (hawk_dove+BoS, 64 runs)"
log "Estimated time: ~67 min"
log "============================================"

python3 run_phase1_5.py --job job1 --workers 6 > "$LOGDIR/job1.log" 2>&1 &
PID_J1=$!
log "  job1 started (PID $PID_J1)"

python3 run_phase1_5.py --job job2 --workers 6 > "$LOGDIR/job2.log" 2>&1 &
PID_J2=$!
log "  job2 started (PID $PID_J2)"

log "  Waiting for batch 1..."
wait $PID_J1
J1_EXIT=$?
wait $PID_J2
J2_EXIT=$?
log "  job1 exited (code $J1_EXIT), job2 exited (code $J2_EXIT)"

M1=$(find "$OUT/phase1" -name metrics.json 2>/dev/null | wc -l)
log "  Batch 1 complete. phase1 metrics.json: $M1"

# ── Phase 2: Batch 2 — job3 + job4 in parallel ────────────────────────────
log "============================================"
log "BATCH 2: job3 (chicken_repro+public_goods, 48 runs) + job4 (threshold ablation, 64 runs)"
log "Estimated time: ~89 min"
log "============================================"

python3 run_phase1_5.py --job job3 --workers 6 > "$LOGDIR/job3.log" 2>&1 &
PID_J3=$!
log "  job3 started (PID $PID_J3)"

python3 run_phase1_5.py --job job4 --workers 6 > "$LOGDIR/job4.log" 2>&1 &
PID_J4=$!
log "  job4 started (PID $PID_J4)"

log "  Waiting for batch 2..."
wait $PID_J3
J3_EXIT=$?
wait $PID_J4
J4_EXIT=$?
log "  job3 exited (code $J3_EXIT), job4 exited (code $J4_EXIT)"

# ── Phase 3: Verify and analyze ────────────────────────────────────────────
log "============================================"
log "PHASE 3: Verification & Analysis"
log "============================================"

TOTAL=$(find "$OUT" -name metrics.json 2>/dev/null | wc -l)
log "Total metrics.json in phase1_5_unified: $TOTAL"

log "--- Per-directory breakdown ---"
for subdir in phase1 chicken_repro phase3_threshold phase4_public_goods; do
    if [ -d "$OUT/$subdir" ]; then
        CNT=$(find "$OUT/$subdir" -name metrics.json 2>/dev/null | wc -l)
        log "  $subdir: $CNT metrics.json"
    fi
done

log "--- Running analyze_results.py ---"
cd /data/lab/gsaca
python3 analyze_results.py > "$LOGDIR/analysis_output.txt" 2>&1
log "Analysis exit code: $?"

log "--- Running nash_equilibrium_analysis.py ---"
python3 nash_equilibrium_analysis.py > "$LOGDIR/nash_analysis_output.txt" 2>&1
log "Nash analysis exit code: $?"

log "============================================"
log "RECOVERY COMPLETE"
log "Total metrics.json: $TOTAL"
log "Logs: $LOGDIR/"
log "Analysis: $LOGDIR/analysis_output.txt"
log "Nash: $LOGDIR/nash_analysis_output.txt"
log "============================================"
