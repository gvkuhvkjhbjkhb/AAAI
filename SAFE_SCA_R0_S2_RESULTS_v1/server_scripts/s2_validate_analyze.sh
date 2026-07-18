#!/usr/bin/env bash
# Wait for S2 campaign (PID in s2_campaign_pid.txt) to finish, then validate + analyze
set -uo pipefail
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:/usr/local/cuda/lib64
export HF_HOME=/data/models/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
export GSACA_ROOT=/data/aaai/safe_sca_replication/g123_augmentation
export FROZEN_CONFIG="$GSACA_ROOT/protocols/s1_safe_sca_frozen.json"
export S2_OUT="$GSACA_ROOT/v2_results_s2/exp_s2_safe_sca_test"
RUN_LOG=/data/aaai/safe_sca_replication/run_logs
ORCH_LOG=$RUN_LOG/s2_validate_analyze.log
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$ORCH_LOG"; }
echo "=== S2 VALIDATE+ANALYZE WAIT START $(ts) ===" > "$ORCH_LOG"
cd "$GSACA_ROOT/code"

S2_PID=$(cat "$RUN_LOG/s2_campaign_pid.txt")
log "Waiting for S2 campaign PID=$S2_PID"
while kill -0 "$S2_PID" 2>/dev/null; do
  cells=$(find "$S2_OUT" -name metrics.json 2>/dev/null | wc -l)
  log "  S2 running... cells=$cells/720"
  sleep 60
done
cells=$(find "$S2_OUT" -name metrics.json 2>/dev/null | wc -l)
log "S2 campaign finished. cells=$cells/720 at $(ts)"

log "STAGE F: S2 validate"
python3 validate_s1_results.py --results "$S2_OUT" --frozen-config "$FROZEN_CONFIG" \
  --seeds 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 \
  > "$RUN_LOG/s2_final_validate.log" 2>&1
tail -15 "$RUN_LOG/s2_final_validate.log" | tee -a "$ORCH_LOG"

log "STAGE G: S2 analyze (safety-margin 0.10, min-recovery 0.30, min-games 2)"
python3 analyze_s1_safe_sca.py --results "$S2_OUT" --safety-margin 0.10 \
  --min-coordination-recovery 0.30 --min-coordination-games 2 \
  > "$RUN_LOG/s2_final_analyze.log" 2>&1
tail -50 "$RUN_LOG/s2_final_analyze.log" | tee -a "$ORCH_LOG"
log "=== S2 COMPLETE $(ts) ==="
echo "ALL_DONE=0" >> "$ORCH_LOG"
