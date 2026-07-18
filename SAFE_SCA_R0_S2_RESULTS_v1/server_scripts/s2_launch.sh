#!/usr/bin/env bash
# S2 launch: route check -> preflight -> S2 campaign (32 workers) -> validate -> analyze
set -uo pipefail
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:/usr/local/cuda/lib64
export HF_HOME=/data/models/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
export GSACA_ROOT=/data/aaai/safe_sca_replication/g123_augmentation
export S1_REFERENCE_ROOT=/data/aaai/s1_reference/g123_augmentation
export FROZEN_CONFIG="$GSACA_ROOT/protocols/s1_safe_sca_frozen.json"
export R0_DIAG_OUT="$GSACA_ROOT/v2_results_r0_diag/exp_r0_safe_sca_test"
export S2_ROOT="$GSACA_ROOT/v2_results_s2"
export S2_OUT="$S2_ROOT/exp_s2_safe_sca_test"
RUN_LOG=/data/aaai/safe_sca_replication/run_logs
ORCH_LOG=$RUN_LOG/s2_final.log
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$ORCH_LOG"; }
echo "=== S2 FINAL CHAIN START $(ts) ===" > "$ORCH_LOG"
cd "$GSACA_ROOT/code"

# Route check (must have 72/72 cells)
CELLS=$(find "$R0_DIAG_OUT" -name metrics.json | wc -l)
log "R0 diag cells: $CELLS/72"
if [ "$CELLS" -lt 72 ]; then log "ABORT: not all R0 diag cells done"; exit 1; fi

log "STAGE C: route-only comparison"
python3 compare_safe_sca_replay.py \
  --reference "$S1_REFERENCE_ROOT/v2_results/exp_s1_safe_sca_test" \
  --replay "$R0_DIAG_OUT" --seeds 62 63 \
  --payoff-tolerance 999.0 --route-mismatch-budget 0 \
  > "$RUN_LOG/r0_diag_final_compare.log" 2>&1
crc=$?
python3 -c "
import json
d=json.load(open('$R0_DIAG_OUT/R0_REPLAY_COMPARISON.json'))
print('route_mismatches:', d['route_mismatches'], 'passed:', d['passed'])
" | tee -a "$ORCH_LOG"
if [ $crc -ne 0 ]; then log "ABORT: route check failed"; exit 3; fi
log "STAGE C DONE: 12/12 routes match at $(ts)"

log "STAGE D: S2 preflight"
python3 preflight_s1.py --out-dir "$S2_OUT" --force > "$RUN_LOG/s2_final_preflight.log" 2>&1
[ $? -ne 0 ] && { log "ABORT: preflight failed"; exit 4; }

log "STAGE E: S2 campaign (720 cells, 32 workers, optimized vLLM)"
E_T0=$(date +%s)
python3 run_safe_sca_campaign.py --campaign s2 --root "$GSACA_ROOT" --results-root "$S2_ROOT" \
  --frozen-config "$FROZEN_CONFIG" \
  --seeds 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 \
  --workers 32 --task-timeout 7200 --max-retries 2 \
  > "$RUN_LOG/s2_final_campaign.log" 2>&1
E_RC=$?
E_DUR=$(( $(date +%s) - E_T0 ))
log "STAGE E DONE: rc=$E_RC duration=${E_DUR}s ($(( E_DUR/60 ))m) at $(ts)"
[ $E_RC -ne 0 ] && { log "ABORT: S2 incomplete"; tail -20 "$RUN_LOG/s2_final_campaign.log" | tee -a "$ORCH_LOG"; exit 5; }

log "STAGE F: S2 validate"
python3 validate_s1_results.py --results "$S2_OUT" --frozen-config "$FROZEN_CONFIG" \
  --seeds 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 \
  > "$RUN_LOG/s2_final_validate.log" 2>&1
tail -12 "$RUN_LOG/s2_final_validate.log" | tee -a "$ORCH_LOG"

log "STAGE G: S2 analyze"
python3 analyze_s1_safe_sca.py --results "$S2_OUT" --safety-margin 0.10 \
  --min-coordination-recovery 0.30 --min-coordination-games 2 \
  > "$RUN_LOG/s2_final_analyze.log" 2>&1
tail -50 "$RUN_LOG/s2_final_analyze.log" | tee -a "$ORCH_LOG"
log "=== S2 FINAL CHAIN COMPLETE $(ts) ==="
echo "ALL_DONE=0" >> "$ORCH_LOG"
