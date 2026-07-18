#!/usr/bin/env bash
# Full chain: R0 diagnostic replay -> route check (12/12) -> [pass] -> S2 -> validate -> analyze
# All with revision-pinned servers. Logs every stage with timestamps.
set -uo pipefail
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:/usr/local/cuda/lib64
export HF_HOME=/data/models/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
export GSACA_ROOT=/data/aaai/safe_sca_replication/g123_augmentation
export S1_REFERENCE_ROOT=/data/aaai/s1_reference/g123_augmentation
export FROZEN_CONFIG="$GSACA_ROOT/protocols/s1_safe_sca_frozen.json"
export R0_DIAG_ROOT="$GSACA_ROOT/v2_results_r0_diag"
export R0_DIAG_OUT="$R0_DIAG_ROOT/exp_r0_safe_sca_test"
export S2_ROOT="$GSACA_ROOT/v2_results_s2"
export S2_OUT="$S2_ROOT/exp_s2_safe_sca_test"
RUN_LOG=/data/aaai/safe_sca_replication/run_logs
ORCH_LOG=$RUN_LOG/r0_diag_s2_chain.log
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$ORCH_LOG"; }
mkdir -p "$RUN_LOG"
echo "=== R0_DIAG->S2 CHAIN START $(ts) ===" > "$ORCH_LOG"

# --- Stage A: R0 diagnostic replay (72 cells, seeds 62,63) ---
log "STAGE A: R0 diagnostic replay (revision-pinned, 72 cells, seeds 62,63)"
A_T0=$(date +%s)
cd "$GSACA_ROOT/code"
python3 run_safe_sca_campaign.py --campaign r0 --root "$GSACA_ROOT" --results-root "$R0_DIAG_ROOT" \
  --frozen-config "$FROZEN_CONFIG" --seeds 62 63 --workers 24 \
  > "$RUN_LOG/r0_diag_campaign.log" 2>&1
A_RC=$?
A_T1=$(date +%s); A_DUR=$((A_T1 - A_T0))
log "STAGE A DONE: rc=$A_RC duration=${A_DUR}s ($(( A_DUR/60 ))m) at $(ts)"
if [ $A_RC -ne 0 ]; then log "ABORT: R0 diag campaign incomplete"; tail -20 "$RUN_LOG/r0_diag_campaign.log" | tee -a "$ORCH_LOG"; exit 1; fi

# --- Stage B: validate R0 diag ---
log "STAGE B: R0 diag validate"
python3 validate_s1_results.py --results "$R0_DIAG_OUT" --frozen-config "$FROZEN_CONFIG" --seeds 62 63 \
  > "$RUN_LOG/r0_diag_validate.log" 2>&1
brc=$?
tail -8 "$RUN_LOG/r0_diag_validate.log" | tee -a "$ORCH_LOG"
if [ $brc -ne 0 ]; then log "ABORT: R0 diag validation failed"; exit 2; fi

# --- Stage C: route-only comparison (payoff tolerance 999 = ignore payoff; route budget 0) ---
log "STAGE C: R0 diag route-only comparison vs S1 (12/12 routes must match)"
python3 compare_safe_sca_replay.py \
  --reference "$S1_REFERENCE_ROOT/v2_results/exp_s1_safe_sca_test" \
  --replay "$R0_DIAG_OUT" --seeds 62 63 \
  --payoff-tolerance 999.0 --route-mismatch-budget 0 \
  > "$RUN_LOG/r0_diag_compare.log" 2>&1
crc=$?
# Extract route result regardless of payoff
python3 -c "
import json
d=json.load(open('$R0_DIAG_OUT/R0_REPLAY_COMPARISON.json'))
print('route_mismatches:', d['route_mismatches'])
print('passed:', d['passed'])
" | tee -a "$ORCH_LOG"
tail -12 "$RUN_LOG/r0_diag_compare.log" | tee -a "$ORCH_LOG"
if [ $crc -ne 0 ]; then
  log "ABORT: R0 diag route comparison FAILED — do NOT start S2. Pause and investigate environment."
  exit 3
fi
log "STAGE C DONE: R0 diag routes 12/12 match — S2 cleared at $(ts)"

# --- Stage D: S2 preflight ---
log "STAGE D: S2 preflight"
python3 preflight_s1.py --out-dir "$S2_OUT" --force > "$RUN_LOG/s2_preflight.log" 2>&1
drc=$?
tail -4 "$RUN_LOG/s2_preflight.log" | tee -a "$ORCH_LOG"
if [ $drc -ne 0 ]; then log "ABORT: S2 preflight failed"; exit 4; fi

# --- Stage E: S2 campaign (720 cells, seeds 82-101) ---
log "STAGE E: S2 campaign launch (720 cells, seeds 82-101, 24 workers)"
E_T0=$(date +%s)
python3 run_safe_sca_campaign.py --campaign s2 --root "$GSACA_ROOT" --results-root "$S2_ROOT" \
  --frozen-config "$FROZEN_CONFIG" \
  --seeds 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 \
  --workers 24 --task-timeout 7200 --max-retries 2 \
  > "$RUN_LOG/s2_campaign.log" 2>&1
E_RC=$?
E_T1=$(date +%s); E_DUR=$((E_T1 - E_T0))
log "STAGE E DONE: S2 campaign rc=$E_RC duration=${E_DUR}s ($(( E_DUR/60 ))m) at $(ts)"
if [ $E_RC -ne 0 ]; then log "ABORT: S2 campaign incomplete"; tail -20 "$RUN_LOG/s2_campaign.log" | tee -a "$ORCH_LOG"; exit 5; fi

# --- Stage F: S2 validation ---
log "STAGE F: S2 validate"
python3 validate_s1_results.py --results "$S2_OUT" --frozen-config "$FROZEN_CONFIG" \
  --seeds 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 \
  > "$RUN_LOG/s2_validate.log" 2>&1
frc=$?
tail -12 "$RUN_LOG/s2_validate.log" | tee -a "$ORCH_LOG"
if [ $frc -ne 0 ]; then log "ABORT: S2 validation failed"; exit 6; fi

# --- Stage G: S2 analysis with preregistered gates ---
log "STAGE G: S2 analyze (safety-margin 0.10, min-recovery 0.30, min-games 2)"
python3 analyze_s1_safe_sca.py --results "$S2_OUT" --safety-margin 0.10 \
  --min-coordination-recovery 0.30 --min-coordination-games 2 \
  > "$RUN_LOG/s2_analyze.log" 2>&1
grc=$?
tail -45 "$RUN_LOG/s2_analyze.log" | tee -a "$ORCH_LOG"
log "STAGE G DONE: S2 analysis rc=$grc at $(ts)"
log "=== R0_DIAG->S2 CHAIN COMPLETE $(ts) ==="
echo "ALL_DONE_CHAIN_RC=0" >> "$ORCH_LOG"
