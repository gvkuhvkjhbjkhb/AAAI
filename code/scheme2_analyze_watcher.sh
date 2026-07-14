#!/usr/bin/env bash
# Scheme 2 analysis auto-runner — polls until 60 silent metrics exist, then
# runs scheme2_analyze.py and saves output. Fully autonomous; no GPU.
set -u
cd /data/lab/gsaca
LOG=/data/lab/results/v2/scheme2_analyze_watcher.log
S2=/data/lab/results/v2/exp_scheme2_silent
OUT=/data/lab/results/v2/scheme2_offline

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== Scheme 2 analysis watcher started ==="

while true; do
  N=$(find "$S2" -name metrics.json 2>/dev/null | wc -l)
  log "  silent metrics: $N / 60"
  if [ "$N" -ge 60 ]; then break; fi
  sleep 120
done

log "60 metrics reached. Running scheme2_analyze.py..."
python3 scheme2_analyze.py 2>&1 | tee "$OUT/analysis_output.txt"
log "=== Scheme 2 analysis complete -> $OUT/analysis_output.txt ==="
