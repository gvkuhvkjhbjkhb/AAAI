#!/usr/bin/env bash
# Scheme 2 auto-launcher — waits until the ongoing run_final_fast.sh experiment
# has FULLY drained (no run_experiment_local procs, driver exited), then launches
# Scheme 2 silent-anti-coord on 2 GPUs. Guarantees zero disturbance to the
# ongoing run.
set -u
cd /data/lab/gsaca
LOG=/data/lab/results/v2/scheme2_launcher.log
V2=/data/lab/results/v2
QWEN=Qwen/Qwen2.5-7B-Instruct
GLM=THUDM/GLM-4-9B-0414

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Scheme 2 waiter started ==="
log "waiting for ongoing run_final_fast.sh to drain (no run_experiment_local procs)..."

while true; do
  NPROC=$(ps aux | grep "run_experiment_local" | grep -v grep | wc -l)
  DRIVER=$(ps aux | grep "run_final_fast.sh" | grep -v grep | wc -l)
  if [ "$NPROC" -eq 0 ] && [ "$DRIVER" -eq 0 ]; then
    break
  fi
  log "  still busy: $NPROC exp procs, driver=$DRIVER — sleep 120s"
  sleep 120
done

log "ongoing run drained. GPUs should be free."
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$LOG"

# wait a little for VRAM to actually release after procs exit
sleep 30
log "launching Scheme 2: 2 workers (GPU0=seeds42-51, GPU1=seeds52-61)"

mkdir -p "$V2/exp_scheme2_silent"
CUDA_VISIBLE_DEVICES=0 nohup python3 scheme2_silent.py \
  --games chicken hawk_dove deadlock --seeds 42 43 44 45 46 47 48 49 50 51 \
  --episodes 30 --log_every 10 --force \
  > "$V2/exp_scheme2_silent/worker0.log" 2>&1 &
W0=$!
sleep 60
CUDA_VISIBLE_DEVICES=1 nohup python3 scheme2_silent.py \
  --games chicken hawk_dove deadlock --seeds 52 53 54 55 56 57 58 59 60 61 \
  --episodes 30 --log_every 10 --force \
  > "$V2/exp_scheme2_silent/worker1.log" 2>&1 &
W1=$!
log "launched worker0 pid=$W0 (GPU0, seeds 42-51), worker1 pid=$W1 (GPU1, seeds 52-61)"

wait $W0; log "worker0 done (rc=$?)"
wait $W1; log "worker1 done (rc=$?)"

N=$(find "$V2/exp_scheme2_silent" -name metrics.json | wc -l)
log "=== Scheme 2 complete: $N metrics (expected 60) ==="
for g in chicken hawk_dove deadlock; do
  echo -n "  $g: " | tee -a "$LOG"
  find "$V2/exp_scheme2_silent/$g" -name metrics.json 2>/dev/null | wc -l | tee -a "$LOG"
done
