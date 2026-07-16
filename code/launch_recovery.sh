#!/bin/bash
# Recovery launcher: 9 missing runs across 4 GPUs, 2 processes/GPU (dual-process).
# Each process loads both models once in 4-bit (~9GB); 2 procs/GPU = ~18GB < 32GB.
set -u
cd /data/lab/gsaca/code || { echo "ERROR: code dir missing"; exit 1; }

OUT="/data/lab/results/v2/exp_b_20seed"
LOGDIR="/data/lab/results/v2/logs"
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)
PIDFILE="$LOGDIR/recovery_pids_${TS}.txt"
GW="Qwen/Qwen2.5-7B-Instruct"
GL="THUDM/GLM-4-9B-0414"
COMMON="--episodes 20 --horizon 5 --memory 2 --out_dir $OUT --models_het $GW $GL --gsaca_warmup 3 --log_every 10 --force"

launch() {
  local gpu=$1 seed=$2 cells=$3 tag=$4
  local log="$LOGDIR/recover_${tag}.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup python3 run_experiment_local.py \
    --games public_goods --seeds $seed --cells $cells $COMMON \
    > "$log" 2>&1 &
  local pid=$!
  echo "GPU$gpu  PID=$pid  seed=$seed  cells=[$cells]  log=$log" | tee -a "$PIDFILE"
}

echo "=== Recovery launch @ $(date) ==="
echo "Missing: gsaca{50,51,54,56,58,60} gated{51} dp{51} notom{51}"
echo

# GPU0: two heavy het_gsaca (~44min each)
launch 0 50  het_gsaca               g0_gsaca_s50
launch 0 54  het_gsaca               g0_gsaca_s54
# GPU1: two heavy het_gsaca
launch 1 56  het_gsaca               g1_gsaca_s56
launch 1 58  het_gsaca               g1_gsaca_s58
# GPU2: gsaca_s60 + gsaca_s51 (bundled with trivial het_notom s51, models loaded once)
launch 2 60  het_gsaca               g2_gsaca_s60
launch 2 51  "het_notom het_gsaca"   g2_notom_gsaca_s51
# GPU3: gated_s51 + dp_s51 (both ~45min)
launch 3 51  het_gated_atom_talk     g3_gated_s51
launch 3 51  het_dp_gated_atom_talk  g3_dp_s51

N=$(grep -c "PID=" "$PIDFILE")
echo
echo "=== Launched $N processes (2 per GPU). PID file: $PIDFILE ==="
echo "ETA: ~70-90 min (heavy cells ~44-46min solo, ~1.7x under dual-process)."
date
