#!/usr/bin/env bash
# Layer 2b: Exp A homogeneous control completion (QQ + GG, 5-seed target).
# Brings Exp A from 76% -> 100% at 5-seed target. QL (needs Llama) deferred.
# hom cells: hom_notom hom_gated_atom_talk hom_dp_gated_atom_talk hom_gsaca
# QQ incomplete: stag_hunt(5), battle_of_the_sexes(5) -> --force 2 games x 4 cells x 5 seeds = 40
# GG incomplete: deadlock(2), stag_hunt(2), battle_of_the_sexes(2) -> --force 3 games x 4 cells x 5 seeds = 60
set -e
cd /data/lab/gsaca
GSACA=/data/lab/gsaca
OUT_QQ=$GSACA/v2_results/exp_a_fix/QQ
OUT_GG=$GSACA/v2_results/exp_a_fix/GG
LOG=$GSACA/results/v2/logs
mkdir -p "$OUT_QQ" "$OUT_GG" "$LOG"
QWEN=Qwen/Qwen2.5-7B-Instruct
GLM=THUDM/GLM-4-9B-0414
SEEDS="42 43 44 45 46"
HOM_CELLS="hom_notom hom_gated_atom_talk hom_dp_gated_atom_talk hom_gsaca"

echo "[$(date '+%H:%M:%S')] Exp A hom completion START (QQ 40 + GG 60 = 100 cells)" | tee "$LOG/exp_a_run.log"

# GPU0: QQ (Qwen x Qwen homogeneous), stag_hunt + battle_of_the_sexes, 4 hom cells, 5 seeds = 40
CUDA_VISIBLE_DEVICES=0 python3 $GSACA/code/run_experiment_local.py \
    --games stag_hunt battle_of_the_sexes --seeds $SEEDS \
    --episodes 30 --horizon 5 --memory 2 \
    --cells $HOM_CELLS --out_dir "$OUT_QQ" \
    --log_every 50 --model_homo $QWEN --force \
    > "$LOG/exp_a_qq_gpu0.log" 2>&1 &
P0=$!
echo "[$(date '+%H:%M:%S')] GPU0 PID $P0 (QQ hom: stag_hunt,BoS)" | tee -a "$LOG/exp_a_run.log"

# GPU1: GG (GLM x GLM homogeneous), deadlock + stag_hunt + battle_of_the_sexes, 4 hom cells, 5 seeds = 60
CUDA_VISIBLE_DEVICES=1 python3 $GSACA/code/run_experiment_local.py \
    --games deadlock stag_hunt battle_of_the_sexes --seeds $SEEDS \
    --episodes 30 --horizon 5 --memory 2 \
    --cells $HOM_CELLS --out_dir "$OUT_GG" \
    --log_every 50 --model_homo $GLM --force \
    > "$LOG/exp_a_gg_gpu1.log" 2>&1 &
P1=$!
echo "[$(date '+%H:%M:%S')] GPU1 PID $P1 (GG hom: deadlock,stag_hunt,BoS)" | tee -a "$LOG/exp_a_run.log"

wait
echo "[$(date '+%H:%M:%S')] Exp A DONE. QQ=$(find $OUT_QQ -name metrics.json|wc -l) GG=$(find $OUT_GG -name metrics.json|wc -l)" | tee -a "$LOG/exp_a_run.log"
