#!/usr/bin/env bash
# Round-2 HetToM experiment: extended matrix (cheap-talk + A-ToM).
#
# Three mechanisms from literature survey, all addressing round-1 problems:
#   1) Cheap-talk channel (Madmoun & Lahlou 2025, EACL 2026) — solves het
#      cooperation collapse (round-1 het_notom payoff 0.580 vs baseline 2.244).
#      Literature reports 0%->96.7% cooperation in Stag Hunt with one-word talk.
#   2) Adaptive ToM (Mu et al. 2026, AAAI 2026) — solves low ToM accuracy
#      (round-1 het_tom tom_acc 0.54). A-ToM estimates partner ToM order and
#      aligns, addressing order-mismatch that impairs coordination.
#   3) Expand seeds (3->6) + add battle_of_the_sexes for generalization.
#
# Extended matrix (11 cells per seed, --extend flag):
#   base 4: hom_notom, hom_tom, het_notom, het_tom
#   +talk 4: *_talk variants of the base 4
#   +atom 2: hom_atom, het_atom
#   combined 1: het_atom_talk (full method)
#
# Models: Qwen2.5-3B-Instruct (homo) + Qwen2.5-3B/1.5B (het) — same as round-1
# for comparability. Seeds 4-6 (stag_hunt extends round-1's 1-3; bos fresh 1-6).
set -u
cd /data/lab/hettom_run
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

MODEL_HOMO="Qwen/Qwen2.5-3B-Instruct"
MODELS_HET="Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-1.5B-Instruct"
EP=30; HORIZON=5; TOM=1; MEM=2
mkdir -p logs

run_job() {
  local game=$1 seed=$2 gpu=$3
  echo "[launch] $game seed=$seed gpu=$gpu @ $(date +%T)"
  CUDA_VISIBLE_DEVICES=$gpu python3 hettom_baseline.py --matrix --extend \
    --game $game --seeds $seed \
    --episodes $EP --horizon $HORIZON --tom_order $TOM --memory $MEM \
    --model_homo "$MODEL_HOMO" --models_het $MODELS_HET \
    --out_dir "results/hettom_layer1/$game" --log_every 10 \
    > "logs/${game}_seed${seed}.log" 2>&1
  echo "[done] $game seed=$seed gpu=$gpu rc=$? @ $(date +%T)"
}

# stag_hunt: seeds 4,5,6 (extend round-1's 1-3 to 6 total)
# battle_of_the_sexes: seeds 1,2,3,4,5,6 (fresh, 6 for generalization)
# Interleave across 2 GPUs; each job = 11 cells x 30 ep.
JOBS=(
  "stag_hunt 4"
  "battle_of_the_sexes 1"
  "stag_hunt 5"
  "battle_of_the_sexes 2"
  "stag_hunt 6"
  "battle_of_the_sexes 3"
  "battle_of_the_sexes 4"
  "battle_of_the_sexes 5"
  "battle_of_the_sexes 6"
)

n=${#JOBS[@]}
i=0
while [ $i -lt $n ]; do
  set -- ${JOBS[$i]}
  run_job $1 $2 0 &
  P1=$!
  if [ $((i+1)) -lt $n ]; then
    set -- ${JOBS[$((i+1))]}
    run_job $1 $2 1 &
    P2=$!
    wait $P2
  fi
  wait $P1
  echo "=== wave (jobs $i,$((i+1))) done @ $(date +%T) ==="
  i=$((i+2))
done

echo "ALL_JOBS_DONE @ $(date +%T)"
