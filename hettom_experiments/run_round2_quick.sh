#!/usr/bin/env bash
# Round-2 QUICK validation: fastest positive-direction test.
# Optimized for speed: only the cells that test the 3 new mechanisms, on
# stag_hunt (same as round-1 for comparability), 3 seeds, reduced episodes.
#
# Speed optimizations vs full round-2 plan:
#   - 7 cells not 11 (drop hom talk variants + hom_atom; keep the cells that
#     directly test cheap-talk rescuing het, A-ToM beating fixed ToM, combined)
#   - 20 episodes/cell not 30 (enough for direction given r=1.0 in round-1)
#   - 3 seeds (round-1 proved r=+-1.0 at n=3; direction is reliable)
#   - stag_hunt only (battle_of_the_sexes added later for generalization)
#
# Cells: hom_notom(baseline) het_notom het_tom het_notom_talk het_tom_talk
#         het_atom het_atom_talk
set -u
cd /data/lab/hettom_run
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

MODEL_HOMO="Qwen/Qwen2.5-3B-Instruct"
MODELS_HET="Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-1.5B-Instruct"
EP=20; HORIZON=5; TOM=1; MEM=2
mkdir -p logs

run_job() {
  local seed=$1 gpu=$2
  echo "[launch] seed=$seed gpu=$gpu @ $(date +%T)"
  CUDA_VISIBLE_DEVICES=$gpu python3 hettom_baseline.py --matrix --extend \
    --game stag_hunt --seeds $seed \
    --episodes $EP --horizon $HORIZON --tom_order $TOM --memory $MEM \
    --model_homo "$MODEL_HOMO" --models_het $MODELS_HET \
    --out_dir "results/hettom_layer1/stag_hunt" --log_every 5 \
    > "logs/stag_hunt_seed${seed}.log" 2>&1
  echo "[done] seed=$seed gpu=$gpu rc=$? @ $(date +%T)"
}

# 3 seeds: wave1 = seed4@GPU0 + seed5@GPU1 (parallel), wave2 = seed6@GPU0
# (seeds 4,5,6 extend round-1's 1,2,3 -> 6 total for analysis)
run_job 4 0 &
P1=$!
run_job 5 1 &
P2=$!
wait $P1
wait $P2
echo "=== wave 1 done @ $(date +%T) ==="
run_job 6 0 &
P3=$!
wait $P3
echo "=== wave 2 done @ $(date +%T) ==="
echo "ALL_JOBS_DONE @ $(date +%T)"