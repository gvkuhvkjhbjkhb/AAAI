#!/usr/bin/env bash
# =============================================================================
# Parallel client driver for the 3-arm paradox/attainment study.
# Identical config/output to run_paradox_attainment.sh, but shards the 20 seeds
# across several concurrent client processes so the two shared vLLM servers stay
# saturated (each cell is self-contained; all 3 arms of a given (game,seed) run
# in the SAME worker, so same-batch same-seed pairing is preserved). Runner
# skips already-present metrics.json, so this safely resumes a partial run.
#
# Concurrency is over (game,seed-shard); the frozen model/seed/episode/arm
# config is unchanged. Latin-square arm order per seed is unchanged.
# =============================================================================
set -uo pipefail
GSACA_ROOT="${GSACA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
RUNNER="$GSACA_ROOT/code/run_experiment_local.py"
OUT="${OUT:-$GSACA_ROOT/v2_results/exp_vllm_paradox_attainment_v1}"
LOG="${LOG:-$OUT/logs}"
mkdir -p "$OUT" "$LOG"

QWEN=Qwen/Qwen2.5-7B-Instruct
GLM=THUDM/GLM-4-9B-0414
ARMS="het_notom het_gated_atom_talk het_dp_gated_atom_talk"
GEN_SEED_BASE="${GEN_SEED_BASE:-1000}"
ALL_SEEDS=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61)
# number of concurrent seed-shards PER game
NSHARD="${NSHARD:-5}"

TWO_PLAYER=(chicken deadlock hawk_dove stag_hunt battle_of_the_sexes)

launch_game () {   # $1=game  $2=episodes
  local game="$1" ep="$2"
  for ((k=0; k<NSHARD; k++)); do
    local shard=()
    for ((i=k; i<${#ALL_SEEDS[@]}; i+=NSHARD)); do shard+=(${ALL_SEEDS[$i]}); done
    python3 "$RUNNER" \
      --use_vllm --latin_square --gen_seed_base $GEN_SEED_BASE \
      --games "$game" --seeds "${shard[@]}" \
      --episodes "$ep" --horizon 5 --memory 2 \
      --cells $ARMS --out_dir "$OUT" \
      --log_every 100 --models_het $QWEN $GLM \
      --gate_trust_threshold 0.6 --gate_ema_alpha 0.3 --atom_warmup 3 \
      > "$LOG/${game}_shard${k}.log" 2>&1 &
  done
}

echo "[$(date '+%H:%M:%S')] PARALLEL START nshard=$NSHARD/game -> $OUT" | tee -a "$LOG/run.log"
for g in "${TWO_PLAYER[@]}"; do launch_game "$g" 30; done
launch_game public_goods 20
echo "[$(date '+%H:%M:%S')] launched $(( (${#TWO_PLAYER[@]}+1) * NSHARD )) client procs" | tee -a "$LOG/run.log"
wait
N=$(find "$OUT" -name metrics.json | wc -l)
echo "[$(date '+%H:%M:%S')] PARALLEL DONE. metrics: $N/360" | tee -a "$LOG/run.log"
