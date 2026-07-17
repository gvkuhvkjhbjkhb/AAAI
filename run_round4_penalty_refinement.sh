#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT_DIR/epymarl/src/main.py" ]]; then
  EPYMARL_DIR="$ROOT_DIR/epymarl"
else
  EPYMARL_DIR="$ROOT_DIR"
fi
OUT_DIR="$ROOT_DIR/results/round4_penalty_refinement_10x10"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

ENV_KEY="lbforaging:Foraging-10x10-3p-3f-v3"
TIME_LIMIT=50
T_MAX=500000
TEST_INTERVAL=100000
LOG_INTERVAL=25000
TEST_NEPISODE=20
MAX_RECORDS=2000
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
PENALTIES="${PENALTIES:-0.0001 0.0002 0.0003 0.0005 0.0007 0.001}"

printf 'Round4 penalty refinement started at %s\n' "$(date)" > "$OUT_DIR/STATUS.txt"
printf 'env=%s t_max=%s seeds=%s penalties=%s\n' "$ENV_KEY" "$T_MAX" "$SEEDS" "$PENALTIES" >> "$OUT_DIR/STATUS.txt"

run_one() {
  local method="$1"
  local seed="$2"
  local config="mappo"
  local extra=""
  if [[ "$method" != "baseline" ]]; then
    config="mappo_llm_fd"
    local penalty="${method#penalty_}"
    extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$penalty llm_fd_intervention_mode=uniform"
  fi
  printf '\n== %s seed=%s start %s ==\n' "$method" "$seed" "$(date)" | tee -a "$OUT_DIR/STATUS.txt"
  (cd "$EPYMARL_DIR" && python3 src/main.py \
    --config="$config" \
    --env-config=gymma \
    with \
    env_args.key="$ENV_KEY" \
    env_args.time_limit="$TIME_LIMIT" \
    t_max="$T_MAX" \
    use_cuda=True \
    test_nepisode="$TEST_NEPISODE" \
    test_interval="$TEST_INTERVAL" \
    log_interval="$LOG_INTERVAL" \
    runner_log_interval="$LOG_INTERVAL" \
    learner_log_interval="$LOG_INTERVAL" \
    seed="$seed" \
    $extra) > "$LOG_DIR/${method}_seed${seed}.log" 2>&1
  local code=$?
  printf '== %s seed=%s exit=%s end %s ==\n' "$method" "$seed" "$code" "$(date)" | tee -a "$OUT_DIR/STATUS.txt"
}

for seed in $SEEDS; do
  run_one baseline "$seed"
  for penalty in $PENALTIES; do
    run_one "penalty_${penalty}" "$seed"
  done
done

python3 "$EPYMARL_DIR/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" --baseline baseline

printf 'Round4 penalty refinement finished at %s\n' "$(date)" >> "$OUT_DIR/STATUS.txt"
