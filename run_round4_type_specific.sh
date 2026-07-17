#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT_DIR/epymarl/src/main.py" ]]; then
  EPYMARL_DIR="$ROOT_DIR/epymarl"
else
  EPYMARL_DIR="$ROOT_DIR"
fi
OUT_DIR="$ROOT_DIR/results/round4_type_specific_10x10"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

ENV_KEY="lbforaging:Foraging-10x10-3p-3f-v3"
TIME_LIMIT=50
T_MAX=500000
TEST_INTERVAL=100000
LOG_INTERVAL=25000
TEST_NEPISODE=20
MAX_RECORDS=2000
BASE_PENALTY=0.0003
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"

printf 'Round4 type-specific 10x10 experiment started at %s\n' "$(date)" > "$OUT_DIR/STATUS.txt"
printf 'env=%s t_max=%s seeds=%s methods=baseline,diagnosis_only,uniform,type_specific,adaptive,random_type\n' "$ENV_KEY" "$T_MAX" "$SEEDS" >> "$OUT_DIR/STATUS.txt"

run_one() {
  local method="$1"
  local seed="$2"
  local config="mappo"
  local extra=""

  case "$method" in
    baseline)
      config="mappo"
      ;;
    diagnosis_only)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=False"
      ;;
    uniform_0.0003)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$BASE_PENALTY llm_fd_intervention_mode=uniform"
      ;;
    type_specific_0.0003)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$BASE_PENALTY llm_fd_intervention_mode=type_specific llm_fd_terminal_bonus=0.0003"
      ;;
    adaptive_0.0003)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$BASE_PENALTY llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0003"
      ;;
    random_type_0.0003)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$BASE_PENALTY llm_fd_intervention_mode=random_type llm_fd_terminal_bonus=0.0003 llm_fd_random_type_use_phase=True"
      ;;
    *)
      printf 'Unknown method: %s\n' "$method" >&2
      return 2
      ;;
  esac

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
  run_one diagnosis_only "$seed"
  run_one uniform_0.0003 "$seed"
  run_one type_specific_0.0003 "$seed"
  run_one adaptive_0.0003 "$seed"
  run_one random_type_0.0003 "$seed"
done

python3 "$EPYMARL_DIR/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" --baseline baseline

printf 'Round4 type-specific 10x10 experiment finished at %s\n' "$(date)" >> "$OUT_DIR/STATUS.txt"
