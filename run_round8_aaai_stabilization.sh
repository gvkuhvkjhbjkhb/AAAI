#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/round8_aaai_stabilization_${STAMP}}"
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/STATUS.txt"
mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"

ENV_10="${ENV_10:-lbforaging:Foraging-10x10-3p-3f-v3}"
ENV_12="${ENV_12:-lbforaging:Foraging-12x12-3p-4f-v3}"
TIME_LIMIT_10="${TIME_LIMIT_10:-50}"
TIME_LIMIT_12="${TIME_LIMIT_12:-50}"
T_MAX="${T_MAX:-500000}"
TEST_INTERVAL="${TEST_INTERVAL:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-25000}"
TEST_NEPISODE="${TEST_NEPISODE:-20}"
MAX_RECORDS="${MAX_RECORDS:-3000}"
SEEDS_MAIN="${SEEDS_MAIN:-1 2 3 4 5 6 7 8}"
METHODS_MAIN="${METHODS_MAIN:-baseline uniform_budget_matched_0.0003_late045 adaptive_0.0003_late045 random_type_budget_matched_0.0003_late045 semantic_shuffled_budget_matched_0.0003_late045 diagnosis_only}"
RUN_10X10="${RUN_10X10:-1}"
RUN_12X12="${RUN_12X12:-1}"
RUN_QWEN_VALIDATION="${RUN_QWEN_VALIDATION:-0}"
QWEN_SAMPLE_SIZE="${QWEN_SAMPLE_SIZE:-120}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3.5-4B}"
LLM_FD_API_BASE="${LLM_FD_API_BASE:-https://api.siliconflow.cn/v1}"
WORKER="${WORKER:-/data/lab/AAAI_worker_round8_stabilization}"
GPU="${GPU:-0}"
ROUND6_SUMMARY="${ROUND6_SUMMARY:-$ROOT_DIR/results/round6_aaai_submission_20260702_134358/summary.csv}"
ROUND7_SUMMARY="${ROUND7_SUMMARY:-$ROOT_DIR/results/round7_decisive_aaai_20260702_round7_decisive/summary.csv}"
PHASE1_SUMMARY="${PHASE1_SUMMARY:-$ROOT_DIR/results/phase1_adaptive_robustness_20260704_phase1_robustness/summary.csv}"

log_status() { printf '%s\n' "$*" | tee -a "$STATUS"; }

prepare_worker() {
  rm -rf "$WORKER"
  mkdir -p "$WORKER"
  tar --exclude='./results' --exclude='./artifacts' --exclude='./.git' -C "$ROOT_DIR" -cf - . | tar -C "$WORKER" -xf -
}

write_plan() {
  cat > "$OUT_DIR/ROUND8_EXPERIMENT_PLAN.md" <<EOF
# Round 8 AAAI Stabilization

Goal: make the submission defensible as **Failure-Triggered Adaptive Reward Shaping for Cooperative Multi-Agent Reinforcement Learning** rather than as an LLM semantic-causality paper.

Main package:
- Environments: $ENV_10 and $ENV_12.
- Methods: $METHODS_MAIN.
- Seeds: $SEEDS_MAIN.
- Budget: $T_MAX timesteps per run, test interval $TEST_INTERVAL, $TEST_NEPISODE test episodes.

Controls:
- uniform_budget_matched uses a label-independent phase schedule with the same nominal penalty and late-phase attenuation as adaptive shaping.
- random_type_budget_matched uses random failure labels with the same trigger stream, confidence scaling, phase schedule, and type-weight budget.
- semantic_shuffled_budget_matched uses matched-frequency shuffled labels from the observed label pool.
- diagnosis_only records failures with no reward intervention.

Optional Qwen validation:
- RUN_QWEN_VALIDATION=1 evaluates $LLM_MODEL offline with cache/retry enabled on $QWEN_SAMPLE_SIZE validation records. It is not part of the online RL main claim.
EOF
}

method_extra() {
  local method="$1" config="mappo" extra=""
  case "$method" in
    baseline)
      config="mappo"; extra="" ;;
    diagnosis_only)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=False" ;;
    uniform_budget_matched_0.0003_late045)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=phase_uniform llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45" ;;
    adaptive_0.0003_late045)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45" ;;
    random_type_budget_matched_0.0003_late045)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=random_type llm_fd_terminal_bonus=0.0003 llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.45" ;;
    semantic_shuffled_budget_matched_0.0003_late045)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=random_type_matched llm_fd_terminal_bonus=0.0003 llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.45" ;;
    *) return 2 ;;
  esac
  printf '%s|%s\n' "$config" "$extra"
}

make_jobs() {
  : > "$OUT_DIR/jobs_all.txt"
  if [[ "$RUN_10X10" == "1" ]]; then
    for seed in $SEEDS_MAIN; do
      for method in $METHODS_MAIN; do
        printf 'main10x10|%s|%s|%s|%s\n' "$ENV_10" "$TIME_LIMIT_10" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"
      done
    done
  fi
  if [[ "$RUN_12X12" == "1" ]]; then
    for seed in $SEEDS_MAIN; do
      for method in $METHODS_MAIN; do
        printf 'generalization12|%s|%s|%s|%s\n' "$ENV_12" "$TIME_LIMIT_12" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"
      done
    done
  fi
}

run_one() {
  local phase="$1" env_key="$2" time_limit="$3" method="$4" seed="$5"
  local safe_env="$(printf '%s' "$env_key" | tr ':/' '__')"
  local log="$LOG_DIR/${phase}_${safe_env}_${method}_seed${seed}.log"
  local done_file="$log.done"
  local fail_file="$log.fail"
  rm -f "$fail_file"
  if [[ -f "$done_file" ]]; then
    log_status "SKIP completed phase=$phase method=$method seed=$seed env=$env_key"
    return 0
  fi
  local config_extra config extra start_ts end_ts elapsed
  config_extra="$(method_extra "$method")" || return 2
  config="${config_extra%%|*}"
  extra="${config_extra#*|}"
  start_ts=$(date +%s)
  log_status "START phase=$phase method=$method seed=$seed env=$env_key at $(date)"
  (
    cd "$WORKER/epymarl" && CUDA_VISIBLE_DEVICES="$GPU" python3 src/main.py \
      --config="$config" \
      --env-config=gymma \
      with \
      env_args.key="$env_key" \
      env_args.time_limit="$time_limit" \
      t_max="$T_MAX" \
      use_cuda=True \
      test_nepisode="$TEST_NEPISODE" \
      test_interval="$TEST_INTERVAL" \
      log_interval="$LOG_INTERVAL" \
      runner_log_interval="$LOG_INTERVAL" \
      learner_log_interval="$LOG_INTERVAL" \
      seed="$seed" \
      $extra
  ) > "$log" 2>&1
  local code=$?
  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  if [[ "$code" -eq 0 ]]; then
    touch "$done_file"
    log_status "DONE phase=$phase method=$method seed=$seed exit=$code elapsed=${elapsed}s at $(date)"
  else
    printf '%s\n' "$code" > "$fail_file"
    log_status "FAIL phase=$phase method=$method seed=$seed exit=$code elapsed=${elapsed}s at $(date)"
  fi
  summarize_partial
  return "$code"
}

run_qwen_validation() {
  if [[ "$RUN_QWEN_VALIDATION" != "1" ]]; then
    return 0
  fi
  if [[ -z "${LLM_FD_API_KEY:-}" ]]; then
    log_status "QWEN_SKIP no LLM_FD_API_KEY set"
    return 0
  fi
  local src="$ROOT_DIR/results/phase2_full_semantic_20260704_175517/phase2_validation_records.jsonl"
  if [[ ! -f "$src" ]]; then
    log_status "QWEN_SKIP validation records missing: $src"
    return 0
  fi
  log_status "QWEN_START offline cached validation sample_size=$QWEN_SAMPLE_SIZE at $(date)"
  (
    cd "$WORKER/epymarl" && \
    LLM_FD_API_KEY="$LLM_FD_API_KEY" \
    LLM_FD_API_BASE="$LLM_FD_API_BASE" \
    LLM_FD_CACHE_PATH="$OUT_DIR/qwen35_4b_cache.json" \
    LLM_FD_API_RETRIES=4 \
    LLM_FD_API_RETRY_SLEEP=3 \
    LLM_FD_ENABLE_THINKING=false \
    LLM_FD_MAX_TOKENS=256 \
    python3 -m src.llm_diagnosis.offline_relabel \
      --input-glob "$src" \
      --output-dir "$OUT_DIR/qwen35_4b_cached" \
      --mode api \
      --model "$LLM_MODEL" \
      --sample-size "$QWEN_SAMPLE_SIZE" \
      --seed 8
  ) > "$OUT_DIR/qwen35_4b_cached.log" 2>&1 || log_status "QWEN_WARN offline validation exited nonzero"
  log_status "QWEN_DONE at $(date)"
}

summarize_partial() {
  python3 "$ROOT_DIR/epymarl/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" --baseline main10x10_lbforaging_Foraging-10x10-3p-3f-v3_baseline >/dev/null 2>&1 || true
  python3 "$ROOT_DIR/epymarl/tools/build_round8_stabilization_report.py" \
    --out-dir "$OUT_DIR" \
    --extra-summary "$ROUND6_SUMMARY" \
    --extra-summary "$ROUND7_SUMMARY" \
    --extra-summary "$PHASE1_SUMMARY" >/dev/null 2>&1 || true
  log_status "SUMMARY_REFRESH completed=$(find "$LOG_DIR" -name '*.done' | wc -l) failed=$(find "$LOG_DIR" -name '*.fail' | wc -l) at $(date)"
}

package_outputs() {
  summarize_partial
  local artifact="$ROOT_DIR/artifacts/AAAI_round8_stabilization_${STAMP}.tar.gz"
  tar -C "$ROOT_DIR" -czf "$artifact" "results/$(basename "$OUT_DIR")" "run_round8_aaai_stabilization.sh" "epymarl/tools/build_round8_stabilization_report.py"
  log_status "ARTIFACT $artifact"
}

main() {
  log_status "Round 8 AAAI stabilization started at $(date)"
  log_status "root=$ROOT_DIR out=$OUT_DIR gpu=$GPU worker=$WORKER"
  write_plan
  prepare_worker
  run_qwen_validation
  make_jobs
  log_status "job_count=$(wc -l < "$OUT_DIR/jobs_all.txt") methods=$METHODS_MAIN seeds=$SEEDS_MAIN t_max=$T_MAX"
  while IFS='|' read -r phase env_key time_limit method seed; do
    run_one "$phase" "$env_key" "$time_limit" "$method" "$seed" || true
  done < "$OUT_DIR/jobs_all.txt"
  package_outputs
  log_status "Round 8 AAAI stabilization finished at $(date)"
}

main "$@"
