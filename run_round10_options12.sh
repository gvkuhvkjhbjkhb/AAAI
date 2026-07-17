#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/round10_options12_${STAMP}}"
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/STATUS.txt"
WORKER="${WORKER:-/data/lab/AAAI_worker_round10_options12}"
GPU="${GPU:-0}"
mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"

NEW_LBF_ENV="${NEW_LBF_ENV:-lbforaging:Foraging-10x10-4p-4f-v3}"
NEW_LBF_TIME_LIMIT="${NEW_LBF_TIME_LIMIT:-50}"
NEW_LBF_TMAX="${NEW_LBF_TMAX:-500000}"
NEW_LBF_SEEDS="${NEW_LBF_SEEDS:-1 2 3 4 5 6 7 8}"
NEW_LBF_METHODS="${NEW_LBF_METHODS:-baseline uniform_budget_matched_0.0003_late045 adaptive_0.0003_late045 random_type_budget_matched_0.0003_late045}"
RUN_NEW_LBF="${RUN_NEW_LBF:-1}"

SENS_ENV="${SENS_ENV:-lbforaging:Foraging-10x10-3p-3f-v3}"
SENS_TIME_LIMIT="${SENS_TIME_LIMIT:-50}"
SENS_TMAX="${SENS_TMAX:-500000}"
SENS_SEEDS="${SENS_SEEDS:-1 2 3 4}"
SENS_METHODS="${SENS_METHODS:-adaptive_0.0002_late045 adaptive_0.0005_late045 adaptive_0.0003_late060 uniform_budget_matched_0.0003_late045 random_type_budget_matched_0.0003_late045}"
RUN_SENSITIVITY="${RUN_SENSITIVITY:-1}"

TEST_INTERVAL="${TEST_INTERVAL:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-25000}"
TEST_NEPISODE="${TEST_NEPISODE:-20}"
MAX_RECORDS="${MAX_RECORDS:-3000}"

log_status() { printf '%s\n' "$*" | tee -a "$STATUS"; }

prepare_worker() {
  rm -rf "$WORKER"
  mkdir -p "$WORKER"
  tar --exclude='./results' --exclude='./artifacts' --exclude='./.git' -C "$ROOT_DIR" -cf - . | tar -C "$WORKER" -xf -
}

write_plan() {
  cat > "$OUT_DIR/ROUND10_EXPERIMENT_PLAN.md" <<EOF
# Round 10 Options 1+2

Goal: implement the safest AAAI package after Round 8/9: mechanism defense plus LBF-family generalization.

Option 1 components:
- Sensitivity on $SENS_ENV with seeds $SENS_SEEDS and methods $SENS_METHODS.
- Budget accounting logs: records, shaping triggers, penalty total, terminal bonus total, shaped episode steps, average penalty per trigger.

Option 2 components:
- New LBF-family task $NEW_LBF_ENV with seeds $NEW_LBF_SEEDS and methods $NEW_LBF_METHODS.

Decision use:
- Main claim remains failure-triggered adaptive reward shaping for sparse cooperative foraging.
- New LBF evidence is used only if adaptive is positive against baseline and competitive with budget-matched/random controls.
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
    adaptive_0.0002_late045)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0002 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0002 llm_fd_late_phase_weight=0.45" ;;
    adaptive_0.0005_late045)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0005 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0005 llm_fd_late_phase_weight=0.45" ;;
    adaptive_0.0003_late060)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.60" ;;
    *) return 2 ;;
  esac
  printf '%s|%s\n' "$config" "$extra"
}

make_jobs() {
  : > "$OUT_DIR/jobs_all.txt"
  if [[ "$RUN_NEW_LBF" == "1" ]]; then
    for seed in $NEW_LBF_SEEDS; do for method in $NEW_LBF_METHODS; do
      printf 'newlbf|%s|%s|%s|%s|%s\n' "$NEW_LBF_ENV" "$NEW_LBF_TIME_LIMIT" "$NEW_LBF_TMAX" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"
    done; done
  fi
  if [[ "$RUN_SENSITIVITY" == "1" ]]; then
    for seed in $SENS_SEEDS; do for method in $SENS_METHODS; do
      printf 'sensitivity10|%s|%s|%s|%s|%s\n' "$SENS_ENV" "$SENS_TIME_LIMIT" "$SENS_TMAX" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"
    done; done
  fi
  if [[ "${SHUFFLE_JOBS:-1}" == "1" ]]; then
    python3 - "$OUT_DIR/jobs_all.txt" "${JOB_SHUFFLE_SEED:-10}" <<'PYSHUFFLE'
import random
import sys
path, seed = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as handle:
    rows = handle.readlines()
rng = random.Random(seed)
rng.shuffle(rows)
with open(path, "w", encoding="utf-8") as handle:
    handle.writelines(rows)
PYSHUFFLE
  fi
}

sanitize_env() { printf '%s' "$1" | tr ':/' '__'; }

run_one() {
  local phase="$1" env_key="$2" time_limit="$3" tmax="$4" method="$5" seed="$6"
  local safe_env log done_file fail_file config_extra config extra start_ts end_ts elapsed code
  safe_env="$(sanitize_env "$env_key")"
  log="$LOG_DIR/${phase}_${safe_env}_${method}_seed${seed}.log"
  done_file="$log.done"
  fail_file="$log.fail"
  rm -f "$fail_file"
  if [[ -f "$done_file" ]]; then
    log_status "SKIP phase=$phase method=$method seed=$seed env=$env_key"
    return 0
  fi
  config_extra="$(method_extra "$method")" || return 2
  config="${config_extra%%|*}"
  extra="${config_extra#*|}"
  start_ts=$(date +%s)
  log_status "START phase=$phase method=$method seed=$seed env=$env_key tmax=$tmax at $(date)"
  (
    cd "$WORKER/epymarl" && CUDA_VISIBLE_DEVICES="$GPU" python3 src/main.py \
      --config="$config" --env-config=gymma with \
      env_args.key="$env_key" env_args.time_limit="$time_limit" \
      t_max="$tmax" use_cuda=True test_nepisode="$TEST_NEPISODE" \
      test_interval="$TEST_INTERVAL" log_interval="$LOG_INTERVAL" \
      runner_log_interval="$LOG_INTERVAL" learner_log_interval="$LOG_INTERVAL" \
      seed="$seed" $extra
  ) > "$log" 2>&1
  code=$?
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

summarize_partial() {
  python3 "$ROOT_DIR/epymarl/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" --baseline newlbf_lbforaging_Foraging-10x10-4p-4f-v3_baseline >/dev/null 2>&1 || true
  python3 "$ROOT_DIR/epymarl/tools/build_round10_options12_report.py" --out-dir "$OUT_DIR" >/dev/null 2>&1 || true
  log_status "SUMMARY_REFRESH completed=$(find "$LOG_DIR" -name '*.done' | wc -l) failed=$(find "$LOG_DIR" -name '*.fail' | wc -l) at $(date)"
}

package_outputs() {
  summarize_partial
  local artifact="$ROOT_DIR/artifacts/AAAI_round10_options12_${STAMP}.tar.gz"
  tar -C "$ROOT_DIR" -czf "$artifact" "results/$(basename "$OUT_DIR")" "run_round10_options12.sh" "epymarl/tools/build_round10_options12_report.py"
  log_status "ARTIFACT $artifact"
}

main() {
  log_status "Round 10 Options 1+2 started at $(date)"
  log_status "root=$ROOT_DIR out=$OUT_DIR gpu=$GPU worker=$WORKER"
  write_plan
  prepare_worker
  make_jobs
  log_status "job_count=$(wc -l < "$OUT_DIR/jobs_all.txt")"
  while IFS='|' read -r phase env_key time_limit tmax method seed; do
    run_one "$phase" "$env_key" "$time_limit" "$tmax" "$method" "$seed" || true
  done < "$OUT_DIR/jobs_all.txt"
  package_outputs
  log_status "Round 10 Options 1+2 finished at $(date)"
}

main "$@"
