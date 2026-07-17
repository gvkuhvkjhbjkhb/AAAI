#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/round11_vmas_calibration_${STAMP}}"
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/STATUS.txt"
WORKER="${WORKER:-/data/lab/AAAI_worker_round11_vmas_calibration}"
GPU="${GPU:-0}"
mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"

VMAS_ENV="${VMAS_ENV:-vmas-navigation}"
TIME_LIMIT="${TIME_LIMIT:-50}"
T_MAX="${T_MAX:-300000}"
SEEDS="${SEEDS:-1 2 3}"
PENALTIES="${PENALTIES:-0.00001 0.00003 0.0001 0.0003}"
METHODS="${METHODS:-baseline adaptive uniform random}"
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

penalty_label() {
  python3 - "$1" <<'PY'
import sys
v = float(sys.argv[1])
text = f"{v:.8f}".rstrip("0").rstrip(".")
print(text.replace("-", "m"))
PY
}

method_extra() {
  local method="$1" penalty="$2" config="mappo" extra=""
  case "$method" in
    baseline)
      config="mappo"; extra="" ;;
    adaptive)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$penalty llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=$penalty llm_fd_late_phase_weight=0.45" ;;
    uniform)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$penalty llm_fd_intervention_mode=phase_uniform llm_fd_terminal_bonus=$penalty llm_fd_late_phase_weight=0.45" ;;
    random)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$penalty llm_fd_intervention_mode=random_type llm_fd_terminal_bonus=$penalty llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.45" ;;
    *) return 2 ;;
  esac
  printf '%s|%s\n' "$config" "$extra"
}

write_plan() {
  cat > "$OUT_DIR/ROUND11_EXPERIMENT_PLAN.md" <<EOF
# Round 11 VMAS Reward-Scale Calibration

Goal: implement Option 3 by testing whether Round 9's VMAS weakness was caused by transferring the LBF-tuned 0.0003 shaping scale to a dense VMAS reward landscape.

Environment: $VMAS_ENV
Timesteps: $T_MAX
Seeds: $SEEDS
Penalties: $PENALTIES
Methods: $METHODS

Decision rule:
- Expand the best penalty to more seeds only if adaptive is positive against baseline and not clearly worse than random-type budget matching.
- If random-type or uniform dominates across penalties, keep VMAS as a transparent reward-scale limitation and do not use it as a main generalization result.
EOF
}

make_jobs() {
  : > "$OUT_DIR/jobs_all.txt"
  for penalty in $PENALTIES; do
    local label
    label="$(penalty_label "$penalty")"
    for seed in $SEEDS; do
      for method in $METHODS; do
        printf 'vmascal|%s|%s|%s|%s|%s|%s|%s\n' "$VMAS_ENV" "$TIME_LIMIT" "$T_MAX" "$method" "$penalty" "$label" "$seed" >> "$OUT_DIR/jobs_all.txt"
      done
    done
  done
  if [[ "${SHUFFLE_JOBS:-1}" == "1" ]]; then
    python3 - "$OUT_DIR/jobs_all.txt" "${JOB_SHUFFLE_SEED:-11}" <<'PYSHUFFLE'
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
  local phase="$1" env_key="$2" time_limit="$3" tmax="$4" method="$5" penalty="$6" label="$7" seed="$8"
  local safe_env log done_file fail_file config_extra config extra start_ts end_ts elapsed code method_name
  safe_env="$(sanitize_env "$env_key")"
  method_name="${method}_p${label}"
  log="$LOG_DIR/${phase}_${safe_env}_${method_name}_seed${seed}.log"
  done_file="$log.done"
  fail_file="$log.fail"
  rm -f "$fail_file"
  if [[ -f "$done_file" ]]; then
    log_status "SKIP phase=$phase method=$method_name seed=$seed env=$env_key"
    return 0
  fi
  config_extra="$(method_extra "$method" "$penalty")" || return 2
  config="${config_extra%%|*}"
  extra="${config_extra#*|}"
  start_ts=$(date +%s)
  log_status "START phase=$phase method=$method_name seed=$seed env=$env_key tmax=$tmax at $(date)"
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
    log_status "DONE phase=$phase method=$method_name seed=$seed exit=$code elapsed=${elapsed}s at $(date)"
  else
    printf '%s\n' "$code" > "$fail_file"
    log_status "FAIL phase=$phase method=$method_name seed=$seed exit=$code elapsed=${elapsed}s at $(date)"
  fi
  summarize_partial
  return "$code"
}

summarize_partial() {
  python3 "$ROOT_DIR/epymarl/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" --baseline vmascal_vmas-navigation_baseline_p0.00001 >/dev/null 2>&1 || true
  python3 "$ROOT_DIR/epymarl/tools/build_round11_vmas_calibration_report.py" --out-dir "$OUT_DIR" >/dev/null 2>&1 || true
  log_status "SUMMARY_REFRESH completed=$(find "$LOG_DIR" -name '*.done' | wc -l) failed=$(find "$LOG_DIR" -name '*.fail' | wc -l) at $(date)"
}

package_outputs() {
  summarize_partial
  local artifact="$ROOT_DIR/artifacts/AAAI_round11_vmas_calibration_${STAMP}.tar.gz"
  tar -C "$ROOT_DIR" -czf "$artifact" "results/$(basename "$OUT_DIR")" "run_round11_vmas_calibration.sh" "epymarl/tools/build_round11_vmas_calibration_report.py"
  log_status "ARTIFACT $artifact"
}

main() {
  log_status "Round 11 VMAS calibration started at $(date)"
  log_status "root=$ROOT_DIR out=$OUT_DIR gpu=$GPU worker=$WORKER"
  write_plan
  prepare_worker
  make_jobs
  log_status "job_count=$(wc -l < "$OUT_DIR/jobs_all.txt")"
  while IFS='|' read -r phase env_key time_limit tmax method penalty label seed; do
    run_one "$phase" "$env_key" "$time_limit" "$tmax" "$method" "$penalty" "$label" "$seed" || true
  done < "$OUT_DIR/jobs_all.txt"
  package_outputs
  log_status "Round 11 VMAS calibration finished at $(date)"
}

main "$@"
