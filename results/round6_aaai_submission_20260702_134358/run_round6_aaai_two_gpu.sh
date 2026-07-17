#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/round6_aaai_submission_${STAMP}}"
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/STATUS.txt"
mkdir -p "$LOG_DIR"

ENV_10="${ENV_10:-lbforaging:Foraging-10x10-3p-3f-v3}"
ENV_GEN="${ENV_GEN:-lbforaging:Foraging-12x12-3p-4f-v3}"
FALLBACK_GEN="${FALLBACK_GEN:-lbforaging:Foraging-10x10-3p-4f-v3}"
TIME_LIMIT="${TIME_LIMIT:-50}"
T_MAX="${T_MAX:-500000}"
TEST_INTERVAL="${TEST_INTERVAL:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-25000}"
TEST_NEPISODE="${TEST_NEPISODE:-20}"
MAX_RECORDS="${MAX_RECORDS:-3000}"
SEEDS_MAIN="${SEEDS_MAIN:-1 2 3 4 5 6 7 8}"
SEEDS_GEN="${SEEDS_GEN:-1 2 3 4 5}"

WORKER0="${WORKER0:-/data/lab/AAAI_worker0_round6}"
WORKER1="${WORKER1:-/data/lab/AAAI_worker1_round6}"

log_status() {
  printf '%s\n' "$*" | tee -a "$STATUS"
}

prepare_worker() {
  local dest="$1"
  rm -rf "$dest"
  mkdir -p "$dest"
  tar --exclude='./results' --exclude='./artifacts' --exclude='./.git' -C "$ROOT_DIR" -cf - . | tar -C "$dest" -xf -
}

write_plan() {
  cat > "$OUT_DIR/EXPERIMENT_PLAN.md" <<EOF
# Round 6 AAAI Submission-Grade Two-GPU Experiment

This run prioritizes the most defensible AAAI claim: diagnosis-conditioned intervention should beat MAPPO and random-type controls, while calibrated uniform shaping is treated as a strong baseline rather than a weak foil.

## Main 10x10 Experiment

- Environment: $ENV_10
- Seeds: $SEEDS_MAIN
- Budget: $T_MAX timesteps
- Methods: baseline, diagnosis_only, uniform_0.0002, uniform_0.0003, type_specific_0.0002, type_specific_0.0003, adaptive_0.0002_late045, adaptive_0.0002_late060, adaptive_0.0003_late045, random_type_0.0002
- Primary metrics: last_test_return, train_auc, stability_gap, best_train_return
- Main acceptance criterion: adaptive or type-specific intervention improves baseline and random_type, and is competitive with or better than the best uniform penalty.

## Generalization Experiment

- Environment: $ENV_GEN, with fallback to $FALLBACK_GEN if unavailable
- Seeds: $SEEDS_GEN
- Budget: $T_MAX timesteps
- Methods: baseline, uniform_0.0002, uniform_0.0003, adaptive_0.0002_late045, adaptive_0.0002_late060

## Compute

- Worker 0 uses CUDA_VISIBLE_DEVICES=0
- Worker 1 uses CUDA_VISIBLE_DEVICES=1
- Each worker has a private source copy to avoid Sacred result-directory races.
EOF
}

make_jobs() {
  local phase="$1"
  local env_key="$2"
  local seeds="$3"
  shift 3
  local methods=("$@")
  for seed in $seeds; do
    for method in "${methods[@]}"; do
      printf '%s|%s|%s|%s\n' "$phase" "$env_key" "$method" "$seed"
    done
  done
}

method_extra() {
  local method="$1"
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
    uniform_0.0002)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0002 llm_fd_intervention_mode=uniform"
      ;;
    uniform_0.0003)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=uniform"
      ;;
    type_specific_0.0002)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0002 llm_fd_intervention_mode=type_specific llm_fd_terminal_bonus=0.0002"
      ;;
    type_specific_0.0003)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=type_specific llm_fd_terminal_bonus=0.0003"
      ;;
    adaptive_0.0002_late045)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0002 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0002 llm_fd_late_phase_weight=0.45"
      ;;
    adaptive_0.0002_late060)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0002 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0002 llm_fd_late_phase_weight=0.60"
      ;;
    adaptive_0.0003_late045)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45"
      ;;
    random_type_0.0002)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0002 llm_fd_intervention_mode=random_type llm_fd_terminal_bonus=0.0002 llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.45"
      ;;
    *)
      return 2
      ;;
  esac
  printf '%s|%s\n' "$config" "$extra"
}

run_one() {
  local worker="$1"
  local gpu="$2"
  local phase="$3"
  local env_key="$4"
  local method="$5"
  local seed="$6"
  local safe_env="$(printf '%s' "$env_key" | tr ':/' '__')"
  local log="$LOG_DIR/${phase}_${safe_env}_${method}_seed${seed}.log"
  local done_file="$log.done"
  local fail_file="$log.fail"
  if [[ -f "$done_file" ]]; then
    log_status "SKIP completed $phase $method seed=$seed env=$env_key"
    return 0
  fi
  local config_extra
  config_extra="$(method_extra "$method")" || return 2
  local config="${config_extra%%|*}"
  local extra="${config_extra#*|}"
  log_status "START gpu=$gpu phase=$phase method=$method seed=$seed env=$env_key at $(date)"
  (
    cd "$worker/epymarl" && CUDA_VISIBLE_DEVICES="$gpu" python3 src/main.py \
      --config="$config" \
      --env-config=gymma \
      with \
      env_args.key="$env_key" \
      env_args.time_limit="$TIME_LIMIT" \
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
  if [[ "$code" -eq 0 ]]; then
    touch "$done_file"
  else
    printf '%s\n' "$code" > "$fail_file"
  fi
  log_status "END gpu=$gpu phase=$phase method=$method seed=$seed exit=$code at $(date)"
  return "$code"
}

worker_loop() {
  local worker="$1"
  local gpu="$2"
  local jobs_file="$3"
  while IFS='|' read -r phase env_key method seed; do
    [[ -z "${phase:-}" ]] && continue
    run_one "$worker" "$gpu" "$phase" "$env_key" "$method" "$seed"
  done < "$jobs_file"
}

summarize() {
  python3 "$ROOT_DIR/epymarl/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" --baseline baseline || true
  python3 - <<'PY' "$OUT_DIR"
import csv, os, sys
out = sys.argv[1]
path = os.path.join(out, 'summary.csv')
if not os.path.exists(path):
    raise SystemExit(0)
rows = list(csv.DictReader(open(path, encoding='utf-8')))
groups = {}
for row in rows:
    key = row['method']
    groups.setdefault(key, []).append(row)
with open(os.path.join(out, 'aaai_decision_snapshot.txt'), 'w', encoding='utf-8') as handle:
    handle.write('AAAI decision snapshot\n\n')
    for method in sorted(groups):
        items = groups[method]
        handle.write(f'method={method} n={len(items)}\n')
        for metric in ['last_test_return', 'train_auc', 'stability_gap', 'best_train_return']:
            vals = [float(r[metric]) for r in items if r.get(metric) not in ('', None)]
            if vals:
                handle.write(f'  {metric}_mean={sum(vals)/len(vals):.6f}\n')
        handle.write('\n')
PY
}

main() {
  log_status "Round 6 AAAI two-GPU experiment started at $(date)"
  log_status "root=$ROOT_DIR out=$OUT_DIR"
  write_plan
  prepare_worker "$WORKER0"
  prepare_worker "$WORKER1"

  MAIN_METHODS=(baseline diagnosis_only uniform_0.0002 uniform_0.0003 type_specific_0.0002 type_specific_0.0003 adaptive_0.0002_late045 adaptive_0.0002_late060 adaptive_0.0003_late045 random_type_0.0002)
  GEN_METHODS=(baseline uniform_0.0002 uniform_0.0003 adaptive_0.0002_late045 adaptive_0.0002_late060)
  make_jobs main10x10 "$ENV_10" "$SEEDS_MAIN" "${MAIN_METHODS[@]}" > "$OUT_DIR/jobs_main.txt"
  make_jobs generalization "$ENV_GEN" "$SEEDS_GEN" "${GEN_METHODS[@]}" > "$OUT_DIR/jobs_gen.txt"
  cat "$OUT_DIR/jobs_main.txt" "$OUT_DIR/jobs_gen.txt" > "$OUT_DIR/jobs_all.txt"
  awk 'NR % 2 == 1' "$OUT_DIR/jobs_all.txt" > "$OUT_DIR/jobs_gpu0.txt"
  awk 'NR % 2 == 0' "$OUT_DIR/jobs_all.txt" > "$OUT_DIR/jobs_gpu1.txt"
  log_status "job_count=$(wc -l < "$OUT_DIR/jobs_all.txt") gpu0=$(wc -l < "$OUT_DIR/jobs_gpu0.txt") gpu1=$(wc -l < "$OUT_DIR/jobs_gpu1.txt")"

  worker_loop "$WORKER0" 0 "$OUT_DIR/jobs_gpu0.txt" &
  pid0=$!
  worker_loop "$WORKER1" 1 "$OUT_DIR/jobs_gpu1.txt" &
  pid1=$!
  wait "$pid0"
  code0=$?
  wait "$pid1"
  code1=$?
  summarize
  cp "$ROOT_DIR/run_round6_aaai_two_gpu.sh" "$OUT_DIR/" || true
  tar -czf "$ROOT_DIR/artifacts/AAAI_round6_aaai_submission_${STAMP}.tar.gz" -C "$ROOT_DIR" "results/$(basename "$OUT_DIR")" "run_round6_aaai_two_gpu.sh" || true
  log_status "Round 6 finished at $(date) worker0=$code0 worker1=$code1 artifact=artifacts/AAAI_round6_aaai_submission_${STAMP}.tar.gz"
  if [[ "$code0" -ne 0 || "$code1" -ne 0 ]]; then
    exit 1
  fi
}

mkdir -p "$ROOT_DIR/artifacts"
main "$@"
