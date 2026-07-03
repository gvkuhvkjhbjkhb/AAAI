#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/round7_decisive_aaai_${STAMP}}"
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/STATUS.txt"
mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"

ENV_10="${ENV_10:-lbforaging:Foraging-10x10-3p-3f-v3}"
ENV_GEN="${ENV_GEN:-lbforaging:Foraging-12x12-3p-4f-v3}"
TIME_LIMIT="${TIME_LIMIT:-50}"
T_MAX="${T_MAX:-500000}"
TEST_INTERVAL="${TEST_INTERVAL:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-25000}"
TEST_NEPISODE="${TEST_NEPISODE:-20}"
MAX_RECORDS="${MAX_RECORDS:-3000}"
SEEDS_MAIN="${SEEDS_MAIN:-1 2 3 4 5 6 7 8}"
SEEDS_GEN="${SEEDS_GEN:-6 7 8}"
WORKER="${WORKER:-/data/lab/AAAI_worker_round7_decisive}"
GPU="${GPU:-0}"
ROUND6_SUMMARY="${ROUND6_SUMMARY:-$ROOT_DIR/results/round6_aaai_submission_20260702_134358/summary.csv}"

log_status() {
  printf '%s\n' "$*" | tee -a "$STATUS"
}

prepare_worker() {
  rm -rf "$WORKER"
  mkdir -p "$WORKER"
  tar --exclude='./results' --exclude='./artifacts' --exclude='./.git' -C "$ROOT_DIR" -cf - . | tar -C "$WORKER" -xf -
}

write_plan() {
  cat > "$OUT_DIR/EXPERIMENT_PLAN.md" <<EOF
# Round 7 Decisive AAAI One-GPU Experiment

This run is designed to close the Round 6 reviewer gap in one pass on a single RTX 5090. It reuses completed Round 6 baseline, uniform, and adaptive seeds, then adds only the decisive missing controls: matched-frequency random type, ordinary random type at matched penalty/late schedule, confidence-gated semantic adaptive, and conservative 12x12 completion seeds.

## Main 10x10 Jobs

- Environment: $ENV_10
- Seeds: $SEEDS_MAIN
- Budget: $T_MAX timesteps
- Methods: random_type_matched_0.0003_late045, random_type_0.0003_late045, random_type_0.0003_late060, semantic_gate_0.0003_late045
- Primary decision: semantic_gate_0.0003_late045 must beat matched random-type and calibrated uniform/adaptive baselines in paired seed comparisons for a strong semantic AAAI claim.

## Conservative 12x12 Completion Jobs

- Environment: $ENV_GEN
- Seeds: $SEEDS_GEN
- Methods: baseline, uniform_0.0002, adaptive_0.0003_late045, semantic_gate_0.0003_late045
- Interpretation: stress test unless semantic_gate clearly wins.

## Expected Runtime

Typical Round 6/7 runtime is 6-7 minutes per 500k run. With 44 queued jobs and per-run summarization, expected wall time is 4.5-5.5 hours. STATUS.txt is updated after every START, DONE, FAIL, and summary refresh.
EOF
}

method_extra() {
  local method="$1"
  local config="mappo"
  local extra=""
  case "$method" in
    baseline)
      config="mappo"
      ;;
    uniform_0.0002)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0002 llm_fd_intervention_mode=uniform"
      ;;
    adaptive_0.0003_late045)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45"
      ;;
    random_type_0.0003_late045)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=random_type llm_fd_terminal_bonus=0.0003 llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.45"
      ;;
    random_type_0.0003_late060)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=random_type llm_fd_terminal_bonus=0.0003 llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.60"
      ;;
    random_type_matched_0.0003_late045)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=random_type_matched llm_fd_terminal_bonus=0.0003 llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.45"
      ;;
    semantic_gate_0.0003_late045)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=semantic_gate llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45 llm_fd_semantic_gate_threshold=0.55 llm_fd_semantic_gate_fallback_weight=1.0 llm_fd_weight_inefficient_exploration=0.65 llm_fd_weight_target_miscoordination=1.45 llm_fd_weight_insufficient_cooperation=1.35 llm_fd_weight_low_value_overcommitment=1.20 llm_fd_weight_timeout_near_success=0.00 llm_fd_weight_unknown=0.75"
      ;;
    *) return 2 ;;
  esac
  printf '%s|%s\n' "$config" "$extra"
}

make_jobs() {
  : > "$OUT_DIR/jobs_all.txt"
  for seed in $SEEDS_MAIN; do
    for method in random_type_matched_0.0003_late045 semantic_gate_0.0003_late045 random_type_0.0003_late045 random_type_0.0003_late060; do
      printf 'main10x10|%s|%s|%s\n' "$ENV_10" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"
    done
  done
  for seed in $SEEDS_GEN; do
    for method in baseline uniform_0.0002 adaptive_0.0003_late045 semantic_gate_0.0003_late045; do
      printf 'generalization|%s|%s|%s\n' "$ENV_GEN" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"
    done
  done
}

run_one() {
  local phase="$1" env_key="$2" method="$3" seed="$4"
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
  python3 "$ROOT_DIR/epymarl/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" --baseline main10x10_lbforaging_Foraging-10x10-3p-3f-v3_semantic_gate_0.0003_late045 >/dev/null 2>&1 || true
  if [[ -f "$ROUND6_SUMMARY" && -f "$OUT_DIR/summary.csv" ]]; then
    python3 "$ROOT_DIR/epymarl/tools/build_round7_decision_report.py" --output "$OUT_DIR/ROUND7_DECISIVE_REPORT.md" "$ROUND6_SUMMARY" "$OUT_DIR/summary.csv" >/dev/null 2>&1 || true
  fi
  python3 - "$OUT_DIR" <<'INNERPY' >/dev/null 2>&1 || true
import csv, os, sys
from collections import defaultdict
out = sys.argv[1]
path = os.path.join(out, 'summary.csv')
if not os.path.exists(path):
    raise SystemExit
def short_name(name):
    if '_lbforaging_Foraging-10x10-3p-3f-v3_' in name:
        return 'main10x10:' + name.split('_lbforaging_Foraging-10x10-3p-3f-v3_', 1)[1]
    if '_lbforaging_Foraging-12x12-3p-4f-v3_' in name:
        return 'generalization:' + name.split('_lbforaging_Foraging-12x12-3p-4f-v3_', 1)[1]
    return name
rows = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r.get('last_test_return')]
groups = defaultdict(list)
for row in rows:
    groups[short_name(row['method'])].append(row)
with open(os.path.join(out, 'ROUND7_LIVE_AGGREGATES.md'), 'w', encoding='utf-8') as h:
    h.write('# Round 7 Live Aggregates\n\n')
    h.write(f'Completed successful logs: {len(rows)}\n\n')
    h.write('| method | n | mean last test | mean train AUC |\n|---|---:|---:|---:|\n')
    for method in sorted(groups):
        items = groups[method]
        def avg(k):
            vals=[float(r[k]) for r in items if r.get(k)]
            return sum(vals)/len(vals) if vals else float('nan')
        h.write(f'| {method} | {len(items)} | {avg("last_test_return"):.4f} | {avg("train_auc"):.4f} |\n')
INNERPY
  log_status "SUMMARY_REFRESH completed=$(find "$LOG_DIR" -name '*.done' | wc -l) failed=$(find "$LOG_DIR" -name '*.fail' | wc -l) at $(date)"
}

main() {
  log_status "Round 7 decisive AAAI experiment started at $(date)"
  log_status "root=$ROOT_DIR out=$OUT_DIR gpu=$GPU"
  write_plan
  prepare_worker
  make_jobs
  log_status "job_count=$(wc -l < "$OUT_DIR/jobs_all.txt")"
  while IFS='|' read -r phase env_key method seed; do
    [[ -z "${phase:-}" ]] && continue
    run_one "$phase" "$env_key" "$method" "$seed"
  done < "$OUT_DIR/jobs_all.txt"
  summarize_partial
  cp "$ROOT_DIR/run_round7_decisive_one_gpu.sh" "$OUT_DIR/" || true
  tar -czf "$ROOT_DIR/artifacts/AAAI_round7_decisive_aaai_${STAMP}.tar.gz" -C "$ROOT_DIR" "results/$(basename "$OUT_DIR")" "run_round7_decisive_one_gpu.sh" || true
  log_status "Round 7 decisive finished at $(date) artifact=artifacts/AAAI_round7_decisive_aaai_${STAMP}.tar.gz"
}

main "$@"
