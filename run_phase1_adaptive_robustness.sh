#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/phase1_adaptive_robustness_${STAMP}}"
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/STATUS.txt"
mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"

ENV_10="${ENV_10:-lbforaging:Foraging-10x10-3p-3f-v3}"
TIME_LIMIT="${TIME_LIMIT:-50}"
T_MAX="${T_MAX:-500000}"
TEST_INTERVAL="${TEST_INTERVAL:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-25000}"
TEST_NEPISODE="${TEST_NEPISODE:-20}"
MAX_RECORDS="${MAX_RECORDS:-3000}"
SEEDS_MAIN="${SEEDS_MAIN:-1 2 3 4 5 6 7 8}"
METHODS_MAIN="${METHODS_MAIN:-adaptive_0.0005_late045 uniform_0.0005 random_type_0.0005_late045}"
WORKER="${WORKER:-/data/lab/AAAI_worker_phase1_robustness}"
GPU="${GPU:-0}"
ROUND6_SUMMARY="${ROUND6_SUMMARY:-$ROOT_DIR/results/round6_aaai_submission_20260702_134358/summary.csv}"
ROUND7_SUMMARY="${ROUND7_SUMMARY:-$ROOT_DIR/results/round7_decisive_aaai_20260702_round7_decisive/summary.csv}"

log_status() {
  printf '%s\n' "$*" | tee -a "$STATUS"
}

prepare_worker() {
  rm -rf "$WORKER"
  mkdir -p "$WORKER"
  tar --exclude='./results' --exclude='./artifacts' --exclude='./.git' -C "$ROOT_DIR" -cf - . | tar -C "$WORKER" -xf -
}

write_plan() {
  cat > "$OUT_DIR/PHASE1_EXPERIMENT_PLAN.md" <<EOF
# Phase 1 Adaptive Robustness Experiment

This run continues Phase 1 of the roadmap: failure-triggered adaptive reward shaping on the 10x10 LBF task. Round 6 already contains eight seeds for adaptive_0.0002_late045 and adaptive_0.0003_late045. This continuation adds the missing upper penalty point adaptive_0.0005_late045 and two calibrated controls at the same nominal intervention magnitude: uniform_0.0005 and random_type_0.0005_late045.

The decision question is whether adaptive shaping remains competitive across penalty values rather than being a single lucky 0.0003 setting. The primary comparisons after merging Round 6, Round 7, and this run are adaptive_0.0005_late045 against baseline, uniform_0.0005, random_type_0.0005_late045, adaptive_0.0003_late045, and uniform_0.0003 on final test return and train AUC.

Environment: $ENV_10
Seeds: $SEEDS_MAIN
Budget: $T_MAX timesteps per run
Methods: $METHODS_MAIN
EOF
}

method_extra() {
  local method="$1"
  local config="mappo_llm_fd"
  local extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True"
  case "$method" in
    adaptive_0.0005_late045)
      extra="$extra llm_fd_failure_penalty=0.0005 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0005 llm_fd_late_phase_weight=0.45"
      ;;
    uniform_0.0005)
      extra="$extra llm_fd_failure_penalty=0.0005 llm_fd_intervention_mode=uniform llm_fd_terminal_bonus=0.0005"
      ;;
    random_type_0.0005_late045)
      extra="$extra llm_fd_failure_penalty=0.0005 llm_fd_intervention_mode=random_type llm_fd_terminal_bonus=0.0005 llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.45"
      ;;
    *) return 2 ;;
  esac
  printf '%s|%s\n' "$config" "$extra"
}

make_jobs() {
  : > "$OUT_DIR/jobs_all.txt"
  for seed in $SEEDS_MAIN; do
    for method in $METHODS_MAIN; do
      printf 'main10x10|%s|%s|%s\n' "$ENV_10" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"
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
  python3 "$ROOT_DIR/epymarl/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" --baseline main10x10_lbforaging_Foraging-10x10-3p-3f-v3_adaptive_0.0005_late045 >/dev/null 2>&1 || true
  python3 - "$OUT_DIR" "$ROUND6_SUMMARY" "$ROUND7_SUMMARY" <<'INNERPY' >/dev/null 2>&1 || true
import csv, os, random, sys
from collections import defaultdict

out, round6, round7 = sys.argv[1:4]
current = os.path.join(out, 'summary.csv')
main_prefix = 'main10x10_lbforaging_Foraging-10x10-3p-3f-v3_'
metrics = ['last_test_return', 'train_auc', 'best_train_return', 'stability_gap']

def load(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as h:
        return list(csv.DictReader(h))

def norm(name):
    return name[len(main_prefix):] if name.startswith(main_prefix) else None

best = {}
for path in [round6, round7, current]:
    for row in load(path):
        method = norm(row.get('method', ''))
        if not method or not row.get('last_test_return'):
            continue
        best[(method, row['seed'])] = dict(row, method_short=method)
rows = list(best.values())

def vals(method, metric):
    return [float(r[metric]) for r in rows if r['method_short'] == method and r.get(metric)]

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def ci(xs, n=5000):
    if not xs:
        return None, None
    rng = random.Random(13)
    ms = []
    for _ in range(n):
        s = [xs[rng.randrange(len(xs))] for _ in xs]
        ms.append(sum(s) / len(s))
    ms.sort()
    return ms[int(.025*n)], ms[int(.975*n)]

def fmt(x):
    return 'NA' if x is None else f'{x:.4f}'

by = defaultdict(dict)
for r in rows:
    by[r['method_short']][r['seed']] = r

methods = sorted(by)
with open(os.path.join(out, 'PHASE1_LIVE_MERGED_REPORT.md'), 'w', encoding='utf-8') as h:
    h.write('# Phase 1 Live Merged Report\n\n')
    h.write('This report merges Round 6, Round 7, and the current Phase 1 robustness run for the 10x10 task.\n\n')
    h.write('| method | n | last test | last test 95% CI | train AUC | best train | stability gap |\n')
    h.write('|---|---:|---:|---:|---:|---:|---:|\n')
    for m in methods:
        x = vals(m, 'last_test_return')
        lo, hi = ci(x)
        h.write(f'| {m} | {len(x)} | {fmt(mean(x))} | [{fmt(lo)}, {fmt(hi)}] | {fmt(mean(vals(m, "train_auc")))} | {fmt(mean(vals(m, "best_train_return")))} | {fmt(mean(vals(m, "stability_gap")))} |\n')
    h.write('\n## Paired Phase 1 Comparisons\n\n')
    h.write('| comparison | metric | n | mean delta | 95% CI | shared seeds |\n')
    h.write('|---|---|---:|---:|---:|---|\n')
    comparisons = [
        ('adaptive_0.0005_late045', 'baseline'),
        ('adaptive_0.0005_late045', 'uniform_0.0005'),
        ('adaptive_0.0005_late045', 'random_type_0.0005_late045'),
        ('adaptive_0.0005_late045', 'adaptive_0.0003_late045'),
        ('adaptive_0.0003_late045', 'baseline'),
        ('adaptive_0.0003_late045', 'uniform_0.0003'),
        ('adaptive_0.0003_late045', 'random_type_matched_0.0003_late045'),
    ]
    for left, right in comparisons:
        for metric in metrics:
            seeds = sorted(set(by.get(left, {})) & set(by.get(right, {})), key=int)
            diffs = [float(by[left][s][metric]) - float(by[right][s][metric]) for s in seeds if by[left][s].get(metric) and by[right][s].get(metric)]
            if not diffs:
                continue
            lo, hi = ci(diffs)
            h.write(f'| {left} - {right} | {metric} | {len(diffs)} | {fmt(mean(diffs))} | [{fmt(lo)}, {fmt(hi)}] | {",".join(seeds)} |\n')
INNERPY
  log_status "SUMMARY_REFRESH completed=$(find "$LOG_DIR" -name '*.done' | wc -l) failed=$(find "$LOG_DIR" -name '*.fail' | wc -l) at $(date)"
}

main() {
  log_status "Phase 1 adaptive robustness started at $(date)"
  log_status "root=$ROOT_DIR out=$OUT_DIR gpu=$GPU"
  write_plan
  prepare_worker
  make_jobs
  log_status "job_count=$(wc -l < "$OUT_DIR/jobs_all.txt") methods=$METHODS_MAIN seeds=$SEEDS_MAIN"
  while IFS='|' read -r phase env_key method seed; do
    [[ -z "${phase:-}" ]] && continue
    run_one "$phase" "$env_key" "$method" "$seed"
  done < "$OUT_DIR/jobs_all.txt"
  summarize_partial
  cp "$ROOT_DIR/run_phase1_adaptive_robustness.sh" "$OUT_DIR/" || true
  tar -czf "$ROOT_DIR/artifacts/AAAI_phase1_adaptive_robustness_${STAMP}.tar.gz" -C "$ROOT_DIR" "results/$(basename "$OUT_DIR")" "run_phase1_adaptive_robustness.sh" || true
  log_status "Phase 1 adaptive robustness finished at $(date) artifact=artifacts/AAAI_phase1_adaptive_robustness_${STAMP}.tar.gz"
}

main "$@"
