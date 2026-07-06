#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/round9_supplement_${STAMP}}"
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/STATUS.txt"
WORKER="${WORKER:-/data/lab/AAAI_worker_round9_supplement}"
GPU="${GPU:-0}"
mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"

SEEDS_LBF12="${SEEDS_LBF12:-9 10 11 12}"
SEEDS_RWARE="${SEEDS_RWARE:-1 2 3 4 5}"
SEEDS_VMAS="${SEEDS_VMAS:-1 2 3 4 5}"
TMAX_LBF12="${TMAX_LBF12:-500000}"
TMAX_RWARE="${TMAX_RWARE:-300000}"
TMAX_VMAS="${TMAX_VMAS:-300000}"
TEST_INTERVAL="${TEST_INTERVAL:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-25000}"
TEST_NEPISODE="${TEST_NEPISODE:-20}"
MAX_RECORDS="${MAX_RECORDS:-3000}"
RUN_LBF12_EXT="${RUN_LBF12_EXT:-1}"
RUN_RWARE="${RUN_RWARE:-1}"
RUN_VMAS="${RUN_VMAS:-1}"
METHODS_LBF12="${METHODS_LBF12:-baseline uniform_budget_matched_0.0003_late045 adaptive_0.0003_late045 random_type_budget_matched_0.0003_late045 semantic_shuffled_budget_matched_0.0003_late045 diagnosis_only}"
METHODS_RWARE="${METHODS_RWARE:-baseline uniform_budget_matched_0.0003_late045 adaptive_0.0003_late045 random_type_budget_matched_0.0003_late045 semantic_shuffled_budget_matched_0.0003_late045 diagnosis_only}"
METHODS_VMAS="${METHODS_VMAS:-baseline uniform_budget_matched_0.0003_late045 adaptive_0.0003_late045 random_type_budget_matched_0.0003_late045 diagnosis_only}"

log_status() { printf '%s\n' "$*" | tee -a "$STATUS"; }

prepare_worker() {
  rm -rf "$WORKER"
  mkdir -p "$WORKER"
  tar --exclude='./results' --exclude='./artifacts' --exclude='./.git' -C "$ROOT_DIR" -cf - . | tar -C "$WORKER" -xf -
}

write_plan() {
  cat > "$OUT_DIR/ROUND9_EXPERIMENT_PLAN.md" <<EOF
# Round 9 Supplemental Stabilization

Goal: address the remaining AAAI risk by adding a cross-domain task package and extending 12x12 LBF seeds. The main claim remains failure-triggered adaptive reward shaping; Qwen/semantic causality is not used as the main causal claim.

Experiments:
- LBF 12x12 seed extension: seeds $SEEDS_LBF12, t_max=$TMAX_LBF12, methods=$METHODS_LBF12.
- RWARE tiny cross-domain: seeds $SEEDS_RWARE, t_max=$TMAX_RWARE, methods=$METHODS_RWARE.
- VMAS navigation cross-domain: seeds $SEEDS_VMAS, t_max=$TMAX_VMAS, methods=$METHODS_VMAS.

Decision use:
- A strong AAAI boost requires adaptive to remain positive against baseline and not lose to phase-uniform/random controls in RWARE or VMAS.
- If cross-domain results are mixed, report them transparently and use them as limitation evidence rather than overclaiming.
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
  if [[ "$RUN_LBF12_EXT" == "1" ]]; then
    for seed in $SEEDS_LBF12; do for method in $METHODS_LBF12; do
      printf 'lbf12ext|lbforaging:Foraging-12x12-3p-4f-v3|50|%s|%s|%s\n' "$TMAX_LBF12" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"
    done; done
  fi
  if [[ "$RUN_RWARE" == "1" ]]; then
    for seed in $SEEDS_RWARE; do for method in $METHODS_RWARE; do
      printf 'rwaretiny|rware:rware-tiny-2ag-v2|50|%s|%s|%s\n' "$TMAX_RWARE" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"
    done; done
  fi
  if [[ "$RUN_VMAS" == "1" ]]; then
    for seed in $SEEDS_VMAS; do for method in $METHODS_VMAS; do
      printf 'vmasnav|vmas-navigation|50|%s|%s|%s\n' "$TMAX_VMAS" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"
    done; done
  fi
  if [[ "${SHUFFLE_JOBS:-1}" == "1" ]]; then
    python3 - "$OUT_DIR/jobs_all.txt" "${JOB_SHUFFLE_SEED:-9}" <<'PYSHUFFLE'
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
  python3 "$ROOT_DIR/epymarl/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" --baseline lbf12ext_lbforaging_Foraging-12x12-3p-4f-v3_baseline >/dev/null 2>&1 || true
  python3 "$ROOT_DIR/epymarl/tools/build_round9_supplement_report.py" --out-dir "$OUT_DIR" >/dev/null 2>&1 || true
  log_status "SUMMARY_REFRESH completed=$(find "$LOG_DIR" -name '*.done' | wc -l) failed=$(find "$LOG_DIR" -name '*.fail' | wc -l) at $(date)"
}

package_outputs() {
  summarize_partial
  local artifact="$ROOT_DIR/artifacts/AAAI_round9_supplement_${STAMP}.tar.gz"
  tar -C "$ROOT_DIR" -czf "$artifact" "results/$(basename "$OUT_DIR")" "run_round9_supplement.sh" "epymarl/tools/build_round9_supplement_report.py"
  log_status "ARTIFACT $artifact"
}

main() {
  log_status "Round 9 supplemental stabilization started at $(date)"
  log_status "root=$ROOT_DIR out=$OUT_DIR gpu=$GPU worker=$WORKER"
  write_plan
  prepare_worker
  make_jobs
  log_status "job_count=$(wc -l < "$OUT_DIR/jobs_all.txt")"
  while IFS='|' read -r phase env_key time_limit tmax method seed; do
    run_one "$phase" "$env_key" "$time_limit" "$tmax" "$method" "$seed" || true
  done < "$OUT_DIR/jobs_all.txt"
  package_outputs
  log_status "Round 9 supplemental stabilization finished at $(date)"
}

main "$@"
