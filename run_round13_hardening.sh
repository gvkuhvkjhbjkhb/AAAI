#!/usr/bin/env bash
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/round13_hardening_${STAMP}}"
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/STATUS.txt"
WORKER="${WORKER:-/data/lab/AAAI_worker_round13_hardening}"
GPU="${GPU:-0}"
mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"

ENV_KEY="${ENV_KEY:-lbforaging:Foraging-10x10-3p-3f-v3}"
TIME_LIMIT="${TIME_LIMIT:-50}"
T_MAX="${T_MAX:-500000}"
SEED_EXT_SEEDS="${SEED_EXT_SEEDS:-9 10 11 12 13 14 15 16}"
SEED_EXT_METHODS="${SEED_EXT_METHODS:-baseline adaptive_0.0003_late045 uniform_budget_matched_0.0003_late045 random_type_budget_matched_0.0003_late045}"
ACTUAL_BUDGET_SEEDS="${ACTUAL_BUDGET_SEEDS:-1 2 3 4 5 6 7 8}"
ACTUAL_BUDGET_METHODS="${ACTUAL_BUDGET_METHODS:-uniform_actual_budget_matched random_actual_budget_matched}"
SENS_EXT_SEEDS="${SENS_EXT_SEEDS:-5 6 7 8}"
SENS_EXT_METHODS="${SENS_EXT_METHODS:-adaptive_0.0002_late045 adaptive_0.0005_late045 adaptive_0.0003_late060}"
RUN_SEED_EXT="${RUN_SEED_EXT:-1}"
RUN_ACTUAL_BUDGET="${RUN_ACTUAL_BUDGET:-1}"
RUN_SENS_EXT="${RUN_SENS_EXT:-1}"
TEST_INTERVAL="${TEST_INTERVAL:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-25000}"
TEST_NEPISODE="${TEST_NEPISODE:-20}"
MAX_RECORDS="${MAX_RECORDS:-3000}"

log_status(){ printf '%s\n' "$*" | tee -a "$STATUS"; }
prepare_worker(){ rm -rf "$WORKER"; mkdir -p "$WORKER"; tar --exclude='./results' --exclude='./artifacts' --exclude='./.git' -C "$ROOT_DIR" -cf - . | tar -C "$WORKER" -xf -; }
write_plan(){ cat > "$OUT_DIR/ROUND13_EXPERIMENT_PLAN.md" <<EOF
# Round 13 AAAI Main-Claim Hardening

Goal: harden the main 10x10 claim, not search for new domains.

Components:
- Seed extension: $SEED_EXT_METHODS on seeds $SEED_EXT_SEEDS.
- Actual-budget controls: $ACTUAL_BUDGET_METHODS on seeds $ACTUAL_BUDGET_SEEDS.
- Sensitivity extension: $SENS_EXT_METHODS on seeds $SENS_EXT_SEEDS.

Actual-budget coefficients are calibrated from prior accounting: uniform 0.00024 and random 0.00027, lower than nominal 0.0003, to better match adaptive's realized shaping budget.
EOF
}
method_extra(){
  local method="$1" config="mappo" extra=""
  case "$method" in
    baseline) config="mappo"; extra="" ;;
    adaptive_0.0003_late045) config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45" ;;
    uniform_budget_matched_0.0003_late045) config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=phase_uniform llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45" ;;
    random_type_budget_matched_0.0003_late045) config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=random_type llm_fd_terminal_bonus=0.0003 llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.45" ;;
    uniform_actual_budget_matched) config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.00024 llm_fd_intervention_mode=phase_uniform llm_fd_terminal_bonus=0.00024 llm_fd_late_phase_weight=0.45" ;;
    random_actual_budget_matched) config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.00027 llm_fd_intervention_mode=random_type llm_fd_terminal_bonus=0.00027 llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.45" ;;
    adaptive_0.0002_late045) config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0002 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0002 llm_fd_late_phase_weight=0.45" ;;
    adaptive_0.0005_late045) config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0005 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0005 llm_fd_late_phase_weight=0.45" ;;
    adaptive_0.0003_late060) config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.60" ;;
    *) return 2 ;;
  esac
  printf '%s|%s\n' "$config" "$extra"
}
make_jobs(){
  : > "$OUT_DIR/jobs_all.txt"
  if [[ "$RUN_SEED_EXT" == "1" ]]; then for seed in $SEED_EXT_SEEDS; do for method in $SEED_EXT_METHODS; do printf 'seedext10|%s|%s|%s|%s|%s\n' "$ENV_KEY" "$TIME_LIMIT" "$T_MAX" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"; done; done; fi
  if [[ "$RUN_ACTUAL_BUDGET" == "1" ]]; then for seed in $ACTUAL_BUDGET_SEEDS; do for method in $ACTUAL_BUDGET_METHODS; do printf 'actualbudget10|%s|%s|%s|%s|%s\n' "$ENV_KEY" "$TIME_LIMIT" "$T_MAX" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"; done; done; fi
  if [[ "$RUN_SENS_EXT" == "1" ]]; then for seed in $SENS_EXT_SEEDS; do for method in $SENS_EXT_METHODS; do printf 'sensitivity10ext|%s|%s|%s|%s|%s\n' "$ENV_KEY" "$TIME_LIMIT" "$T_MAX" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"; done; done; fi
  if [[ "${SHUFFLE_JOBS:-1}" == "1" ]]; then python3 - "$OUT_DIR/jobs_all.txt" "${JOB_SHUFFLE_SEED:-13}" <<'PYSHUFFLE'
import random, sys
p, seed = sys.argv[1], int(sys.argv[2])
rows = open(p, encoding='utf-8').readlines(); random.Random(seed).shuffle(rows); open(p,'w',encoding='utf-8').writelines(rows)
PYSHUFFLE
  fi
}
sanitize_env(){ printf '%s' "$1" | tr ':/' '__'; }
run_one(){
  local phase="$1" env_key="$2" time_limit="$3" tmax="$4" method="$5" seed="$6"
  local safe_env log done_file fail_file config_extra config extra start end elapsed code
  safe_env="$(sanitize_env "$env_key")"; log="$LOG_DIR/${phase}_${safe_env}_${method}_seed${seed}.log"; done_file="$log.done"; fail_file="$log.fail"; rm -f "$fail_file"
  if [[ -f "$done_file" ]]; then log_status "SKIP phase=$phase method=$method seed=$seed"; return 0; fi
  config_extra="$(method_extra "$method")" || return 2; config="${config_extra%%|*}"; extra="${config_extra#*|}"
  start=$(date +%s); log_status "START phase=$phase method=$method seed=$seed env=$env_key tmax=$tmax at $(date)"
  (cd "$WORKER/epymarl" && CUDA_VISIBLE_DEVICES="$GPU" python3 src/main.py --config="$config" --env-config=gymma with env_args.key="$env_key" env_args.time_limit="$time_limit" t_max="$tmax" use_cuda=True test_nepisode="$TEST_NEPISODE" test_interval="$TEST_INTERVAL" log_interval="$LOG_INTERVAL" runner_log_interval="$LOG_INTERVAL" learner_log_interval="$LOG_INTERVAL" seed="$seed" $extra) > "$log" 2>&1
  code=$?; end=$(date +%s); elapsed=$((end-start))
  if [[ "$code" -eq 0 ]]; then touch "$done_file"; log_status "DONE phase=$phase method=$method seed=$seed exit=$code elapsed=${elapsed}s at $(date)"; else printf '%s\n' "$code" > "$fail_file"; log_status "FAIL phase=$phase method=$method seed=$seed exit=$code elapsed=${elapsed}s at $(date)"; fi
  summarize_partial; return "$code"
}
summarize_partial(){ python3 "$ROOT_DIR/epymarl/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" --baseline seedext10_lbforaging_Foraging-10x10-3p-3f-v3_baseline >/dev/null 2>&1 || true; python3 "$ROOT_DIR/epymarl/tools/build_round13_hardening_report.py" --out-dir "$OUT_DIR" >/dev/null 2>&1 || true; log_status "SUMMARY_REFRESH completed=$(find "$LOG_DIR" -name '*.done' | wc -l) failed=$(find "$LOG_DIR" -name '*.fail' | wc -l) at $(date)"; }
package_outputs(){ summarize_partial; local artifact="$ROOT_DIR/artifacts/AAAI_round13_hardening_${STAMP}.tar.gz"; tar -C "$ROOT_DIR" -czf "$artifact" "results/$(basename "$OUT_DIR")" "run_round13_hardening.sh" "epymarl/tools/build_round13_hardening_report.py"; log_status "ARTIFACT $artifact"; }
main(){ log_status "Round 13 hardening started at $(date)"; log_status "root=$ROOT_DIR out=$OUT_DIR gpu=$GPU worker=$WORKER"; write_plan; prepare_worker; make_jobs; log_status "job_count=$(wc -l < "$OUT_DIR/jobs_all.txt")"; while IFS='|' read -r phase env_key time_limit tmax method seed; do run_one "$phase" "$env_key" "$time_limit" "$tmax" "$method" "$seed" || true; done < "$OUT_DIR/jobs_all.txt"; package_outputs; log_status "Round 13 hardening finished at $(date)"; }
main "$@"
