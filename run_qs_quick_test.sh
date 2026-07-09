#!/usr/bin/env bash
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="qs_quick_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/results/${STAMP}"
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/STATUS.txt"
WORKER_BASE="/data/lab/AAAI_worker_qs"
GPU="0"
PARALLEL=6
TEST_INTERVAL=50000
LOG_INTERVAL=25000
TEST_NEPISODE=5
MAX_RECORDS=500
mkdir -p "$LOG_DIR"
log_status(){ printf '%s\n' "$*" | tee -a "$STATUS"; }
prepare_worker(){ local idx="$1"; local w="${WORKER_BASE}_${idx}"; rm -rf "$w"; mkdir -p "$w"; tar --exclude='./results' --exclude='./artifacts' --exclude='./.git' -C "$ROOT_DIR" -cf - . | tar -C "$w" -xf -; }
method_extra(){
  local method="$1" config="mappo" extra=""
  case "$method" in
    baseline) config="mappo"; extra="" ;;
    fa_qs_l002) config="mappo_llm_fd"; extra="llm_fd_enabled=True llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=False llm_fd_intervention_mode=uniform qs_enabled=True qs_lambda=0.02 qs_alpha=0.01 qs_beta=0.05 qs_adaptive=True" ;;
    qs_random_features_l002) config="mappo_llm_fd"; extra="llm_fd_enabled=False llm_fd_apply_reward_shaping=False llm_fd_intervention_mode=uniform qs_enabled=True qs_lambda=0.02 qs_alpha=0.0 qs_beta=0.0 qs_adaptive=False qs_random_features=True" ;;
    *) return 2 ;;
  esac
  printf '%s|%s\n' "$config" "$extra"
}
make_jobs(){
  : > "$OUT_DIR/jobs_all.txt"
  for seed in 1 9; do
    for method in baseline fa_qs_l002 qs_random_features_l002; do
      printf 'qs|lbforaging:Foraging-10x10-3p-3f-v3|50|100000|%s|%s\n' "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"
    done
  done
}
run_one(){
  local worker_idx="$1" env_key="$2" time_limit="$3" tmax="$4" method="$5" seed="$6"
  local worker="${WORKER_BASE}_${worker_idx}" log done_file config_extra config extra
  log="$LOG_DIR/qs_${method}_seed${seed}.log"; done_file="$log.done"
  config_extra="$(method_extra "$method")"; config="${config_extra%%|*}"; extra="${config_extra#*|}"
  (cd "$worker/epymarl" && CUDA_VISIBLE_DEVICES="$GPU" python3 src/main.py --config="$config" --env-config=gymma with env_args.key="$env_key" env_args.time_limit="$time_limit" t_max="$tmax" use_cuda=True test_nepisode="$TEST_NEPISODE" test_interval="$TEST_INTERVAL" log_interval="$LOG_INTERVAL" runner_log_interval="$LOG_INTERVAL" learner_log_interval="$LOG_INTERVAL" seed="$seed" $extra) > "$log" 2>&1
  local code=$?
  if [[ "$code" -eq 0 ]]; then touch "$done_file"; log_status "DONE method=$method seed=$seed"; else log_status "FAIL method=$method seed=$seed exit=$code"; fi
}
main(){
  log_status "Q-Shaping quick test started at $(date)"
  for ((i=0;i<PARALLEL;i++)); do prepare_worker "$i"; done
  make_jobs
  log_status "job_count=$(wc -l < "$OUT_DIR/jobs_all.txt")"
  split -n l/$PARALLEL -d --additional-suffix=.txt "$OUT_DIR/jobs_all.txt" "$OUT_DIR/w_"
  local pids=()
  for ((i=0;i<PARALLEL;i++)); do
    f=$(printf "%s/w_%02d.txt" "$OUT_DIR" "$i")
    [ -f "$f" ] || continue
    ( while IFS='|' read -r phase env_key tl tmax method seed; do run_one "$i" "$env_key" "$tl" "$tmax" "$method" "$seed" || true; done < "$f" ) & pids+=($!)
  done
  wait "${pids[@]}"
  log_status "Q-Shaping quick test finished at $(date)"
}
main "$@"
