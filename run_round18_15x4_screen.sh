#!/usr/bin/env bash
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="$ROOT_DIR/results/round18_twohour_screen_${STAMP}"
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/STATUS.txt"
WORKER_BASE="/data/lab/AAAI_worker_round18_screen"
GPU="${GPU:-0}"
PARALLEL="${PARALLEL:-10}"
TEST_INTERVAL="${TEST_INTERVAL:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-50000}"
TEST_NEPISODE="${TEST_NEPISODE:-8}"
MAX_RECORDS="${MAX_RECORDS:-1200}"
mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"
log_status(){ printf '%s\n' "$*" | tee -a "$STATUS"; }
prepare_worker(){ local idx="$1"; local w="${WORKER_BASE}_${idx}"; rm -rf "$w"; mkdir -p "$w"; tar --exclude='./results' --exclude='./artifacts' --exclude='./.git' -C "$ROOT_DIR" -cf - . | tar -C "$w" -xf -; }
sanitize_env(){ printf '%s' "$1" | tr ':/' '__'; }
method_extra(){
  local method="$1" config="mappo_llm_fd" extra=""
  case "$method" in
    baseline) config="mappo"; extra="" ;;
    fa_pbrs_l002) config="mappo_llm_fd"; extra="llm_fd_enabled=True llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_intervention_mode=fa_pbrs llm_fd_pbrs_lambda=0.02 llm_fd_pbrs_alpha=0.01 llm_fd_pbrs_beta=0.05 llm_fd_pbrs_adaptive=True" ;;
    pbrs_fixed_l002) config="mappo_llm_fd"; extra="llm_fd_enabled=False llm_fd_apply_reward_shaping=True llm_fd_intervention_mode=fa_pbrs llm_fd_pbrs_lambda=0.02 llm_fd_pbrs_alpha=0.0 llm_fd_pbrs_beta=0.0 llm_fd_pbrs_adaptive=False" ;;
    pbrs_random_features_l002) config="mappo_llm_fd"; extra="llm_fd_enabled=False llm_fd_apply_reward_shaping=True llm_fd_intervention_mode=fa_pbrs llm_fd_pbrs_lambda=0.02 llm_fd_pbrs_alpha=0.0 llm_fd_pbrs_beta=0.0 llm_fd_pbrs_adaptive=False llm_fd_pbrs_random_features=True" ;;
    *) return 2 ;;
  esac
  printf '%s|%s\n' "$config" "$extra"
}
make_jobs(){
  : > "$OUT_DIR/jobs_all.txt"
  for seed in 1 9; do
    for method in baseline fa_pbrs_l002 pbrs_random_features_l002; do
      printf 'env300fix|lbforaging:Foraging-15x15-3p-4f-v3|50|300000|%s|%s\n' "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"
    done
  done
}
summarize_partial(){ python3 "$ROOT_DIR/epymarl/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" >/dev/null 2>&1 || true; }
run_one(){
  local worker_idx="$1" phase="$2" env_key="$3" time_limit="$4" tmax="$5" method="$6" seed="$7"
  local worker="${WORKER_BASE}_${worker_idx}" safe_env log done_file fail_file config_extra config extra start end elapsed code
  safe_env="$(sanitize_env "$env_key")"; log="$LOG_DIR/${phase}_${safe_env}_${method}_seed${seed}.log"; done_file="$log.done"; fail_file="$log.fail"; rm -f "$fail_file"
  config_extra="$(method_extra "$method")" || return 2; config="${config_extra%%|*}"; extra="${config_extra#*|}"
  start=$(date +%s)
  (cd "$worker/epymarl" && CUDA_VISIBLE_DEVICES="$GPU" python3 src/main.py --config="$config" --env-config=gymma with env_args.key="$env_key" env_args.time_limit="$time_limit" t_max="$tmax" use_cuda=True test_nepisode="$TEST_NEPISODE" test_interval="$TEST_INTERVAL" log_interval="$LOG_INTERVAL" runner_log_interval="$LOG_INTERVAL" learner_log_interval="$LOG_INTERVAL" seed="$seed" $extra) > "$log" 2>&1
  code=$?; end=$(date +%s); elapsed=$((end-start))
  if [[ "$code" -eq 0 ]]; then touch "$done_file"; printf 'DONE phase=%s env=%s method=%s seed=%s elapsed=%ss\n' "$phase" "$env_key" "$method" "$seed" "$elapsed" >> "$STATUS"; else printf '%s\n' "$code" > "$fail_file"; printf 'FAIL phase=%s env=%s method=%s seed=%s exit=%s elapsed=%ss\n' "$phase" "$env_key" "$method" "$seed" "$code" "$elapsed" >> "$STATUS"; fi
}
worker_loop(){
  local idx="$1" jobs_file="$2"
  while IFS='|' read -r phase env_key time_limit tmax method seed; do
    [ -z "${phase:-}" ] && continue
    run_one "$idx" "$phase" "$env_key" "$time_limit" "$tmax" "$method" "$seed" || true
  done < "$jobs_file"
}
package_outputs(){ summarize_partial; local artifact="$ROOT_DIR/artifacts/AAAI_round18_twohour_screen_${STAMP}.tar.gz"; tar -C "$ROOT_DIR" -czf "$artifact" "results/$(basename "$OUT_DIR")" "run_round18_twohour_screen.sh"; log_status "ARTIFACT $artifact"; }
main(){
  log_status "Round 18 two-hour screen started at $(date) PARALLEL=$PARALLEL"
  for ((i=0;i<PARALLEL;i++)); do prepare_worker "$i"; done
  make_jobs
  log_status "job_count=$(wc -l < "$OUT_DIR/jobs_all.txt") parallel=$PARALLEL"
  split -n l/$PARALLEL -d --additional-suffix=.txt "$OUT_DIR/jobs_all.txt" "$OUT_DIR/jobs_worker_"
  local pids=()
  for ((i=0;i<PARALLEL;i++)); do
    f=$(printf "%s/jobs_worker_%02d.txt" "$OUT_DIR" "$i")
    [ -f "$f" ] || continue
    worker_loop "$i" "$f" & pids+=($!)
  done
  wait "${pids[@]}"
  package_outputs
  log_status "Round 18 two-hour screen finished at $(date)"
}
main "$@"
