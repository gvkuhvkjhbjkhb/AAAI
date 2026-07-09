#!/usr/bin/env bash
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/dg_pbs_pilot_${STAMP}}"
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/STATUS.txt"
WORKER_BASE="${WORKER:-/data/lab/AAAI_worker_dg_pbs}"
GPU="${GPU:-0}"
PARALLEL="${PARALLEL:-4}"
mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"
ENV_KEY="${ENV_KEY:-lbforaging:Foraging-10x10-3p-3f-v3}"
TIME_LIMIT="${TIME_LIMIT:-50}"
T_MAX="${T_MAX:-500000}"
SEEDS="${SEEDS:-1 2 3 4}"
METHODS="${METHODS:-baseline fa_pbrs_l002 dg_pbs_l002}"
TEST_INTERVAL="${TEST_INTERVAL:-50000}"
LOG_INTERVAL="${LOG_INTERVAL:-25000}"
TEST_NEPISODE="${TEST_NEPISODE:-10}"
MAX_RECORDS="${MAX_RECORDS:-3000}"
log_status(){ printf '%s\n' "$*" | tee -a "$STATUS"; }
prepare_worker(){ local idx="$1"; local w="${WORKER_BASE}_${idx}"; rm -rf "$w"; mkdir -p "$w"; tar --exclude='./results' --exclude='./artifacts' --exclude='./.git' -C "$ROOT_DIR" -cf - . | tar -C "$w" -xf -; }
method_extra(){
  local method="$1" config="mappo_llm_fd" extra=""
  case "$method" in
    baseline) config="mappo"; extra="" ;;
    fa_pbrs_l002) extra="llm_fd_enabled=True llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_intervention_mode=fa_pbrs llm_fd_pbrs_lambda=0.02 llm_fd_pbrs_alpha=0.01 llm_fd_pbrs_beta=0.05 llm_fd_pbrs_adaptive=True" ;;
    dg_pbs_l002) extra="llm_fd_enabled=True llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_intervention_mode=dg_pbs llm_fd_pbrs_lambda=0.02 llm_fd_pbrs_alpha=0.01 llm_fd_pbrs_beta=0.05 llm_fd_pbrs_adaptive=True llm_fd_dg_window=50 llm_fd_dg_warmup=20 llm_fd_dg_smoothing=0.8 llm_fd_dg_temperature=5.0" ;;
    *) return 2 ;;
  esac
  printf '%s|%s\n' "$config" "$extra"
}
make_jobs(){
  : > "$OUT_DIR/jobs_all.txt"
  for seed in $SEEDS; do for method in $METHODS; do printf 'dgpbs|%s|%s|%s|%s|%s\n' "$ENV_KEY" "$TIME_LIMIT" "$T_MAX" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"; done; done
  if [[ "${SHUFFLE_JOBS:-1}" == "1" ]]; then python3 - "$OUT_DIR/jobs_all.txt" "${JOB_SHUFFLE_SEED:-21}" <<'PYSHUFFLE'
import random, sys
p, seed = sys.argv[1], int(sys.argv[2])
rows = open(p, encoding='utf-8').readlines(); random.Random(seed).shuffle(rows); open(p,'w',encoding='utf-8').writelines(rows)
PYSHUFFLE
  fi
}
sanitize_env(){ printf '%s' "$1" | tr ':/' '__'; }
summarize_partial(){ python3 "$ROOT_DIR/epymarl/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" >/dev/null 2>&1 || true; }
run_one(){
  local worker_idx="$1" phase="$2" env_key="$3" time_limit="$4" tmax="$5" method="$6" seed="$7"
  local worker="${WORKER_BASE}_${worker_idx}"
  local safe_env log done_file fail_file config_extra config extra start end elapsed code
  safe_env="$(sanitize_env "$env_key")"; log="$LOG_DIR/${phase}_${safe_env}_${method}_seed${seed}.log"; done_file="$log.done"; fail_file="$log.fail"; rm -f "$fail_file"
  if [[ -f "$done_file" ]]; then return 0; fi
  config_extra="$(method_extra "$method")" || return 2; config="${config_extra%%|*}"; extra="${config_extra#*|}"
  start=$(date +%s)
  (cd "$worker/epymarl" && CUDA_VISIBLE_DEVICES="$GPU" python3 src/main.py --config="$config" --env-config=gymma with env_args.key="$env_key" env_args.time_limit="$time_limit" t_max="$tmax" use_cuda=True test_nepisode="$TEST_NEPISODE" test_interval="$TEST_INTERVAL" log_interval="$LOG_INTERVAL" runner_log_interval="$LOG_INTERVAL" learner_log_interval="$LOG_INTERVAL" seed="$seed" $extra) > "$log" 2>&1
  code=$?; end=$(date +%s); elapsed=$((end-start))
  if [[ "$code" -eq 0 ]]; then touch "$done_file"; printf 'DONE worker=%s method=%s seed=%s elapsed=%ss\n' "$worker_idx" "$method" "$seed" "$elapsed" >> "$STATUS"; else printf '%s\n' "$code" > "$fail_file"; printf 'FAIL worker=%s method=%s seed=%s exit=%s\n' "$worker_idx" "$method" "$seed" "$code" >> "$STATUS"; fi
  return "$code"
}
worker_loop(){
  local idx="$1" jobs_file="$2"
  while true; do
    local line
    line=$(flock "$jobs_file.lock" python3 -c "
import sys
lines=open('$jobs_file',encoding='utf-8').readlines()
if not lines: sys.exit(1)
line=lines.pop(0)
open('$jobs_file','w',encoding='utf-8').writelines(lines)
sys.stdout.write(line)
" 2>/dev/null) || break
    IFS='|' read -r phase env_key time_limit tmax method seed <<< "$line"
    run_one "$idx" "$phase" "$env_key" "$time_limit" "$tmax" "$method" "$seed" || true
  done
}
package_outputs(){ summarize_partial; local artifact="$ROOT_DIR/artifacts/AAAI_dg_pbs_pilot_${STAMP}.tar.gz"; tar -C "$ROOT_DIR" -czf "$artifact" "results/$(basename "$OUT_DIR")" "run_dg_pbs_pilot.sh"; log_status "ARTIFACT $artifact"; }
main(){
  log_status "DG-PBS pilot started at $(date) GPU=$GPU PARALLEL=$PARALLEL methods=$METHODS seeds=$SEEDS t_max=$T_MAX"
  for ((i=0;i<PARALLEL;i++)); do prepare_worker "$i"; done
  make_jobs
  local total=$(wc -l < "$OUT_DIR/jobs_all.txt")
  log_status "job_count=$total parallel=$PARALLEL"
  cp "$OUT_DIR/jobs_all.txt" "$OUT_DIR/jobs_queue.txt"
  local pids=()
  for ((i=0;i<PARALLEL;i++)); do worker_loop "$i" "$OUT_DIR/jobs_queue.txt" & pids+=($!); done
  wait "${pids[@]}"
  package_outputs
  log_status "DG-PBS pilot finished at $(date)"
}
main "$@"
