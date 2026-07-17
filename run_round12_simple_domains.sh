#!/usr/bin/env bash
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/round12_simple_domains_${STAMP}}"
LOG_DIR="$OUT_DIR/logs"
STATUS="$OUT_DIR/STATUS.txt"
WORKER="${WORKER:-/data/lab/AAAI_worker_round12_simple_domains}"
GPU="${GPU:-0}"
mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"

RUN_SMALLCOOP="${RUN_SMALLCOOP:-1}"
RUN_MAINCOOP="${RUN_MAINCOOP:-1}"
RUN_SCALE15="${RUN_SCALE15:-0}"
SMALLCOOP_SEEDS="${SMALLCOOP_SEEDS:-1 2 3 4 5 6 7 8}"
MAINCOOP_SEEDS="${MAINCOOP_SEEDS:-1 2 3 4}"
SCALE15_SEEDS="${SCALE15_SEEDS:-1 2 3 4}"
SMALLCOOP_TMAX="${SMALLCOOP_TMAX:-300000}"
MAINCOOP_TMAX="${MAINCOOP_TMAX:-300000}"
SCALE15_TMAX="${SCALE15_TMAX:-300000}"
METHODS="${METHODS:-baseline uniform_budget_matched_0.0003_late045 adaptive_0.0003_late045 random_type_budget_matched_0.0003_late045}"
TEST_INTERVAL="${TEST_INTERVAL:-100000}"
LOG_INTERVAL="${LOG_INTERVAL:-25000}"
TEST_NEPISODE="${TEST_NEPISODE:-20}"
MAX_RECORDS="${MAX_RECORDS:-3000}"

log_status(){ printf '%s\n' "$*" | tee -a "$STATUS"; }
prepare_worker(){ rm -rf "$WORKER"; mkdir -p "$WORKER"; tar --exclude='./results' --exclude='./artifacts' --exclude='./.git' -C "$ROOT_DIR" -cf - . | tar -C "$WORKER" -xf -; }
method_extra(){
  local method="$1" config="mappo" extra=""
  case "$method" in
    baseline) config="mappo"; extra="" ;;
    uniform_budget_matched_0.0003_late045) config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=phase_uniform llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45" ;;
    adaptive_0.0003_late045) config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45" ;;
    random_type_budget_matched_0.0003_late045) config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=random_type llm_fd_terminal_bonus=0.0003 llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.45" ;;
    *) return 2 ;;
  esac
  printf '%s|%s\n' "$config" "$extra"
}
write_plan(){ cat > "$OUT_DIR/ROUND12_EXPERIMENT_PLAN.md" <<EOF
# Round 12 Simple Cooperative Domains

Goal: find simple sparse cooperative domains likely to yield positive evidence without relying on VMAS/RWARE/Qwen.

Domains:
- smallcoop: lbforaging:Foraging-8x8-2p-2f-coop-v3, seeds $SMALLCOOP_SEEDS, tmax $SMALLCOOP_TMAX.
- maincoop: lbforaging:Foraging-10x10-3p-3f-coop-v3, seeds $MAINCOOP_SEEDS, tmax $MAINCOOP_TMAX.
- scale15 optional: lbforaging:Foraging-15x15-3p-4f-v3, seeds $SCALE15_SEEDS, tmax $SCALE15_TMAX.

Methods: $METHODS
EOF
}
make_jobs(){
  : > "$OUT_DIR/jobs_all.txt"
  if [[ "$RUN_SMALLCOOP" == "1" ]]; then for seed in $SMALLCOOP_SEEDS; do for method in $METHODS; do printf 'smallcoop|lbforaging:Foraging-8x8-2p-2f-coop-v3|50|%s|%s|%s\n' "$SMALLCOOP_TMAX" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"; done; done; fi
  if [[ "$RUN_MAINCOOP" == "1" ]]; then for seed in $MAINCOOP_SEEDS; do for method in $METHODS; do printf 'maincoop|lbforaging:Foraging-10x10-3p-3f-coop-v3|50|%s|%s|%s\n' "$MAINCOOP_TMAX" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"; done; done; fi
  if [[ "$RUN_SCALE15" == "1" ]]; then for seed in $SCALE15_SEEDS; do for method in $METHODS; do printf 'scale15|lbforaging:Foraging-15x15-3p-4f-v3|50|%s|%s|%s\n' "$SCALE15_TMAX" "$method" "$seed" >> "$OUT_DIR/jobs_all.txt"; done; done; fi
  if [[ "${SHUFFLE_JOBS:-1}" == "1" ]]; then python3 - "$OUT_DIR/jobs_all.txt" "${JOB_SHUFFLE_SEED:-12}" <<'PYSHUFFLE'
import random, sys
p, seed=sys.argv[1], int(sys.argv[2])
rows=open(p, encoding='utf-8').readlines(); random.Random(seed).shuffle(rows); open(p,'w',encoding='utf-8').writelines(rows)
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
summarize_partial(){ python3 "$ROOT_DIR/epymarl/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" --baseline smallcoop_lbforaging_Foraging-8x8-2p-2f-coop-v3_baseline >/dev/null 2>&1 || true; python3 "$ROOT_DIR/epymarl/tools/build_round12_simple_domain_report.py" --out-dir "$OUT_DIR" >/dev/null 2>&1 || true; log_status "SUMMARY_REFRESH completed=$(find "$LOG_DIR" -name '*.done' | wc -l) failed=$(find "$LOG_DIR" -name '*.fail' | wc -l) at $(date)"; }
package_outputs(){ summarize_partial; local artifact="$ROOT_DIR/artifacts/AAAI_round12_simple_domains_${STAMP}.tar.gz"; tar -C "$ROOT_DIR" -czf "$artifact" "results/$(basename "$OUT_DIR")" "run_round12_simple_domains.sh" "epymarl/tools/build_round12_simple_domain_report.py"; log_status "ARTIFACT $artifact"; }
main(){ log_status "Round 12 simple domains started at $(date)"; log_status "root=$ROOT_DIR out=$OUT_DIR gpu=$GPU worker=$WORKER"; write_plan; prepare_worker; make_jobs; log_status "job_count=$(wc -l < "$OUT_DIR/jobs_all.txt")"; while IFS='|' read -r phase env_key time_limit tmax method seed; do run_one "$phase" "$env_key" "$time_limit" "$tmax" "$method" "$seed" || true; done < "$OUT_DIR/jobs_all.txt"; package_outputs; log_status "Round 12 simple domains finished at $(date)"; }
main "$@"
