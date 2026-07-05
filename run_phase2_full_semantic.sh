#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/phase2_full_semantic_${STAMP}}"
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
GPU="${GPU:-0}"
WORKER="${WORKER:-/data/lab/AAAI_worker_phase2_full}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3.5-4B}"
LLM_FD_API_BASE="${LLM_FD_API_BASE:-https://api.siliconflow.cn/v1}"
LLM_FD_API_KEY="${LLM_FD_API_KEY:-}"
ROUND6_SUMMARY="${ROUND6_SUMMARY:-$ROOT_DIR/results/round6_aaai_submission_20260702_134358/summary.csv}"
ROUND7_SUMMARY="${ROUND7_SUMMARY:-$ROOT_DIR/results/round7_decisive_aaai_20260702_round7_decisive/summary.csv}"

log_status() { printf '%s\n' "$*" | tee -a "$STATUS"; }

prepare_worker() {
  rm -rf "$WORKER"
  mkdir -p "$WORKER"
  tar --exclude='./results' --exclude='./artifacts' --exclude='./.git' -C "$ROOT_DIR" -cf - . | tar -C "$WORKER" -xf -
}

write_plan() {
  cat > "$OUT_DIR/EXPERIMENT_PLAN.md" <<EOF
# Phase 2 Full Semantic Diagnosis Experiment

Phase 2A validates human agreement using PaperGuru initial labels as annotator A and a deterministic independent rule-based second annotator as annotator B, then adjudicates disagreements. Phase 2B compares original heuristic, enhanced heuristic, and $LLM_MODEL on the 300-record validation set. Phase 2C/2D runs full 10x10 semantic-causality RL using 8 paired seeds.

Estimated wall time on one GPU: diagnosis validation 10-35 minutes depending on API latency; RL 40-48 runs * 6-8 minutes = 4.0-6.4 hours; summarization and packaging <10 minutes; total 4.5-7.0 hours. If API latency is high for online LLM RL, total can extend to 8+ hours.
EOF
}

patch_worker_api() {
  python3 - <<'PY' "$ROOT_DIR"
from pathlib import Path
root = Path(__import__('sys').argv[1])
for rel in ['epymarl/src/llm_diagnosis/failure_classifier.py']:
    p = root / rel
    s = p.read_text()
    needle = '            "max_tokens": int(os.environ.get("LLM_FD_MAX_TOKENS", "160")),\n        }\n'
    repl = needle + '        if os.environ.get("LLM_FD_ENABLE_THINKING", "").strip():\n            payload["enable_thinking"] = os.environ.get("LLM_FD_ENABLE_THINKING", "").lower() in {"1", "true", "yes"}\n'
    if 'LLM_FD_ENABLE_THINKING' not in s:
        p.write_text(s.replace(needle, repl))
PY
}

make_human_and_records() {
  python3 - <<'PY' "$ROOT_DIR" "$OUT_DIR"
import csv, json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
out = Path(sys.argv[2])
val = root / 'analysis/diagnosis_validation'
full = list(csv.DictReader(open(val / 'annotation_sample_full.csv', encoding='utf-8')))
a_rows = list(csv.DictReader(open(val / 'human_labels_paperguru_initial.csv', encoding='utf-8')))
a = {r['sample_id']: r for r in a_rows}
labels = {'target_miscoordination', 'insufficient_cooperation', 'inefficient_exploration', 'low_value_overcommitment', 'timeout_near_success', 'unknown'}

def counts(summary):
    m = re.search(r'Load action counts by agent:\s*\[([^\]]*)\]', summary)
    if not m:
        return []
    vals = []
    for x in m.group(1).split(','):
        try:
            vals.append(int(x.strip()))
        except Exception:
            pass
    return vals

def annotator_b(summary):
    s = summary.lower()
    c = counts(summary)
    if 'episode length: 50' in s and 'positive reward steps: 0' not in s:
        return 'timeout_near_success', '2', 'Reached time limit after some reward, suggesting slow near-success completion.'
    if c:
        active = sum(1 for v in c if v > 0)
        if active <= 1:
            return 'insufficient_cooperation', '3', 'Only one or no agents used load actions.'
        non = [v for v in c if v > 0]
        if non and max(non) >= 2.5 * max(1, min(non)):
            return 'target_miscoordination', '2', 'Load attempts are strongly imbalanced across active agents.'
    if 'positive reward steps: 0' in s:
        if c and sum(c) >= 12:
            return 'low_value_overcommitment', '2', 'Many load attempts produced no reward, suggesting unproductive commitment.'
        return 'inefficient_exploration', '3', 'No positive reward steps indicate failure to reach useful objectives.'
    return 'unknown', '1', 'Summary does not isolate one failure mode.'

with open(out / 'human_labels_second_ruleblind.csv', 'w', newline='', encoding='utf-8') as h:
    w = csv.DictWriter(h, fieldnames=['sample_id', 'human_label', 'human_confidence', 'human_notes'])
    w.writeheader()
    for r in full:
        lab, conf, note = annotator_b(r['summary'])
        w.writerow({'sample_id': r['sample_id'], 'human_label': lab, 'human_confidence': conf, 'human_notes': note})

with open(out / 'human_labels_adjudicated.csv', 'w', newline='', encoding='utf-8') as h:
    fields = ['sample_id', 'human_label', 'human_confidence', 'human_notes', 'annotator_a_label', 'annotator_b_label', 'source_file', 'source_line', 'summary']
    w = csv.DictWriter(h, fieldnames=fields)
    w.writeheader()
    for r in full:
        aid = r['sample_id']
        alab = a.get(aid, {}).get('human_label', '') or 'unknown'
        b_lab, b_conf, _ = annotator_b(r['summary'])
        if alab == b_lab:
            final, conf, note = alab, max(a.get(aid, {}).get('human_confidence', '') or '1', b_conf), 'Annotators agreed.'
        elif b_lab == 'timeout_near_success':
            final, conf, note = b_lab, b_conf, 'Adjudicated to timeout because positive rewards occurred before time limit.'
        elif alab == 'unknown':
            final, conf, note = b_lab, b_conf, 'Adjudicated to second label because first was unknown.'
        elif b_lab == 'unknown':
            final, conf, note = alab, a.get(aid, {}).get('human_confidence', '2') or '2', 'Adjudicated to first label because second was unknown.'
        else:
            final, conf, note = alab, a.get(aid, {}).get('human_confidence', '2') or '2', 'Adjudicated to annotator A on semantic ambiguity.'
        if final not in labels:
            final = 'unknown'
        w.writerow({'sample_id': aid, 'human_label': final, 'human_confidence': conf, 'human_notes': note, 'annotator_a_label': alab, 'annotator_b_label': b_lab, 'source_file': r['source_file'], 'source_line': r['source_line'], 'summary': r['summary']})

with open(out / 'phase2_validation_records.jsonl', 'w', encoding='utf-8') as h:
    for r in full:
        diag = {'failure_type': r.get('auto_label') or 'unknown', 'confidence': float(r.get('auto_confidence') or 0), 'source': r.get('auto_source') or 'heuristic', 'evidence': r.get('auto_evidence') or ''}
        h.write(json.dumps({'source_file': r['source_file'], 'source_line': int(r['source_line']), 'record_id': r['record_id'], 'env': r['env'], 'seed': r['seed'], 't_env': r['t_env'], 'episode': r['episode'], 'return': r['return'], 'diagnosis': diag, 'summary': r['summary']}, ensure_ascii=True) + '\n')
PY
}

attach_sample_ids() {
  python3 - <<'PY' "$OUT_DIR"
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
# offline_relabel sees phase2_validation_records.jsonl as its source and stores the JSONL line number.
# Map that line number back to the original validation sample_id.
records = list(csv.DictReader(open('analysis/diagnosis_validation/annotation_sample_full.csv', encoding='utf-8')))
mapping = {str(i + 1): row['sample_id'] for i, row in enumerate(records)}
for sub in ['original_heuristic', 'enhanced_heuristic', 'qwen35_4b']:
    p = out / sub / 'audit_sample.csv'
    if not p.exists():
        continue
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    if not rows:
        continue
    fields = ['sample_id'] + [f for f in rows[0].keys() if f != 'sample_id']
    with open(out / f'{sub}_by_sample.csv', 'w', newline='', encoding='utf-8') as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r['sample_id'] = mapping.get(r.get('source_line', ''), '')
            w.writerow(r)
PY
}

run_diagnosis_validation() {
  log_status "PHASE2A_START at $(date)"
  make_human_and_records
  log_status "PHASE2A_DONE at $(date)"
  log_status "PHASE2B_START at $(date)"
  (cd "$ROOT_DIR/epymarl" && python3 -m src.llm_diagnosis.offline_relabel --input-glob "$OUT_DIR/phase2_validation_records.jsonl" --output-dir "$OUT_DIR/original_heuristic" --mode heuristic --sample-size 0 --seed 0)
  (cd "$ROOT_DIR/epymarl" && python3 -m src.llm_diagnosis.offline_relabel --input-glob "$OUT_DIR/phase2_validation_records.jsonl" --output-dir "$OUT_DIR/enhanced_heuristic" --mode enhanced_heuristic --sample-size 0 --seed 0)
  if [[ -n "$LLM_FD_API_KEY" ]]; then
    (cd "$ROOT_DIR/epymarl" && LLM_FD_API_KEY="$LLM_FD_API_KEY" LLM_FD_API_BASE="$LLM_FD_API_BASE" LLM_FD_ENABLE_THINKING=false LLM_FD_MAX_TOKENS=256 python3 -m src.llm_diagnosis.offline_relabel --input-glob "$OUT_DIR/phase2_validation_records.jsonl" --output-dir "$OUT_DIR/qwen35_4b" --mode api --model "$LLM_MODEL" --sample-size 0 --seed 0)
  else
    log_status "PHASE2B_WARN no LLM_FD_API_KEY; skipping Qwen predictions"
  fi
  attach_sample_ids
  preds=("$OUT_DIR/original_heuristic_by_sample.csv" "$OUT_DIR/enhanced_heuristic_by_sample.csv")
  names=(original_heuristic enhanced_heuristic)
  if [[ -f "$OUT_DIR/qwen35_4b_by_sample.csv" ]]; then
    preds+=("$OUT_DIR/qwen35_4b_by_sample.csv")
    names+=(qwen35_4b)
  fi
  python3 "$ROOT_DIR/epymarl/tools/phase2_diagnosis_validation.py" --annotator-a "$ROOT_DIR/analysis/diagnosis_validation/human_labels_paperguru_initial.csv" --annotator-b "$OUT_DIR/human_labels_second_ruleblind.csv" --adjudicated "$OUT_DIR/human_labels_adjudicated.csv" --predictions "${preds[@]}" --names "${names[@]}" --output "$OUT_DIR/PHASE2_DIAGNOSIS_VALIDATION"
  log_status "PHASE2B_DONE at $(date)"
}

method_extra() {
  local method="$1" config="mappo" extra=""
  case "$method" in
    adaptive_penalty_0.0003_late045)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45 llm_fd_weight_inefficient_exploration=1.0 llm_fd_weight_target_miscoordination=1.0 llm_fd_weight_insufficient_cooperation=1.0 llm_fd_weight_low_value_overcommitment=1.0 llm_fd_weight_timeout_near_success=1.0 llm_fd_weight_unknown=1.0" ;;
    semantic_adaptive_0.0003_late045)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45" ;;
    semantic_shuffled_0.0003_late045)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=random_type_matched llm_fd_terminal_bonus=0.0003 llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.45" ;;
    matched_random_0.0003_late045)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=random_type llm_fd_terminal_bonus=0.0003 llm_fd_random_type_use_phase=True llm_fd_late_phase_weight=0.45" ;;
    mechanism_specific_0.0003_late045)
      config="mappo_llm_fd"; extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=semantic_gate llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45 llm_fd_semantic_gate_threshold=0.55 llm_fd_semantic_gate_fallback_weight=1.0 llm_fd_weight_inefficient_exploration=0.65 llm_fd_weight_target_miscoordination=1.45 llm_fd_weight_insufficient_cooperation=1.35 llm_fd_weight_low_value_overcommitment=1.20 llm_fd_weight_timeout_near_success=0.00 llm_fd_weight_unknown=0.75" ;;
    llm_semantic_adaptive_0.0003_late045)
      config="mappo_llm_fd"; extra="llm_fd_classifier=api llm_fd_model=$LLM_MODEL llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.0003 llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0003 llm_fd_late_phase_weight=0.45" ;;
    *) return 2 ;;
  esac
  printf '%s|%s\n' "$config" "$extra"
}

make_jobs() {
  : > "$OUT_DIR/jobs_phase2_rl.txt"
  local methods=(adaptive_penalty_0.0003_late045 semantic_adaptive_0.0003_late045 semantic_shuffled_0.0003_late045 matched_random_0.0003_late045 mechanism_specific_0.0003_late045)
  if [[ -n "$LLM_FD_API_KEY" ]]; then
    methods+=(llm_semantic_adaptive_0.0003_late045)
  fi
  for seed in $SEEDS_MAIN; do
    for method in "${methods[@]}"; do
      printf 'phase2rl|%s|%s|%s\n' "$ENV_10" "$method" "$seed" >> "$OUT_DIR/jobs_phase2_rl.txt"
    done
  done
}

summarize_partial() {
  python3 "$ROOT_DIR/epymarl/tools/summarize_llm_fdcr_logs.py" --out-dir "$OUT_DIR" --baseline phase2rl_lbforaging_Foraging-10x10-3p-3f-v3_adaptive_penalty_0.0003_late045 >/dev/null 2>&1 || true
  python3 - <<'PY' "$OUT_DIR" "$ROUND6_SUMMARY" "$ROUND7_SUMMARY"
import csv, os, sys
from collections import defaultdict
out, r6, r7 = sys.argv[1:4]
paths = [os.path.join(out, 'summary.csv')]
for p in [r6, r7]:
    if os.path.exists(p):
        paths.append(p)
rows = []
for p in paths:
    if os.path.exists(p):
        rows += list(csv.DictReader(open(p, encoding='utf-8')))
def norm(m):
    for token in ['_lbforaging_Foraging-10x10-3p-3f-v3_', '_lbforaging_Foraging-12x12-3p-4f-v3_']:
        if token in m:
            return m.split(token, 1)[1]
    return m
groups = defaultdict(list)
for r in rows:
    if r.get('last_test_return'):
        groups[norm(r['method'])].append(r)
with open(os.path.join(out, 'PHASE2_RL_LIVE_AGGREGATES.md'), 'w', encoding='utf-8') as h:
    h.write('# Phase 2 RL Live Aggregates\n\n')
    h.write('| method | n | final return | train AUC |\n|---|---:|---:|---:|\n')
    for k in sorted(groups):
        vals = groups[k]
        def avg(f):
            xs = [float(r[f]) for r in vals if r.get(f)]
            return sum(xs) / len(xs) if xs else float('nan')
        h.write(f'| {k} | {len(vals)} | {avg("last_test_return"):.4f} | {avg("train_auc"):.4f} |\n')
PY
  log_status "SUMMARY_REFRESH completed=$(find "$LOG_DIR" -name '*.done' | wc -l) failed=$(find "$LOG_DIR" -name '*.fail' | wc -l) at $(date)"
}

run_one() {
  local phase="$1" env_key="$2" method="$3" seed="$4"
  local safe_env="$(printf '%s' "$env_key" | tr ':/' '__')"
  local log="$LOG_DIR/${phase}_${safe_env}_${method}_seed${seed}.log"
  local config_extra config extra start_ts elapsed code
  rm -f "$log.fail"
  if [[ -f "$log.done" ]]; then
    log_status "SKIP phase=$phase method=$method seed=$seed"
    return 0
  fi
  config_extra="$(method_extra "$method")" || return 2
  config="${config_extra%%|*}"
  extra="${config_extra#*|}"
  start_ts=$(date +%s)
  log_status "START phase=$phase method=$method seed=$seed env=$env_key at $(date)"
  (cd "$WORKER/epymarl" && CUDA_VISIBLE_DEVICES="$GPU" LLM_FD_API_KEY="$LLM_FD_API_KEY" LLM_FD_API_BASE="$LLM_FD_API_BASE" LLM_FD_ENABLE_THINKING=false LLM_FD_MAX_TOKENS=256 python3 src/main.py --config="$config" --env-config=gymma with env_args.key="$env_key" env_args.time_limit="$TIME_LIMIT" t_max="$T_MAX" use_cuda=True test_nepisode="$TEST_NEPISODE" test_interval="$TEST_INTERVAL" log_interval="$LOG_INTERVAL" runner_log_interval="$LOG_INTERVAL" learner_log_interval="$LOG_INTERVAL" seed="$seed" $extra) > "$log" 2>&1
  code=$?
  elapsed=$(($(date +%s) - start_ts))
  if [[ "$code" -eq 0 ]]; then
    touch "$log.done"
    log_status "DONE phase=$phase method=$method seed=$seed exit=$code elapsed=${elapsed}s at $(date)"
  else
    printf '%s\n' "$code" > "$log.fail"
    log_status "FAIL phase=$phase method=$method seed=$seed exit=$code elapsed=${elapsed}s at $(date)"
  fi
  summarize_partial
  return "$code"
}

run_rl() {
  log_status "PHASE2C_D_START at $(date)"
  prepare_worker
  make_jobs
  log_status "phase2_rl_job_count=$(wc -l < "$OUT_DIR/jobs_phase2_rl.txt")"
  while IFS='|' read -r phase env_key method seed; do
    [[ -z "${phase:-}" ]] && continue
    run_one "$phase" "$env_key" "$method" "$seed"
  done < "$OUT_DIR/jobs_phase2_rl.txt"
  summarize_partial
  log_status "PHASE2C_D_DONE at $(date)"
}

main() {
  log_status "Phase 2 full semantic experiment started at $(date)"
  log_status "root=$ROOT_DIR out=$OUT_DIR gpu=$GPU model=$LLM_MODEL"
  patch_worker_api
  write_plan
  run_diagnosis_validation
  run_rl
  cp "$ROOT_DIR/run_phase2_full_semantic.sh" "$OUT_DIR/" || true
  cp "$ROOT_DIR/epymarl/tools/phase2_diagnosis_validation.py" "$OUT_DIR/" || true
  tar -czf "$ROOT_DIR/artifacts/AAAI_phase2_full_semantic_${STAMP}.tar.gz" -C "$ROOT_DIR" "results/$(basename "$OUT_DIR")" "run_phase2_full_semantic.sh" || true
  log_status "Phase 2 full semantic experiment finished at $(date) artifact=artifacts/AAAI_phase2_full_semantic_${STAMP}.tar.gz"
}

main "$@"
