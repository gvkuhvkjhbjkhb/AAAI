#!/usr/bin/env bash
set -u

ROOT_DIR="$(pwd)"
OUT_DIR="$ROOT_DIR/results/formal_llm_fdcr_8x8"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

ENV_KEY="lbforaging:Foraging-8x8-2p-2f-coop-v3"
TIME_LIMIT=50
T_MAX=200000
TEST_INTERVAL=50000
LOG_INTERVAL=10000
RUNNER_LOG_INTERVAL=10000
LEARNER_LOG_INTERVAL=10000
TEST_NEPISODE=20

printf 'Formal LLM-FDCR experiment started at %s\n' "$(date)" > "$OUT_DIR/STATUS.txt"
printf 'env=%s time_limit=%s t_max=%s\n' "$ENV_KEY" "$TIME_LIMIT" "$T_MAX" >> "$OUT_DIR/STATUS.txt"

run_one() {
  local method="$1"
  local seed="$2"
  local config="mappo"
  local extra=""

  if [[ "$method" == "fd" ]]; then
    config="mappo_llm_fd"
    extra="llm_fd_classifier=heuristic llm_fd_max_records=1000"
  elif [[ "$method" == "fd_shaping" ]]; then
    config="mappo_llm_fd"
    extra="llm_fd_classifier=heuristic llm_fd_max_records=1000 llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.001"
  fi

  printf '\n== %s seed=%s start %s ==\n' "$method" "$seed" "$(date)" | tee -a "$OUT_DIR/STATUS.txt"
  python3 src/main.py \
    --config="$config" \
    --env-config=gymma \
    with \
    env_args.key="$ENV_KEY" \
    env_args.time_limit="$TIME_LIMIT" \
    t_max="$T_MAX" \
    use_cuda=True \
    test_nepisode="$TEST_NEPISODE" \
    test_interval="$TEST_INTERVAL" \
    log_interval="$LOG_INTERVAL" \
    runner_log_interval="$RUNNER_LOG_INTERVAL" \
    learner_log_interval="$LEARNER_LOG_INTERVAL" \
    seed="$seed" \
    $extra > "$LOG_DIR/${method}_seed${seed}.log" 2>&1
  local code=$?
  printf '== %s seed=%s exit=%s end %s ==\n' "$method" "$seed" "$code" "$(date)" | tee -a "$OUT_DIR/STATUS.txt"
}

for seed in 1 2 3; do
  run_one baseline "$seed"
  run_one fd "$seed"
  run_one fd_shaping "$seed"
done

python3 - <<'PY'
import csv
import glob
import json
import os
import re

out_dir = "results/formal_llm_fdcr_8x8"
log_dir = os.path.join(out_dir, "logs")
rows = []

for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
    name = os.path.basename(path).replace(".log", "")
    method, seed_text = name.rsplit("_seed", 1)
    text = open(path, encoding="utf-8", errors="ignore").read()
    returns = [float(x) for x in re.findall(r"return_mean:\s+([-0-9.]+)", text)]
    tests = [float(x) for x in re.findall(r"test_return_mean:\s+([-0-9.]+)", text)]
    records = [float(x) for x in re.findall(r"llm_fd_records:\s+([-0-9.]+)", text)]
    rows.append({
        "method": method,
        "seed": seed_text,
        "last_train_return": returns[-1] if returns else "",
        "best_train_return": max(returns) if returns else "",
        "last_test_return": tests[-1] if tests else "",
        "best_test_return": max(tests) if tests else "",
        "last_llm_fd_records": records[-1] if records else "",
    })

summary_csv = os.path.join(out_dir, "summary.csv")
with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["method"])
    writer.writeheader()
    writer.writerows(rows)

record_summaries = []
for path in sorted(glob.glob("results/llm_fd/*/failure_records.jsonl")):
    counts = {}
    lines = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            lines += 1
            payload = json.loads(line)
            failure_type = payload["diagnosis"]["failure_type"]
            counts[failure_type] = counts.get(failure_type, 0) + 1
    record_summaries.append((path, lines, counts))

with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as handle:
    handle.write("Formal LLM-FDCR summary\n\n")
    for row in rows:
        handle.write(str(row) + "\n")
    handle.write("\nFailure record files\n")
    for item in record_summaries:
        handle.write(str(item) + "\n")
PY

printf 'Formal LLM-FDCR experiment finished at %s\n' "$(date)" >> "$OUT_DIR/STATUS.txt"
