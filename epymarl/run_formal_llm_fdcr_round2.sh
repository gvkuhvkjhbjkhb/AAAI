#!/usr/bin/env bash
set -u

ROOT_DIR="$(pwd)"
OUT_DIR="$ROOT_DIR/results/formal_llm_fdcr_round2"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

TEST_INTERVAL=100000
LOG_INTERVAL=25000
RUNNER_LOG_INTERVAL=25000
LEARNER_LOG_INTERVAL=25000
TEST_NEPISODE=20

printf 'Formal LLM-FDCR round2 started at %s\n' "$(date)" > "$OUT_DIR/STATUS.txt"
printf 'experiments: 8x8 500k and 10x10 300k, baseline vs fd_shaping, seeds 1 2 3\n' >> "$OUT_DIR/STATUS.txt"

run_one() {
  local env_label="$1"
  local env_key="$2"
  local time_limit="$3"
  local t_max="$4"
  local method="$5"
  local seed="$6"
  local config="mappo"
  local extra=""

  if [[ "$method" == "fd_shaping" ]]; then
    config="mappo_llm_fd"
    extra="llm_fd_classifier=heuristic llm_fd_max_records=2000 llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=0.001"
  fi

  printf '\n== %s %s seed=%s start %s ==\n' "$env_label" "$method" "$seed" "$(date)" | tee -a "$OUT_DIR/STATUS.txt"
  python3 src/main.py \
    --config="$config" \
    --env-config=gymma \
    with \
    env_args.key="$env_key" \
    env_args.time_limit="$time_limit" \
    t_max="$t_max" \
    use_cuda=True \
    test_nepisode="$TEST_NEPISODE" \
    test_interval="$TEST_INTERVAL" \
    log_interval="$LOG_INTERVAL" \
    runner_log_interval="$RUNNER_LOG_INTERVAL" \
    learner_log_interval="$LEARNER_LOG_INTERVAL" \
    seed="$seed" \
    $extra > "$LOG_DIR/${env_label}_${method}_seed${seed}.log" 2>&1
  local code=$?
  printf '== %s %s seed=%s exit=%s end %s ==\n' "$env_label" "$method" "$seed" "$code" "$(date)" | tee -a "$OUT_DIR/STATUS.txt"
}

for seed in 1 2 3; do
  run_one "lbf8x8" "lbforaging:Foraging-8x8-2p-2f-coop-v3" 50 500000 baseline "$seed"
  run_one "lbf8x8" "lbforaging:Foraging-8x8-2p-2f-coop-v3" 50 500000 fd_shaping "$seed"
done

for seed in 1 2 3; do
  run_one "lbf10x10" "lbforaging:Foraging-10x10-3p-3f-v3" 50 300000 baseline "$seed"
  run_one "lbf10x10" "lbforaging:Foraging-10x10-3p-3f-v3" 50 300000 fd_shaping "$seed"
done

python3 - <<'PY'
import csv
import glob
import json
import os
import re

out_dir = "results/formal_llm_fdcr_round2"
log_dir = os.path.join(out_dir, "logs")
rows = []

for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
    name = os.path.basename(path).replace(".log", "")
    parts = name.split("_")
    env_label = parts[0]
    if parts[1] == "fd" and parts[2] == "shaping":
        method = "fd_shaping"
        seed_text = parts[3].replace("seed", "")
    else:
        method = parts[1]
        seed_text = parts[2].replace("seed", "")
    text = open(path, encoding="utf-8", errors="ignore").read()
    returns = [float(x) for x in re.findall(r"return_mean:\s+([-0-9.]+)", text)]
    tests = [float(x) for x in re.findall(r"test_return_mean:\s+([-0-9.]+)", text)]
    records = [float(x) for x in re.findall(r"llm_fd_records:\s+([-0-9.]+)", text)]
    rows.append({
        "env": env_label,
        "method": method,
        "seed": seed_text,
        "last_train_return": returns[-1] if returns else "",
        "best_train_return": max(returns) if returns else "",
        "last_test_return": tests[-1] if tests else "",
        "best_test_return": max(tests) if tests else "",
        "last_llm_fd_records": records[-1] if records else "",
    })

summary_csv = os.path.join(out_dir, "summary.csv")
fieldnames = ["env", "method", "seed", "last_train_return", "best_train_return", "last_test_return", "best_test_return", "last_llm_fd_records"]
with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

groups = {}
for row in rows:
    key = (row["env"], row["method"])
    groups.setdefault(key, []).append(row)

def mean(values):
    values = [float(v) for v in values if v != ""]
    return sum(values) / len(values) if values else ""

with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as handle:
    handle.write("Formal LLM-FDCR round2 summary\n\n")
    for row in rows:
        handle.write(str(row) + "\n")
    handle.write("\nGrouped means\n")
    for key, items in sorted(groups.items()):
        handle.write(str({
            "env": key[0],
            "method": key[1],
            "mean_last_train_return": mean([r["last_train_return"] for r in items]),
            "mean_best_train_return": mean([r["best_train_return"] for r in items]),
            "mean_last_test_return": mean([r["last_test_return"] for r in items]),
            "mean_best_test_return": mean([r["best_test_return"] for r in items]),
        }) + "\n")
PY

printf 'Formal LLM-FDCR round2 finished at %s\n' "$(date)" >> "$OUT_DIR/STATUS.txt"
