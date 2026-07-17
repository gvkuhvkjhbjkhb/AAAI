#!/usr/bin/env bash
set -u

ROOT_DIR="$(pwd)"
OUT_DIR="$ROOT_DIR/results/round4_penalty_refinement_10x10"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

ENV_KEY="lbforaging:Foraging-10x10-3p-3f-v3"
TIME_LIMIT=50
T_MAX=500000
TEST_INTERVAL=100000
LOG_INTERVAL=25000
TEST_NEPISODE=20
MAX_RECORDS=2000
SEEDS="4 5"
PENALTIES="0.0002 0.0003 0.0005 0.0007 0.001"

printf 'Round4 penalty refinement started at %s\n' "$(date)" > "$OUT_DIR/STATUS.txt"
printf 'env=%s t_max=%s seeds=%s penalties=%s\n' "$ENV_KEY" "$T_MAX" "$SEEDS" "$PENALTIES" >> "$OUT_DIR/STATUS.txt"

run_one() {
  local method="$1"
  local seed="$2"
  local config="mappo"
  local extra=""
  if [[ "$method" != "baseline" ]]; then
    config="mappo_llm_fd"
    local penalty="${method#penalty_}"
    extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$penalty llm_fd_intervention_mode=uniform"
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
    runner_log_interval="$LOG_INTERVAL" \
    learner_log_interval="$LOG_INTERVAL" \
    seed="$seed" \
    $extra > "$LOG_DIR/${method}_seed${seed}.log" 2>&1
  local code=$?
  printf '== %s seed=%s exit=%s end %s ==\n' "$method" "$seed" "$code" "$(date)" | tee -a "$OUT_DIR/STATUS.txt"
}

for seed in $SEEDS; do
  run_one baseline "$seed"
  for penalty in $PENALTIES; do
    run_one "penalty_${penalty}" "$seed"
  done
done

python3 - <<'PY'
import csv
import glob
import math
import os
import re
from collections import defaultdict

out_dir = "results/round4_penalty_refinement_10x10"
rows = []
for path in sorted(glob.glob(os.path.join(out_dir, "logs", "*.log"))):
    name = os.path.basename(path).replace(".log", "")
    method, seed = name.rsplit("_seed", 1)
    text = open(path, encoding="utf-8", errors="ignore").read()
    returns = [float(x) for x in re.findall(r"return_mean:\s+([-0-9.]+)", text)]
    tests = [float(x) for x in re.findall(r"test_return_mean:\s+([-0-9.]+)", text)]
    records = [float(x) for x in re.findall(r"llm_fd_records:\s+([-0-9.]+)", text)]
    rows.append({
        "method": method,
        "seed": seed,
        "last_train_return": returns[-1] if returns else "",
        "best_train_return": max(returns) if returns else "",
        "train_auc": sum(returns) / len(returns) if returns else "",
        "stability_gap": (max(returns) - returns[-1]) if returns else "",
        "last_test_return": tests[-1] if tests else "",
        "best_test_return": max(tests) if tests else "",
        "last_llm_fd_records": records[-1] if records else "",
    })

fieldnames = ["method", "seed", "last_train_return", "best_train_return", "train_auc", "stability_gap", "last_test_return", "best_test_return", "last_llm_fd_records"]
with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def mean(values):
    values = [float(v) for v in values if v != ""]
    return sum(values) / len(values) if values else ""

def std(values):
    values = [float(v) for v in values if v != ""]
    if len(values) < 2:
        return ""
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

groups = defaultdict(list)
for row in rows:
    groups[row["method"]].append(row)

with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as handle:
    handle.write("Round4 penalty refinement 10x10 summary\n\n")
    for row in rows:
        handle.write(str(row) + "\n")
    handle.write("\nGrouped means and standard deviations\n")
    for method, items in sorted(groups.items()):
        payload = {"method": method, "n": len(items)}
        for metric in ["last_train_return", "best_train_return", "train_auc", "stability_gap", "last_test_return", "best_test_return"]:
            vals = [r[metric] for r in items]
            payload[f"mean_{metric}"] = mean(vals)
            payload[f"std_{metric}"] = std(vals)
        handle.write(str(payload) + "\n")
PY

printf 'Round4 penalty refinement finished at %s\n' "$(date)" >> "$OUT_DIR/STATUS.txt"
