#!/usr/bin/env bash
set -u

ROOT_DIR="$(pwd)"
OUT_DIR="$ROOT_DIR/results/round4_type_specific_10x10"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

ENV_KEY="lbforaging:Foraging-10x10-3p-3f-v3"
TIME_LIMIT=50
T_MAX=500000
TEST_INTERVAL=100000
LOG_INTERVAL=25000
TEST_NEPISODE=20
MAX_RECORDS=2000
BASE_PENALTY=0.0003
SEEDS="1 2 3 4 5"

printf 'Round4 type-specific 10x10 experiment started at %s\n' "$(date)" > "$OUT_DIR/STATUS.txt"
printf 'env=%s t_max=%s seeds=%s methods=baseline,diagnosis_only,uniform,type_specific,adaptive,random_type\n' "$ENV_KEY" "$T_MAX" "$SEEDS" >> "$OUT_DIR/STATUS.txt"

run_one() {
  local method="$1"
  local seed="$2"
  local config="mappo"
  local extra=""

  case "$method" in
    baseline)
      config="mappo"
      ;;
    diagnosis_only)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=False"
      ;;
    uniform_0.0003)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$BASE_PENALTY llm_fd_intervention_mode=uniform"
      ;;
    type_specific_0.0003)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$BASE_PENALTY llm_fd_intervention_mode=type_specific llm_fd_terminal_bonus=0.0003"
      ;;
    adaptive_0.0003)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$BASE_PENALTY llm_fd_intervention_mode=adaptive llm_fd_terminal_bonus=0.0003"
      ;;
    random_type_0.0003)
      config="mappo_llm_fd"
      extra="llm_fd_classifier=enhanced_heuristic llm_fd_max_records=$MAX_RECORDS llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$BASE_PENALTY llm_fd_intervention_mode=random_type llm_fd_terminal_bonus=0.0003"
      ;;
    *)
      printf 'Unknown method: %s\n' "$method" >&2
      return 2
      ;;
  esac

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
  run_one diagnosis_only "$seed"
  run_one uniform_0.0003 "$seed"
  run_one type_specific_0.0003 "$seed"
  run_one adaptive_0.0003 "$seed"
  run_one random_type_0.0003 "$seed"
done

python3 - <<'PY'
import csv
import glob
import math
import os
import re
from collections import defaultdict

out_dir = "results/round4_type_specific_10x10"
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
    handle.write("Round4 type-specific 10x10 summary\n\n")
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

printf 'Round4 type-specific 10x10 experiment finished at %s\n' "$(date)" >> "$OUT_DIR/STATUS.txt"
