#!/usr/bin/env bash
set -u

ROOT_DIR="$(pwd)"
OUT_DIR="$ROOT_DIR/results/formal_10x10_tuning"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

ENV_KEY="lbforaging:Foraging-10x10-3p-3f-v3"
TIME_LIMIT=50
T_MAX=500000
TEST_INTERVAL=100000
LOG_INTERVAL=25000
TEST_NEPISODE=20

printf '10x10 tuning started at %s\n' "$(date)" > "$OUT_DIR/STATUS.txt"
printf 'env=%s t_max=%s seeds=1,2,3 penalties=baseline,0.0001,0.0003,0.001\n' "$ENV_KEY" "$T_MAX" >> "$OUT_DIR/STATUS.txt"

run_one() {
  local method="$1"
  local seed="$2"
  local config="mappo"
  local extra=""
  if [[ "$method" != "baseline" ]]; then
    config="mappo_llm_fd"
    local penalty="${method#penalty_}"
    extra="llm_fd_classifier=heuristic llm_fd_max_records=2000 llm_fd_apply_reward_shaping=True llm_fd_failure_penalty=$penalty"
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

for seed in 1 2 3; do
  run_one baseline "$seed"
  run_one penalty_0.0001 "$seed"
  run_one penalty_0.0003 "$seed"
  run_one penalty_0.001 "$seed"
done

python3 - <<'PY'
import csv
import glob
import os
import re

out_dir = "results/formal_10x10_tuning"
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
        "last_test_return": tests[-1] if tests else "",
        "best_test_return": max(tests) if tests else "",
        "last_llm_fd_records": records[-1] if records else "",
    })

fieldnames = ["method", "seed", "last_train_return", "best_train_return", "last_test_return", "best_test_return", "last_llm_fd_records"]
with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

groups = {}
for row in rows:
    groups.setdefault(row["method"], []).append(row)

def mean(values):
    values = [float(v) for v in values if v != ""]
    return sum(values) / len(values) if values else ""

with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as handle:
    handle.write("10x10 tuning summary\n\n")
    for row in rows:
        handle.write(str(row) + "\n")
    handle.write("\nGrouped means\n")
    for method, items in sorted(groups.items()):
        handle.write(str({
            "method": method,
            "mean_last_train_return": mean([r["last_train_return"] for r in items]),
            "mean_best_train_return": mean([r["best_train_return"] for r in items]),
            "mean_last_test_return": mean([r["last_test_return"] for r in items]),
            "mean_best_test_return": mean([r["best_test_return"] for r in items]),
        }) + "\n")
PY

if command -v ollama >/dev/null 2>&1 && ollama list >/dev/null 2>&1; then
  model="$(ollama list | awk 'NR==2 {print $1}')"
  if [[ -n "$model" ]]; then
    python3 -m src.llm_diagnosis.offline_relabel --mode ollama --model "$model" --sample-size 1000 --output-dir results/offline_relabel_tuning
  else
    python3 -m src.llm_diagnosis.offline_relabel --mode enhanced_heuristic --sample-size 1000 --output-dir results/offline_relabel_tuning
  fi
else
  if python3 - <<'PY'
from huggingface_hub import model_info
model_info('Qwen/Qwen2.5-0.5B-Instruct')
PY
  then
    python3 -m src.llm_diagnosis.offline_relabel --mode hf --model Qwen/Qwen2.5-0.5B-Instruct --sample-size 300 --output-dir results/offline_relabel_tuning
  else
    python3 -m src.llm_diagnosis.offline_relabel --mode enhanced_heuristic --sample-size 1000 --output-dir results/offline_relabel_tuning
  fi
fi

printf '10x10 tuning finished at %s\n' "$(date)" >> "$OUT_DIR/STATUS.txt"
