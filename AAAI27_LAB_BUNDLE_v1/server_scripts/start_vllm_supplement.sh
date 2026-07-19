#!/usr/bin/env bash
# Start the supplemental-control servers with the exact S2/P3 revision pins.
set -euo pipefail

SUPPLEMENT_ROOT="${SUPPLEMENT_ROOT:?Set SUPPLEMENT_ROOT to AAAI27_SUPPLEMENTAL_EXPERIMENTS_v1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-$SUPPLEMENT_ROOT/logs/vllm_supplement}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
GLM_MODEL="${GLM_MODEL:-THUDM/GLM-4-9B-0414}"
QWEN_REVISION="${QWEN_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}"
GLM_REVISION="${GLM_REVISION:-645b8482494e31b6b752272bf7f7f273ef0f3caf}"

if [[ -n "${VLLM_EXTRA_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${VLLM_EXTRA_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$VLLM_LOG_DIR"

for port in 8000 8001; do
  if curl --silent --fail --max-time 2 "http://localhost:${port}/v1/models" >/dev/null; then
    echo "Port ${port} is already serving a model; verify it manually or stop it first." >&2
    exit 2
  fi
done

start_server() {
  local gpu="$1" port="$2" model="$3" revision="$4" log="$5"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$model" --served-model-name "$model" --revision "$revision" \
    --tokenizer "$model" --tokenizer-revision "$revision" \
    --port "$port" --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 --enforce-eager --api-key dummy --trust-remote-code >"$log" 2>&1 &
  echo $!
}

PID0="$(start_server 0 8000 "$QWEN_MODEL" "$QWEN_REVISION" "$VLLM_LOG_DIR/gpu0_qwen.log")"
PID1="$(start_server 1 8001 "$GLM_MODEL" "$GLM_REVISION" "$VLLM_LOG_DIR/gpu1_glm.log")"
printf 'qwen_pid=%s\nglm_pid=%s\nqwen_model=%s\nqwen_revision=%s\nglm_model=%s\nglm_revision=%s\nstarted_utc=%s\n' \
  "$PID0" "$PID1" "$QWEN_MODEL" "$QWEN_REVISION" "$GLM_MODEL" "$GLM_REVISION" "$(date -u +%FT%TZ)" \
  > "$VLLM_LOG_DIR/pids_and_revisions.env"

for attempt in $(seq 1 90); do
  qwen="$(curl --silent --max-time 2 http://localhost:8000/v1/models || true)"
  glm="$(curl --silent --max-time 2 http://localhost:8001/v1/models || true)"
  if [[ "$qwen" == *"$QWEN_MODEL"* && "$glm" == *"$GLM_MODEL"* ]]; then
    echo "Supplement servers ready: Qwen PID=$PID0; GLM PID=$PID1."
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    exit 0
  fi
  sleep 10
done

echo "Timed out waiting for supplemental vLLM servers." >&2
tail -n 40 "$VLLM_LOG_DIR/gpu0_qwen.log" || true
tail -n 40 "$VLLM_LOG_DIR/gpu1_glm.log" || true
exit 1
