#!/usr/bin/env bash
# Start the exact two-server vLLM topology used for S1/R0/S2.
#
# Required environment: GSACA_ROOT points to g123_augmentation and PYTHON_BIN
# selects the environment containing vLLM 0.25.1, PyTorch 2.11.0+cu128, and
# Transformers 5.14.1. This script never kills an existing server; if either
# port is already live it stops so the operator can verify its model identity.
set -euo pipefail

GSACA_ROOT="${GSACA_ROOT:?Set GSACA_ROOT to the extracted g123_augmentation directory}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-$GSACA_ROOT/logs/vllm_s2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
GLM_MODEL="${GLM_MODEL:-THUDM/GLM-4-9B-0414}"

if [[ -n "${VLLM_EXTRA_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${VLLM_EXTRA_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$VLLM_LOG_DIR"

for port in 8000 8001; do
  if curl --silent --fail --max-time 2 "http://localhost:${port}/v1/models" >/dev/null; then
    echo "Port ${port} already serves vLLM. Stop it or verify it manually; refusing to mix servers." >&2
    exit 2
  fi
done

start_server() {
  local gpu="$1"
  local port="$2"
  local model="$3"
  local log="$4"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$model" \
    --served-model-name "$model" \
    --port "$port" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --enforce-eager \
    --api-key dummy \
    --trust-remote-code >"$log" 2>&1 &
  echo $!
}

PID0="$(start_server 0 8000 "$QWEN_MODEL" "$VLLM_LOG_DIR/gpu0_qwen.log")"
PID1="$(start_server 1 8001 "$GLM_MODEL" "$VLLM_LOG_DIR/gpu1_glm.log")"
printf 'gpu0_pid=%s\ngpu1_pid=%s\nstarted_utc=%s\n' "$PID0" "$PID1" "$(date -u +%FT%TZ)" \
  >"$VLLM_LOG_DIR/pids.env"
echo "Started Qwen PID=$PID0 (GPU0) and GLM PID=$PID1 (GPU1)."

for attempt in $(seq 1 90); do
  qwen="$(curl --silent --max-time 2 http://localhost:8000/v1/models || true)"
  glm="$(curl --silent --max-time 2 http://localhost:8001/v1/models || true)"
  if [[ "$qwen" == *"$QWEN_MODEL"* && "$glm" == *"$GLM_MODEL"* ]]; then
    echo "Both vLLM servers are ready after ${attempt} checks."
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    exit 0
  fi
  sleep 10
done

echo "Timed out waiting for vLLM. Recent logs:" >&2
tail -n 40 "$VLLM_LOG_DIR/gpu0_qwen.log" || true
tail -n 40 "$VLLM_LOG_DIR/gpu1_glm.log" || true
exit 1
