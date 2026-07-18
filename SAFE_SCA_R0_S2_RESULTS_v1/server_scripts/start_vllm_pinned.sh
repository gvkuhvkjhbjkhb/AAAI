#!/usr/bin/env bash
# Start two vLLM servers with EXPLICITLY PINNED model revisions.
# See SERVER_ENVIRONMENT_APPENDIX.json for the frozen environment record.
set -euo pipefail

export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:/usr/local/cuda/lib64
export HF_HOME=/data/models/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0

VLLM_LOG_DIR=/data/aaai/safe_sca_replication/g123_augmentation/logs/vllm_pinned
mkdir -p "$VLLM_LOG_DIR"

QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
QWEN_REV="a09a35458c702b33eeacc393d103063234e8bc28"
GLM_MODEL="THUDM/GLM-4-9B-0414"
GLM_REV="645b8482494e31b6b752272bf7f7f273ef0f3caf"

for port in 8000 8001; do
  if curl --silent --fail --max-time 2 "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
    echo "Port ${port} already serves. Refusing to mix servers." >&2
    exit 2
  fi
done

start_server() {
  local gpu="$1"; local port="$2"; local model="$3"; local rev="$4"; local log="$5"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$model" --served-model-name "$model" --port "$port" \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 2048 \
    --dtype bfloat16 --enforce-eager --trust-remote-code \
    --revision "$rev" --tokenizer-revision "$rev" \
    >"$log" 2>&1 &
  echo $!
}

PID0="$(start_server 0 8000 "$QWEN_MODEL" "$QWEN_REV" "$VLLM_LOG_DIR/gpu0_qwen.log")"
PID1="$(start_server 1 8001 "$GLM_MODEL" "$GLM_REV" "$VLLM_LOG_DIR/gpu1_glm.log")"
printf 'gpu0_pid=%s\ngpu1_pid=%s\nqwen_rev=%s\nglm_rev=%s\nstarted_utc=%s\n' \
  "$PID0" "$PID1" "$QWEN_REV" "$GLM_REV" "$(date -u +%FT%TZ)" >"$VLLM_LOG_DIR/pids.env"
echo "Started Qwen PID=$PID0 (GPU0, rev=$QWEN_REV) and GLM PID=$PID1 (GPU1, rev=$GLM_REV)."

for attempt in $(seq 1 90); do
  qwen="$(curl --silent --max-time 2 http://localhost:8000/v1/models || true)"
  glm="$(curl --silent --max-time 2 http://localhost:8001/v1/models || true)"
  if [[ "$qwen" == *"$QWEN_MODEL"* && "$glm" == *"$GLM_MODEL"* ]]; then
    echo "Both vLLM servers ready after ${attempt} checks."
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    exit 0
  fi
  sleep 10
done
echo "Timed out waiting for vLLM." >&2
tail -n 40 "$VLLM_LOG_DIR/gpu0_qwen.log" || true
tail -n 40 "$VLLM_LOG_DIR/gpu1_glm.log" || true
exit 1
