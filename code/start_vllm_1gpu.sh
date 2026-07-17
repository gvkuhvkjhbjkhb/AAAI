#!/bin/bash
# Co-locate BOTH bf16 models on a single 32GB GPU (RTX 5090).
# Qwen2.5-7B-Instruct @ :8000, GLM-4-9B-0414 @ :8001, both on GPU0.
# Short max-model-len keeps KV cache tiny (prompts are memory=2/horizon=5).
set -e
LOG=/data/lab
QWEN_UTIL="${QWEN_UTIL:-0.42}"
GLM_UTIL="${GLM_UTIL:-0.50}"
MAXLEN="${MAXLEN:-2048}"

echo "[$(date '+%H:%M:%S')] Starting BOTH vLLM servers on GPU0 (bf16, colocated)..."

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name Qwen/Qwen2.5-7B-Instruct \
    --port 8000 --tensor-parallel-size 1 \
    --gpu-memory-utilization $QWEN_UTIL \
    --max-model-len $MAXLEN --dtype bfloat16 \
    --api-key dummy --trust-remote-code --disable-log-requests \
    > $LOG/vllm_qwen.log 2>&1 &
echo "[$(date '+%H:%M:%S')] Qwen PID=$! util=$QWEN_UTIL"

# stagger start so the two loaders don't race for VRAM during init
sleep 45

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model THUDM/GLM-4-9B-0414 \
    --served-model-name THUDM/GLM-4-9B-0414 \
    --port 8001 --tensor-parallel-size 1 \
    --gpu-memory-utilization $GLM_UTIL \
    --max-model-len $MAXLEN --dtype bfloat16 \
    --api-key dummy --trust-remote-code --disable-log-requests \
    > $LOG/vllm_glm.log 2>&1 &
echo "[$(date '+%H:%M:%S')] GLM PID=$! util=$GLM_UTIL"

echo "[$(date '+%H:%M:%S')] Waiting for both servers /v1/models ..."
for i in $(seq 1 180); do
    R0=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null || echo 000)
    R1=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/v1/models 2>/dev/null || echo 000)
    if [ "$R0" = "200" ] && [ "$R1" = "200" ]; then
        echo "[$(date '+%H:%M:%S')] BOTH READY (attempt $i)"
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
        exit 0
    fi
    echo "  [$i] Qwen=$R0 GLM=$R1"; sleep 10
done
echo "TIMEOUT. Qwen log:"; tail -30 $LOG/vllm_qwen.log
echo "GLM log:"; tail -30 $LOG/vllm_glm.log
exit 1
