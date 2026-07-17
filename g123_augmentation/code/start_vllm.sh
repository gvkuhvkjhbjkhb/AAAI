#!/bin/bash
# Start 2 vLLM servers: one model per GPU.
# GPU 0: Qwen2.5-7B-Instruct @ localhost:8000
# GPU 1: GLM-4-9B-0414   @ localhost:8001
set -e

export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib/:$LD_LIBRARY_PATH"
# RTX 5090 is sm_120 (Blackwell); FlashInfer's JIT rejects sm_120, so disable
# the FlashInfer sampler (fall back to the native vLLM top-k/top-p sampler).
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "[$(date '+%H:%M:%S')] Starting vLLM servers..."

# GPU 0: Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048 \
    --dtype bfloat16 \
    --enforce-eager \
    --api-key dummy \
    --trust-remote-code \
    > /data/lab/vllm_gpu0.log 2>&1 &
PID0=$!
echo "[$(date '+%H:%M:%S')] vLLM GPU0 (Qwen) PID=$PID0"

# GPU 1: GLM-4-9B-0414
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model THUDM/GLM-4-9B-0414 \
    --served-model-name THUDM/GLM-4-9B-0414 \
    --port 8001 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048 \
    --dtype bfloat16 \
    --enforce-eager \
    --api-key dummy \
    --trust-remote-code \
    > /data/lab/vllm_gpu1.log 2>&1 &
PID1=$!
echo "[$(date '+%H:%M:%S')] vLLM GPU1 (GLM)  PID=$PID1"

echo "[$(date '+%H:%M:%S')] Waiting for servers to be ready..."
for i in $(seq 1 120); do
    R0=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models 2>/dev/null || echo "000")
    R1=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/v1/models 2>/dev/null || echo "000")
    if [ "$R0" = "200" ] && [ "$R1" = "200" ]; then
        echo "[$(date '+%H:%M:%S')] Both vLLM servers ready! (attempt $i)"
        echo "GPU0: http://localhost:8000/v1 (Qwen2.5-7B-Instruct)"
        echo "GPU1: http://localhost:8001/v1 (GLM-4-9B-0414)"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
        exit 0
    fi
    echo "  [$i] GPU0=$R0 GPU1=$R1 ..."
    sleep 10
done

echo "[$(date '+%H:%M:%S')] TIMEOUT waiting for vLLM servers"
echo "=== GPU0 log tail ==="
tail -20 /data/lab/vllm_gpu0.log
echo "=== GPU1 log tail ==="
tail -20 /data/lab/vllm_gpu1.log
exit 1
