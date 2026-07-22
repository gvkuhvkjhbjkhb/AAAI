#!/bin/bash
echo "=== B-3 Unknown-Payoff Experiment ==="
echo "Model pair: Qwen2.5-7B + GLM-4-9B (on ports 8000/8001)"
echo "K values: 5, 10, 20 probe rounds"
echo "20 matrices × 10 seeds × 3 K = 600 episodes"

python3 /data/lab/run_b3.py \
    --url1 http://localhost:8000/v1 \
    --url2 http://localhost:8001/v1 \
    --model1 Qwen/Qwen2.5-7B-Instruct \
    --model2 THUDM/glm-4-9b-chat \
    --dist uniform \
    --probe_ks "5,10,20" \
    --n_matrices 20 --seeds 10 \
    --out /data/lab/res_b3_qwen_glm.jsonl

echo "=== B-3 DONE $(date) ==="
