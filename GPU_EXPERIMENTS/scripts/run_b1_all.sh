#!/bin/bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
SCRIPT=/data/lab/run_experiment_b1.py
U1=http://localhost:8000/v1
U2=http://localhost:8001/v1
echo "=== B-1 ALL $(date) ==="
for dist in uniform integer adversarial; do
    echo "--- ${dist} $(date) ---"
    python3 "$SCRIPT" --url1 "$U1" --url2 "$U2" \
        --model1 Qwen/Qwen2.5-7B-Instruct \
        --model2 THUDM/glm-4-9b-chat \
        --dist "${dist}" --n_matrices 30 --seeds 15 \
        --out "/data/lab/res_b1_qwen_glm_${dist}.jsonl"
    echo "done ${dist} $(date)"
done
echo "=== B-1 ALL DONE $(date) ==="
