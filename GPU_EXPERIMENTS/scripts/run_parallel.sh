#!/bin/bash
set -e
export FLASHINFER_DISABLE_VERSION_CHECK=1

MODEL1="$1"; MODEL2="$2"; GPU="$3"
echo "=== GPU${GPU}: ${MODEL1} + ${MODEL2} ==="
for dist in uniform integer adversarial; do
    echo "--- GPU${GPU} dist=$dist $(date) ---"
    CUDA_VISIBLE_DEVICES=${GPU} python3 /data/lab/run_generalization_rollouts.py \
        --backend vllm \
        --model1 "${MODEL1}" --model2 "${MODEL2}" \
        --dist "${dist}" \
        --n_matrices 30 --seeds 15 \
        --out "/data/lab/res_${dist}_gpu${GPU}.jsonl"
    echo "  done dist=$dist on GPU${GPU} at $(date)"
done
echo "=== GPU${GPU}: ALL DONE at $(date) ==="
