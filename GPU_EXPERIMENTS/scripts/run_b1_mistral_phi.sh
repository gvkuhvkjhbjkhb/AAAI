#!/bin/bash
export FLASHINFER_DISABLE_VERSION_CHECK=1
SCRIPT=/data/lab/run_experiment_b1.py
U1=http://localhost:8000/v1
U2=http://localhost:8001/v1
echo "=== B-1 Mistral+Phi ALL $(date) ==="
for dist in uniform integer adversarial; do
    echo "--- ${dist} $(date) ---"
    python3 "$SCRIPT" --url1 "$U1" --url2 "$U2" \
        --model1 mistralai/Mistral-7B-Instruct-v0.3 \
        --model2 microsoft/Phi-3.5-mini-instruct \
        --dist "${dist}" --n_matrices 30 --seeds 15 \
        --out "/data/lab/res_b1_mistral_phi_${dist}.jsonl"
    echo "done ${dist} $(date)"
done
echo "=== B-1 Mistral+Phi ALL DONE $(date) ==="
