#!/bin/bash
set -e
export FLASHINFER_DISABLE_VERSION_CHECK=1
U1=http://localhost:8000/v1
U2=http://localhost:8001/v1
SCRIPT=/data/lab/run_b3.py
LOG=/data/lab/b3_run.log
echo "=== B-3 Comprehensive Experiment Started $(date) ===" | tee -a "$LOG"

for dist in uniform integer adversarial; do
    echo "--- B-3 ${dist} $(date) ---" | tee -a "$LOG"
    python3 "$SCRIPT" \
        --url1 "$U1" --url2 "$U2" \
        --model1 Qwen/Qwen2.5-7B-Instruct \
        --model2 THUDM/glm-4-9b-chat \
        --dist "${dist}" \
        --probe_ks "5,10,20" \
        --n_matrices 20 --seeds 10 \
        --out "/data/lab/res_b3_qwen_glm_${dist}.jsonl" 2>&1 | tee -a "$LOG"
    echo "done ${dist} $(date)" | tee -a "$LOG"
done

echo "=== B-3 ALL DONE $(date) ===" | tee -a "$LOG"
