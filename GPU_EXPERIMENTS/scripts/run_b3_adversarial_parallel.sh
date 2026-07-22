#!/bin/bash
set -e
export FLASHINFER_DISABLE_VERSION_CHECK=1
U1=http://localhost:8000/v1
U2=http://localhost:8001/v1
SCRIPT=/data/lab/run_b3.py
DIST=adversarial

echo "=== B3 Adversarial 4-Worker Parallel $(date) ==="

# Worker 1: matrices 0-4
python3 "$SCRIPT" --url1 "$U1" --url2 "$U2" \
    --model1 Qwen/Qwen2.5-7B-Instruct --model2 THUDM/glm-4-9b-chat \
    --dist "$DIST" --probe_ks "5,10,20" --n_matrices 5 --seeds 10 \
    --out /data/lab/res_b3_adv_part0.jsonl 2>/tmp/b3_adv_w0.log &
echo "Worker0 (matrices 0-4) PID=$!"

# Worker 2: matrices 5-9
python3 "$SCRIPT" --url1 "$U1" --url2 "$U2" \
    --model1 Qwen/Qwen2.5-7B-Instruct --model2 THUDM/glm-4-9b-chat \
    --dist "$DIST" --probe_ks "5,10,20" --n_matrices 5 --seeds 10 \
    --out /data/lab/res_b3_adv_part1.jsonl 2>/tmp/b3_adv_w1.log &
echo "Worker1 (matrices 5-9) PID=$!"

# Worker 3: matrices 10-14
python3 "$SCRIPT" --url1 "$U1" --url2 "$U2" \
    --model1 Qwen/Qwen2.5-7B-Instruct --model2 THUDM/glm-4-9b-chat \
    --dist "$DIST" --probe_ks "5,10,20" --n_matrices 5 --seeds 10 \
    --out /data/lab/res_b3_adv_part2.jsonl 2>/tmp/b3_adv_w2.log &
echo "Worker2 (matrices 10-14) PID=$!"

# Worker 4: matrices 15-19
python3 "$SCRIPT" --url1 "$U1" --url2 "$U2" \
    --model1 Qwen/Qwen2.5-7B-Instruct --model2 THUDM/glm-4-9b-chat \
    --dist "$DIST" --probe_ks "5,10,20" --n_matrices 5 --seeds 10 \
    --out /data/lab/res_b3_adv_part3.jsonl 2>/tmp/b3_adv_w3.log &
echo "Worker3 (matrices 15-19) PID=$!"

echo "All 4 workers launched. Waiting..."
wait
echo "All workers done $(date)"

# Merge parts into final
cat /data/lab/res_b3_adv_part0.jsonl /data/lab/res_b3_adv_part1.jsonl \
    /data/lab/res_b3_adv_part2.jsonl /data/lab/res_b3_adv_part3.jsonl \
    > /data/lab/res_b3_qwen_glm_adversarial.jsonl
echo "Merged into res_b3_qwen_glm_adversarial.jsonl ($(wc -l < /data/lab/res_b3_qwen_glm_adversarial.jsonl) lines)"
echo "=== DONE $(date) ==="
