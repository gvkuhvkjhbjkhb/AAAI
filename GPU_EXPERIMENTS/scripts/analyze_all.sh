#!/bin/bash
echo "================================" | tee -a /data/lab/analysis_report.txt
echo "B-1 Analysis Report - $(date)" | tee -a /data/lab/analysis_report.txt
echo "================================" | tee -a /data/lab/analysis_report.txt

# Combine all jsonl files for Qwen+GLM
cat /data/lab/res_b1_qwen_glm_*.jsonl > /tmp/b1_qwen_glm_all.jsonl
echo "" | tee -a /data/lab/analysis_report.txt
echo "### Qwen+GLM All Distributions Combined" | tee -a /data/lab/analysis_report.txt
python3 /data/lab/analyze_rollouts.py /tmp/b1_qwen_glm_all.jsonl 2>&1 | tee -a /data/lab/analysis_report.txt

# For Mistral+Phi, only if files exist
if ls /data/lab/res_b1_mistral_phi_*.jsonl 2>/dev/null | wc -l | grep -q '[1-9]'; then
    cat /data/lab/res_b1_mistral_phi_*.jsonl > /tmp/b1_mistral_phi_all.jsonl
    echo "" | tee -a /data/lab/analysis_report.txt
    echo "### Mistral+Phi All Distributions Combined" | tee -a /data/lab/analysis_report.txt
    python3 /data/lab/analyze_rollouts.py /tmp/b1_mistral_phi_all.jsonl 2>&1 | tee -a /data/lab/analysis_report.txt
fi

echo "" | tee -a /data/lab/analysis_report.txt
echo "=== Analysis complete $(date) ===" | tee -a /data/lab/analysis_report.txt
