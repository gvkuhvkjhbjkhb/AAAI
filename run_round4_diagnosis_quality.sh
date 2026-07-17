#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT_DIR/epymarl/src/main.py" ]]; then
  EPYMARL_DIR="$ROOT_DIR/epymarl"
else
  EPYMARL_DIR="$ROOT_DIR"
fi
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/round4_diagnosis_quality}"
INPUT_GLOB="${INPUT_GLOB:-results/llm_fd/*/failure_records.jsonl}"
SAMPLE_SIZE="${SAMPLE_SIZE:-300}"
SEED="${SEED:-0}"
MODEL="${MODEL:-Qwen3.5-4B}"
HUMAN_LABELS="${HUMAN_LABELS:-}"
mkdir -p "$OUT_DIR"

printf 'Round4 diagnosis-quality experiment started at %s\n' "$(date)" > "$OUT_DIR/STATUS.txt"
printf 'input=%s sample_size=%s model=%s\n' "$INPUT_GLOB" "$SAMPLE_SIZE" "$MODEL" >> "$OUT_DIR/STATUS.txt"

(cd "$EPYMARL_DIR" && python3 -m src.llm_diagnosis.offline_relabel \
  --input-glob "$INPUT_GLOB" \
  --output-dir "$OUT_DIR/enhanced_heuristic" \
  --mode enhanced_heuristic \
  --sample-size "$SAMPLE_SIZE" \
  --seed "$SEED" \
  --human-labels "$HUMAN_LABELS")

(cd "$EPYMARL_DIR" && python3 -m src.llm_diagnosis.offline_relabel \
  --input-glob "$INPUT_GLOB" \
  --output-dir "$OUT_DIR/qwen35_4b" \
  --mode api \
  --model "$MODEL" \
  --sample-size "$SAMPLE_SIZE" \
  --seed "$SEED" \
  --human-labels "$HUMAN_LABELS")

python3 "$EPYMARL_DIR/tools/compare_diagnosis_labels.py" \
  --predictions "$OUT_DIR/enhanced_heuristic/audit_sample.csv" "$OUT_DIR/qwen35_4b/audit_sample.csv" \
  --names enhanced_heuristic qwen35_4b \
  --output "$OUT_DIR/diagnosis_quality_summary.txt"

printf 'Round4 diagnosis-quality experiment finished at %s\n' "$(date)" >> "$OUT_DIR/STATUS.txt"
