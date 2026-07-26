#!/usr/bin/env bash
# node-a : Mistral-7B-Instruct-v0.3 on one RTX 4090, serving role 0.
#
# The revision pin is MANDATORY.  The original P6R run did not record one in a
# way that could be checked afterwards, which is precisely why its low-fidelity
# cell cannot be attributed today.  This script refuses to start without it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTOCOL="${PROTOCOL:-$HERE/protocols/p6r_replay_frozen.json}"
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
PORT="${PORT:-8002}"

REV="$(python3 -c "
import json,sys
p=json.load(open('$PROTOCOL'))
print(p['model_pair']['revisions']['$MODEL'])
")"
if [[ -z "$REV" || "$REV" == PIN_EXACT_COMMIT* ]]; then
  echo "REFUSING TO START: pin the commit SHA for $MODEL in $PROTOCOL first." >&2
  echo "  python3 -c \"from huggingface_hub import HfApi; print(HfApi().model_info('$MODEL').sha)\"" >&2
  exit 2
fi

export VLLM_API_KEY="${VLLM_API_KEY:-p6r-replay}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Deterministic-as-possible serving: no prefix caching (its hit pattern depends
# on arrival order, which would make outputs depend on scheduling).
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --revision "$REV" \
  --served-model-name "$MODEL" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --api-key "$VLLM_API_KEY" \
  --dtype bfloat16 \
  --seed 20260726 \
  --max-model-len 2048 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --disable-log-requests
