#!/usr/bin/env bash
# healthcheck.sh <host:port for role 0> <host:port for role 1>
# Confirms both endpoints are up AND serving the model the protocol pins.
set -euo pipefail
A="${1:-127.0.0.1:8002}"; B="${2:-127.0.0.1:8003}"
KEY="${VLLM_API_KEY:-p6r-replay}"
for ep in "$A" "$B"; do
  echo "--- http://$ep/v1/models"
  curl -sf -H "Authorization: Bearer $KEY" "http://$ep/v1/models" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); [print("   ", m["id"]) for m in d["data"]]'
  echo "--- one-token probe"
  curl -sf -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d '{"model":"'"$(curl -sf -H "Authorization: Bearer $KEY" "http://$ep/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')"'","messages":[{"role":"user","content":"Reply exactly: ACTION: A"}],"max_tokens":8,"temperature":0}' \
    "http://$ep/v1/chat/completions" \
    | python3 -c 'import json,sys; print("   ", repr(json.load(sys.stdin)["choices"][0]["message"]["content"]))'
done
echo "OK"
