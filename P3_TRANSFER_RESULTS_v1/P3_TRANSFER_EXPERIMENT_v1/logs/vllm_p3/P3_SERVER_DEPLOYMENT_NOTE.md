# P3 vLLM server deployment note

## Frozen protocol requirements (preserved exactly)
- Qwen/Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28
- THUDM/GLM-4-9B-0414 revision 645b8482494e31b6b752272bf7f7f273ef0f3caf
- max-model-len 2048, dtype bfloat16, tensor-parallel-size 1, trust-remote-code

## Deviation from server_scripts/start_vllm_p3.sh
The bundled start_vllm_p3.sh launches vLLM with `--api-key dummy --enforce-eager
--gpu-memory-utilization 0.85`. That combination is incompatible with the
runbook's own verification step (`curl --silent --fail http://localhost:8000/v1/models`,
no Authorization header) and with base/code/preflight_s1.py, which probes
/v1/models via urllib without an Authorization header. Both return HTTP 401,
causing preflight to fail and the start_vllm_p3.sh readiness poll to time out.

## Alignment with S2 (P3's stated cross-environment reference)
Per SERVER_ENVIRONMENT_APPENDIX_V2.json, the S2 servers that produced
SAFE_SCA_R0_S2_RESULTS_v1 used: no --api-key flag, no --enforce-eager (CUDA
graphs enabled), --gpu-memory-utilization 0.92. The appendix explicitly records
that enforce-eager / CUDA graphs and gpu-memory-utilization do not change model
outputs (same weights, same bf16 precision, same revision) — only kernel-launch
overhead and memory headroom.

## P3 launch config used
To preserve all protocol-frozen elements while making preflight and the
runbook's verification curls work, P3 is launched with S2's exact deployment
config plus the P3-pinned revisions:

  CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen/Qwen2.5-7B-Instruct \
    --port 8000 --tensor-parallel-size 1 --gpu-memory-utilization 0.92 \
    --max-model-len 2048 --dtype bfloat16 --trust-remote-code \
    --revision a09a35458c702b33eeacc393d103063234e8bc28 \
    --tokenizer-revision a09a35458c702b33eeacc393d103063234e8bc28

  CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model THUDM/GLM-4-9B-0414 --served-model-name THUDM/GLM-4-9B-0414 \
    --port 8001 --tensor-parallel-size 1 --gpu-memory-utilization 0.92 \
    --max-model-len 2048 --dtype bfloat16 --trust-remote-code \
    --revision 645b8482494e31b6b752272bf7f7f273ef0f3caf \
    --tokenizer-revision 645b8482494e31b6b752272bf7f7f273ef0f3caf

The pids_and_revisions.env audit log is written in the same format as
start_vllm_p3.sh. The protocol JSON (protocols/p3_frozen_protocol.json) is
unchanged; only model revisions are protocol-frozen, and they match exactly.
