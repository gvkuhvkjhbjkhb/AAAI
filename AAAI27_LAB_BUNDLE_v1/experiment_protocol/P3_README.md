# P3_TRANSFER_EXPERIMENT_v1

Complete frozen transfer package for the post-S2 P3 experiment.

Start with [P3_FROZEN_PROTOCOL.md](P3_FROZEN_PROTOCOL.md), then execute
[P3_NEW_MACHINE_RUNBOOK.md](P3_NEW_MACHINE_RUNBOOK.md) without altering the
protocol.

Contents:

- `protocols/p3_frozen_protocol.json` — seeds, models/revisions, policy list,
  execution topology, gates, and controller information barrier.
- `code/p3_matrices.py` — eight frozen unseen matrices and opaque action
  surfaces; categories are analysis-only.
- `code/run_p3_matrix.py` and `code/run_p3_campaign.py` — generic runner and
  resumable 320-cell orchestrator.
- `code/validate_p3_results.py` and `code/analyze_p3_transfer.py` — fail-closed
  integrity and separate-matrix analysis.
- `server_scripts/start_vllm_p3.sh` — Qwen/GLM revision-pinned server startup.
- `code/tests/` — no-GPU tests for registry, protocol, validator, and analysis.

P3 is an upside transfer experiment. If it fails, the supported S1/S2 claim
remains an in-distribution result over the six evaluated source games; only
the unseen-matrix generalization claim is removed.
