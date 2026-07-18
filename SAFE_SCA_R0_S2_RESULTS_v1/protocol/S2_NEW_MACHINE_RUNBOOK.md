# R0 + S2 Safe-SCA runbook for a new machine

This document is an operational protocol, not an exploratory notebook. It
runs (1) the same-seed R0 execution replay and then (2) the independent S2
seed-block replication, both with the **frozen S1 Safe-SCA configuration**.
Do not change a threshold, warm-up length, model, prompt, episode count,
policy list, or analysis gate after R0/S2 starts.

## 1. What must be copied to the new machine

Use two inputs, kept separate:

```text
PROJECT_ROOT/
  g123_augmentation/                       current source tree from this package
  g123_augmentation/code/run_safe_sca_campaign.py
  g123_augmentation/code/compare_safe_sca_replay.py
  g123_augmentation/code/start_vllm_s2.sh
  g123_augmentation/protocols/s1_safe_sca_frozen.json

S1_REFERENCE_ROOT/
  g123_augmentation/                       extracted g123_augmentation_s1_results.zip
    v2_results/s1_safe_sca_frozen.json
    v2_results/exp_s1_safe_sca_test/       all 720 original S1 cells
```

Do **not** point an R0 or S2 output directory into `S1_REFERENCE_ROOT`. The
reference data is read-only. New experiments use new result roots.

## 2. Hardware and software acceptance checklist

Required before any generation:

| Item | Requirement |
|---|---|
| GPUs | exactly 2 visible NVIDIA RTX 5090 GPUs (32 GB class) |
| Model service | Qwen2.5-7B-Instruct on `localhost:8000`; GLM-4-9B-0414 on `localhost:8001` |
| vLLM | `0.25.1` |
| PyTorch | `2.11.0+cu128` |
| Transformers | `5.14.1` |
| Precision | bf16, eager mode, `VLLM_USE_FLASHINFER_SAMPLER=0` |
| Disk | at least 80 GB free for models, logs, trajectories, and two new campaigns |
| Network | required only if model weights/packages are not already cached |

Use the validated lab environment/container when possible. Do not silently
substitute later library versions: `preflight_s1.py` is intentionally exact
and must pass without `--allow-version-mismatch`.

## 3. Shell setup and CPU-only checks

Run as the experiment user on Linux. Replace paths, but keep the directory
roles unchanged.

```bash
export PROJECT_ROOT=/data/aaai/safe_sca_replication
export GSACA_ROOT="$PROJECT_ROOT/g123_augmentation"
export S1_REFERENCE_ROOT=/data/aaai/s1_reference/g123_augmentation
export PYTHON_BIN=/data/venvs/safe-sca/bin/python
export VLLM_LOG_DIR="$GSACA_ROOT/logs/vllm_s2"

test -f "$GSACA_ROOT/code/run_experiment_local.py"
test -f "$GSACA_ROOT/code/run_safe_sca_campaign.py"
test -f "$GSACA_ROOT/protocols/s1_safe_sca_frozen.json"
test -f "$S1_REFERENCE_ROOT/v2_results/s1_safe_sca_frozen.json"
test -d "$S1_REFERENCE_ROOT/v2_results/exp_s1_safe_sca_test"
cmp "$GSACA_ROOT/protocols/s1_safe_sca_frozen.json" \
  "$S1_REFERENCE_ROOT/v2_results/s1_safe_sca_frozen.json"

cd "$GSACA_ROOT/code"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py' -v
"$PYTHON_BIN" -m py_compile \
  safe_sca.py run_experiment_local.py preflight_s1.py \
  validate_s1_results.py analyze_s1_safe_sca.py \
  run_safe_sca_campaign.py compare_safe_sca_replay.py

chmod +x "$GSACA_ROOT/code/start_vllm_s2.sh"
```

Expected result: all CPU tests pass and all listed files compile. A failure is
a stop condition; do not use `--force` or edit the frozen configuration to
make a test pass.

## 4. Start and verify the two vLLM servers

```bash
"$GSACA_ROOT/code/start_vllm_s2.sh"

curl -s http://localhost:8000/v1/models
curl -s http://localhost:8001/v1/models
nvidia-smi
```

The script records PIDs and logs under `$VLLM_LOG_DIR`. If a port is already
occupied, the script stops rather than attaching to an unknown model server.
Verify the process and model identity manually before deciding whether to
reuse it.

## 5. R0: same-seed execution replay (72 cells)

R0 checks whether the S1 result survives a new-machine execution path. It is
not an opportunity to tune. The default comparison requires exact total
payoff and exact Safe-SCA routes. If an alternative tolerance is scientifically
necessary, write it into a dated protocol note **before** launching R0.

```bash
export FROZEN_CONFIG="$GSACA_ROOT/protocols/s1_safe_sca_frozen.json"
export R0_ROOT="$GSACA_ROOT/v2_results_r0"
export R0_OUT="$R0_ROOT/exp_r0_safe_sca_test"

# 0 GPU generation: exact package/GPU/model endpoint check.
"$PYTHON_BIN" "$GSACA_ROOT/code/preflight_s1.py" --out-dir "$R0_OUT"

# Inspect the precise command first. This writes nothing and contacts no model.
"$PYTHON_BIN" "$GSACA_ROOT/code/run_safe_sca_campaign.py" \
  --campaign r0 --root "$GSACA_ROOT" --results-root "$R0_ROOT" \
  --frozen-config "$FROZEN_CONFIG" --seeds 62 63 --workers 24 --dry-run

# Actual 6 games × 2 seeds × 6 policies = 72 cells.
"$PYTHON_BIN" "$GSACA_ROOT/code/run_safe_sca_campaign.py" \
  --campaign r0 --root "$GSACA_ROOT" --results-root "$R0_ROOT" \
  --frozen-config "$FROZEN_CONFIG" --seeds 62 63 --workers 24

# Completeness/provenance first, comparison second.
"$PYTHON_BIN" "$GSACA_ROOT/code/validate_s1_results.py" \
  --results "$R0_OUT" --frozen-config "$FROZEN_CONFIG" --seeds 62 63
"$PYTHON_BIN" "$GSACA_ROOT/code/compare_safe_sca_replay.py" \
  --reference "$S1_REFERENCE_ROOT/v2_results/exp_s1_safe_sca_test" \
  --replay "$R0_OUT" --seeds 62 63 \
  --payoff-tolerance 0.0 --route-mismatch-budget 0
```

Success artifacts are `CAMPAIGN_SNAPSHOT.json`,
`CAMPAIGN_EXECUTION_REPORT.json`, `S1_INTEGRITY_REPORT.json`, and
`R0_REPLAY_COMPARISON.json` inside `$R0_OUT`. If the comparison fails, do not
start S2. Preserve logs and investigate model version, endpoint identity,
request seeds, server flags, and concurrency before any rerun.

## 6. S2: independent seed-block confirmation (720 cells)

S2 reuses the frozen S1 configuration unchanged. It uses seeds 82–101 and a
new result root. The campaign launcher schedules each `(game, seed)` as an
atomic six-policy task, preserving the actual Latin-square policy order.

```bash
export S2_ROOT="$GSACA_ROOT/v2_results_s2"
export S2_OUT="$S2_ROOT/exp_s2_safe_sca_test"

"$PYTHON_BIN" "$GSACA_ROOT/code/preflight_s1.py" --out-dir "$S2_OUT"

"$PYTHON_BIN" "$GSACA_ROOT/code/run_safe_sca_campaign.py" \
  --campaign s2 --root "$GSACA_ROOT" --results-root "$S2_ROOT" \
  --frozen-config "$FROZEN_CONFIG" \
  --seeds 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 \
  --workers 24 --dry-run

"$PYTHON_BIN" "$GSACA_ROOT/code/run_safe_sca_campaign.py" \
  --campaign s2 --root "$GSACA_ROOT" --results-root "$S2_ROOT" \
  --frozen-config "$FROZEN_CONFIG" \
  --seeds 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 \
  --workers 24 --task-timeout 7200 --max-retries 2

"$PYTHON_BIN" "$GSACA_ROOT/code/validate_s1_results.py" \
  --results "$S2_OUT" --frozen-config "$FROZEN_CONFIG" \
  --seeds 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101
"$PYTHON_BIN" "$GSACA_ROOT/code/analyze_s1_safe_sca.py" \
  --results "$S2_OUT" --safety-margin 0.10 \
  --min-coordination-recovery 0.30 --min-coordination-games 2
```

Expected volume is 6 games × 20 seeds × 6 policies = 720 cells. On the S1
hardware, the earlier 24-worker execution took about 2.5 hours, but treat the
new machine's first 20 completed tasks as the real throughput calibration.

## 7. Monitoring, recovery, and stop conditions

```bash
# Terminal A: task-level progress and errors
tail -f "$S2_OUT/logs_campaign"/*.log

# Terminal B: GPU utilization and memory
watch -n 5 nvidia-smi

# Terminal C: completed metric count (must eventually be 720)
find "$S2_OUT" -path '*/metrics.json' | wc -l
```

- The launcher may be safely re-run with the **identical command** after an
  interruption; completed `metrics.json` cells are skipped and the immutable
  campaign snapshot rejects protocol drift.
- Do not delete partial cells, alter seeds, or use the runner's `--force`.
- If a task exceeds 7,200 seconds it is terminated, logged, and retried up to
  two times. A remaining failure is a stop condition, not a missing value to
  omit.
- If GPU memory errors occur, stop and record the logs. Changing server memory
  utilization is an environment/protocol change and requires a new campaign
  directory plus an explicit deviation note.

## 8. Preregistered S2 decision rule

Run final analysis only after integrity validation passes. S2 confirms the
method result only when both conditions hold:

1. Safe-SCA's paired bootstrap 95% lower bound versus NoAlign is at least
   `-0.10` in each of Chicken, Deadlock, and Hawk-Dove.
2. Safe-SCA recovers at least 30% of a positive Always-Gated gain in at least
   two of Stag Hunt, Battle of the Sexes, and Public Goods.

Report S1 and S2 separately; do not pool p-values across environment runs.
Only after R0 and S2 pass should the 320-cell unseen-matrix/action-surface
transfer suite begin.
