# P3 new-machine runbook — 320-cell frozen transfer suite

This is the only execution procedure for P3. Read
[`P3_FROZEN_PROTOCOL.md`](P3_FROZEN_PROTOCOL.md) before starting. P3 is
authorized by the S2-confirmed paper plan, but it is not permitted to tune
Safe-SCA, select matrices, change the worker topology, or hide failed cells.

## 1. Required layout

On the Linux GPU machine, place this package next to the tested S1/S2 source:

```text
/data/aaai/safe_sca_replication/
├── g123_augmentation/                 # S1/S2 base project and Python code
└── P3_TRANSFER_EXPERIMENT_v1/          # this package, unchanged
```

The base project must contain `code/run_experiment_local.py`,
`code/hettom_baseline.py`, `code/safe_sca.py`, and `code/preflight_s1.py`.
Use the same Python environment that completed S2, including vLLM, PyTorch,
Transformers, NumPy, and the OpenAI-compatible client. Do not run P3 from the
Windows planning machine.

```bash
export PROJECT_ROOT=/data/aaai/safe_sca_replication
export BASE_ROOT="$PROJECT_ROOT/g123_augmentation"
export P3_ROOT="$PROJECT_ROOT/P3_TRANSFER_EXPERIMENT_v1"
export P3_RESULTS_ROOT="$PROJECT_ROOT/v2_results_p3"
export P3_OUT="$P3_RESULTS_ROOT/exp_p3_transfer_test"
export P3_PROTOCOL="$P3_ROOT/protocols/p3_frozen_protocol.json"
export PYTHON_BIN=/data/venvs/safe-sca/bin/python
```

## 2. Offline checks before any GPU generation

```bash
test -f "$BASE_ROOT/code/run_experiment_local.py"
test -f "$P3_PROTOCOL"
"$PYTHON_BIN" -m py_compile "$P3_ROOT/code"/*.py
PYTHONPATH="$P3_ROOT/code" "$PYTHON_BIN" -m unittest discover \
  -s "$P3_ROOT/code/tests" -v
"$PYTHON_BIN" "$P3_ROOT/code/run_p3_campaign.py" --help
"$PYTHON_BIN" "$P3_ROOT/code/analyze_p3_transfer.py" --help
```

Expected: all unit tests pass. The test suite uses only synthetic metrics and
does not load a model or contact vLLM.

## 3. Start revision-pinned vLLM servers

The S1 revision cannot be recovered. P3 intentionally matches the revision
pins recorded for S2; it is therefore a cross-environment transfer test.

```bash
chmod +x "$P3_ROOT/server_scripts/start_vllm_p3.sh"
P3_ROOT="$P3_ROOT" PYTHON_BIN="$PYTHON_BIN" \
  "$P3_ROOT/server_scripts/start_vllm_p3.sh"

curl --silent --fail http://localhost:8000/v1/models
curl --silent --fail http://localhost:8001/v1/models
cat "$P3_ROOT/logs/vllm_p3/pids_and_revisions.env"
```

The script refuses to reuse an occupied port. Do not kill an unknown server or
silently mix a non-pinned server into P3.

## 4. Strict preflight and immutable dry run

The campaign refuses to start until the strict S1/S2 environment preflight
passes for this exact output directory.

```bash
"$PYTHON_BIN" "$BASE_ROOT/code/preflight_s1.py" --out-dir "$P3_OUT"

"$PYTHON_BIN" "$P3_ROOT/code/run_p3_campaign.py" \
  --base-root "$BASE_ROOT" --protocol "$P3_PROTOCOL" \
  --results-root "$P3_RESULTS_ROOT" --dry-run
```

Check the dry-run output before proceeding: it must show 320 cells, 80
matrix-seed tasks, 32 workers, the frozen protocol hash, and the matrix
registry hash. If any value differs, stop; do not edit the protocol after the
preflight directory was created.

## 5. Launch P3

```bash
"$PYTHON_BIN" "$P3_ROOT/code/run_p3_campaign.py" \
  --base-root "$BASE_ROOT" --protocol "$P3_PROTOCOL" \
  --results-root "$P3_RESULTS_ROOT"
```

Each task is one `(matrix, seed)` run containing all four policy cells in a
seed-keyed Latin-square order. The command resumes only wholly missing tasks;
it never overwrites a metrics file. A task gets one initial attempt plus at
most two retries. The 32-worker topology is frozen because it is part of the
cross-environment P3 execution condition.

Useful monitoring commands:

```bash
tail -f "$P3_OUT/logs_campaign"/*.log
find "$P3_OUT" -path '*/metrics.json' | wc -l
find "$P3_OUT" -path '*/het_safe_sca/decision.json' | wc -l
cat "$P3_OUT/P3_CAMPAIGN_EXECUTION_REPORT.json"
```

Expected final counts: 320 metrics and 80 Safe-SCA decision files. Runtime is
hardware/load dependent; compared with S2, P3 has fewer tasks and no
four-agent Public-Goods tail, but do not treat a timing estimate as a timeout
override.

## 6. Stop conditions and recovery

Stop the campaign and preserve logs if:

- a task exhausts its retries;
- GPU OOM, model-server restart, or endpoint identity issue occurs;
- preflight/config/registry validation fails; or
- a human proposes changing models, revisions, prompts, protocol values,
  matrix registry, workers, or timeout after launch.

For an ordinary process interruption, rerun the **same** launch command; it
resumes completed tasks. Do not use `--force`, do not delete failed metrics,
and do not run selected matrices with a different configuration.

## 7. Validate before analysis

```bash
"$PYTHON_BIN" "$P3_ROOT/code/validate_p3_results.py" \
  --results "$P3_OUT" --protocol "$P3_PROTOCOL"
```

This writes `P3_INTEGRITY_REPORT.json`. It must show exactly 320 checked
metrics, zero missing, zero errors, and `ready_for_analysis=true`.

## 8. Run the frozen analysis

```bash
"$PYTHON_BIN" "$P3_ROOT/code/analyze_p3_transfer.py" \
  --results "$P3_OUT" --protocol "$P3_PROTOCOL"
```

The script writes `p3_transfer_summary.json` and
`p3_transfer_summary.md`, then returns non-zero if P3 fails any gate. This is
intentional: a failed P3 means **no unseen-matrix generalization claim**, not
that S1/S2 should be discarded.

P3 passes only if all four anti matrices satisfy the `−0.10` paired-CI safety
margin, anti false-align is `0/40`, and at least two positive-Gated
coordination/boundary matrices recover at least 30% of the Gated gain.

## 9. Archive exact artifacts

After validation and analysis, preserve the entire `exp_p3_transfer_test`
directory, the P3 package, vLLM revision log, and the server logs. A portable
archive can be created without modifying results:

```bash
cd "$P3_RESULTS_ROOT"
zip -r "P3_TRANSFER_RESULTS_v1.zip" "exp_p3_transfer_test"
```

The paper must report each matrix separately and must retain the matrix
registry, protocol snapshot, campaign report, integrity report, and every
decision record in its supplement.
