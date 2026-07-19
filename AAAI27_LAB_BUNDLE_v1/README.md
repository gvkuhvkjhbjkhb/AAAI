# AAAI27 Lab Bundle v1

Frozen experiment **data + protocol (方案)** for the AAAI 2027 submission, assembled on the lab machine from the four source archives produced by the supplement and P3-transfer campaigns.

## Layout

| Path | Contents |
| --- | --- |
| `experiment_protocol/` | Frozen protocols, runbooks (CN + EN), package QA, README-first |
| `code/supplement/` | Supplement campaign code: `safe_sca.py`, `hettom_baseline.py`, `run_supplement_campaign.py`, `run_supplement_task.py`, `analyze_supplement_results.py`, `validate_supplement_results.py`, `p3_*` |
| `code/p3_transfer/` | P3 transfer code: `run_p3_campaign.py`, `run_p3_matrix.py`, `analyze_p3_transfer.py`, `validate_p3_results.py`, `p3_protocol.py`, `tests/` |
| `server_scripts/` | vLLM launchers (`start_vllm_supplement.sh`, `start_vllm_p3.sh`, wrapper), `install_safe_sca_env.sh`, frozen requirements |
| `results/supplement_v1/` | Supplement results: `p0_payoff_prompt`, `p1_label_swap`, `p2_teammean_bandit_p3` + campaign snapshot/execution report + analysis |
| `results/p3_transfer/` | P3 transfer results: `p3_m01`..`p3_m08` + campaign snapshot/execution report + summary |
| `analysis/` | Convenience copies of the key analysis outputs (CSV/JSON/MD) |

## Integrity (machine-verified)

- **Supplement** — `results/supplement_v1/SUPPLEMENT_INTEGRITY_REPORT.json`: 640/640 metrics checked (p0/p3 = 320, p1/p3 = 240, p2/p3 = 80), 0 missing, 0 errors, `ready_for_analysis: true`.
- **P3 transfer** — `results/p3_transfer/P3_INTEGRITY_REPORT.json`: 320/320 cells checked, 0 missing, 0 errors, `ready_for_analysis: true`.

## Provenance

Assembled by `/data/lab/build_bundle.py` from:
- `AAAI27_SUPPLEMENTAL_RESULTS_v1.zip`
- `P3_TRANSFER_RESULTS_v1.zip`
- `supplement_package.zip` (supplement code + protocol)
- `P3_TRANSFER_EXPERIMENT_v1.zip` (P3 code + protocol)
- `analyze_supplement_results_fixed.py`, `validate_supplement_results_fixed.py` (overwrite the shipped versions)

See `experiment_protocol/README_FIRST.md` for the recommended reading order and `experiment_protocol/NEW_MACHINE_RUNBOOK_CN.md` / `P3_NEW_MACHINE_RUNBOOK.md` for reproducing on a fresh machine.
