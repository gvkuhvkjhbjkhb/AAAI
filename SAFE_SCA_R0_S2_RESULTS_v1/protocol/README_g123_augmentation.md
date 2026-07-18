# G1/G2/G3 Augmentation Experiments

This directory holds the G1/G2/G3 follow-up experiments run on 2026-07-17 and
the S1 Coverage-Certified Safe-SCA protocol added after the AAAI pivot review.
They build on the same stack-B
config (Qwen2.5-7B + GLM-4-9B, bf16 vLLM, 6 games × 20 seeds) as
`v2_results/exp_vllm_paradox_attainment_v1/`.

All artifacts are self-contained here and do **not** modify any other directory
in the repository.

## Contents

```
g123_augmentation/
├── README.md                              this file
├── S1_SAFE_SCA_PROTOCOL.md                preregistered S1 protocol and commands
├── code/                                  G1/G2/G3 runner + analysis (new/modified files)
│   ├── run_g123.py                        unified parallel driver (G1+G2+G3 concurrently)
│   ├── run_experiment_local.py            patched: +het_bandit cell, +top_p arg, +auto_episodes
│   ├── hettom_baseline.py                 patched: +top_p plumbing (G2 ablation)
│   ├── start_vllm.sh                      patched: +enforce-eager, +5090 (sm_120) flashinfer disable
│   ├── analyze_g1_bandit.py               G1 analysis vs ground_truth_paradox.json
│   ├── analyze_g2_topp1.py                G2 top_p=1.0 ablation analysis
│   └── analyze_g3_detect.py               G3 stack-B detection analysis
│   └── safe_sca.py                        S1 label-free coverage certificate
│   └── run_s1_safe_sca.py                 S1 two-stage launcher
│   └── select_s1_config.py                development-only threshold freezer
│   └── analyze_s1_safe_sca.py             held-out S1 analysis
│   └── preflight_s1.py                    environment/endpoint manifest checker
│   └── validate_s1_results.py             completeness/provenance gate
└── v2_results/
    ├── G123_RESULTS.md                    consolidated results + verdicts (start here)
    ├── CONFIG_SNAPSHOT_G123.txt           frozen protocol (frozen before run)
    ├── ground_truth_paradox.json          per (game,seed) NoToM/Gated payoffs, oracle, SCA (from exp_vllm_paradox_attainment_v1)
    ├── trajectories.tar.gz                all 210 per-cell trajectories.jsonl (gzipped; extract with `tar xzf`)
    ├── exp_g1_bandit/                     G1: het_bandit, 6 games × 20 seeds = 120 cells
    │   ├── */seed_*/*/metrics.json        per-cell metrics (bandit_chosen_arm, probe means, commit payoff)
    │   └── g1_bandit_summary.{md,json}    analysis output
    ├── exp_g2_topp1/                      G2: NoToM+Gated, 3 anti-coord × 10 seeds = 60 cells, top_p=1.0
    │   ├── */seed_*/*/metrics.json
    │   └── g2_topp1_summary.{md,json}
    └── exp_g3_detect/                     G3: het_gsaca, 6 games × 5 seeds = 30 cells
        ├── */seed_*/*/metrics.json
        └── g3_detect_summary.{md,json}
```

## How to reproduce

```bash
# 1. start the two bf16 vLLM servers (one model per 5090 GPU)
bash g123_augmentation/code/start_vllm.sh

# 2. run all three experiments concurrently (20 workers, ~55 min on 2×5090)
cd /path/to/repo && python3 g123_augmentation/code/run_g123.py

# 3. analyze
python3 g123_augmentation/code/analyze_g1_bandit.py \
    --bandit g123_augmentation/v2_results/exp_g1_bandit \
    --gt    g123_augmentation/v2_results/ground_truth_paradox.json
python3 g123_augmentation/code/analyze_g2_topp1.py \
    --topp1 g123_augmentation/v2_results/exp_g2_topp1 \
    --gt    g123_augmentation/v2_results/ground_truth_paradox.json
python3 g123_augmentation/code/analyze_g3_detect.py \
    --detect g123_augmentation/v2_results/exp_g3_detect
```

## Headline results (see G123_RESULTS.md for full detail)

| Exp | Question | Result | Verdict |
|---|---|---|---|
| G1 | Can the offline bandit (110/120, 2.49 vs SCA 2.23) be reproduced end-to-end? | accuracy 48.3% (58/120); commit mean 2.333 ≥ SCA 2.227 | PARTIAL — offline is an upper bound; probe/commit decoupling (Gated EMA warmup) halves accuracy |
| G2 | Is the anti-coord flip a top_p=0.9 truncation artifact? | chicken flips at both 0.9 and 1.0; deadlock/hawk_dove never flip | SAMPLING-ROBUST — flip attributable to precision/template, not sampling |
| G3 | Does the GSACA detector hold on stack B (Prop 1)? | 22/30 (73%); misses where warm-up profiles are low-diversity | Prop 1 precondition empirically fails; §6.6 boundary now demonstrated |

## S1: Coverage-Certified Safe-SCA (completed)

The preregistered S1 result is complete: 720/720 held-out cells, integrity
validation passed, and the frozen safety-plus-utility gate returned
`method_paper_pass=true`. The result archive is
`g123_augmentation_s1_results.zip`; its frozen configuration is copied here as
[`protocols/s1_safe_sca_frozen.json`](protocols/s1_safe_sca_frozen.json).

The next work is not threshold tuning. It is an R0 same-seed execution replay
followed by an S2 independent seed-block replication. See
[`S2_NEW_MACHINE_RUNBOOK.md`](S2_NEW_MACHINE_RUNBOOK.md) for the complete
new-machine procedure and [`../../AAAI_PIVOT_OUTLINE_AND_PLAN_V3_S1_VALIDATED.md`](../../AAAI_PIVOT_OUTLINE_AND_PLAN_V3_S1_VALIDATED.md)
for the paper decision tree. The original S1 preregistration remains in
[`S1_SAFE_SCA_PROTOCOL.md`](S1_SAFE_SCA_PROTOCOL.md).
