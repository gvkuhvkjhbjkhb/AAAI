# SAFE_SCA_R0_S2_RESULTS_v1

R0 execution replay + S2 independent seed-block replication for the
Coverage-Certified Safe-SCA experiment, run on 2× NVIDIA RTX 5090.

## Contents

```
SAFE_SCA_R0_S2_RESULTS_v1/
├── README.md                          this file
├── protocol/                          dated protocol amendments & appendices
│   ├── R0_S2_PROTOCOL_AMENDMENT.md    primary amendment: route reproduction
│   │                                  as R0 criterion, S2 independent seeds
│   ├── R0_DEVIATION_NOTE.md           (withdrawn) earlier tolerance note
│   ├── SERVER_ENVIRONMENT_APPENDIX.json      V1 env record
│   ├── SERVER_ENVIRONMENT_APPENDIX_V2.json   V2 optimized env record
│   ├── S2_NEW_MACHINE_RUNBOOK.md      original runbook
│   ├── S1_SAFE_SCA_PROTOCOL.md        S1 preregistration
│   └── README_g123_augmentation.md    original g123 README
├── code/                              experiment code (Python + tests)
├── server_scripts/                    vLLM start + orchestration scripts
├── protocols/
│   └── s1_safe_sca_frozen.json        frozen S1 configuration (do not edit)
├── r0_original/                       R0 strict result (PAYOFF FAILED, ROUTE REPRODUCED)
│   ├── R0_STRICT_PAYOFF_FAILED_ROUTE_REPRODUCED   marker
│   ├── R0_REPLAY_COMPARISON.json      strict comparison (55/72 differ, 0/12 routes)
│   ├── ENVIRONMENT_MANIFEST_S1.json
│   ├── CAMPAIGN_EXECUTION_REPORT.json
│   └── <game>/seed_<s>/<cell>/{metrics,decision}.json
├── r0_diagnostic/                     R0 diagnostic replay (revision-pinned)
│   ├── R0_REPLAY_COMPARISON.json      12/12 routes match, max payoff diff 0.16
│   ├── ENVIRONMENT_MANIFEST_S1.json
│   └── <game>/seed_<s>/<cell>/{metrics,decision}.json
├── s2_results/                        S2 independent replication (720/720, PASS)
│   ├── s1_safe_sca_summary.json       full analysis (method_paper_pass=TRUE)
│   ├── s1_safe_sca_summary.md         human-readable summary
│   ├── R0... no — S2_INTEGRITY_REPORT.json
│   ├── CAMPAIGN_EXECUTION_REPORT.json
│   ├── CAMPAIGN_SNAPSHOT.json
│   ├── ENVIRONMENT_MANIFEST_S1.json
│   └── <game>/seed_<s>/<cell>/{metrics,decision,trajectories}.jsonl
├── s1_reference/                      S1 original (for comparison only)
│   ├── s1_safe_sca_summary.json
│   └── S1_EXPERIMENT_REPORT.md
└── logs/                              campaign & analysis logs
```

## Key results

| Experiment | Result |
|---|---|
| R0 strict payoff (tolerance 0.0) | FAILED: 55/72 differ, max \|diff\|=0.20 |
| R0 route reproduction | PASSED: 0/12 mismatches |
| R0 diagnostic (revision-pinned) | PASSED: 12/12 routes match |
| S2 safety gate | PASSED: all 3 anti-games LB ≥ -0.10 |
| S2 utility gate | PASSED: stag_hunt 42.3%, BoS 47.9% recovery |
| S2 method_paper_pass | **TRUE** |
| S2 anti false-align | 0/60 |
| S2 coord false-abstain | 3/60 |

## Classification

S2 is a **cross-environment independent replication**, not a same-environment
replay. S1's vLLM logs were unavailable, so model revision identity with S1
cannot be independently verified. See R0_S2_PROTOCOL_AMENDMENT.md §4.
