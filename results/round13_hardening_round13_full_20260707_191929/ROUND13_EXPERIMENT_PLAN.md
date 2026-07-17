# Round 13 AAAI Main-Claim Hardening

Goal: harden the main 10x10 claim, not search for new domains.

Components:
- Seed extension: baseline adaptive_0.0003_late045 uniform_budget_matched_0.0003_late045 random_type_budget_matched_0.0003_late045 on seeds 9 10 11 12 13 14 15 16.
- Actual-budget controls: uniform_actual_budget_matched random_actual_budget_matched on seeds 1 2 3 4 5 6 7 8.
- Sensitivity extension: adaptive_0.0002_late045 adaptive_0.0005_late045 adaptive_0.0003_late060 on seeds 5 6 7 8.

Actual-budget coefficients are calibrated from prior accounting: uniform 0.00024 and random 0.00027, lower than nominal 0.0003, to better match adaptive's realized shaping budget.
