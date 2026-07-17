# Round 10 Options 1+2

Goal: implement the safest AAAI package after Round 8/9: mechanism defense plus LBF-family generalization.

Option 1 components:
- Sensitivity on lbforaging:Foraging-10x10-3p-3f-v3 with seeds 1 2 3 4 and methods adaptive_0.0002_late045 adaptive_0.0005_late045 adaptive_0.0003_late060 uniform_budget_matched_0.0003_late045 random_type_budget_matched_0.0003_late045.
- Budget accounting logs: records, shaping triggers, penalty total, terminal bonus total, shaped episode steps, average penalty per trigger.

Option 2 components:
- New LBF-family task lbforaging:Foraging-10x10-4p-4f-v3 with seeds 1 2 3 4 5 6 7 8 and methods baseline uniform_budget_matched_0.0003_late045 adaptive_0.0003_late045 random_type_budget_matched_0.0003_late045.

Decision use:
- Main claim remains failure-triggered adaptive reward shaping for sparse cooperative foraging.
- New LBF evidence is used only if adaptive is positive against baseline and competitive with budget-matched/random controls.
