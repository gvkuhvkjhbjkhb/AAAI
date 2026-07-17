# Round 12 Simple Cooperative Domains

Goal: find simple sparse cooperative domains likely to yield positive evidence without relying on VMAS/RWARE/Qwen.

Domains:
- smallcoop: lbforaging:Foraging-8x8-2p-2f-coop-v3, seeds 1 2 3 4 5 6 7 8, tmax 300000.
- maincoop: lbforaging:Foraging-10x10-3p-3f-coop-v3, seeds 1 2 3 4 5 6 7 8, tmax 300000.
- scale15 optional: lbforaging:Foraging-15x15-3p-4f-v3, seeds 1 2 3 4, tmax 300000.

Methods: baseline uniform_budget_matched_0.0003_late045 adaptive_0.0003_late045 random_type_budget_matched_0.0003_late045
