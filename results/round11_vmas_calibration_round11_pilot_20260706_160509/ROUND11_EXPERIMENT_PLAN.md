# Round 11 VMAS Reward-Scale Calibration

Goal: implement Option 3 by testing whether Round 9's VMAS weakness was caused by transferring the LBF-tuned 0.0003 shaping scale to a dense VMAS reward landscape.

Environment: vmas-navigation
Timesteps: 300000
Seeds: 1 2 3
Penalties: 0.00001 0.00003 0.0001 0.0003
Methods: baseline adaptive uniform random

Decision rule:
- Expand the best penalty to more seeds only if adaptive is positive against baseline and not clearly worse than random-type budget matching.
- If random-type or uniform dominates across penalties, keep VMAS as a transparent reward-scale limitation and do not use it as a main generalization result.
