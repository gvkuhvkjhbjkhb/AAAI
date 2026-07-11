#!/usr/bin/env python3
"""
GSACA: Game-Structure-Adaptive Conditional Alignment — Launcher.
Calls run_round4.py with adaptive gating modes.
Uses GameStructureEstimator + AdaptiveAlignmentPolicy + EquilibriumSelectionTracker
to auto-detect game structure and pick the best alignment strategy.

Design documented in FINAL_REPORT.md (§6: GSACA upgrade)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print("[GSACA] Use 'launch_gsaca.py' for the optimizer entry point")
