# AAAI Experimental Conclusion After Rounds 8-14

The full hardening sequence is complete. The experiments improve the credibility of the project by preventing an overclaim, but they do not support an AAAI-level positive method paper in the current form.

## Completed Evidence

- Round 8: strong positive 8-seed result for `adaptive_0.0003_late045` on 10x10 LBF.
- Round 9: cross-domain and larger LBF tests were mixed or negative.
- Round 10: budget accounting showed adaptive used less shaping budget, but LBF-family stress tests were only directional.
- Round 11: VMAS calibration did not produce robust transfer.
- Round 12: simple cooperative domains did not produce useful positive evidence.
- Round 13: the main 10x10 result failed under seeds 9-16 and merged seeds 1-16.
- Round 14: the late060 salvage variant also failed to beat baseline or matched controls at 16 seeds.

## Final Main-Task Numbers

| method | n | final return | train AUC | interpretation |
|---|---:|---:|---:|---|
| baseline | 16 | 0.2460 | 0.1899 | reference |
| adaptive_0.0003_late045 | 16 | 0.2341 | 0.1838 | original adaptive, not robust |
| adaptive_0.0003_late060 | 16 | 0.2325 | 0.1831 | salvage adaptive, not robust |
| uniform_budget_matched_0.0003_late060 | 16 | 0.2346 | 0.1847 | strong matched control |
| random_type_budget_matched_0.0003_late060 | 16 | 0.2259 | 0.1799 | strong matched control |

## Recommendation

Do not submit the current manuscript as a positive AAAI method paper. A credible AAAI path now requires either: (1) redesigning the adaptive intervention and rerunning a preregistered 16-seed evaluation with the current controls, or (2) reframing the paper as a careful negative/mixed empirical study showing why adaptive reward shaping can appear promising under small seed counts but fail under stronger controls and larger seed sets.