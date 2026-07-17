# Round 6 AAAI Submission Experiment Summary

## Status

Round 6 completed on the Lab machine before the Lab connection dropped. The run executed 105 planned jobs with two NVIDIA RTX 5090 GPUs and reported 105 completed runs with zero failed runs. The completed Lab artifact reported by the run script is `artifacts/AAAI_round6_aaai_submission_20260702_134358.tar.gz`, and the result directory reported by the run script is `results/round6_aaai_submission_20260702_134358`.

The Lab connection became unavailable before the repository could be committed and pushed from the Lab machine. Therefore this file records the verified console-reported metrics and the required follow-up actions, but the actual Lab artifact must be pushed after reconnecting to Lab.

## Experiment Matrix

The main experiment used `lbforaging:Foraging-10x10-3p-3f-v3`, 500000 environment timesteps, and seeds 1-8. It compared baseline MAPPO, diagnosis-only logging, calibrated uniform penalties at 0.0002 and 0.0003, type-specific penalties at 0.0002 and 0.0003, adaptive schedules at 0.0002 with late weights 0.45 and 0.60, adaptive 0.0003 with late weight 0.45, and random-type 0.0002. The generalization experiment used `lbforaging:Foraging-12x12-3p-4f-v3`, 500000 environment timesteps, and seeds 1-5. It compared baseline, uniform 0.0002, uniform 0.0003, adaptive 0.0002 late 0.45, and adaptive 0.0002 late 0.60.

## Verified Aggregate Results

The strongest 10x10 method was `adaptive_0.0003_late045` among the semantic/adaptive methods. It achieved mean final test return 0.3042 and train AUC 0.2263, compared with baseline final test return 0.2258 and train AUC 0.1827. It also exceeded `uniform_0.0003`, which achieved mean final test return 0.2892 and train AUC 0.2092. This supports a defensible primary-task claim that a late-preserving adaptive schedule improves both final return and sample efficiency over MAPPO and over a calibrated uniform shaping baseline.

The strongest overall 10x10 result was the random-type control `random_type_0.0002`, which achieved mean final test return 0.3048 and train AUC 0.2308. This creates an important reviewer-facing limitation: the present Round 6 evidence does not yet prove that semantic failure labels are the causal source of the gain. The safest paper framing is that failure-triggered adaptive reward shaping is effective, while semantic diagnosis requires additional validation and matched random-control experiments.

The 12x12 generalization task showed weak transfer. Baseline achieved mean final test return 0.1030 and train AUC 0.0888. `uniform_0.0002` achieved the best mean final test return at 0.1093 and train AUC 0.0911. Both adaptive 0.0002 schedules achieved mean final test return 0.1034 and train AUC 0.0893. This supports only a conservative generalization statement: the method transfers without catastrophic degradation, but current adaptive settings do not yet deliver a strong 12x12 improvement.

## Method-Level Results Recorded From Lab Output

| phase | method | n | mean last test | mean train AUC |
|---|---|---:|---:|---:|
| main10x10 | baseline | 8 | 0.2258 | 0.1827 |
| main10x10 | diagnosis_only | 8 | 0.2258 | 0.1827 |
| main10x10 | uniform_0.0002 | 8 | 0.2680 | 0.2095 |
| main10x10 | uniform_0.0003 | 8 | 0.2892 | 0.2092 |
| main10x10 | type_specific_0.0002 | 8 | 0.2514 | 0.1981 |
| main10x10 | type_specific_0.0003 | 8 | 0.2755 | 0.2072 |
| main10x10 | adaptive_0.0002_late045 | 8 | 0.2435 | 0.1900 |
| main10x10 | adaptive_0.0002_late060 | 8 | 0.2432 | 0.1899 |
| main10x10 | adaptive_0.0003_late045 | 8 | 0.3042 | 0.2263 |
| main10x10 | random_type_0.0002 | 8 | 0.3048 | 0.2308 |
| generalization | baseline | 5 | 0.1030 | 0.0888 |
| generalization | uniform_0.0002 | 5 | 0.1093 | 0.0911 |
| generalization | uniform_0.0003 | 5 | 0.1073 | 0.0887 |
| generalization | adaptive_0.0002_late045 | 5 | 0.1034 | 0.0893 |
| generalization | adaptive_0.0002_late060 | 5 | 0.1034 | 0.0893 |

## Conclusions

Round 6 moves the project substantially closer to an AAAI-submission-level empirical package because it provides eight-seed evidence on the primary 10x10 task and includes calibrated uniform and random-type controls. The main positive result is that `adaptive_0.0003_late045` improves mean final test return by about 34.7 percent relative to baseline and about 5.2 percent relative to `uniform_0.0003`, while improving train AUC by about 23.9 percent relative to baseline and about 8.2 percent relative to `uniform_0.0003`. The diagnosis-only condition matches baseline, which is useful because it shows the logging and diagnosis pipeline does not alter learning unless reward intervention is enabled.

The main limitation is that `random_type_0.0002` is marginally stronger than the best adaptive semantic method. This prevents a strong causal claim that semantic failure labels are already better than randomized type assignment. For AAAI, the current evidence should be framed as calibrated failure-triggered adaptive shaping with diagnosis-conditioned mechanisms, not as proof that the LLM-derived semantics are independently sufficient. A reviewer will likely ask why random types match adaptive semantics, so the next round must directly address this question.

The generalization result is not yet strong. Uniform shaping transfers slightly better than baseline, while adaptive settings are nearly tied with baseline. The paper can include 12x12 as a stress test, but it should not currently claim strong cross-map adaptive generalization. The next round should either improve the adaptive transfer setting or describe 12x12 as a negative/neutral robustness result.

## Immediate Follow-Up

After reconnecting to Lab, push the actual result directory, artifact, run script, and this summary. The Lab-reported files to commit are `run_round6_aaai_two_gpu.sh`, `results/round6_aaai_submission_20260702_134358`, and `artifacts/AAAI_round6_aaai_submission_20260702_134358.tar.gz`.
