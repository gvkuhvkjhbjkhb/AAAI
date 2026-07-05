# Phase 2A/2B Diagnosis Validation Corrected

The first auto-generated report mapped source lines incorrectly after converting the validation set to JSONL. This corrected report maps JSONL line numbers back to `sample_id`.

Human-human n=300 raw_agreement=0.6367 cohen_kappa=0.4322

annotator_a_counts={'target_miscoordination': 19, 'inefficient_exploration': 114, 'low_value_overcommitment': 128, 'insufficient_cooperation': 39}

annotator_b_counts={'target_miscoordination': 28, 'low_value_overcommitment': 228, 'inefficient_exploration': 13, 'insufficient_cooperation': 31}

| classifier | n | accuracy | macro-F1 | kappa | note |
|---|---:|---:|---:|---:|---|
| original_heuristic | 300 | 0.4867 | 0.2497 | 0.2062 | original repository heuristic |
| enhanced_heuristic | 300 | 0.3433 | 0.2038 | 0.1060 | enhanced rule heuristic |
| qwen35_4b_completed_138 | 138 | 0.0580 | 0.0224 | 0.0051 | Qwen records completed before API timeout |
| qwen35_4b_timeout_missing_as_unknown | 300 | 0.0267 | 0.0158 | -0.0363 | full 300 with timed-out missing records counted as unknown |

## Counts and Confusion

### original_heuristic

gold_counts={'target_miscoordination': 19, 'inefficient_exploration': 114, 'low_value_overcommitment': 128, 'insufficient_cooperation': 39}

pred_counts={'inefficient_exploration': 268, 'insufficient_cooperation': 32}

confusion={'target_miscoordination': {'inefficient_exploration': 19}, 'inefficient_exploration': {'inefficient_exploration': 114}, 'low_value_overcommitment': {'inefficient_exploration': 128}, 'insufficient_cooperation': {'inefficient_exploration': 7, 'insufficient_cooperation': 32}}

per_class_f1={'target_miscoordination': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 19}, 'insufficient_cooperation': {'precision': 1.0, 'recall': 0.8205128205128205, 'f1': 0.9014084507042254, 'support': 39}, 'inefficient_exploration': {'precision': 0.4253731343283582, 'recall': 1.0, 'f1': 0.5968586387434555, 'support': 114}, 'low_value_overcommitment': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 128}, 'timeout_near_success': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}, 'unknown': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}}

### enhanced_heuristic

gold_counts={'target_miscoordination': 19, 'inefficient_exploration': 114, 'low_value_overcommitment': 128, 'insufficient_cooperation': 39}

pred_counts={'insufficient_cooperation': 126, 'inefficient_exploration': 165, 'target_miscoordination': 9}

confusion={'target_miscoordination': {'insufficient_cooperation': 4, 'target_miscoordination': 5, 'inefficient_exploration': 10}, 'inefficient_exploration': {'insufficient_cooperation': 49, 'inefficient_exploration': 65}, 'low_value_overcommitment': {'insufficient_cooperation': 40, 'inefficient_exploration': 88}, 'insufficient_cooperation': {'insufficient_cooperation': 33, 'inefficient_exploration': 2, 'target_miscoordination': 4}}

per_class_f1={'target_miscoordination': {'precision': 0.5555555555555556, 'recall': 0.2631578947368421, 'f1': 0.35714285714285715, 'support': 19}, 'insufficient_cooperation': {'precision': 0.2619047619047619, 'recall': 0.8461538461538461, 'f1': 0.4, 'support': 39}, 'inefficient_exploration': {'precision': 0.3939393939393939, 'recall': 0.5701754385964912, 'f1': 0.4659498207885305, 'support': 114}, 'low_value_overcommitment': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 128}, 'timeout_near_success': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}, 'unknown': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}}

### qwen35_4b_completed_138

gold_counts={'target_miscoordination': 16, 'inefficient_exploration': 62, 'low_value_overcommitment': 53, 'insufficient_cooperation': 7}

pred_counts={'insufficient_cooperation': 137, 'low_value_overcommitment': 1}

confusion={'target_miscoordination': {'insufficient_cooperation': 16}, 'inefficient_exploration': {'insufficient_cooperation': 62}, 'low_value_overcommitment': {'insufficient_cooperation': 52, 'low_value_overcommitment': 1}, 'insufficient_cooperation': {'insufficient_cooperation': 7}}

per_class_f1={'target_miscoordination': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 16}, 'insufficient_cooperation': {'precision': 0.051094890510948905, 'recall': 1.0, 'f1': 0.09722222222222222, 'support': 7}, 'inefficient_exploration': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 62}, 'low_value_overcommitment': {'precision': 1.0, 'recall': 0.018867924528301886, 'f1': 0.037037037037037035, 'support': 53}, 'timeout_near_success': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}, 'unknown': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}}

### qwen35_4b_timeout_missing_as_unknown

gold_counts={'target_miscoordination': 19, 'inefficient_exploration': 114, 'low_value_overcommitment': 128, 'insufficient_cooperation': 39}

pred_counts={'insufficient_cooperation': 137, 'low_value_overcommitment': 1, 'unknown': 162}

confusion={'target_miscoordination': {'insufficient_cooperation': 16, 'unknown': 3}, 'inefficient_exploration': {'insufficient_cooperation': 62, 'unknown': 52}, 'low_value_overcommitment': {'insufficient_cooperation': 52, 'low_value_overcommitment': 1, 'unknown': 75}, 'insufficient_cooperation': {'insufficient_cooperation': 7, 'unknown': 32}}

per_class_f1={'target_miscoordination': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 19}, 'insufficient_cooperation': {'precision': 0.051094890510948905, 'recall': 0.1794871794871795, 'f1': 0.07954545454545456, 'support': 39}, 'inefficient_exploration': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 114}, 'low_value_overcommitment': {'precision': 1.0, 'recall': 0.0078125, 'f1': 0.015503875968992248, 'support': 128}, 'timeout_near_success': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}, 'unknown': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}}

