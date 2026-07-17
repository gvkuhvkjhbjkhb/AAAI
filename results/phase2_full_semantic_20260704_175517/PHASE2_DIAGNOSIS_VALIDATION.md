# Phase 2A/2B Diagnosis Validation

Human-human n=300 raw_agreement=0.6367 cohen_kappa=0.4322

| classifier | n | accuracy | macro-F1 | kappa |
|---|---:|---:|---:|---:|
| original_heuristic | 300 | 0.0000 | 0.0000 | 0.0000 |
| enhanced_heuristic | 300 | 0.0000 | 0.0000 | 0.0000 |
| qwen35_4b | 300 | 0.0000 | 0.0000 | 0.0000 |

## Counts and Confusions

### original_heuristic

gold_counts={'target_miscoordination': 19, 'inefficient_exploration': 114, 'low_value_overcommitment': 128, 'insufficient_cooperation': 39}

pred_counts={'unknown': 300}

confusion={'target_miscoordination': {'unknown': 19}, 'insufficient_cooperation': {'unknown': 39}, 'inefficient_exploration': {'unknown': 114}, 'low_value_overcommitment': {'unknown': 128}}

per_class_f1={'target_miscoordination': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 19}, 'insufficient_cooperation': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 39}, 'inefficient_exploration': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 114}, 'low_value_overcommitment': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 128}, 'timeout_near_success': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}, 'unknown': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}}

### enhanced_heuristic

gold_counts={'target_miscoordination': 19, 'inefficient_exploration': 114, 'low_value_overcommitment': 128, 'insufficient_cooperation': 39}

pred_counts={'unknown': 300}

confusion={'target_miscoordination': {'unknown': 19}, 'insufficient_cooperation': {'unknown': 39}, 'inefficient_exploration': {'unknown': 114}, 'low_value_overcommitment': {'unknown': 128}}

per_class_f1={'target_miscoordination': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 19}, 'insufficient_cooperation': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 39}, 'inefficient_exploration': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 114}, 'low_value_overcommitment': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 128}, 'timeout_near_success': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}, 'unknown': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}}

### qwen35_4b

gold_counts={'target_miscoordination': 19, 'inefficient_exploration': 114, 'low_value_overcommitment': 128, 'insufficient_cooperation': 39}

pred_counts={'unknown': 300}

confusion={'target_miscoordination': {'unknown': 19}, 'insufficient_cooperation': {'unknown': 39}, 'inefficient_exploration': {'unknown': 114}, 'low_value_overcommitment': {'unknown': 128}}

per_class_f1={'target_miscoordination': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 19}, 'insufficient_cooperation': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 39}, 'inefficient_exploration': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 114}, 'low_value_overcommitment': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 128}, 'timeout_near_success': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}, 'unknown': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}}

