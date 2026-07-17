# Round 7 Decisive AAAI Optimization Plan

Round 7 directly tests the Round 6 reviewer risk that random-type controls can match semantic adaptive reward shaping. It adds matched-frequency random-type controls, ordinary random-type controls at the same penalty and late schedule, a confidence-gated semantic method, and a conservative 12x12 completion panel.

The main 10x10 queue contains eight seeds for `random_type_matched_0.0003_late045`, `semantic_gate_0.0003_late045`, `random_type_0.0003_late045`, and `random_type_0.0003_late060`. The matched control preserves the empirical semantic label frequency pool but shuffles failure-label assignment, providing a fairer causal test than ordinary random labels.

The 12x12 queue completes seeds 6--8 for `baseline`, `uniform_0.0002`, `adaptive_0.0003_late045`, and `semantic_gate_0.0003_late045`. These results are interpreted as a stress test unless semantic_gate clearly beats baseline, uniform, and adaptive controls.

The decisive submission rule is conservative. A strong semantic diagnosis claim is allowed only if semantic_gate beats matched random-type, ordinary random-type, adaptive, uniform, and baseline controls on paired final return or train AUC. If matched random remains tied or stronger, the AAAI framing should pivot to calibrated failure-triggered adaptive reward shaping with semantic diagnosis as interpretability and analysis.
