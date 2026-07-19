# 实验完成后的论文回填模板

## 摘要必须加入

先报告 P3 的事实，再加 P0 的同网格基线；不要只报告通过的矩阵：

> On the same 8-matrix, 10-seed grid, adding the full payoff table to an otherwise identical NoAlign prompt changed team payoff by Δ=__ (hierarchical 95% CI [__, __]); its regret to the better concurrently rerun fixed arm was __ [__, __].

若 P1 通过：

> Safe-SCA retained __/80 routes under a preregistered action-label permutation, and all normalized policy-effect interactions satisfied the ±0.10 equivalence margin.

若 P1 不通过：

> Action-label permutation preserved __/80 routes but materially moderated policy effects on matrices __; therefore we restrict transfer claims to the tested prompt surfaces.

## 方法新增小节

标题建议：`Matched Payoff-Visibility and Label-Surface Controls`。正文必须写清：

- P0 四臂在同一 P3 matrix-seed context 内运行；
- `PayoffPrompt − NoAlign` 只操纵完整 payoff table 是否进入 prompt；
- 固定臂是同批重跑，旧 P3 只做 temporal sensitivity；
- P1 保持 tensor/index payoff 不变，只交换 action strings；
- P2 的主结果包含 probe 成本；
- 所有分析统一 `team_mean_payoff`。

## 结果表

主文至少增加一张 8 行矩阵表：

| Matrix | PayoffPrompt−NoAlign | PayoffPrompt regret | Safe route label agreement | Gated label interaction | Safe label interaction |
|---|---:|---:|---:|---:|---:|

P2 放一张紧凑表：整体 selection accuracy（Wilson CI）、online gain vs NoAlign、online regret；逐矩阵完整表放附录。

## 允许的最强结论

- P0 正：payoff visibility can improve utility on the matched P3 grid。
- P0 零/负：payoff visibility alone is insufficient to recover utility。
- P1 通过：policy effects are robust to the preregistered label permutation。
- P2 正：small online probes recover some utility without converting the safety certificate into a utility certificate。

始终保留：安全外推和效用外推是两个 estimand；P3 的 policy-arm 效应是 context 内识别的，但 source→P3 的跨网格差异不是 payoff-only 因果效应。
