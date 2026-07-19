# AAAI-27 补充实验包 V1

本包把投稿前仍需补的实验分成三个互不混淆的问题，并提供冻结协议、完整运行器、并发编排、完整性校验和统计分析代码。

## 先做什么

1. **P0（必做，最高优先级）**：在原 P3 的 8×10 同一网格中，同时重跑 NoAlign、Always-Gated、Safe-SCA 与 payoff-in-prompt。P0 的核心对比 `payoff-in-prompt - NoAlign` 只改变 prompt 是否展示完整收益矩阵，是最干净的单因素对照；Safe-SCA 同批重跑提供直接外部基线比较。
2. **P1（强烈建议）**：保持 P3 收益张量和种子不变，只交换两个不透明 action label 与 action index 的文字映射，检验路由和策略效应是否依赖标签表面。
3. **P2（建议）**：用 K=5/arm 的在线探测选择 NoAlign 或 Gated；选择依据和最终报告统一使用 `team_mean_payoff`，主结果包含 10 个探测 episode 的成本。

主块共 **640 个新 cell**：P0 320、P1 240、P2-P3 80。可选的 P2-S2 源网格另有 120 个 cell。

## 文件导航

- `EXPERIMENT_PROTOCOL_CN.md`：科学问题、假设、变量、estimand、统计规则和论文解释边界。
- `NEW_MACHINE_RUNBOOK_CN.md`：从空机器到结果归档的逐条命令。
- `PAPER_UPDATE_TEMPLATE_CN.md`：不同实验结果对应的论文回填模板。
- `protocols/supplement_frozen_protocol.json`：唯一允许执行的冻结协议。
- `code/run_supplement_campaign.py`：32 worker、可恢复、不可覆盖的总编排器。
- `code/run_supplement_task.py`：单个 matrix/game-seed 任务运行器。
- `code/validate_supplement_results.py`：只做完整性检查，不看结论。
- `code/analyze_supplement_results.py`：冻结统计分析和表格导出。
- `server_scripts/start_vllm_supplement.sh`：revision-pinned 双 vLLM 服务。

现有 P3 结果不会被修改。本包写入一个全新的 `results-root`；分析时只读原始 `exp_p3_transfer_test` 作为时序敏感性和 P1/P2 参照。
