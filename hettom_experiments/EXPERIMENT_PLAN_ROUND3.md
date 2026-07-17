# HetToM Round-3 实验方案：门控信号-信念仲裁 + 改进 A-ToM

> 生成时间：2026-07-09
> 基于 Round-2 结果：cheap-talk 是最强对齐机制（het_notom_talk payoff +1.165, p=0.024, r=+1.0），但 ToM+talk 朴素组合失败（het_tom_talk 0.890 < het_tom 1.259），A-ToM 单独无效果（het_atom - het_tom = +0.001, p=1.0）。
> 关联代码：`hettom_baseline.py`（gated_talk_tom 标志 + 改进 A-ToM bandit）、`analyze_layer1.py`（round-3 判定）、`run_round3.py`

---

## 0  Round-2 暴露的两个问题

| 问题 | Round-2 数据 | 根因 |
|---|---|---|
| A. ToM+talk 朴素组合失败 | het_tom_talk 0.890 < het_tom 1.259 | 把 ToM 预测和 cheap-talk 信号都塞进 prompt，无仲裁；信号与信念冲突时相互干扰，重新引入不稳定性 |
| B. A-ToM 无效果 | het_atom - het_tom = +0.001, p=1.0, r=0 | (i) 每格仅 20 episode，历史太短；(ii) 规则过粗（hit rate<0.4 加深 / >0.75 降低），在 2 动作博弈中随机基线=0.5，<0.4 几乎不触发、>0.75 罕见，且只能在 1↔3 间振荡，无法学到稳定的最优阶数 |

## 1  对症方案

### 方案 A：门控 talk+ToM 仲裁（het_gated_talk_tom）—— 解决问题 A
- **机制**：cheap-talk 先产生公开意图（signal）；ToM 只判断该信号是否可信；可信则跟随 signal，不可信则回退到 ToM 预测。
  - 信任条件（满足其一即信任信号）：(a) 一致性——信号与 ToM 信念相同；(b) 可靠性——该队友历史 signal-action 匹配 EMA ≥ 阈值（默认 0.6）。
  - 仲裁后只把**单一门控信念**喂给动作 prompt（而非同时塞入信号与信念），消除干扰。
- **文献依据**：Madmoun & Lahlou 2025（cheap-talk 在 Stag Hunt 提升 0%→96.7% 合作）证明 cheap-talk 有效；但 El Mir, Takáč & Lahlou 2026《Byzantine Cheap Talk》(arXiv:2606.07790) 与 Yao, Zou & Hawkins 2026《Talk is Cheap, Communication is Hard》(arXiv:2605.01750) 指出 cheap-talk 信号可能不可靠、需要修复机制。门控仲裁正是用 ToM 信念作为信号的信任校验，是二者的自然综合。
- **对照设计（受控变量）**：
  - vs `het_tom`：唯一差别是加入仲裁（信号源）
  - vs `het_tom_talk`：唯一差别是把"双源塞入"换成"单一门控信念"——直接检验仲裁是否优于朴素组合

### 方案 B：改进 A-ToM（per-order bandit）—— 解决问题 B
- **旧规则缺陷**：`<0.4 加深 / >0.75 降低` + 仅看最近 10 次。在 2 动作博弈（随机基线 0.5）中：<0.4 几乎不触发（需差于随机），>0.75 罕见；估计仅能在 1↔3 振荡，不建模"哪个阶数最优"；20 episode×5=100 轮但每轮只记 1 个命中，数据稀疏。
- **新设计**：contextual bandit over ToM orders {1,2,3}。
  - 对每个 (队友, 阶数) 维护命中率的 EMA（用**全部**历史，非仅最近 10，短历史也稳定）。
  - 选择：epsilon-greedy（默认 ε=0.15）+ warmup（每阶数先采样 ≥3 次再利用）。
  - **关键加速**：每轮把链中**所有**阶数（1..选中阶数）的预测都对实际动作打分（旧规则只记选中那一个），单轮获得多阶数据，bandit 在短历史上快速收敛。
  - 这是对 Mu et al. 2026（阶数失配损害协调）"对齐阶数"思想的更稳健实现。
- **同时扩 episode**：20→50（horizon 5 ⇒ 250 轮历史，warmup 9 轮后 240 轮利用），直接解决"历史太短"。

## 2  Round-3 实验矩阵（8 格，全部 50 episode/cell）

| Cell | 同质 | ToM | talk | A-ToM | 门控 | 验证 |
|---|---|---|---|---|---|---|
| hom_notom | 同 | 无 | 无 | — | — | 基线（复现陷阱）|
| het_notom | 异 | 无 | 无 | — | — | 异质摧毁协作 |
| het_tom | 异 | 固定1阶 | 无 | — | — | Round-1 方法（固定 ToM）|
| het_notom_talk | 异 | 无 | 有 | — | — | cheap-talk 单独（Round-2 最强）|
| het_tom_talk | 异 | 固定1阶 | 有 | — | — | 朴素组合（Round-2 失败）|
| het_atom | 异 | 自适应 | 无 | 改进 | — | 改进 A-ToM 单独 |
| **het_gated_talk_tom** | 异 | 固定1阶 | 有 | — | **有** | **方案 A 核心：门控仲裁** |
| **het_gated_atom_talk** | 异 | 自适应 | 有 | 改进 | **有** | **方案 A+B 完整方法** |

- 模型：Qwen2.5-3B-Instruct（同质）/ Qwen2.5-3B + Qwen2.5-1.5B（异质），与 Round-1/2 一致以便对比。
- 种子：4,5,6,7,8（n=5；Round-2 扩展格仅 n=3，扩到 5 以提升统计功效）。
- 输出目录：`results/hettom_layer1/stag_hunt_round3/`（独立目录，不覆盖 Round-1/2 数据）。
- 算力：1× RTX 5090，串行运行（单 GPU；避免 Round-2 因 2 进程抢 1 GPU 导致 seed5 慢 10×）。
- 预估：~24 min/seed × 5 ≈ 2 小时。

## 3  预注册判定规则

**陷阱被打破（POSITIVE）**，当且仅当门控方法满足：
- 在视角多样性 AND 协作回报上均 > `hom_notom`
- 协作回报 Mann-Whitney p < 0.05
- 差值 bootstrap CI 排除 0

**机制级子判定**：
- **门控仲裁有效**：`het_gated_talk_tom` payoff > `het_tom_talk`（p<0.05）—— 仲裁优于朴素组合
- **门控优于 talk-only**：`het_gated_talk_tom` payoff > `het_notom_talk`（p<0.05）
- **门控优于固定 ToM**：`het_gated_talk_tom` payoff > `het_tom`（p<0.05）
- **A-ToM 在门控下增益**：`het_gated_atom_talk` payoff > `het_gated_talk_tom`（p<0.05）
- **完整方法打破陷阱**：`het_gated_atom_talk` 满足 POSITIVE 全部条件

**门控机制的诊断指标**（新增）：
- `gate_trust_rate`：信任信号的比例
- `gated_prediction_accuracy`：门控信念命中率
- `signal_accuracy`：原始信号命中率
- `tom_belief_accuracy_in_gated`：ToM 信念命中率
- 预期：`gated_prediction_accuracy > max(signal_accuracy, tom_belief_accuracy)` ⇒ 仲裁选择了更优源

## 4  代码改动清单

| 文件 | 改动 |
|---|---|
| `hettom_baseline.py` | LLMAgent 加 `gated_talk_tom` 标志 + 门控参数；`_gate_signals()` 仲裁；`update_signal_history()` 信号信任 EMA；`act()` 返回链+门控决策并把单一门控信念喂 prompt；`_decide_tom_orders()` 重写为 per-order EMA bandit + warmup；`update_tom_history()` 对链中所有阶数打分；`compute_metrics()` 加 4 个门控指标；`make_matrix_configs()` 加 `het_gated_talk_tom`/`het_gated_atom_talk` + `--cells` 过滤 |
| `analyze_layer1.py` | CELLS 加 2 格；METRICS 加 4 个门控指标；comparisons 加 6 条 round-3 对比；DECISION 加 round-3 门控判定 |
| `run_round3.py` | 新增：聚焦 8 格、可恢复（skip 已有 metrics）、串行单 GPU 的运行器 |

## 5  参考文献

[1] Madmoun, H. & Lahlou, S. (2025). Communication Enables Cooperation in LLM Agents. arXiv:2510.05748.
[2] Mu, C. et al. (2026). Adaptive Theory of Mind for LLM-based Multi-Agent Coordination. arXiv:2603.16264.
[3] El Mir, A., Takáč, M. & Lahlou, S. (2026). Byzantine Cheap Talk: Adversarial Resilience and Topology Effects in LLM Coordination Games. arXiv:2606.07790.
[4] Yao, Y., Zou, C. & Hawkins, R.D. (2026). Talk is Cheap, Communication is Hard: Dynamic Grounding Failures and Repair in Multi-Agent Negotiation. arXiv:2605.01750.
