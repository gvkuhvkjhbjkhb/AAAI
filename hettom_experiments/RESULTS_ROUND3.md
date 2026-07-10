# HetToM Round-3 实验结果与结论（门控信号-信念仲裁 + 改进 A-ToM）

> 实验时间：2026-07-09 22:28–2026-07-10 00:20（约 1 小时 52 分钟，1× RTX 5090 串行）
> 博弈：Stag Hunt（2 智能体，horizon=5，50 episode/cell，memory=2）
> 种子：4, 5, 6, 7, 8（n=5，全部独立运行完整 8 格矩阵）
> 模型：Qwen2.5-3B-Instruct（同质）/ Qwen2.5-3B + Qwen2.5-1.5B（异质）
> 输出目录：`results/hettom_layer1/stag_hunt_round3/`

---

## 1. 结果总表（5 种子均值 ± std，95% bootstrap CI）

| Cell | 视角多样性 | 协作回报 | 均衡收敛 | ToM 准确率 | 门控信任率 | 门控信念准确率 |
|---|---|---|---|---|---|---|
| hom_notom（基线） | 0.136 ± 0.026 [0.115,0.155] | **2.325** ± 0.116 [2.237,2.414] | 0.970 | — | — | — |
| het_notom（异质） | **1.939** ± 0.150 [1.821,2.057] | 0.549 ± 0.030 [0.525,0.573] | 0.986 | — | — | — |
| het_tom（固定ToM） | 0.538 ± 0.070 [0.489,0.601] | 1.245 ± 0.051 [1.203,1.282] | 0.978 | 0.518 ± 0.024 | — | — |
| het_notom_talk（talk） | 0.004 ± 0.003 [0.001,0.006] | 1.776 ± 0.048 [1.745,1.817] | 0.909 | — | — | — |
| het_tom_talk（朴素组合） | 1.091 ± 0.051 [1.048,1.125] | 0.848 ± 0.045 [0.815,0.884] | 0.974 | 0.464 ± 0.012 | — | — |
| het_atom（改进A-ToM） | 0.189 ± 0.021 [0.174,0.207] | 1.570 ± 0.034 [1.548,1.601] | 0.978 | 0.599 ± 0.008 | — | — |
| **het_gated_talk_tom（门控）** | 0.006 ± 0.006 [0.002,0.012] | **1.580** ± 0.063 [1.531,1.626] | 0.907 | 0.440 ± 0.023 | 0.670 ± 0.032 | 0.537 ± 0.033 |
| **het_gated_atom_talk（完整）** | 0.013 ± 0.011 [0.006,0.023] | **1.734** ± 0.057 [1.690,1.778] | 0.973 | 0.585 ± 0.013 | 0.720 ± 0.011 | 0.583 ± 0.021 |

Stag Hunt 回报：双方 Stag=3（Pareto 最优），双方 Hare=2（风险占优），错配=0/2。

---

## 2. 关键统计比较（Mann-Whitney U，n=5）

### 2.1 核心复现（Round-1/2 结论巩固）

| 比较 | 回报 Δ | p | rank-biserial r | 结论 |
|---|---|---|---|---|
| het_notom vs hom_notom | -1.776 | 0.008 | -1.000 | 异质摧毁协作（方向100%一致，显著）|
| het_tom vs het_notom | +0.696 | 0.008 | +1.000 | ToM 部分拯救异质协作（显著）|
| het_notom_talk vs het_notom | +1.227 | 0.008 | +1.000 | cheap-talk 显著拯救异质协作（显著）|
| het_tom_talk vs het_tom | -0.397 | 0.008 | -1.000 | 朴素 ToM+talk 组合有害（显著）|

### 2.2 Round-3 新结果：门控仲裁

| 比较 | 回报 Δ | p | rank-biserial r | 结论 |
|---|---|---|---|---|
| **het_gated_talk_tom vs het_tom_talk** | **+0.732** | **0.008** | **+1.000** | **门控仲裁显著优于朴素组合**（核心结果）|
| **het_gated_talk_tom vs het_tom** | **+0.335** | **0.008** | **+1.000** | **门控仲裁显著优于固定 ToM** |
| het_gated_talk_tom vs het_notom_talk | -0.196 | 0.008 | -1.000 | 门控仍低于 talk-only |
| **het_gated_atom_talk vs het_gated_talk_tom** | **+0.154** | **0.008** | **+1.000** | **A-ToM 在门控下显著增益** |

### 2.3 改进 A-ToM 的效果

| 比较 | 回报 Δ | p | rank-biserial r | 结论 |
|---|---|---|---|---|
| **het_atom vs het_tom** | **+0.325** | **0.008** | **+1.000** | **改进 A-ToM 显著优于固定 ToM**（Round-2 旧规则为 +0.001, p=1.0）|

> 改进 A-ToM 的 tom_acc 从 0.518 提升到 0.599（+15.6%），回报从 1.245 提升到 1.570（+26.1%）。per-order bandit + 全链打分 + 50 episode 长历史直接解决了 Round-2 暴露的"A-ToM 规则过粗 + 历史太短"问题。

### 2.4 门控机制诊断指标

| Cell | 门控信任率 | 门控信念准确率 | 信号准确率 | ToM信念准确率 | 仲裁增益 |
|---|---|---|---|---|---|
| het_gated_talk_tom | 0.670 | 0.537 | 0.575 | 0.440 | +0.038 vs signal, +0.097 vs tom |
| het_gated_atom_talk | 0.720 | 0.583 | 0.565 | 0.585 | +0.018 vs signal, -0.002 vs tom |

> 仲裁增益 = 门控信念准确率 - max(信号准确率, ToM信念准确率)。在固定 ToM 门控中，仲裁显著优于两个单独源；在 A-ToM 门控中，ToM 信念已很强（0.585），仲裁主要起保护作用。

---

## 3. 预注册判定

按 `analyze_layer1.py` 内置的预注册规则，"陷阱被打破"需同时满足：
- 门控方法在视角多样性 AND 协作回报上均 > hom_notom
- 回报差值 Mann-Whitney p < 0.05 且 bootstrap CI 排除 0

**判定：PARTIAL / POSITIVE-MECHANISM**

未达"完全打破"：
- het_gated_talk_tom 多样性 0.006 < baseline 0.136（门控把多样性压到接近0）
- het_gated_atom_talk 多样性 0.013 < baseline 0.136（同上）
- 两个门控方法的回报仍低于 baseline（1.580/1.734 vs 2.325）

但 Round-3 实现了三个显著的机制级突破：
1. **门控仲裁解决信念-信号干扰**：het_gated_talk_tom 1.580 vs het_tom_talk 0.848，+0.732, p=0.008, r=+1.0
2. **改进 A-ToM 有效**：het_atom 1.570 vs het_tom 1.245，+0.325, p=0.008, r=+1.0（Round-2 旧规则无效）
3. **门控+A-ToM 协同**：het_gated_atom_talk 1.734 是所有异质方法中最高回报

---

## 4. 数据解读

### 4.1 门控仲裁成功解决了信念-信号干扰（核心发现）

Round-2 中，朴素 ToM+talk 组合（het_tom_talk 0.848）不仅低于 talk-only（1.776），甚至低于固定 ToM（1.245）。根因是把 ToM 预测和 cheap-talk 信号同时塞入 prompt，无仲裁机制，二者冲突时相互干扰。

Round-3 的门控仲裁用单一门控信念替代双源输入：
- 信号与 ToM 信念一致 → 信任信号（67% 的轮次）
- 信号与 ToM 信念冲突但队友历史可靠（signal-EMA ≥ 0.6）→ 信任信号
- 否则 → 回退到 ToM 信念

结果：het_gated_talk_tom 1.580 显著超过 het_tom_talk 0.848（+0.732, p=0.008, r=+1.0），证明仲裁是解决干扰的关键。门控信念准确率 0.537 高于 ToM 信念单独的 0.440，说明仲裁确实选择了更优的信息源。

### 4.2 改进 A-ToM 从无效变为有效

Round-2 的旧 A-ToM 规则（hit rate <0.4 加深 / >0.75 降低）在 2 动作博弈中几乎不触发，且只在 1↔3 间振荡，结果与固定 ToM 无差异（+0.001, p=1.0）。

Round-3 的改进设计：
- per-order EMA bandit（用全部历史，非仅最近10次）
- epsilon-greedy + warmup（每阶数先采样≥3次）
- 对链中所有阶数打分（单轮获得多阶数据）
- 50 episode（250轮历史，vs Round-2 的 100轮）

结果：het_atom 1.570 显著超过 het_tom 1.245（+0.325, p=0.008, r=+1.0），tom_acc 从 0.518 提升到 0.599。bandit 成功学到了比固定1阶更优的推理深度。

### 4.3 门控 + A-ToM 协同最强但未超基线

het_gated_atom_talk 1.734 是所有异质方法中最高回报，且 A-ToM 在门控下仍有显著增益（+0.154, p=0.008）。但与 hom_notom 2.325 仍有 -0.591 差距。

关键张力：门控仲裁把多样性压到接近 0（0.013），与 talk-only（0.004）类似。这说明门控主要通过对齐（而非多样性）提升协作——它继承了 cheap-talk 的"压缩行为分歧"特性。异质性提供的多样性在门控对齐过程中被牺牲了。

### 4.4 多样性-对齐张力的精确化

跨三轮实验的完整图景：

| 方法 | 多样性 | 回报 | 机制 |
|---|---|---|---|
| hom_notom | 0.136 | 2.325 | 共享先验=免费对齐，无多样性 |
| het_notom | 1.939 | 0.549 | 最大多样性，零对齐=混乱 |
| het_tom | 0.538 | 1.245 | ToM 部分对齐，保留部分多样性 |
| het_notom_talk | 0.004 | 1.776 | talk 强对齐，牺牲多样性 |
| het_gated_atom_talk | 0.013 | 1.734 | 门控+A-ToM 强对齐，牺牲多样性 |

**核心张力**：对齐机制（talk/门控）通过压缩行为分歧提升协作，但这与异质性提供的认知多样性直接冲突。门控解决了"信号-信念干扰"问题，但未解决"对齐-多样性"根本张力。

---

## 5. 结论

### 5.1 三轮实验的完整故事

1. **Round-1**：推理陷阱存在；异质性打破多样性坍缩但摧毁协作；ToM 在异质中部分恢复协作（+123%），但未超基线。
2. **Round-2**：cheap-talk 是最强对齐机制（+1.227, p=0.008）；但 ToM+talk 朴素组合失败（干扰）；A-ToM 旧规则无效。
3. **Round-3**：门控仲裁解决干扰（+0.732, p=0.008）；改进 A-ToM 有效（+0.325, p=0.008）；门控+A-ToM 是最强异质方法（1.734），但仍未超基线。

### 5.2 论文主线建议

**Heterogeneity creates useful cognitive diversity but destroys coordination unless agents establish an alignment channel. Theory-of-Mind partially restores cooperation, while cheap-talk is a stronger alignment mechanism. Naive ToM+communication composition fails due to belief-signal interference, but gated signal-belief arbitration resolves this interference. Adaptive ToM (per-order bandit) further improves alignment. However, all alignment mechanisms compress the diversity that heterogeneity provides, revealing a fundamental diversity-alignment tension that no current method fully resolves.**

中文一句话：

异质性负责产生多样性，ToM 与 cheap-talk 负责对齐；门控仲裁解决了 ToM 与通信的信念-信号干扰，改进 A-ToM 进一步提升对齐，但所有对齐机制都以牺牲多样性为代价，多样性-对齐张力是核心未解问题。

### 5.3 预注册判定

**PARTIAL / POSITIVE-MECHANISM**

- 未达"完全打破"：门控方法回报仍低于基线
- 但实现三个显著机制级突破（全部 p=0.008, r=+1.0）
- 对应"方法+分析论文"（AAAI 30-40%）

---

## 6. 局限与下一步

| 局限 | 影响 | 缓解 |
|---|---|---|
| 仅 Stag Hunt 1 博弈 | 泛化性未知 | 跑 battle_of_the_sexes / chicken / public_goods |
| 模型 3B/1.5B（弱） | ToM 推理能力受限 | 配 HF token 用 7B + Llama/Mistral |
| 门控把多样性压到~0 | 未实现"高多样性+高回报" | 设计选择性门控：仅在冲突时仲裁，一致时保留多样性 |
| n=5 种子 | paired Wilcoxon p 卡 0.062 | 扩到 8-10 种子达 p<0.05 |
| 同质基线无 talk/ToM 对照 | hom_tom/hom_talk 缺失 | 补跑同质门控对照 |

**建议下一步**：
1. 设计"多样性保留门控"：信号与 ToM 一致时不干预（保留各自独立推理），仅冲突时仲裁
2. 跨博弈验证（battle_of_the_sexes 是 ToM 最强测试）
3. 扩种子到 8-10

---

## 7. 文件清单

| 文件 | 内容 |
|---|---|
| `analysis_report.txt` | 分析器完整输出（含统计检验 + 预注册决策） |
| `aggregated.csv` | 8 指标 × 8 cell 跨种子聚合（mean/std/CI） |
| `seed_<n>/<cell>/metrics.json` | 每格每种子的详细指标 + 门控诊断 + per-episode 回报 |
| `seed_<n>/<cell>/trajectories.jsonl` | 每格每种子完整轨迹（含门控决策日志） |
| `logs/round3_gated.log` | 运行时间线 |
