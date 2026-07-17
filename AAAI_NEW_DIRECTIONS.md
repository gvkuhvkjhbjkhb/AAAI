# AAAI 方案重整：放弃旧方法，调研新方向

> 生成时间：2026-07-09。基于 18 轮、~910 次实验的全部证据 + 2024-2026 文献调研。

---

## 一、现有方案与结果总结（为什么放弃）

### 1.1 资产盘点

| 资产 | 规模 | 详情 |
|---|---|---|
| 实验轮次 | 18 轮 | Round 1-10（语义诊断路线）+ Round 11-18（FA-PBS 路线）|
| 总运行数 | ~910 次 | 894 个 `.log.done` + 16 个 8-seed 验证 |
| 代码库 | EPyMARL fork | 含 MAPPO/QMIX/VDN/IPPO 等 + LLM 诊断模块 + FA-PBS + Q-Shaping |
| 环境 | 5 个 | LBF（3p-3f, 4p-4f, 多尺度）、VMAS、RWARE |
| GPU | 1× RTX 5090 | 32GB，单 run ~2GB，可 6 并行 |
| 已修复 bug | 2 个 | trajectory_recorder 路径、unique_token 清洗 |

### 1.2 两条已走死的方法路线

**路线 A：LLM 语义诊断 → 放弃**
- Qwen3.5-4B 在 120 样本上全部预测 `insufficient_cooperation`，accuracy 0.09，macro-F1 0.04
- 标签坍缩，无法作为因果机制
- 语义标签特异性不成立：random_type 与 shuffled 控制具竞争力

**路线 B：Failure-Aware PBRS (FA-PBS) → 核心指标失败**
- 8 种子 1M 步最终结果：

| 指标 | Δ | 95% CI | 结论 |
|---|---:|---:|---|
| Final Test Return | +0.0378 | [−0.0587, +0.1251] | **不显著** |
| Train AUC | +0.0397 | [+0.0050, +0.0733] | 边缘显著（下界仅 0.005）|
| 机制隔离（vs random features）| +0.0725 | [0.0384, 0.1054] | 16 seeds 显著 |

**致命问题**：
1. 头条指标（Final Return）失败
2. 方法新颖性（adaptive weight）无独立贡献（vs fixed PBS：+0.0115，不显著）
3. 单一任务族：LBF 10×10-3p-3f，跨域全失败（VMAS/RWARE/4p-4f）
4. 种子敏感性：5/8 正、3/8 负，方法不可靠
5. PBRS 理论上不改变最优策略，"提升 AUC 不提升 final"部分是理论预测，惊喜度不足

### 1.3 放弃结论

以当前证据投 AAAI 主会概率 **10-20%**。方法论文的核心创新点（failure-aware adaptive weighting）没有独立贡献，头条指标失败，跨域全负。继续在这个框架内修补（加种子、延长训练）无法从根本上解决新颖性和鲁棒性问题。**放弃 FA-PBS 作为方法论文的主线。**

### 1.4 仍可复用的资产

| 资产 | 价值 |
|---|---|
| 种子敏感性数据（5/8 vs 3/8）| MARL 评估方法论的实证素材 |
| 视界效应数据（500k→1M，baseline 2.86×）| 同上 |
| 894 次实验的完整日志 | 可挖掘的负面对照 |
| EPyMARL 代码 + FA-PBS/Q-Shaping 实现 | 基础设施，可复用 |
| 16-seed 机制隔离结果 | 作为负面/中性发现可引用 |
| Lab + GPU + 已验证的实验流水线 | 快速迭代能力 |

---

## 二、文献调研结论（2024-2026）

### 2.1 两个关键的新发现（改变格局的论文）

1. **ARMS (Abboud & Gal, 2026, arXiv:2605.23562)**：首个博弈论均衡保持的自动 reward shaping 框架。**证明单智能体 PBRS 的策略不变性保证在多智能体非平稳性下失效**。这是 MARL reward shaping 领域的"危机论文"——打开了新的攻击面。

2. **Akella (2025, arXiv:2511.00034)**：实证证明去中心化可学习 shaping 无法弥合与集中式训练的差距（26 分 gap）。指出"个体奖励优化与全局目标的错位"是关键障碍，但**没有提供诊断方法**。

### 2.2 最尖锐的文献空白

**"诊断驱动的 reward shaping"在 MARL 中是零工作：**
- 唯一的 failure-aware RL 论文（Li et al., 2026, arXiv:2601.07821）是**单智能体**机器人操作
- ARMS 识别了振荡失败模式，但只用更多探索来缓解，**没有诊断哪个 agent/transition 导致失败**
- Safe-MARL 约束的是动作，不是 shaping 信号
- "检测哪个 agent 的局部奖励与全局回报错位并定向重塑"——**无直接前作**

### 2.3 MARL 评估方法论是真空地带

- 搜索"MARL benchmark evaluation reproducibility seed variance"几乎**无 MARL 专属方法论论文**
- 新 benchmark（StarCraft+, HLSMAC, POGEMA, Craftax）是被动反应，没有量化 seed/horizon 方差
- **你已有的种子敏感性和视界效应数据，正好填补这个空白**

### 2.4 Benchmark 现状

| Benchmark | 2026 状态 | 单 GPU 可行？ |
|---|---|---|
| SMAC（原版）| 过时/饱和 | 是 |
| SMACv2 | 标准但有争议 | 是（小图）|
| LBF | 标准（小规模）| 是 ✓ |
| RWARE | 标准 | 是 |
| MAMuJoCo | 标准 | 是 |
| VMAS | 新兴，GPU 友好 | 是 ✓ |
| POGEMA | 新兴标准 | 是 |
| MPE | 过时，审稿人会 dismiss | — |

---

## 三、可行的 AAAI 方案候选（按推荐度排序）

### 方案 1：MARL 评估方法论实证研究 ★★★★★（最推荐）

**标题方向**：*How Many Seeds Does It Take? A Diagnostic Study of Evaluation Variance in Cooperative MARL*

**核心主张**：系统量化 seed 数量、训练视界、benchmark 饱和度对合作 MARL 报告结果的影响，提出可操作的评估协议。

**为什么可行**：
- **你已有核心数据**：种子敏感性（5/8 vs 3/8）、视界效应（500k→1M, 2.86×）、16-seed 机制隔离
- **填补文献真空**：无 MARL 专属评估方法论论文
- **低计算风险**：用现有 EPyMARL + LBF/RWARE，补跑 SMACv2/MAMuJoCo 的 seed×horizon 网格
- **AAAI 偏好**：实证/可复现研究在 AAAI 有专门 track，审稿人对"暴露评估陷阱"类工作接受度高

**实验计划（2-3 周）**：
- Week 1：MAPPO/QMIX 在 LBF/RWARE/SMACv2-small 上跑 5/8/16/32 seeds × {300k, 500k, 1M, 2M} horizons
- Week 2：计算方差比、最小种子数要求、饱和曲线、效应量；提出评估协议
- Week 3：写论文 + 复现检查

**AAAI 概率**：25-35%（实证研究，但选题新颖且数据扎实）

**风险**：纯实证研究在 AAAI 主会接受率本就偏低；需要极强的写作让审稿人觉得"这个协议必须被采用"。

---

### 方案 2：诊断驱动的定向 Reward Shaping ★★★★

**标题方向**：*Diagnosis-Driven Reward Shaping: Detecting and Fixing Per-Agent Reward Misalignment in Cooperative MARL*

**核心主张**：检测每个 agent 的 shaping 信号与真实全局价值贡献的偏离，只对错位的组件定向重塑——而非全局重学整个 shaping 函数。

**为什么可行**：
- **文献空白最尖锐**：MARL 中零前作
- **建立在最新危机论文上**：直接回应 ARMS（2026）的振荡失败 + Akella（2025）的错位障碍
- **代码可复用**：EPyMARL + 现有 FA-PBS 模块改造为"偏离检测器 + 门控重塑"
- **单 GPU 可行**：LBF/RWARE/MPE-simple_spread

**实验计划（3 周）**：
- Week 1：实现 per-agent advantage-vs-shaping-gap 偏离检测器，门控重塑
- Week 2：在 LBF/RWARE/SMACv2-small 上对比 MAPPO + ARMS-style 全局重学 + 本方法
- Week 3：消融 + 写论文

**AAAI 概率**：30-40%（若方法在 ≥2 个环境显著优于 baseline + ARMS）

**风险**：方法设计有不确定性；需要 ≥2 个环境的正向结果；3 周较紧。

---

### 方案 3：方案 1 + 方案 2 组合（评估协议 + 诊断方法）★★★★

**逻辑**：用方案 1 的评估协议作为方法验证标准（自证方法在严格协议下仍有效），方案 2 的诊断方法作为算法贡献。两者互相加强。

**AAAI 概率**：35-45%（若方法在严格协议下 ≥2 环境显著）

**风险**：3 周内同时做方法和评估协议，时间风险高。建议优先方案 2，若时间允许补方案 1 的协议部分。

---

### 方案 4：协同探索的失败模式感知触发 ★★★

**标题方向**：*Failure-Mode-Aware Exploration: Targeted Intrinsic Motivation via Shaping Divergence in Cooperative MARL*

**核心主张**：用 shaping 信号与真实回报的偏离作为内在探索 bonus，只对导致失败的 agent/区域触发——定向而非全局探索。

**为什么可行**：
- ARMS 用无方向探索缓解失败，浪费样本；本方法定向触发
- 单 GPU 可行，2-3 周
- 比 FA-PBS 更新颖（从"shaping"转向"exploration trigger"）

**AAAI 概率**：25-35%

**风险**：与方案 2 高度相似，需明确区分"重塑 reward"vs"触发探索"。

---

### 方案 5：动态角色发现与相位感知重特化 ★★★

**标题方向**：*Phase-Aware Role Discovery and Re-Specialization in Cooperative MARL*

**核心主张**：在线发现角色并在任务相位转换时合并/分裂角色，无需预定义角色数。

**为什么可行**：
- ROIS（2025）、GHQ（2024）假设固定角色词汇表，明确留 online re-configuration 为 future work
- LBF/RWARE 单 GPU 可行

**AAAI 概率**：25-35%

**风险**：需与 ROIS 明确区分；角色不稳定可能导致负面结果。

---

## 四、最终推荐

### 首选：方案 2（诊断驱动的定向 Reward Shaping）

**理由**：
1. **新颖性最高**：MARL 中零前作，直接建立在 2026 最新危机论文上
2. **代码复用度最高**：现有 FA-PBS 的"失败检测"模块可改造为"偏离检测器"
3. **故事最完整**：ARMS 指出问题（振荡失败）→ Akella 指出障碍（错位）→ 本方法提供诊断+修复
4. **AAAI 概率最高**：30-40%（若 ≥2 环境显著）

### 备选：方案 1（评估方法论）若方案 2 方法设计受阻

**理由**：
1. 已有核心数据，最低风险
2. 填补明确文献真空
3. 但 AAAI 主会对纯实证研究接受率本就偏低

### 不推荐：继续 FA-PBS 路线

头条指标已失败，新颖性无独立贡献，跨域全负。修补无法解决根本问题。

---

## 五、关键文献（2024-2026，按重要性）

| 论文 | 年份 | 关键性 |
|---|---|---|
| ARMS (Abboud & Gal, arXiv:2605.23562) | 2026 | 证明 PBRS 多智能体保证失效，本方法的立论基础 |
| Akella (arXiv:2511.00034) | 2025 | 去中心化 shaping 局限性，指出错位障碍 |
| ISA (Han et al., arXiv:2505.08630) | 2025 | 联合 credit+exploration，influence scope |
| VLM-PBRS (Müller & Kudenko, arXiv:2606.27180) | 2026 | VLM 引导势能构造，单智能体 |
| Q-Shaping (Wu, arXiv:2410.01458) | 2024 | 无偏 Q 值初始化，单智能体 |
| Failure-Aware RL (Li et al., arXiv:2601.07821) | 2026 | 唯一 failure-aware RL，单智能体 |
| PRD-MAPPO (Kapoor et al., arXiv:2408.04295) | 2024 | MAPPO 部分奖励解耦 |
| Multi-level Advantage (Zhao & Xie, arXiv:2508.06836) | 2025 | 多层级 credit assignment |
| StarCraft+ (Li et al., 2025) | 2025 | SMAC 饱和批判 |
| HLSMAC (Hong et al., 2025) | 2025 | SMAC 高层决策批判 |
| POGEMA (Skrynnik et al., 2024) | 2024 | 可扩展协同 MAPF benchmark |

---

## 六、下一步

1. 确认选择方案 2（诊断驱动）还是方案 1（评估方法论）
2. 若选方案 2：先实现偏离检测器原型，在 LBF 上做 smoke test
3. 若选方案 1：整理现有 seed/horizon 数据，补跑 SMACv2/MAMuJoCo 网格
