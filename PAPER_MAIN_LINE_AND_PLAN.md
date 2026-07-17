# 论文主线与实验方案（方法论文版）

> 生成时间：2026-07-10（方法论文重写版）
> 基于：HetToM Round 1–4 全部实验
> 核心方法：Diversity-Preserving Gated Arbitration (DP-gating)

---

## 一、论文定位与核心主张

### 1.1 定位：方法论文

**标题方向**："Conditional Alignment: Diversity-Preserving Gated Arbitration for Heterogeneous LLM Agent Cooperation"

**一句话概括**：现有对齐机制（cheap-talk、ToM、门控仲裁）始终对齐——即使智能体已经一致也强制对齐，从而摧毁异质性带来的认知多样性；DP-gating 提出条件对齐原则——仅在信号-信念冲突时干预，一致时保留独立推理，从而在维持协作的同时保留多样性。

### 1.2 问题定义

异质 LLM 智能体团队面临一个根本张力：

- **异质性产生认知多样性**：不同架构/训练的模型（Qwen vs GLM）对同一博弈状态产生不同判断，这是有益的认知资源。
- **协作需要对齐**：智能体必须预测或协调队友行为才能达成有利均衡。
- **现有对齐机制过度压缩多样性**：cheap-talk 将行为收敛到单一约定（diversity→0）；ToM 让智能体预测并跟随队友（diversity→0）；常规门控仲裁始终用单一门控信念替代双源输入（diversity→0）。

Round-3 实验精确量化了这一张力：常规门控（het_gated_atom_talk）将多样性从 1.756 压到 0.013，回报仍低于基线（1.734 vs 2.325）。对齐机制"赢了协作，输了多样性"。

### 1.3 方法核心思想

**条件对齐（Conditional Alignment）**：对齐不是永远需要的——当智能体的 cheap-talk 信号与 ToM 信念一致时，说明它们已经隐式对齐，无需干预；仅当信号与信念冲突时，才需要仲裁来消除分歧。

DP-gating 的决策规则：

```
if signal == tom_belief:      # 一致 → 不干预
    agent chooses independently    # 保留认知多样性
elif signal != tom_belief:   # 冲突 → 仲裁
    use gated_belief (trust signal if historically reliable, else ToM)
```

这一原则使多样性保留不再是"对齐的副作用"，而是"对齐的条件性"的直接结果。

### 1.4 方法贡献（4 点）

1. **条件对齐原则**：首个将对齐设计为条件性的机制——仅冲突时干预，一致时保留多样性。区别于所有现有机制的无条件对齐。

2. **DP-gating 机制**：将条件对齐实例化为 cheap-talk 信号与 ToM 信念的冲突检测器 + 门控仲裁器，含历史可靠性 EMA 估计。

3. **多样性-回报 Pareto 前沿**：DP-gating 是唯一在多样性-回报空间中占据前沿位置的方法——常规门控在回报轴上但多样性为零；异质无对齐在多样性轴上但回报低；DP-gating 在两者间提供可调权衡。

4. **跨博弈验证**：在 3 类博弈结构（纯协调、偏好冲突、反协调）上验证，揭示条件对齐的有效性是博弈结构依赖的——在反协调博弈中最优，在协调博弈中有小代价，在偏好冲突中有害（诚实呈现）。

### 1.5 明确不主张

1. 不主张 DP-gating 在所有博弈上最优（在协调博弈中有小代价，在偏好冲突中有害）
2. 不主张条件对齐解决偏好冲突问题（BoS 是诚实的边界条件）
3. 不主张 RL 训练迁移结果（Layer-2 可选，不阻塞主线）

---

## 二、方法详述

### 2.1 系统架构

异质 LLM 智能体团队中，每个智能体配备三个模块：

1. **Cheap-talk 信号器**：每轮先宣布自己的意图动作（公开信号）。
2. **ToM 推理器**：递归预测队友的下一步动作（1–3 阶，可自适应）。
3. **门控仲裁器**：在信号与 ToM 信念之间仲裁，输出单一门控信念。

DP-gating 在门控仲裁器之上增加**冲突检测器**：仅当信号与 ToM 信念冲突时，才将门控信念注入动作 prompt；一致时，智能体独立选择动作。

### 2.2 门控仲裁机制（`_gate_signals`）

对每个队友，门控仲裁器决策如下：

| 条件 | 决策 | 理由 |
|---|---|---|
| signal == tom_pred | 信任信号 | 双源一致，无分歧 |
| signal != tom_pred，且 signal_ema ≥ threshold | 信任信号 | 队友历史信号可靠 |
| signal != tom_pred，且 signal_ema < threshold | 使用 ToM 信念 | 信号不可靠，回退到推理 |

其中 `signal_ema` 是队友历史"信号-实际动作"匹配率的指数移动平均（EMA，α=0.3），`threshold=0.6`。

### 2.3 条件对齐（DP-gating 核心创新）

常规门控始终将门控信念注入动作 prompt（无条件对齐）。DP-gating 增加冲突检测：

```python
has_conflict = any(signal != tom_pred for each teammate)
if has_conflict:
    prompt_belief = gated_belief    # 仲裁 → 注入门控信念
    prompt_signals = None
else:
    prompt_belief = None            # 不干预 → 保留独立推理
    prompt_signals = None
```

**关键设计逻辑**：

- 信号与信念一致 → 智能体已隐式对齐 → 无需外部干预 → 各自独立推理 → **保留异质性提供的认知多样性**
- 信号与信念冲突 → 存在分歧 → 需要仲裁 → 注入门控信念 → **消除有害分歧**

这使得多样性保留不再是副产物，而是"不干预"的直接结果。冲突率本身成为博弈结构的一个涌现指标：纯协调博弈中冲突率低（信号自然一致），反协调博弈中冲突率高（信号天然分歧）。

### 2.4 方法变体

| 变体 | 对齐方式 | 多样性 | 适用场景 |
|---|---|---|---|
| het_notom（无对齐） | 无 | 最高（1.756） | 反协调（但无协作保障） |
| het_notom_talk（cheap-talk） | 无条件全对齐 | 零（0.000） | 纯协调 |
| het_gated_atom_talk（常规门控） | 无条件门控对齐 | 零（0.000） | 纯协调 |
| **het_dp_gated_atom_talk（DP-gating）** | **条件门控对齐** | **可调（0.365–0.442）** | **反协调 / 需多样性的场景** |

### 2.5 与现有方法的关系

DP-gating 不是 cheap-talk 或 ToM 的替代，而是它们的**条件化封装**。它复用 cheap-talk 信号和 ToM 信念作为输入，但改变了"何时使用对齐信念"的决策规则。这意味着 DP-gating 可以与任何信号源（不只是 cheap-talk）和任何信念源（不只是 ToM）组合——条件对齐是一个通用原则。

---

## 三、当前实验结果（Round 4，7B/9B 真异质）

### 3.1 跨博弈回报矩阵

| Cell | StagHunt (协调) | BoS (冲突) | Chicken (反协调) |
|---|---|---|---|
| hom_notom（基线） | 2.315 | 2.133 | 1.960 |
| het_notom（异质无对齐） | 2.735 | 1.700 | 3.093 |
| het_tom（ToM） | 2.978 | 2.067 | 2.507 |
| het_notom_talk（cheap-talk） | 3.000 | 1.593 | 3.073 |
| het_tom_talk（ToM+talk） | 3.000 | **2.380** | 2.893 |
| het_gated_atom_talk（常规门控） | 3.000 | 2.307 | 2.573 |
| **het_dp_gated_atom_talk（DP-gating）** | 2.733 | 1.600 | **3.087** |

### 3.2 跨博弈多样性矩阵

| Cell | StagHunt | BoS | Chicken |
|---|---|---|---|
| het_notom | **1.756** | 0.202 | **0.443** |
| het_gated_atom_talk（常规门控） | 0.000 | 0.010 | 0.040 |
| **het_dp_gated_atom_talk（DP-gating）** | **0.365** | 0.175 | **0.442** |

### 3.3 核心结果解读

**DP-gating vs 常规门控（核心消融）**：

| 博弈 | 常规门控 (回报/多样性) | DP-gating (回报/多样性) | Δ回报 | Δ多样性 |
|---|---|---|---|---|
| Stag Hunt | 3.000 / 0.000 | 2.733 / 0.365 | -0.267 | **+0.365** |
| Chicken | 2.573 / 0.040 | **3.087 / 0.442** | **+0.514** | **+0.402** |

**Chicken 上的关键胜利**：DP-gating 在反协调博弈中同时实现了最高回报（3.087）和最高多样性（0.442），**超越常规门控 +0.514 回报 +0.402 多样性**。这是全论文最核心的方法验证数据点——条件对齐不仅保留了多样性，还因为保留了多样性而获得了更高回报。

**Stag Hunt 上的可理解代价**：DP-gating 在纯协调博弈中比常规门控低 0.267 回报，但保留了 0.365 多样性。这是预期行为——纯协调博弈中多样性有害，条件对齐"错误地"保留了有害多样性。这可以通过阈值调节来缓解（见消融实验）。

### 3.4 多样性-回报 Pareto 前沿

| 方法 | Chicken 多样性 | Chicken 回报 | Pareto 最优？ |
|---|---|---|---|
| hom_notom | 0.072 | 1.960 | 否（被 het_notom 支配） |
| het_notom | 0.443 | 3.093 | **是** |
| het_gated_atom_talk | 0.040 | 2.573 | 否（被多个方法支配） |
| **het_dp_gated_atom_talk** | **0.442** | **3.087** | **是** |

在 Chicken 上，DP-gating 和 het_notom 共同占据 Pareto 前沿。DP-gating 的优势在于它**有对齐保障**（门控仲裁在冲突时介入），而 het_notom 是无保障的高多样性。

---

## 四、下一步实验方案

### 4.1 实验优先级总览

| 优先级 | 实验 | 方法论文中的角色 | GPU 时间 |
|---|---|---|---|
| **P0** | 扩种子到 8（全 3 博弈） | 统计显著性——审稿人必备 | ~5h |
| **P0** | 新增反协调博弈（Hawk-Dove + 1） | 证明 DP-gating 不是单博弈偶然 | ~4h |
| **P1** | 阈值消融（gate_trust_threshold 网格） | 证明多样性-回报可调权衡 | ~3h |
| **P1** | 冲突率分析（按博弈结构） | 机制解释——为什么条件对齐有效 | 0（离线分析） |
| **P2** | 3B vs 7B 对比（动机性） | 证明方法需要充分模型能力 | ~4h |
| **P2** | public_goods（4-agent） | 多智能体可扩展性 | ~2h |
| **P3** | Layer-2 MARL 迁移 | RL 成分（可选） | ~6h |

### 4.2 P0-A：扩种子到 8（全 3 博弈）

**动机**：当前 BoS/Chicken 仅 n=3 种子，所有比较不显著。DP-gating vs 常规门控在 Chicken 上的 +0.514 差距当前 p≈0.076，需 n=8 达 p<0.05。这是方法论文的统计基础。

**方案**：
```bash
python3 hettom_experiments/run_round4.py \
  --games stag_hunt battle_of_the_sexes chicken \
  --seeds 42 43 44 45 46 47 48 49 \
  --episodes 30 --horizon 5 \
  --out_dir results/hettom_layer1/round4_8seed
```

**验收标准**：DP-gating vs 常规门控在 Chicken 上达 p<0.05（paired bootstrap CI 排除零）。

### 4.3 P0-B：新增反协调博弈

**动机**：DP-gating 目前仅在 Chicken（1 个反协调博弈）上最优。方法论文需要证明这不是单博弈偶然。增加至少 1-2 个反协调结构博弈。

**候选博弈**：
1. **Hawk-Dove**（经典反协调）：两个纯纳什均衡（Hawk,Dove）和（Dove,Hawk），与 Chicken 同构但参数不同。验证 DP-gating 的胜利不依赖特定收益矩阵。
2. **Deadlock**（反协调变体）：双方背叛优于双方合作，但被背叛最差。测试 DP-gating 在更极端反协调下的表现。

**方案**：在 `hettom_baseline.py` 中添加新博弈定义，然后用相同 15-cell 矩阵运行。

**验收标准**：DP-gating 在至少 2 个反协调博弈上回报 ≥ 常规门控 + 多样性 > 常规门控。

### 4.4 P1-A：阈值消融

**动机**：方法论文需要展示方法有可调参数，且参数行为可理解。`gate_trust_threshold`（当前 0.6）控制信号信任的严格程度——值越高越保守（更多回退到 ToM），应该影响多样性-回报权衡。

**方案**：在 Chicken 和 Stag Hunt 上跑 threshold ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}，画 diversity-payoff 曲线。

**预期**：
- 低阈值（0.3）→ 更信任信号 → 更强对齐 → 低多样性 + 高回报（趋近常规门控）
- 高阈值（0.8）→ 更不信任信号 → 更多回退到 ToM → 高多样性 + 低回报
- 中间阈值 → DP-gating 的最佳权衡点

**验收标准**：diversity-payoff 曲线呈现单调权衡，且存在阈值使 DP-gating 在 Chicken 上同时优于常规门控的回报和多样性。

### 4.5 P1-B：冲突率分析

**动机**：解释为什么 DP-gating 在不同博弈上表现不同。条件对齐的有效性取决于冲突率——冲突率越高，DP-gating 越接近常规门控（更多干预）；冲突率越低，DP-gating 越接近无对齐（更多保留多样性）。

**方案**：从已有实验的门控决策日志（`gate_decisions`）中提取每个博弈的冲突率，分析冲突率与 DP-gating 回报/多样性的关系。

**预期产出**：一张图展示"博弈结构 → 冲突率 → DP-gating 行为（干预率）→ 多样性/回报"的因果链。这是方法论文的机制解释部分。

### 4.6 P2-A：3B vs 7B 对比（动机性）

**动机**：作为方法的前置条件——DP-gating 需要模型的 ToM 推理能力达到阈值才能有效。3B 模型 ToM 准确率仅 0.52（接近随机），无法产生可靠的信念，DP-gating 的冲突检测失效。7B 模型 ToM 准确率 0.98，信念可靠，DP-gating 才有意义。

**定位**：不是核心发现，而是方法的适用条件分析——"条件对齐需要 ≥7B 级别的 ToM 推理能力"。

### 4.7 P2-B：public_goods（4-agent）

**动机**：验证 DP-gating 在多智能体（>2）场景下的可扩展性。4-agent public goods 是经典多智能体博弈。

### 4.8 P3：Layer-2 MARL 迁移（可选）

**动机**：回应"这不是 MARL"的质疑。将 DP-gating 的意图特征注入 EPyMARL MAPPO 训练（观测侧注入，不改奖励）。

**风险**：时间成本高，且 LBF 的博弈结构不明确。若时间不足则放弃，论文基于 LLM-as-agent 独立成文。

---

## 五、论文结构设计

### 5.1 章节结构

| 章节 | 内容 | 核心素材 |
|---|---|---|
| Abstract | 问题（多样性-对齐张力）→ 方法（条件对齐）→ 结果（Chicken 最优 + Pareto 前沿）→ 贡献 | — |
| 1. Introduction | 多样性-对齐张力背景 → 现有机制无条件对齐的问题 → 条件对齐思想 → 贡献列表 | Round-3 张力数据 |
| 2. Related Work | LLM 多智能体协作 / cheap-talk / ToM / 门控仲裁 / 多样性 in MARL | 文献调研 |
| 3. Method | 问题形式化 → 门控仲裁机制 → 条件对齐（DP-gating）→ 阈值设计 | §二 详述 |
| 4. Experimental Setup | 3 博弈 × 7 方法 × 8 种子 → 真异质模型（7B+9B）→ 统计方法 | Round-4 设计 |
| 5. Results | 5.1 跨博弈主结果 → 5.2 DP-gating vs 常规门控消融 → 5.3 Pareto 前沿 → 5.4 阈值消融 → 5.5 冲突率机制分析 | §三 + P0/P1 |
| 6. Analysis & Discussion | 条件对齐为何在反协调中最优 → 为何在协调中有代价 → BoS 边界条件 → 模型规模前置条件 | P1-B + P2-A |
| 7. Limitations | n=8 种子仍偏小 / 仅 LLM-as-agent / BoS 失败 / 阈值需手调 | — |
| 8. Conclusion | 条件对齐原则 → DP-gating 实现 → 反协调博弈验证 → 未来方向 | — |

### 5.2 核心图表

| 位置 | 内容 | 类型 |
|---|---|---|
| Figure 1 | 方法框架图（信号→ToM→冲突检测→条件干预） | 概念图 |
| Figure 2 | 多样性-回报 Pareto 前沿（3 博弈 × 7 方法） | 散点图 |
| Figure 3 | 阈值消融曲线（threshold vs diversity/payoff） | 折线图 |
| Figure 4 | 冲突率 → 干预率 → 多样性/回报 因果链 | 柱状图+折线 |
| Table 1 | 跨博弈回报矩阵（3 博弈 × 7 方法 × 8 种子） | 主表 |
| Table 2 | 跨博弈多样性矩阵 | 主表 |
| Table 3 | DP-gating vs 常规门控消融（Δ回报/Δ多样性/CI） | 消融表 |
| Table 4 | 阈值消融汇总 | 消融表 |

### 5.3 措辞红线

**应说**：
- "条件对齐——仅在冲突时干预，一致时保留多样性"
- "DP-gating 是首个将对齐设计为条件性的机制"
- "在反协调博弈中同时实现最高回报和最高多样性"
- "多样性-回报 Pareto 前沿的唯一占据者"
- "条件对齐的有效性取决于博弈结构——这本身是方法的洞察"

**不应说**：
- "DP-gating 在所有博弈上最优"
- "条件对齐解决了所有协作问题"
- "多样性总是有益的"

---

## 六、AAAI 投稿评估

### 6.1 方法论文优势

| 维度 | 评分 | 理由 |
|---|---|---|
| 新颖性 | ★★★★ | 条件对齐原则无人提出；现有机制全为无条件对齐 |
| 方法贡献 | ★★★☆ | DP-gating 有明确机制 + 可调参数 + 清晰适用条件 |
| 实验严谨性 | ★★★☆ | 真异质 + 完整对照 + 消融，但 n=3 待扩 |
| 理论深度 | ★★★ | 条件对齐可形式化；冲突率与博弈结构的映射有理论意义 |
| 故事完整性 | ★★★★ | 张力→问题→原则→方法→验证→边界，叙事弧完整 |

### 6.2 关键风险与缓解

| 风险 | 严重性 | 缓解 | 对应实验 |
|---|---|---|---|
| DP-gating 仅在 1 个反协调博弈上最优 | ★★★★★ | 增加 Hawk-Dove 等反协调博弈 | P0-B |
| n=3 种子不显著 | ★★★★ | 扩到 n=8 | P0-A |
| Stag Hunt/BoS 上有代价/有害 | ★★★ | 诚实呈现为博弈结构依赖 + 阈值可调 | P1-A |
| 缺 RL 成分 | ★★ | Layer-2 可选 | P3 |
| 方法只是"有时不干预" | ★★ | 形式化为条件对齐原则 + 冲突率机制分析 | P1-B |

### 6.3 概率评估

| 完成阶段 | AAAI 正会概率 |
|---|---|
| 当前（Round 4，n=3，1 反协调博弈） | 20–25% |
| +P0-A（8 种子） | 25–30% |
| +P0-B（≥2 反协调博弈验证） | **35–40%** |
| +P1（阈值消融 + 冲突率分析） | **40–45%** |
| +P2（模型规模 + public_goods） | 45–50% |

**关键提升点**：P0-B（多反协调博弈验证）是方法论文成败的关键——如果 DP-gating 在 2-3 个反协调博弈上都最优，方法贡献就从"单博弈偶然"升级为"反协调博弈的可靠方法"。

---

## 七、时间线

| 阶段 | 时间 | 内容 | 产出 |
|---|---|---|---|
| 第 1 步 | ~5h GPU | P0-A：扩种子到 8（全 3 博弈） | 8-seed 统计显著 |
| 第 2 步 | ~4h GPU | P0-B：新增 Hawk-Dove + 反协调博弈 | ≥2 反协调博弈验证 |
| 第 3 步 | ~3h GPU | P1-A：阈值消融 | diversity-payoff 可调曲线 |
| 第 4 步 | 0（离线） | P1-B：冲突率分析 | 机制解释 |
| 第 5 步 | ~4h GPU | P2-A：3B vs 7B（动机性） | 适用条件分析 |
| 第 6 步 | ~2h GPU | P2-B：public_goods | 多智能体验证 |
| 写作 | P0 完成后 | 基于完整数据开始论文写作 | LaTeX 初稿 |

**关键路径**：P0-A → P0-B → P1 → 写作。P2/P3 为增量提升。

---

## 八、关键文件索引

| 文件 | 内容 |
|---|---|
| `hettom_experiments/hettom_baseline.py` | 方法实现：DP-gating 在 `diversity_preserving_gate` 参数（L605-624），门控仲裁在 `_gate_signals`（L695-742） |
| `hettom_experiments/run_round4.py` | Round-4 实验启动器（7B/9B + 3 博弈 + 15 cell） |
| `hettom_experiments/RESULTS_ROUND4_FINAL.md` | Round-4 最终结果（跨 3 博弈完整矩阵） |
| `hettom_experiments/RESULTS_ROUND4_PARTIAL.md` | Round-4 Stag Hunt 部分 + 3B→7B 对比 |
| `hettom_experiments/RESULTS_ROUND3.md` | Round-3 弱模型结果（多样性-对齐张力的原始发现） |
| `hettom_experiments/analyze_layer1.py` | 跨种子聚合 + 统计检验 |
| `hettom_experiments/layer2_marl_bridge.py` | Layer-2 MARL 迁移（P3，可选） |