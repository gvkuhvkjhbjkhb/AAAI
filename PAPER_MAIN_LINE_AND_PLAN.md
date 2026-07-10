# 论文主线与实验方案定稿

> 生成时间：2026-07-10
> 基于：HetToM Round 1–4 全部实验 + Round 1–18 旧实验（FA-PBS，已证伪并放弃）
> 状态：Round-4 实验已完成，本文档为论文写作与下一步实验的最终依据

---

## 一、项目历程与主线收敛

### 1.1 两条路线的兴衰

| 阶段 | 路线 | 核心主张 | 结局 |
|---|---|---|---|
| Round 1–10 | LLM 语义诊断 + FA-PBS 奖励 shaping | 失败触发自适应奖励 shaping 改善合作 MARL | **放弃**：16 种子下头条指标不显著，自适应权重无独立贡献，跨域全负 |
| Round 11–18 | FA-PBS 硬化 + 跨域验证 | 在更严格控制下证明 FA-PBS 有效 | **放弃**：Round 13 主结果在种子 9–16 下崩溃，Round 14 salvage 变体同样失败 |
| HetToM Round 1–3 | LLM-as-Agent 矩阵博弈（3B/1.5B） | 异质性→多样性→摧毁协作→ToM/门控部分拯救 | **被 Round-4 推翻**：弱模型假象 |
| HetToM Round 4 | LLM-as-Agent 矩阵博弈（7B/9B，3 博弈） | 方法有效性是模型规模与博弈结构的条件函数 | **最终定稿主线** |

### 1.2 主线收敛逻辑

旧路线（FA-PBS）失败的根因：方法新颖性无独立贡献 + 单一任务族 + 种子敏感性。这些是方法论文的致命伤，无法通过补实验修复。

HetToM 路线的转折点在 Round 4：从 3B/1.5B 升级到 7B/9B（真架构异质性：Qwen2.5-7B + GLM-4-9B），并从单一 Stag Hunt 扩展到三个博弈（Stag Hunt / BoS / Chicken）。这一升级产生了三个颠覆性发现，直接推翻了 Round 1–3 的核心结论，确立了新的论文主线。

---

## 二、论文核心主张（定稿）

### 2.1 标题方向

**"When Does Heterogeneity Help LLM Agent Cooperation? A Scale- and Structure-Dependent Analysis"**

### 2.2 核心主张（三大发现）

异质 LLM 智能体协作中，异质性与对齐机制的有效性**不是 universal 属性**，而是**模型规模与博弈结构的条件函数**。三大发现支撑这一主张：

**发现 1 — 规模依赖性（Scale Dependence）**：
弱模型（3B/1.5B）的"异质性摧毁协作"结论是模型能力不足的假象。从 3B 升级到 7B/9B，异质团队回报跃升 5 倍（het_notom：0.549 → 2.735），ToM 推理准确率从 0.52 跃升至 0.98。Round 1–3 的三个"机制级突破"（信念-信号干扰、A-ToM 有效、门控必需）在强模型下全部消失——它们解决的纯粹是弱模型制造的问题。

**发现 2 — 结构依赖性（Structure Dependence）**：
没有 universal 最优对齐机制。方法排名随博弈结构完全翻转：
- 纯协调（Stag Hunt）：talk / ToM / gating 全部达到完美协作（3.000），DP-gating 反而有代价（2.733）
- 偏好冲突（BoS）：没有任何方法显著超越基线；ToM+talk 是唯一正向方向（+0.247）；DP-gating 有害（1.600）
- 反协调（Chicken）：异质性单独最强（3.093）；DP-gating 达到最优多样性-回报权衡（3.087 + 0.442）

**发现 3 — 多样性-对齐张力的结构依赖反转**：
多样性-对齐张力——在 Round 3 中被识别为"根本性"问题——本身是结构依赖的：
- 纯协调博弈：张力存在（对齐压缩多样性，完美协调需零多样性）
- 偏好冲突博弈：张力加剧（多样性 = 分歧 = 有害）
- **反协调博弈：张力反转**（多样性本身有利，与回报对齐而非冲突）

### 2.3 方法贡献

**DP-gating（多样性保留门控仲裁）**：唯一能根据博弈结构自适应调节多样性保留的机制。设计原则——信号与 ToM 信念一致时不干预（保留各自独立推理），仅冲突时仲裁。在反协调博弈中唯一实现高多样性（0.442）+ 高回报（3.087），但在协调/冲突博弈中有代价或有害。

**定位**：DP-gating 不是 universal 解决方案，而是反协调博弈的正确工具。论文不主张"DP-gating 总是最优"，而主张"机制选择必须以博弈结构为条件"。

### 2.4 明确不主张的三件事

1. 不主张 DP-gating 是 universal 最优方法（在 BoS 上有害，在 Stag Hunt 上有代价）
2. 不主张方法解决了偏好冲突博弈（BoS 是诚实的负面结果——LLM 协作的真瓶颈）
3. 不主张 RL 训练迁移结果（Layer 2 未完成，论文基于 LLM-as-agent 矩阵博弈）

### 2.5 论文定位

**实证分析论文**（empirical analysis paper），非方法论文。理由：
- 最强发现是诊断性/分析性的（规模假象、结构翻转、张力反转），非方法性能
- DP-gating 非 universal 最优，无法支撑"我们提出的方法总是最好"叙事
- AAAI 接受有 surprising 发现的实证研究，尤其当发现推翻前人结论时

---

## 三、证据层级与当前结果汇总

### 3.1 证据层级

```
★★★★★ (最强)    : 3B→7B 的 5x 跃升 (het_notom: 0.549→2.735) — Stag Hunt, n=4/5
★★★★  (核心)    : 跨博弈方法分化 — 3 博弈 × 7 方法完整矩阵
★★★★  (理论)    : 多样性-对齐张力的结构依赖反转
★★★   (方法)    : DP-gating 在 Chicken 上最优 (3.087 + 0.442)
★★★   (诚实)    : BoS 上无方法有效 — LLM 协作真瓶颈
★★    (待强化)  : BoS/Chicken 的 n=3 种子统计不显著
```

### 3.2 跨博弈回报矩阵（Round 4，7B/9B）

| Cell | StagHunt (协调) | BoS (冲突) | Chicken (反协调) | 跨博弈均值 |
|---|---|---|---|---|
| hom_notom (基线) | 2.315 | 2.133 | 1.960 | 2.136 |
| het_notom (异质) | **2.735** | 1.700 | **3.093** | 2.509 |
| het_tom (ToM) | 2.978 | 2.067 | 2.507 | 2.517 |
| het_notom_talk (talk) | 3.000 | 1.593 | 3.073 | 2.556 |
| het_tom_talk (ToM+talk) | 3.000 | **2.380** | 2.893 | **2.758** |
| het_gated_atom_talk (门控) | 3.000 | 2.307 | 2.573 | 2.627 |
| het_dp_gated_atom_talk (DP门控) | 2.733 | 1.600 | 3.087 | 2.473 |

### 3.3 跨博弈多样性矩阵

| Cell | StagHunt | BoS | Chicken |
|---|---|---|---|
| hom_notom | 0.022 | 0.120 | 0.072 |
| het_notom | **1.756** | 0.202 | **0.443** |
| het_tom | 0.008 | 0.014 | 0.171 |
| het_tom_talk | 0.000 | 0.026 | 0.028 |
| het_gated_atom_talk | 0.000 | 0.010 | 0.040 |
| het_dp_gated_atom_talk | **0.365** | 0.175 | **0.442** |

### 3.4 模型规模对比（3B/1.5B vs 7B/9B，Stag Hunt）

| Cell | 3B/1.5B | 7B/9B | 变化 | 倍率 |
|---|---|---|---|---|
| hom_notom | 2.325 | 2.315 | -0.010 | 1.0x |
| **het_notom** | **0.549** | **2.735** | **+2.186** | **5.0x** |
| het_tom | 1.245 | 2.978 | +1.732 | 2.4x |
| het_notom_talk | 1.776 | 3.000 | +1.224 | 1.7x |
| het_tom_talk | 0.848 | 3.000 | +2.152 | 3.5x |
| het_gated_atom_talk | 1.734 | 3.000 | +1.266 | 1.7x |

### 3.5 三个博弈的三种模式

| 模式 | 博弈 | 核心特征 | 最强方法 | 多样性角色 |
|---|---|---|---|---|
| 对齐有效 | Stag Hunt | 7B/9B 几乎解决协调，多方法达 3.000 | talk/tom/gated 全部=3.000 | **有害** — 对齐需零多样性 |
| 无解 | BoS | 无方法显著超基线，异质反而有害 | het_tom_talk=2.380 (仍 ns) | **有害** — 多样性=分歧 |
| 多样性有利 | Chicken | 异质单独极强，DP-gating 最优 | het_notom=3.093 ≈ DP-gated=3.087 | **有利** — 需要做不同的事 |

---

## 四、下一步实验方案

### 4.1 实验优先级总览

| 优先级 | 实验 | 目的 | 预期概率提升 | 工作量 | GPU 时间 |
|---|---|---|---|---|---|
| **P0** | 扩种子到 8（BoS+Chicken） | 统计显著性 | +5% | 代码微调 | ~3h |
| **P0** | 3B vs 7B 全 3 博弈正式对比 | 最强发现的多博弈验证 | +5% | 代码微调 | ~4h |
| **P1** | 博弈结构分类器/决策规则 | 将"无 universal 方法"转为正面贡献 | +5% | 代码+实验 | ~2h |
| **P2** | 补 public_goods（4-agent） | 多智能体验证 | +3% | 代码微调 | ~2h |
| **P2** | Layer-2 MARL 迁移（≥1 博弈） | RL 成分 | +5% | 1-2 天 | ~6h |

### 4.2 P0-A：扩种子到 8（BoS + Chicken）

**动机**：当前 BoS/Chicken 仅 n=3 种子，Mann-Whitney 最小 p=0.0765，所有比较不显著。这是审稿人最易攻击的弱点。扩到 n=8 可使多数比较达 p<0.05。

**方案**：
```bash
python3 hettom_experiments/run_round4.py \
  --games battle_of_the_sexes chicken \
  --seeds 42 43 44 45 46 47 48 49 \
  --episodes 30 --horizon 5 \
  --out_dir results/hettom_layer1/round4_8seed
```

**预期**：het_notom vs hom_notom 在 Chicken 上的 +1.133 差距（当前 p≈0.076）应在 n=8 下达 p<0.05。BoS 上的方向性差异（het_tom_talk +0.247）也可能变显著。

**验收标准**：至少 3 个核心比较的 paired bootstrap CI 排除零。

### 4.3 P0-B：3B vs 7B 全 3 博弈正式对比

**动机**：5x 跃升是全论文最有冲击力的数据点，但当前仅在 Stag Hunt 上有对比。必须在 BoS 和 Chicken 上验证：
- 跃升是否在所有博弈上一致？（预期：是，因为能力提升是博弈无关的）
- 弱模型的"异质摧毁协作"在 BoS/Chicken 上是否也是假象？

**方案**：
```bash
# 3B/1.5B 异质，全 3 博弈
python3 hettom_experiments/run_round4.py \
  --games stag_hunt battle_of_the_sexes chicken \
  --seeds 42 43 44 45 46 47 48 49 \
  --episodes 30 --horizon 5 \
  --models_het Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
  --model_homo Qwen/Qwen2.5-3B-Instruct \
  --out_dir results/hettom_layer1/round4_3b_3games
```

**预期产出**：一张 2×3 表（2 模型规模 × 3 博弈 × 关键 cell），展示规模效应的跨博弈一致性。

**验收标准**：het_notom 回报在所有 3 博弈上 7B/9B > 3B/1.5B，且至少 2 个博弈达 p<0.05。

### 4.4 P1：博弈结构分类器 / 决策规则

**动机**：将"没有 universal 最优方法"从负面发现转为正面贡献——"我们能根据博弈结构预测哪种方法最优"。

**方案**：
1. 从 3 博弈的完整矩阵中提取博弈结构特征：协调度（Pareto 最优 vs 风险占优的对齐程度）、冲突度（均衡间偏好分歧）、反协调度（需要做不同的事）。
2. 训练一个简单的决策规则（决策树或规则表）：给定博弈结构特征 → 推荐最优方法。
3. 在 public_goods 博弈上验证决策规则的预测能力。

**预期决策规则**（基于当前数据的假设）：

| 博弈结构 | 推荐方法 | 理由 |
|---|---|---|
| 高协调度（Pareto 对齐） | het_notom_talk 或 het_tom_talk | 对齐有效，压缩多样性无害 |
| 高冲突度（偏好分歧） | het_tom_talk | ToM 推断偏好 + talk 均衡选择 |
| 高反协调度（需差异化） | het_notom 或 het_dp_gated | 多样性本身有利，DP-gating 保留多样性 |

**验收标准**：决策规则在 3 个已知博弈上预测正确，在 public_goods 上给出可检验预测。

### 4.5 P2-A：补 public_goods（4-agent）

**动机**：当前仅 2-agent 博弈。4-agent public goods 验证多智能体可扩展性。

**方案**：
```bash
python3 hettom_experiments/run_round4.py \
  --games public_goods \
  --seeds 42 43 44 45 46 47 48 49 \
  --episodes 30 --horizon 5 \
  --out_dir results/hettom_layer1/round4_public_goods
```

### 4.6 P2-B：Layer-2 MARL 迁移（≥1 博弈）

**动机**：当前纯 LLM-as-agent，无 RL 训练。补 RL 成分回应"这不是 MARL"的质疑。

**方案**：将 Layer-1 最优配置的 ToM 意图特征离线注入 EPyMARL MAPPO 训练（`layer2_marl_bridge.py`），在 LBF 10×10-3p-3f 上验证。不修改奖励信号（避免重蹈 FA-PBS 覆辙），仅在观测侧注入意图特征。

**风险**：时间成本高（1-2 天），且 Layer-1 发现是博弈结构依赖的，LBF 是否对应某种博弈结构未知。若时间不足则放弃，论文基于 Layer-1 独立成文。

---

## 五、AAAI 投稿评估与提升路径

### 5.1 当前评估

| 维度 | 评分 | 理由 |
|---|---|---|
| 新颖性 | ★★★★ | 跨博弈分化 + 模型规模对比无人做过 |
| 方法贡献 | ★★☆ | DP-gating 非 universal 最优，仅 Chicken 突出 |
| 实验严谨性 | ★★★☆ | 真异质模型 + 3 博弈 + 完整对照，但 n=3 种子不足 |
| 理论深度 | ★★★ | 多样性-对齐张力的结构依赖性可形式化 |
| 故事完整性 | ★★★★ | 从发现到方法到限制，叙事弧完整 |
| **AAAI 正会概率** | **25–30%** | 实证强但方法贡献不够 universal + 种子不足 |
| **AAAI Workshop** | **55–65%** | 负面结果 + 机制分析 + 跨博弈验证 |

### 5.2 致命弱点与缓解

| 弱点 | 严重性 | 缓解方案 | 对应实验 |
|---|---|---|---|
| n=3 种子（BoS/Chicken）不显著 | ★★★★★ | 扩到 n=8 | P0-A |
| 3B→7B 仅 Stag Hunt 对比 | ★★★★ | 补全 3 博弈 | P0-B |
| DP-gating 非 universal 最优 | ★★★ | 重新定位为"结构条件方法选择" | P1 |
| 缺 RL 成分 | ★★ | Layer-2 迁移 | P2-B |
| BoS 全方法失败 | ★★ | 诚实呈现为 LLM 协作真瓶颈 | — |

### 5.3 概率提升路径

| 完成阶段 | 累积概率提升 | AAAI 正会概率 |
|---|---|---|
| 当前（Round 4 完成） | — | 25–30% |
| +P0-A（8 种子） | +5% | 30–35% |
| +P0-B（3B 全博弈） | +5% | 35–40% |
| +P1（结构分类器） | +5% | 40–45% |
| +P2（public_goods + RL） | +5–8% | 45–50% |

### 5.4 措辞红线

**应说**：
- "方法的有效性是博弈结构的条件函数"
- "弱模型的协作失败是能力假象（5x 跃升）"
- "多样性-对齐张力的结构依赖反转"
- "DP-gating 在反协调博弈中唯一实现高多样性 + 高回报"
- "BoS 是 LLM 协作的真瓶颈，现有机制无法解决"（诚实负面）

**不应说**：
- "DP-gating 是 universal 最优方法"
- "异质性总是改善/摧毁协作"
- "多样性-对齐张力是 universal 的"
- "方法在所有博弈上有效"

---

## 六、时间线

| 阶段 | 时间 | 内容 | 产出 |
|---|---|---|---|
| 即时 | 现在 | 提交本文档 + Round-4 结果到 GitHub | 仓库更新 |
| 第 1 步 | ~3h GPU | P0-A：扩种子到 8（BoS+Chicken） | 8-seed 统计显著结果 |
| 第 2 步 | ~4h GPU | P0-B：3B vs 7B 全 3 博弈 | 2×3 规模对比表 |
| 第 3 步 | ~2h | P1：博弈结构分类器 | 决策规则 + public_goods 预测 |
| 第 4 步 | ~2h GPU | P2-A：public_goods 4-agent | 多智能体验证 |
| 第 5 步 | 1-2 天 | P2-B：Layer-2 MARL 迁移（时间允许） | RL 成分 |
| 写作 | P0 完成后 | 基于完整数据开始论文写作 | LaTeX 初稿 |

**关键路径**：P0-A → P0-B → 写作开始。P1/P2 为增量提升，不阻塞写作。

---

## 七、关键文件索引

| 文件 | 内容 |
|---|---|
| `hettom_experiments/RESULTS_ROUND4_FINAL.md` | Round-4 最终实验结论（跨 3 博弈完整结果） |
| `hettom_experiments/RESULTS_ROUND4_PARTIAL.md` | Round-4 Stag Hunt 部分结果（含 3B→7B 对比） |
| `hettom_experiments/RESULTS_ROUND3.md` | Round-3 弱模型结果（被 Round-4 推翻） |
| `hettom_experiments/RESULTS_ROUND2.md` | Round-2 cheap-talk + A-ToM 结果 |
| `hettom_experiments/run_round4.py` | Round-4 实验启动器（7B/9B + 4 博弈 + 15 cell） |
| `hettom_experiments/hettom_baseline.py` | Layer-1 核心：矩阵博弈 + 递归 ToM + DP-gating |
| `hettom_experiments/analyze_layer1.py` | 跨种子聚合 + 统计检验 + 预注册决策 |
| `hettom_experiments/layer2_marl_bridge.py` | Layer-2 MARL 迁移桥接（P2-B） |
| `AAAI_NEW_DIRECTIONS.md` | 旧路线放弃 + 新方向调研（文献空白分析） |
| `AAAI_FINAL_EXPERIMENTAL_CONCLUSION.md` | 旧路线（FA-PBS）最终负面结论 |
| `FINAL_EXPERIMENT_PLAN.md` | 旧路线最终实验方案（已废弃，仅存档参考） |
