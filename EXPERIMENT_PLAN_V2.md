# 实验方案 V2 — 回应五点致命批评

> 目标：补齐证据缺口，使"何时对齐可从交互中学出"这一核心主张获得与非显然性匹配的实验支撑。
> 代码状态：所有需要的代码改动**已实现并上传至 Lab** (`/data/lab/gsaca/`)，经语法检查通过；烟雾测试运行中。
> 硬件：2× NVIDIA RTX 5090 (32GB)，4-bit 本地推理，6 并行 worker (3/GPU)。

---

## 批评→回应映射

| # | 致命批评 | 回应实验 | 代码状态 |
|---|---|---|---|
| 1 | 标题主打 heterogeneous 却只测 Qwen+GLM，无同质对照 | **Exp A**：加 Qwen+Qwen / GLM+GLM 同质对照 + 第二异质对 | ✅ `hom_*` cell 已加 |
| 2 | n=5 太少，声称 paired 却用 MWU | **Exp B**：种子 5→20，改用 Wilcoxon signed-rank | ✅ 仅需改分析脚本 |
| 3 | 确定性博弈下 split_score 天然分离，30/30 重言式 | **Exp D**：噪声收益 + Matching Pennies 压力测试 | ✅ `--payoff_noise_std` + `matching_pennies` 已加 |
| 4 | 全是自家消融，缺 payoff-in-prompt 基线 | **Exp C**：`het_payoff_prompt` 让 LLM 直接看收益矩阵 | ✅ `payoff_in_prompt` 已加 |
| 5 | 超参零敏感性分析 | **Exp E**：θ/α/W 扫描 | ✅ CLI flag 已加 |
| 故事 | "反协调有害"对博弈论审稿人是常识；可学习性证据最弱 | **Exp D** + **Proposition** 正式化 | 见下 |
| 投稿 | Eq 6 溢出、代码链接占位 | 已修 | ✅ |

---

## Exp A — 模型对与同质对照（隔离异质性）

### 动机
当前所有结果来自单一异质对 (Qwen+GLM)。无法回答：(a) 异质性本身是否是 CGA/GSACA 有效的必要条件？(b) 同质团队是否已无冲突（信号-信念天然一致）从而 CGA 退化为 no-op？这是标题 "heterogeneous" 的立论根基。

### 配置
新增 3 个模型对维度：
| 对 | agent1 | agent2 | 类型 |
|---|---|---|---|
| QG-het (现有) | Qwen2.5-7B | GLM-4-9B | 异质 |
| QQ-hom (新) | Qwen2.5-7B | Qwen2.5-7B | 同质 |
| GG-hom (新) | GLM-4-9B | GLM-4-9B | 同质 |
| QL-het (新, 可选) | Qwen2.5-7B | Llama-3.1-8B | 第二异质对 |

`QL-het` 需下载 Llama-3.1-8B-Instruct (~16GB，磁盘 6.5TB 充足)。CLI：`--models_het Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct`。

### 运行
- 博弈：5 个 2×2 博弈（不含 public_goods，多智能体单列）
- 种子：先 5（与现有一致），Exp B 完成后扩 20
- cells：`hom_notom hom_gated_atom_talk hom_dp_gated_atom_talk hom_gsaca`（同质对）+ 现有 `het_*`
- episodes：30, horizon 5, gsaca_warmup 5
- **新 cell 数**：5 博弈 × 5 seeds × 4 cells × 2 同质对 = 200 cells（+ 可选 QL-het 100 cells）
- **时间**：~3 min/cell × 200 / 6 并行 ≈ 1.7h（+ QL-het 下载 20min + 0.8h）

### 成功标准
1. **异质性隔离**：QG-het 上 CGA 的反协调增益显著大于 QQ-hom / GG-hom（若同质对上 CGA≈Gated，证明异质性是 CGA 有效的必要条件——直接支撑标题）。
2. **同质对的 split_score**：预期同质对信号-信念天然一致 → `dp_conflict_rate≈0` → CGA 几乎不触发 → split_score 仍能检测但干预率趋零。
3. 若 QL-het 也复现反协调增益，证明结论不限于特定模型对。

---

## Exp B — 种子扩展 + 统计方法修正

### 动机
n=5 时 Wilcoxon 最小 p=0.0625（5 同符号），无法达 p<0.05；当前用 MWU（独立样本）回避了 paired 性质但引入方法学漏洞。审稿人必抓。

### 配置
- 种子：`--seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61`（20 个）
- 统计：改用 **Wilcoxon signed-rank**（配对，n=20 时最小 p≈9.5×10⁻⁷），辅以 BCa CI + Cohen's d + 配对种子胜率
- 重跑全部 6 博弈 × 4 原始 cell × 20 seeds = 480 cells（含 public_goods 20ep）
- **时间**：现有 120 cells 用了 6.5h；480 cells / 6 并行 ≈ 6.5h（与原全量同级，因 public_goods 占比小）

### 成功标准
1. n=20 下 Wilcoxon 在 Chicken/Deadlock/BoS 上达 p<0.01（n=5 时受限于 0.0625 上限的case 现在可正式显著）。
2. 效应量（Cohen's d）方向与 n=5 一致，95% CI 不跨零。
3. 配对胜率 ≥16/20（80%）。

### 分析脚本改动
新建 `analyze_v2.py`：用 `scipy.stats.wilcoxon` 替换 MWU；对 Δ=0 边界用 `wilcoxon(..., zero_method='wilcox', correction=True)`。

---

## Exp C — Payoff-in-Prompt 基线（最致命的缺失基线）

### 动机
当前 4 个 cell 全是同一对齐栈的消融。最朴素的替代方案——把完整收益矩阵写进 prompt 让 LLM 自己推理——从未对比。若该基线已接近 CGA/GSACA，整个对齐机制的价值存疑。

### 配置
- 新 cell：`het_payoff_prompt`（无 ToM / 无 cheap-talk / 无门控，但 action prompt 含完整 2×2 收益矩阵）
- 代码：`_build_action_prompt` 在 `payoff_in_prompt=True` 时渲染矩阵（已实现，见 `hettom_baseline.py:_build_action_prompt`）
- 运行：6 博弈 × 5(→20) seeds × 1 cell = 30(→120) cells
- **时间**：~3min/cell × 30 / 6 ≈ 15min（20 seeds 时 1h）

### 成功标准（关键）
1. **若 payoff-prompt 基线在反协调博弈上显著低于 CGA**：证明对齐机制提供了"看矩阵也推不出"的价值（LLM 即使有完整信息也无法稳定达成分裂均衡）——强支撑 CGA。
2. **若 payoff-prompt 基线接近 CGA**：需重新定位贡献——从"机制优于无对齐"转向"机制以更低 prompt 成本达到同等效果"或"异质团队即使看矩阵也协调失败"。
3. 无论结果如何，**必须如实报告**——这是审稿人最先查的对照。

---

## Exp D — 结构估计器压力测试（核心主张的真正考验）

### 动机
当前 30/30 检测率近乎重言式：确定性 2×2 博弈中，分裂 vs 对称的团队收益差距巨大且零方差，split_score 必然完美分离。"可学习性"主张需要**信息更少、噪声更大、结构更微妙**的设定。

### D1：噪声收益
- 配置：`--payoff_noise_std` ∈ {0.0, 0.5, 1.0, 2.0}（收益量级 0–5，故 σ=2 是重噪）
- 代码：`wrap_payoff_noise` 在 payoff 上加 iid 高斯（已实现）
- 运行：仅 `het_gsaca` cell（检测-focused），6 博弈 × 5 seeds × 4 噪声级 = 120 cells
- **关键问题**：split_score 在 σ 多大时开始误判？检测准确率随 σ 的退化曲线？
- 成功标准：σ≤0.5 时 ≥90% 准确；给出 σ 与准确率的明确退化关系（即使在高噪下失败，也是诚实的边界）。

### D2：Matching Pennies（零和、常量团队收益）
- 博弈：`matching_pennies`——纯反协调但**无纯 NE**，且**团队收益恒为 0**（(1,−1) 或 (−1,1)）
- **预期失败模式**：split_score = mean(diff) − mean(same) = 0 − 0 = 0 → 判为 "coord" → GSACA 错误地关闭门控。这是一个**真实的、可暴露的估计器局限**。
- 运行：matching_pennies × 5 seeds × 4 cells × 3 模型对 = 60 cells
- 成功标准（含失败）：诚实报告 GSACA 在 MP 上的误判，将其定位为 split_score 的已知局限（基于团队收益的检测器无法处理零和/常量团队收益博弈），并在 Limitations 中讨论。**这比隐藏它更能建立可信度。**

### D3（可选）：结构微妙博弈
- 加一个"对称 NE 但分裂更优"的博弈（如偏斜的 Stag Hunt 变体），测试 split_score 在结构模糊时的鲁棒性。

---

## Exp E — 超参敏感性

### 动机
θ、α、W 全用默认值，零扫描。审稿人会质疑结果是否依赖调参。

### 配置
| 超参 | CLI flag | 扫描值 | 固定其余 |
|---|---|---|---|
| θ (信任阈值) | `--gate_trust_threshold` | {0.3, 0.45, 0.6, 0.75, 0.9} | α=0.3, W=5 |
| α (EMA) | `--gate_ema_alpha` | {0.1, 0.2, 0.3, 0.5} | θ=0.6, W=5 |
| W (GSACA warmup) | `--gsaca_warmup` | {2, 3, 5, 8, 10} | θ=0.6, α=0.3 |
- 博弈：选 1 反协调 (Chicken) + 1 协调 (BoS)（代表性）
- seeds：5（敏感性扫描不需高统计功效，看趋势）
- cell：`het_dp_gated_atom_talk`（θ/α）+ `het_gsaca`（W）
- **cell 数**：θ 5×2×5 + α 4×2×5 + W 5×2×5 = 140 cells
- **时间**：~1.2h

### 成功标准
1. CGA/GSACA 的 Δ 方向在全部超参值上不变（即不依赖调参）。
2. 给出超参→payoff 的平坦曲线图（robustness figure）。

---

## 执行优先级与时间表

| 阶段 | 实验 | 预估时间（6并行） | 依赖 |
|---|---|---|---|
| 1 | Exp C (payoff-prompt, 5 seed) | 15min | 烟雾测试通过 |
| 2 | Exp A (同质对照, 5 seed) | 1.7h | 烟雾测试通过 |
| 3 | Exp D1+D2 (压力测试) | 2h | 无 |
| 4 | Exp E (超参) | 1.2h | 无 |
| 5 | Exp B (20-seed 全量重跑) | 6.5h | Exp A/C 确认方向后 |
| **合计** | | **~12h**（可跨多 session） | |

阶段 1–4 可在 ~5h 内完成，足以决定故事是否成立；阶段 5 是统计加固，最后跑。

---

## 故事重构（回应"证据重心与故事重心错位"）

### 当前问题
- 故事重心：在线可学习性（非显然、是贡献）
- 证据重心：反协调博弈的对齐危害（显然、是常识）

### 重构后
1. **承认显然性**：Intro 明确"对博弈论背景的审稿人，反协调博弈中强制收敛有害近乎常识；本文的贡献不在于发现这一点，而在于…"——主动回应 "isn't this obvious"。
2. **转移证据重心到可学习性**：Exp D（噪声 + Matching Pennies）成为 Results 的核心，展示 split_score 的**适用边界与失效条件**——这才是非显然的、有价值的知识。
3. **Proposition 正式化**（见初稿 Discussion 升级）：将"split_score 在确定性博弈下必然分离"提升为带证明的 Proposition，将"零和/常量团队收益下失效"提升为带反例的 Proposition——把"一个统计量 + 一条 if"变成有理论深度的分析。
4. **标题**：去掉 "Learning"（GSACA 不是学习，是估计+切换）。改为更诚实的表述，如 "Detecting When to Align" 或 "Structure-Conditional Alignment"。

---

## 烟雾测试清单（已完成 ✓）

```
[✓] matching_pennies 加载，team payoff 恒为 0
[✓] het_payoff_prompt @ chicken seed42 5ep → payoff=2.92, prompt 含矩阵, 26s
[✓] hom_notom @ chicken seed42 5ep → payoff=2.12, 同质对加载单模型, 6s
[✓] het_gsaca @ matching_pennies seed42 5ep → split=0.000, detect=coord(oracle=anti_coord, MISS)
    → Proposition 2 经验验证成功：GSACA 在零和博弈上误判，正如理论预测
```

**关键发现**：Matching Pennies 烟雾测试完美复现了 Proposition 2 的预测——split_score 坍缩为 0，GSACA 将反协调博弈误判为协调。这验证了估计器的失效模式是真实的、可暴露的，而非理论臆测。
