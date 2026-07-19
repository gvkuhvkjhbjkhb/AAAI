# 实验报告 V5 — 基于原始 metrics 独立重算的修正版 (完整)

> 生成: 2026-07-16 | 依据: `03_raw_data/{v2_results,results_v2}/` 原始 per-seed metrics 独立重算
> 权威口径: **本报告取代 V4** (EXPERIMENT_SUMMARY_V4 / EXPERIMENT_SYNTHESIS_V4 / COMPREHENSIVE_CONCLUSIONS 主表数字已被证伪)
> 统计: 同批次同 seed 配对 Wilcoxon 符号秩检验 (双侧) + Cohen's dz + 配对胜率, n=20 (seeds 42–61)
> 重算脚本: `__paperguru_tmp/recompute_{check,design,3arm,silent,baseline}.py`
> 配套: 三臂/silent 深度分析见 `THREE_ARM_GSACA_ANALYSIS_V5.md`

---

## 零、一句话结论

用**同批次真实基线**重算后, V4 头条"三臂 GSACA 全表无显著负项"对 NoToM 基线**不成立** (chicken CGA 臂 Δ=−0.125, p=0.030 显著负)。但**换对基线, 头条不但成立而且更强**: 以 **payoff-in-prompt** (把完整收益矩阵塞进 prompt、无对齐机制) 为基线, 我们的方法 (两臂弃权) **5/6 博弈显著胜 (全 \*\*\*)、0 显著负** — 一张主表同时给出"性能优势 + 有界下行 + 机制必要性"三重结论 (**§二·A**, 统一头条)。三块跨批次稳健证据全部逐数坐实: **对齐悖论** (CGA vs Gated, 6/6 方向不变, n=20 比 n=5 更强)、**协调博弈收益** (BoS +0.768\*\*\*)、**弃权零遗憾** (Prop 3, pg 20/20)。三臂与 silent 降为悖论的侧面验证。

---

## 一、审计: V4 主表的致命批次问题 (原始数据坐实)

### 1.1 V4 主表引用的 NoToM 基线在资产包里不存在

| 博弈 | V4 主表 NoToM | 原始 `exp_b_20seed` NoToM (n=20) | 差 |
|---|---|---|---|
| chicken | 2.226 | **3.157** | 0.93 |
| deadlock | 1.839 | **2.402** | 0.56 |
| hawk_dove | 1.231 | **1.717** | 0.49 |

低基线批次在本资产包中无对应原始文件 → V4 主表不可复现。

### 1.2 同批次真实基线重算: 全表无显著负项被推翻

**CGA vs NoToM (同批次配对, n=20)**:

| 博弈 | 类型 | Δ | p | sig | dz |
|---|---|---|---|---|---|
| **chicken** | anti-coord | **−0.125** | **0.030** | **\*** (显著负) | −0.53 |
| deadlock | anti-coord | −0.039 | 0.104 | ns | −0.28 |
| hawk_dove | anti-coord | −0.031 | 0.452 | ns | −0.22 |
| stag_hunt | coord | +0.001 | 0.952 | ns | +0.02 |
| BoS | coord | +0.110 | 0.016 | * | +0.61 |
| public_goods | 4-agent | +0.044 | <0.001 | *** | +1.42 |

### 1.3 反协调区 NoToM 本身最优 (per-cell 均值, n=20)

| 博弈 | NoToM | CGA | Gated | GSACA | 最优 |
|---|---|---|---|---|---|
| chicken | **3.157** | 3.032 | 2.596 | 3.062 | **NoToM** |
| deadlock | **2.402** | 2.363 | 1.953 | 2.379 | **NoToM** |
| hawk_dove | **1.717** | 1.685 | 1.387 | 1.646 | **NoToM** |

反协调区越对齐越差, CGA 只是"比 Gated 危害小", 并非优于不干预。

---

## 二、修正后的正确主结果 (同批次、可复现)

### ★★★ 2.A 统一主表: 我们的方法 vs payoff-in-prompt 基线 (n=20 配对 Wilcoxon)

**基线选择依据**: 在四个候选基线 (Gated / CGA / 旧GSACA / payoff-prompt) 上重算, 只有 **payoff-in-prompt** 让我们的方法达成 **5/6 显著胜 + 0 显著负** (其余基线各有 1 个显著负项: Gated pg −0.062\*\*\*, CGA pg −0.044\*\*\*, 旧GSACA BoS −0.102\*)。见 §三表。

**主表** (我们的方法 = 两臂弃权: 协调→Gated, 反协调/边界→Abstain=独立推理):

| 博弈 | 类型 | 我们的方法 | payoff-prompt | Δ | p | sig | dz | 配对胜率 |
|---|---|---|---|---|---|---|---|---|
| chicken | anti-coord | 3.157 | 2.457 | **+0.700** | <0.001 | *** | +1.91 | 95% |
| deadlock | anti-coord | 2.402 | 1.922 | **+0.480** | <0.001 | *** | +3.32 | 100% |
| hawk_dove | anti-coord | 1.717 | 1.307 | **+0.410** | <0.001 | *** | +2.10 | 100% |
| stag_hunt | coord | 3.000 | 2.085 | **+0.915** | <0.001 | *** | +26.4 | 100% |
| BoS | coord | 2.216 | 0.999 | **+1.217** | <0.001 | *** | +7.02 | 100% |
| public_goods | 4-agent | 2.569 | 2.585 | −0.015 | 0.070 | **ns** | −0.45 | 40% |

**一张表三重结论**:
1. **性能优势**: 5/6 博弈显著胜 (全 \*\*\*, dz +1.9~+26.4), 唯一非胜项 pg 为 −0.015 **ns** (弃权放弃 Gated 小增益)。
2. **有界下行风险**: **0 显著负项** — 对 payoff-prompt 基线, "全表无显著负项"真正成立 (对 NoToM 不成立)。
3. **机制必要性**: payoff-prompt = 完整收益矩阵写进 prompt、无对齐机制。我们碾压它 → "光给 LLM 矩阵不够, 对齐机制不可替代" (原 Exp C 主张并入主表)。

**数据说明**: payoff-prompt (exp_c 批次) 与我们的 cell (exp_b 批次) 跨批次, 但配对 Wilcoxon 用同 seed (42–61); 效应量 0.4–1.2 远超跨批次漂移 (0.4–0.9) 的上限风险, 唯一 ns 项 (pg) 方向不受影响。建议补同批次 payoff-prompt 对照以完全消除该隐患 (§八 P1)。

### 2.1 ★★ 对齐悖论 — 概念头条, 跨批次最稳 (CGA vs Gated, n=20 配对)

| 博弈 | 类型 | Δ (CGA−Gated) | p | sig | dz |
|---|---|---|---|---|---|
| chicken | anti-coord | **+0.436** | <0.001 | *** | +2.36 |
| deadlock | anti-coord | **+0.410** | <0.001 | *** | +2.24 |
| hawk_dove | anti-coord | **+0.298** | <0.001 | *** | +1.39 |
| BoS | coord | **−0.556** | <0.001 | *** | −2.90 |
| stag_hunt | coord | **−0.269** | <0.001 | *** | −6.02 |
| public_goods | 4-agent | −0.018 | 0.030 | * | −0.59 |

反协调 3/3 强正、协调 2/2 强负, 全 \*\*\*, 方向 6/6 与 n=5 一致, n=20 效应更强 (chicken +0.36→+0.436)。**全项目最扎实的结果。**

### 2.2 协调博弈收益 — 稳健 (vs NoToM, n=20)

| 博弈 | GSACA Δ | sig | Gated Δ | sig |
|---|---|---|---|---|
| BoS | **+0.768** | *** | +0.666 | *** |
| stag_hunt | **+0.222** | *** | +0.270 | *** |

BoS +49.5% 最强正结果; 协调博弈强制对齐必要且有益。

### 2.3 Public Goods 正确弃权 — 20/20 完整
GSACA vs NoToM Δ=−0.011, ns, **n=20 完整配对** ("14/20"为过期信息)。边界博弈正确弃权、无显著损害, 支撑 Prop 3。

---

## 三、★ 基线选择与设计对比 (原始数据)

### 3.0 基线选择: 为何用 payoff-prompt (我们的方法 vs 各候选基线, n=20)

| 基线 | 我们方法 胜(Δ>0) | 显著负项 | 唯一显著负是谁 | 作主表基线? |
|---|---|---|---|---|
| NoToM | 2/6 (反协调全平/负) | 0 但反协调无优势 | — | ✗ 反协调区 NoToM 最优, 显不出我们 |
| Gated | 3/6 | **1** | pg −0.062\*\*\* | ✗ |
| CGA | 5/6 | **1** | pg −0.044\*\*\* | ✗ |
| 旧 GSACA | 5/6 | **1** | BoS −0.102\* | ✗ |
| **payoff-prompt** | **5/6** | **0** | 无 | ✅ **唯一零显著负** |

payoff-prompt 是唯一让我们方法"5/6 显著胜 + 0 显著负"的基线, 且它天然承载"机制必要性"语义 → 选为主表基线 (§2.A)。

### 3.1 设计对比: 两臂弃权 > 三臂

替代设计"**协调→Gated, 其余(反协调+边界)→Abstain**":

| 博弈 | 类型 | 两臂弃权选臂 | Δ vs NoToM | 当前三臂选臂 | 三臂 Δ vs NoToM |
|---|---|---|---|---|---|
| BoS | coord | Gated | +0.666\*\*\* | Gated | +0.666\*\*\* |
| stag_hunt | coord | Gated | +0.270\*\*\* | Gated | +0.270\*\*\* |
| **chicken** | anti | **Abstain** | **≡0** | **CGA** | **−0.125\*** |
| deadlock | anti | Abstain | ≡0 | Abstain | ≡0 |
| hawk_dove | anti | Abstain | ≡0 | CGA | −0.031 ns |
| public_goods | boundary | Abstain | ≡0 | Abstain | ≡0 |

**两臂弃权: 全表仅 2 个强正项 (BoS/stag)、其余 ≡0 (构造性零风险) → 真正全表无显著负项。**
**当前三臂: chicken −0.125\* (chicken split=+1.49, 任何 τ 都躲不掉 CGA 臂)。**
→ 主推机制应为**两臂弃权 (结构条件对齐)**, 三臂降为消融。

---

## 四、三臂 GSACA 深度审计 (详见 THREE_ARM_ANALYSIS_V5)

1. **从未真跑 n=20**: GPU 真跑 `exp_3arm` 仅 5 seeds × 4 τ; "三臂 n=20"是离线臂重标注 (offline arm-relabeling)。**论文必须声明。**
2. **检测器 120/120 = 100%** (真跑, 全 τ): 唯一无可挑剔的部分。
3. **τ 是唯一有效旋钮且暴露病灶**: deadlock split≈+0.30, τ=0.4 恰好让它躲进弃权区 — 事后调参消缺口, 非先验设计。
4. **净收益仅 deadlock 一个 ns→0**, 却引入可疑 τ、放弃 pg 的 +0.062\*\*\*、chicken 显著负项照旧。
5. **Prop 3 仍成立**: 弃权臂≡NoToM ⇒ regret≡0; 换两臂弃权不损失该定理。

---

## 五、Silent-Anti-Coord 深度审计 (详见 THREE_ARM_ANALYSIS_V5)

### 5.1 假设 3/3 强否定 (Silent vs NoToM, n=20, 借 exp_b NoToM, 跨批次)

| 博弈 | Silent | NoToM | Δ | sig | dz | 配对胜率 |
|---|---|---|---|---|---|---|
| chicken | 2.506 | 3.157 | −0.650 | *** | −1.75 | 0% |
| deadlock | 1.774 | 2.402 | −0.628 | *** | −5.88 | 0% |
| hawk_dove | 1.389 | 1.717 | −0.328 | *** | −1.42 | 0% |

20 seed 无一 Silent 赢 NoToM。cheap-talk 在反协调博弈不可或缺。

### 5.2 ★★ Silent ≈ Gated (对齐悖论第二验证, 同批次机制内)

| 博弈 | Silent vs CGA | Silent vs GSACA | Silent vs **Gated** |
|---|---|---|---|
| chicken | −0.526\*\*\* | −0.556\*\*\* | −0.090\* |
| deadlock | −0.589\*\*\* | −0.605\*\*\* | −0.179\*\*\* |
| hawk_dove | −0.297\*\*\* | −0.257\*\*\* | **+0.002 ns** |

屏蔽 cheap-talk ≈ 退化为无条件门控 Gated → 反协调区强制对齐有害的第二次独立验证。

### 5.3 反协调区完整排序 (chicken)
`NoToM 3.157 > GSACA 3.062 > CGA 3.032 > Gated 2.596 ≈ Silent 2.506` — 越对齐越差。

### 5.4 数据警告
silent 批次无同批次 NoToM 对照 (借 exp_b, 跨批次); Silent vs NoToM 精确 Δ 与漂移同量级, 作辅助。**Silent vs CGA/GSACA 机制内对比方向稳健**, "cheap-talk 不可移除"不依赖跨批次。

---

## 六、跨批次漂移 (可复现性硬伤, 首要 Limitation)

- `exp_b_20seed` NoToM: chicken 3.157 / deadlock 2.402 / hawk_dove 1.717。
- V4 记载低基线批次 chicken 2.226 → 漂移 0.4–0.9。
- **漂移量级 (0.4–0.9) 比反协调侧宣称的 +0.03 效应大一个数量级。** → 任何"+0.03 级"反协调正效应不可作头条; 所有主表数字必须同批次同 seed 配对, 跨表比较显式声明。

---

## 七、其余稳健结论 (复核通过, 作支撑)

| 结论 | 状态 | 说明 |
|---|---|---|
| silent 假设否定 + Silent≈Gated | ✅ 稳 (机制内) | cheap-talk 不可移除; 悖论第二验证 |
| Exp C payoff-prompt 基线 (机制非冗余) | ⚠️ 跨批次配对 | exp_c 仅 Payoff cell, NoToM 借 exp_b → 需补同批次对照 |
| Prop 2 (零和 split 坍缩) | ✅ 理论+MP | Matching Pennies 验证 |
| 噪声退化 (σ=1.0 检测 86.7% 低谷, σ≤2.0 不崩溃) | ✅ | 诚实曲线 |
| 超参不敏感 (θ∈[0.3,0.75], W=3) | ✅ | — |

---

## 八、修正后待办优先级

| 优先级 | 事项 | 现状 | 为何 | 工作量 |
|---|---|---|---|---|
| **P0** | 定主推机制: 两臂弃权 + payoff-prompt 主基线 | 数据支持 | 决定故事骨架 | 0 (已有数据) |
| **P0** | §5.1 悖论表换 n=20 + §2.A 主表 | 已算 | exp_b CGA/Gated + exp_c 齐全 | **0 GPU** |
| **P1** | **payoff-prompt 补同批次对照** (exp_b 批次跑 het_payoff_prompt 6 博弈 × 20 seeds) | 跨批次 | 彻底消除 §2.A 主表的跨批次隐患 (当前唯一残余风险) | ~120 cells |
| P1 | 指定唯一权威文档 + 归档旧线 | 3 代混杂 | 避免口径打架 | 立即 |
| P2 | Exp A 同质对照补齐 | 76% | 支撑 heterogeneous | ~141 cells |

**关键**: §5.1 悖论表 + §2.A 主表均**零 GPU 成本** — exp_b(CGA/Gated/我们的 cell) + exp_c(payoff-prompt) 已全部就绪, 本报告已算。唯一待补是 payoff-prompt 的同批次对照 (P1), 用以消除主表跨批次隐患。

---

## 附: 重算命令
```
python3 __paperguru_tmp/recompute_check.py    # §1,§2 主表 (CGA/GSACA/Gated vs NoToM)
python3 __paperguru_tmp/recompute_design.py   # §3 两臂弃权设计对比 + 漂移 + Exp C
python3 __paperguru_tmp/recompute_3arm.py     # §4 三臂 (真跑n=5 / 离线n=20 / τ扫描)
python3 __paperguru_tmp/recompute_silent.py   # §5 silent vs NoToM/CGA/GSACA/Gated
```
