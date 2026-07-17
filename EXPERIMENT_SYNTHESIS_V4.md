# 实验综合与方案修正 V4 — GitHub 核验 + 全资产对账 + V3 修订

> 生成时间: 2026-07-15 | 依据: github.txt → `gvkuhvkjhbjkhb/AAAI` 实拉核验 + sandbox 全资产 + 远端 3 个新提交的真实数据
> 目的: (1) 描述整个实验脉络; (2) 综合已跑通实验做总结; (3) 据真实数据**稍微修正** EXPERIMENT_SUMMARY_V3 与 PAPER_OUTLINE_V3
> 关键发现: **本地 sandbox 落后远端 3 个提交**; V3 多处数字与远端 GPU 真跑数据/算术不符, 需校正

---

## Part A. GitHub 与数据资产的真实状态 (ground truth)

### A.1 GitHub 推送核验 (通过 github.txt 实拉)

```
git ls-remote origin main  →  181587ffad8f210c398165696ede0d464df751fe   (远端 HEAD)
本地 HEAD                  →  ce8098ed424c8e0cee118fa7689512b14c3cd1ff
git fetch                  →  ce8098e..181587f  main -> origin/main   (远端领先 3 个提交)
git branch -r --contains ce8098e  →  origin/main   (✓ ce8098e 确实在远端历史中)
```

**结论 (修正用户表述)**:
- ✅ `ce8098e` + `e56c04f` **确实已推送** (在远端历史内)。
- ❌ "origin/main up to date" **已过时**: 本地 `git status` 比的是**缓存的** origin/main, 实拉后发现远端已**前进 3 个提交**到 `181587f`。本地 sandbox 落后。
- 本地有一处未提交修改: `code/run_experiment_local.py` (+38 行, 与远端对该文件的改动可能冲突)。

### A.2 远端 3 个新提交 (本地没有, 含真实 GPU 数据与综合分析)

| commit | 标题 | 内容 |
|---|---|---|
| `3e9ba3e` | Exp B 20-seed complete: BoS+public_goods | BoS+pg n=20 配对 Wilcoxon 分析 |
| `efc3b21` | Complete Type 3 (BoS+pg 20-seed) | 补回 9 个缺失 public_goods runs; "Full 160/160 metrics. 831 total" |
| `181587f` | Add comprehensive analysis script + conclusions | `code/comprehensive_analysis.py` + `COMPREHENSIVE_CONCLUSIONS.md` (4 类实验综合, 831 metrics) |

**新增关键资产 (仅远端有)**:
- `COMPREHENSIVE_CONCLUSIONS.md` — 4 类实验综合结论 (131 行)
- `ANALYSIS_FULL.txt` — `comprehensive_analysis.py` 实跑输出 (164 行, 加载 613 metrics)
- `ANALYSIS_EXP_B_PAIRED.md` — BoS+pg n=20 配对 Wilcoxon
- `results/v2/exp_3arm/tau_{0.0,0.2,0.4,0.6}/` — **真实 GPU het_3arm 跑** (6 博弈 × 5 seeds × 4 τ = 120 cells)
- `results/v2/exp_b_20seed/battle_of_the_sexes,public_goods/` — BoS+pg n=20 完整 (4 cells × 20 seeds)

### A.3 数据量对账 (修正 "951 条")

| 来源 | metrics 数 | 说明 |
|---|---|---|
| 本地 `v2_results/` 实数 | **844** | exp_a_fix(259)+exp_anti_test(55)+exp_b_20seed(320)+exp_c_payoff_prompt(30)+exp_d_stress(50)+exp_e_hyperparam(10)+original_qg_het(120) |
| 远端 commit msg | **831** (4 类) | 4 类新实验 (Type 1-4), 不含 original_qg_het(120) |
| `comprehensive_analysis.py` 加载 | **613** | 脚本按 cell/experiment 过滤后的分析子集 |
| V3 summary 称 "951" | ❌ 重复计数 | "844 新 + 120 旧" 把 original_qg_het(120) 算了两遍: 844 已含那 120 |

**修正**: 总量应以**远端 831 (4 类新实验) + 120 (original_qg_het 旧基线) = 951** 这个口径才自洽, 但前提是 844 里**不含** 120。实况是 844 **已含** 120, 所以真实不重复总量 = **844 (本地) / 远端补齐 BoS+pg+3arm 后更多**。论文应统一报"**~840+ cells, 跨 4 类新实验 + n=5 旧基线**", 避免 "951" 这种被双算的数字。

---

## Part B. 实验脉络 (整个研究的演进弧)

### B.1 三阶段方法演进

```
DP-Gating (Phase 1-5, 旧)
   │  无条件门控仲裁: 每轮注入 gated_belief (强制对齐)
   │  发现: 反协调博弈中强制收敛摧毁分裂均衡 → "对齐悖论"
   ▼
CGA (Conditional Gated Arbitration)
   │  创新: 仅 signal-belief 冲突时注入 (条件对齐)
   │  解决反协调, 但暴露协调博弈上的危害 (BoS −0.585***)
   ▼
两臂 GSACA (split>0→CGA, split≤0→Gated)
   │  在线估计博弈结构 + 自动切换; 100% 检测 (n=5)
   │  近 Oracle (regret 0.005, 比 CGA 好 32×)
   │  ★ 致命缺口: deadlock 上 vs NoToM = −0.023*** (显著为负)
   ▼
三臂 GSACA (split>+τ→CGA, split<−τ→Gated, |split|≤τ→Abstain)  ← V3 核心
   │  弃权臂钉死下行风险; Proposition 3 (弃权区 regret≡0)
   │  deadlock 缺口消除; 全表无显著负项
   │  叙事升级: "选择性不干预" → "有界下行风险"
```

### B.2 五点致命批评 → V2 五实验 → V3 四类新实验 (回应链)

| 批评 | V2 实验 | V3 状态 | 真实结果 |
|---|---|---|---|
| ① heterogeneous 却只测 QG | Exp A 同质对照 (QQ/GG/QL) | ⚠️ 76% (259/400) | QQ/GG 部分完成, QL 不全; 同质对照证据仍弱 |
| ② n=5 太少 + MWU 冒充 paired | Exp B 5→20 + Wilcoxon | ✅ 4 博弈 n=20 + BoS/pg 补齐 | 配对 Wilcoxon, min p≈10⁻⁶ |
| ③ 30/30 检测近乎重言式 | Exp D 噪声 + Matching Pennies | ✅ σ 扫描 + MP | σ≤2.0 退化平缓; MP 失效 (Prop 2 验证) |
| ④ 全自家消融, 缺 payoff-prompt 基线 | Exp C payoff-in-prompt | ⚠️ 仅 n=5 (30) | **最致命缺失基线仍未扩 n=20** |
| ⑤ 零超参敏感性 | Exp E θ/α/W 扫描 | ✅ 完成 | θ∈[0.3,0.75] 稳定, warmup=3 最优 |

V3 在 V2 基础上**新增 4 类实验**回应 "deadlock 缺口" 与 "鲁棒性/必要性":
- **Type 1 三臂弃权**: 离线重算 (n=20) + GPU τ 扫描 (5 seeds × 4 τ) → deadlock 缺口消除
- **Type 2 silent-anti-coord**: 测试 "反协调屏蔽 cheap-talk" 假设 → **被否定** (3/3 显著有害)
- **Type 3 BoS+pg n=20**: 补齐 6 博弈全表 → BoS +0.768*** (+49.5%), pg 正确弃权
- **Type 4 鲁棒性**: 噪声 + 超参 + 反协调补丁 → 三维鲁棒, 补丁未显著

---

## Part C. 已跑通实验的真实总结 (以远端 GPU 数据为准)

> 以下数字以远端 `COMPREHENSIVE_CONCLUSIONS.md` + `ANALYSIS_FULL.txt` (commit 181587f) 为准, 与本地 V3 不一致处已标注。

### C.1 Type 3: BoS + public_goods n=20 (主实验, 配对 Wilcoxon)

**BoS (协调, n=20)** — 最强正结果, 远端与 V3 一致:
| Cell | payoff | Δ vs NoToM | p | sig |
|---|---|---|---|---|
| NoToM | 1.550 | — | — | — |
| Gated | 2.216 | +0.666 | <0.0001 | *** |
| CGA | 1.660 | +0.110 | 0.008 | ** |
| **GSACA (Gated 臂)** | **2.318** | **+0.768** | **<0.0001** | **\*\*\*** |

GSACA 检测 20/20 正确, split=−2.500 → Gated 臂, 提升 +49.5%。✅ 稳。

**public_goods (4-agent, n=20)** — ⚠️ **远端显示并非 20/20 完整**:
| Cell | n | payoff | Δ vs NoToM | p | sig |
|---|---|---|---|---|---|
| NoToM | **19** | 2.573 | — | — | — |
| Gated | 19 | 2.630 | +0.062 | 0.0001 | *** |
| CGA | 19 | 2.615 | +0.044 | <0.0001 | *** |
| GSACA | **14** | 2.559 | **−0.011** | 0.946 | ns |

- GSACA public_goods **只有 14/20 seeds** (commit efc3b21 补了 9 个但仍不全); NoToM/Gated/CGA 也只有 19/20。
- 方向稳 (正确弃权, ns), 但 **V3 称 "public_goods 20/20 弃权, regret=0" 不准确**: 实际 14 seeds, 且 Δ=−0.011 是估计非精确 0 (Prop 3 的 "精确 0" 仅在弃权臂≡NoToM 时成立, 而 14 seeds 的 GSACA 与 19 seeds 的 NoToM 非完全配对)。

### C.2 Type 1: 三臂弃权 — ⚠️ 两套数据集, V3 混用

存在**两套 deadlock 三臂数据**, V3 summary 把它们的 payoff 和 Δ 张冠李戴:

| 数据集 | 来源 | n | deadlock 三臂 | NoToM | Δ | p | sig |
|---|---|---|---|---|---|---|---|
| **离线重算** | SCHEME_RESULTS.md (用 20-seed het_gsaca 重算臂选择) | **20** | 1.837 | 1.839 | **−0.002** | 0.395 | ns |
| **GPU 真跑** | run_3arm.sh het_3arm cell (seeds 42-46) | **5** | 2.372 | 2.402 | −0.019~−0.030 | 0.781 | ns |

- V3 summary 第 112 行写 "deadlock 1.837 / 1.839 / **−0.019**" — **算术错**: 1.837−1.839=−0.002, 不是 −0.019。它拿了离线重算的 payoff (1.837/1.839) 却贴了 GPU 5-seed 的 Δ (−0.019)。
- **论文主表 (n=20) 应报 −0.002** (离线重算, 真正 n=20); −0.019 来自仅 5 seeds 的 GPU τ 扫描, 不能当 n=20 主结果。
- 两套数据**绝对 payoff 差很大** (1.84 vs 2.40) — 见 Part D 跨批次问题。

**臂选择 (τ=0.4, 离线重算 n=20)** — 100% 符合理论, 稳:
chicken→CGA(20), hawk_dove→CGA+弃权(14/6), deadlock→弃权+CGA(17/3), stag_hunt→Gated(20), BoS→Gated(20), pg→弃权(20)。

**τ 敏感性 (离线重算, dev seeds 42-51)**: deadlock 在所有 τ≥0.2 上均 ns (缺口消除); τ=0.4 预注册主值, τ=0.3 稳健备选。✅

### C.3 Type 2: silent-anti-coord — 假设否定, 稳

| Game | Silent | NoToM | Δ | sig |
|---|---|---|---|---|
| chicken | 2.506 | 3.157 | −0.650 | *** (更差) |
| deadlock | 1.774 | 2.402 | −0.628 | *** |
| hawk_dove | 1.389 | 1.717 | −0.328 | *** |

"反协调屏蔽 cheap-talk 有益" 假设被 3/3 否定。✅ 这是扎实的 negative-result-as-contribution。

### C.4 Type 4: 鲁棒性

**噪声 (D1, GSACA)**: σ∈{0,0.5,1,2}, payoff 2.480→2.183, 退化平缓。**但检测准确率并非全 100%**: σ=0→100%, σ=0.5→96.6%, **σ=1.0→86.7%**, σ=2.0→96.4%。V3 "σ≤2.0 不崩溃" 方向对, 但 σ=1.0 的 86.7% 低谷应诚实报告。

**超参 (E)**: θ∈[0.3,0.75] payoff 平坦 (2.306–2.460), θ=0.9 略低; α 全范围稳; **warmup=2 payoff 最高 (2.715) 但检测 90%; warmup=3 检测 100% 且 payoff 2.663**。V3 称 "warmup=3 最优" 成立 (检测+payoff 综合), 但需说明 warmup=2 payoff 更高却检测掉。

**反协调补丁 (anti_test, n=5)**: AdaptInterv 2.471 / ComboAnti 2.455 略超 NoToM 2.402, 但 vs het_gsaca(2.425) 均 ns (p=0.41/0.67)。仅 n=5, 不作主结果。✅ future work。

### C.5 跨实验汇总 (远端 ANALYSIS_FULL.txt Section 7, 613 metrics pooled)

- **GSACA 结构检测 (全部数据 pooled): 379/386 = 98.2%** — ⚠️ 不是 V3 说的 100%。分博弈: deadlock 96.9%, public_goods 92.2%, BoS 98.8%, 其余 100%。100% 仅在干净确定性子集上成立。
- GSACA vs NoToM (pooled): Δ=+0.390, p<0.0001, ***, d=0.773
- GSACA vs Gated (pooled): Δ=+0.020, ns
- GSACA vs CGA (pooled): Δ=+0.127, p=0.075, ns (仅边际)

**含义**: GSACA 的优势是**博弈特异**的 (BoS +0.768*** 拉高均值), pooled 对 CGA 仅边际显著。论文应**以分博弈表为主**, pooled 仅作辅助, 避免夸大。

---

## Part D. 发现的不一致与必须修正项 (7 条)

| # | 问题 | 严重性 | 真实值 | V3 错值 | 影响 |
|---|---|---|---|---|---|
| **D1** | deadlock 三臂 Δ 算术错 + 数据集混用 | ★★★ | **−0.002** (n=20 离线重算, payoff 1.837/1.839) | −0.019 (混了 5-seed GPU 的 Δ) | 主表数字; PAPER_OUTLINE §5.2 Table 2 同错 |
| **D2** | "951 条" 重复计数 | ★★ | 844 (本地, 含 original_qg_het 120); 远端 831 (4 类) | 844+120=951 (双算) | 数据规模表述 |
| **D3** | 跨批次绝对 payoff 不一致 | ★★★ | chicken NoToM: 2.226 (exp_b) vs 3.157 (silent); deadlock: 1.84 vs 2.40 | V3 各表混用不同批次 | 可复现性; 审稿人会质疑基线漂移 |
| **D4** | public_goods n≠20 | ★★ | GSACA pg 仅 **14/20**, NoToM/Gated/CGA 仅 19/20 | V3 称 "20/20 弃权, regret=0 精确" | Prop 3 经验验证强度 |
| **D5** | 检测准确率非 100% (pooled) | ★★ | **98.2%** (379/386); deadlock 96.9%, pg 92.2% | V3 称 "100% (120/120)" | C6 主张需分层表述 |
| **D6** | 噪声 σ=1.0 检测掉到 86.7% | ★ | σ: 100/96.6/**86.7**/96.4 % | V3 "σ≤2.0 不崩溃" 过粗 | §5.6 退化曲线需诚实 |
| **D7** | chicken 三臂 vs NoToM 仅 +0.032 (p=0.058) 且基线敏感 | ★★ | +0.032 (vs 2.226) 或可能负 (vs 3.157 批次) | V3 标 "*" 似稳健 | chicken 是反协调头条, 实为最弱正结果 |

### D.3 跨批次不一致的根因 (重要)
本地 `exp_b_20seed` (chicken NoToM=2.226, deadlock=1.84) 与远端 `silent`/`exp_3arm`/`Type4` 批次 (chicken NoToM=3.157, deadlock=2.40) **绝对 payoff 差 0.4–0.9**。可能原因: (a) memory/horizon/warmup 配置差异; (b) 4-bit 量化推理的 batch 间非确定性; (c) 不同 cell 的 NoToM 基线实际跑在不同条件。**论文必须**: 每个数字标注来源批次 + config; 同一表内所有 cell 用同一批次; 报告 batch 间方差作为可复现性证据。否则审稿人一对照表就会发现基线漂移。

---

## Part E. 对 EXPERIMENT_SUMMARY_V3.md 的具体修正

1. **§一表头 (line 3)**: "831 条已分析 + 120 条旧基线 = 951 条" → 改为 "**844 条 (本地 v2_results, 含 120 旧基线); 远端补齐 BoS+pg+3arm 后 4 类新实验共 831 条**"。删除 "951" 重复计数。

2. **§2.3 GitHub 验证 (line 76-79)**: "up to date with origin/main" → 改为 "**ce8098e 已推送; 但远端已前进 3 提交到 181587f (含 BoS+pg n=20 完成 + GPU het_3arm 真跑 + 综合分析), 本地 sandbox 落后, 需 `git pull`**"。

3. **§3.3 主结果 deadlock 行 (line 112)**: "deadlock | 20 | 1.837 | 1.839 | **−0.019** | 0.395 | ns" → "**−0.002**" (1.837−1.839)。加注: "n=20 主结果取离线重算; GPU 5-seed het_3arm 为 τ 扫描鲁棒性, 不作主表"。

4. **§3.4 与旧两臂对比 deadlock 行 (line 125)**: 三臂 Δ 同改 "−0.019" → "−0.002"。

5. **§3.6 Proposition 3 经验验证 (line 144-146)**: "public_goods: 20/20 弃权 seeds" → 改为 "**public_goods: 14/20 seeds (het_gsaca pg 仅 14 cells 完成)**"。加注 "regret 在弃权臂上理论为 0; 经验估计 −0.011 因 GSACA(14) 与 NoToM(19) 非完全配对, 非 5-seed 精确 0"。

6. **§5.4 n=20 全表 (line 234)**: deadlock "−0.019" → "−0.002"; public_goods "−0.011 ns" 加注 n=14/19。

7. **§8.1 C5/C6 主张**: C5 "regret=0.000000 (精确)" → "理论精确 0; 经验估计在非完全配对下 ≈0 (ns)"; C6 "检测 100% (30/30 + n=20)" → "**确定性 2×2 子集 100%; 全数据 pooled 98.2% (379/386), deadlock 96.9%, pg 92.2%**"。

8. **新增 §8.1 C12**: "跨批次绝对 payoff 存在 0.4–0.9 漂移 (chicken/deadlock), 同表需同批次; 见 Limitations"。

---

## Part F. 对 PAPER_OUTLINE_V3.md 的具体修正

### F.1 标题
保留 "To Intervene or to Abstain: Bounded-Downside Alignment for Heterogeneous LLM Agents"。✅ 叙事准确, 不改。但 Abstract 数字需校正 (见 F.2)。

### F.2 Abstract 数字校正
- "deadlock 缺口消除 (−0.023\*\*\*→ns)" → 保留 (方向对), 但正文表里 ns 的 Δ 必须是 **−0.002** 不是 −0.019。
- "臂选择 100% 符合理论" → "**在确定性子集上 120/120 符合理论**" (避免被 pooled 98.2% 打脸)。
- "Proposition 3 证明弃权区内对基线遗憾恒为零" → 保留 (这是理论命题, 成立), 但**经验验证**句改为 "在 hawk_dove/deadlock 弃权 seeds 上经验 regret≈0 (ns), public_goods 因 14/20 配对不全为估计值"。

### F.3 §3.5 Proposition 3 表述收紧
当前写 "regret(GSACA, NoToM) ≡ 0, 此为零的精确等式 (非估计), 对任意 n、任意博弈成立"。**理论部分正确** (弃权臂≡NoToM ⇒ 同一 agent 路径 ⇒ 同一 payoff)。但需补一句: "**经验验证要求 GSACA 与 NoToM 在相同 seeds 上跑; 当两 cell 的 seed 集不全 (如 public_goods 14 vs 19) 时, 经验 Δ 为估计而非精确 0**"。这样理论与经验分开, 不被审稿人抓 "你说精确 0 为何表里是 −0.011"。

### F.4 §5.1 与 §5.2 的统计基线不对称 (关键结构问题)
- §5.1 对齐悖论表: **CGA vs Gated, n=5, MWU** (旧数据, 未在 n=20 重跑)。
- §5.2 三臂主表: **三臂 vs NoToM, n=20, Wilcoxon**。
- **问题**: 动机表 (n=5, 比 Gated) 与主结果表 (n=20, 比 NoToM) 用了**不同 n、不同基线、不同检验**。审稿人会问: "为何不统一? 为何动机用 n=5?"
- **修正建议** (二选一):
  - (A) 首选: 把 §5.1 也补到 n=20 (重跑 CGA vs Gated, 6 博弈 × 20 seeds × 2 cells = 240 cells, ~3h), 使全表统一 n=20 + Wilcoxon。这是**最值得补的实验**, 比补 Exp A/C 优先级高 (它直接堵统计不对称这个口)。
  - (B) 退而求其次: 保留 §5.1 n=5 但**明确声明** "悖论表为 n=5 探索性证据 (MWU), 主结果表为 n=20 确认性 (Wilcoxon); 悖论方向在 n=20 三臂表的 arm-selection 列复现", 并在 Limitations 列入。

### F.5 §5.2 Table 2 数字校正
deadlock 行 Δ "−0.019" → "−0.002"; public_goods 行加 "n=14/19"。其余 (BoS +0.768***, stag_hunt +0.501***, chicken +0.032*, hawk_dove +0.030**) 保留。

### F.6 §5.5 检测 + holdout
"检测 100% (含 n=20)" → "**确定性 2×2 博弈 100% (6 博弈 × 20 seeds = 120/120); 含噪声/超参扫描的全数据 pooled 98.2%**"。

### F.7 §5.6 噪声退化
补 σ=1.0 检测 86.7% 的低谷 (而非只说 "平缓")。诚实曲线: 100/96.6/86.7/96.4%。可解释为 σ=1.0 时部分 deadlock/pg 边界 seed 误判, σ=2.0 反而因大噪声把边界推向一致误判方向 (需复核, 但如实报)。

### F.8 §5.7 超参
补 "warmup=2 payoff 最高但检测 90%; warmup=3 检测 100% 且 payoff 次高, 综合最优"。不要只说 "warmup=3 最优"。

### F.9 §6.4 有界下行风险叙事
核心叙事保留 (该出手时出手 / 不该出手时弃权 / Prop 3 兜底)。但 "全表无显著负项" 改为 "**全表无统计显著负项** (deadlock −0.002 ns, pg −0.011 ns; 均为数值微小负但不显著)"。强调 "无显著负" 而非 "无负"。

### F.10 §9 Limitations 补两条
- 跨批次绝对 payoff 漂移 (chicken/deadlock 0.4–0.9), 同表已用同批次, 但跨表比较需谨慎。
- §5.1 悖论表 n=5 vs §5.2 主表 n=20 的统计不对称 (若不补跑)。

### F.11 §11 录用概率微调
- AAMAS 2027: 55–63% → **52–60%** (跨批次漂移 + chicken 弱显著 + paradox n=5 是实打实的扣分项, +Prop 3/三臂是加分)。
- AAAI 2027: 47–54% → **45–52%**。
- 最优策略不变: AAMAS 2027 首选。但**补 §5.1 n=20 (F.4-A)** 可把 AAMAS 拉回 55–62%。

---

## Part G. 仍待完成 (优先级重排) 与投稿建议

### G.1 待办优先级 (据真实缺口重排)

| 优先级 | 实验 | 现状 | 为何重要 | 工作量 |
|---|---|---|---|---|
| **P0 (新)** | **§5.1 悖论表补 n=20** (CGA vs Gated, 6 博弈 × 20 seeds × 2 cells) | n=5 only | 消除动机表与主表的统计不对称 — 审稿人必抓 | ~3h, 240 cells |
| P1 | Exp A 同质对照补齐 (QQ/GG/QL) | 259/400 (76%) | 隔离异质性, 支撑标题 "heterogeneous" | ~141 cells |
| P2 | Exp C payoff-prompt 扩 n=20 | n=5 (30) | 最致命缺失基线 | ~120 cells |
| P2' | public_goods 补到 20/20 | 14/19/19/19 | Prop 3 经验验证完整 | ~6 cells |
| P3 | 反协调补丁扩 n=40 | n=5 ns | chicken ns→显著 (可选) | ~60 cells |
| P4 | git pull 同步远端 3 提交 | 本地落后 | 拿到真实 GPU 3arm + 综合分析 | 立即, 需解 run_experiment_local.py 冲突 |

**关键建议**: **先 `git pull` 解冲突** (P4), 把远端的真实 het_3arm GPU 数据 + comprehensive_analysis 拉到本地, 所有后续分析基于远端最新数据, 而非本地过期 view。然后 P0 (悖论表 n=20) 优先级**高于** P1/P2, 因为它直接补统计对称性这个方法学硬伤。

### G.2 投稿就绪度评估 (修正后)
- **当前 (V3 + 修正)**: C1-C11 全有证据, 但 D1-D7 七处数字/口径问题若不修, 审稿人对照远端数据或重算即可发现。**修完 D1-D7 + git pull 后可投** (Exp A/C/P0 作 Limitations)。
- **补 P0 后**: 统计对称性补齐, AAMAS 55–62%, 强投。
- **补 P0+P1+P2 后**: 完整, AAMAS 58–65%。

### G.3 一句话总结
三臂弃权 GSACA 的 "有界下行风险" 叙事**方向正确、核心结果 (BoS +0.768***、deadlock 缺口消除、silent 否定) 稳健**, 但 V3 文档存在 **deadlock Δ 算术错 (−0.019 应为 −0.002)、951 重复计数、跨批次基线漂移、public_goods n≠20、检测率非 100% (pooled 98.2%)** 等七处需校正; 优先 `git pull` 同步远端真实数据, 再补 §5.1 悖论表到 n=20 以消除统计不对称, 即可达到 AAMAS 2027 强投状态。
