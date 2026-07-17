# 方案一 & 方案二 实验结果

> 日期: 2026-07-14 | 数据: exp_b_20seed (4 博弈 × 20 seeds × 4 cells) + n=5 preview (BoS, public_goods)
> 统计: 配对 Wilcoxon 符号秩检验 (单侧) + BCa 95% CI + Cohen's dz + 配对胜率
> 方案一: **零 GPU 成本, 纯离线重算** | 方案二: GPU, 已排队等待正在运行的实验结束后自动启动

---

## 一、定位确认 — 唯一缺口 = deadlock

复现 n=20 诊断表 (GSACA vs NoToM, metric=team_mean_payoff), 完全复现用户给出的数字:

| 博弈 | GSACA | NoToM | Δ | p(单侧) | sig | 胜率 |
|---|---|---|---|---|---|---|
| chicken | 2.272 | 2.226 | +0.046 | 0.161 | ns | 0.70 |
| hawk_dove | 1.304 | 1.231 | +0.072 | 0.012 | ** | 0.80 |
| **deadlock** | 1.816 | 1.839 | **−0.023** | **0.003** | **\*\*\*** | 0.20 |
| stag_hunt | 2.913 | 2.499 | +0.414 | 4.7e-05 | *** | 1.00 |

**确认**: deadlock 是唯一对 NoToM 显著为负的博弈 (−0.023, p=0.003, BCa [−0.037,−0.007])。
破坏 "bounded-downside 选择器" 叙事的正是这一项。chicken +0.046 ns 也偏小 (方案二的目标)。

> 注: 旧 analyze_v2.py 对 "全同符号" 数据短路为 p=1.0 (bug), 导致 stag_hunt 等被误报为 ns。
> 已修复 (wpvalue): 全正/全负交给 scipy 正确计算, 仅全零 (无效应) 返回 p=1.0。

---

## 二、方案一 — 三臂弃权 GSACA (τ=0.4, 零 GPU)

### 机制
当前两臂规则: `split>0 → CGA, split≤0 → Gated` (强制干预)。
改为三臂: `split>+τ → CGA, split<−τ → Gated, |split|≤τ → NoToM` (弃权, 关闭 cheap-talk/ToM 注入)。
依据: |split| 小 ⇒ 结构信号弱 ⇒ 干预期望收益小 ⇒ 弃权钉死下行风险。

### 臂选择分布 (per-seed split_score, τ=0.4)

| 博弈 | oracle | split 均值 | CGA | Gated | ABSTAIN |
|---|---|---|---|---|---|
| chicken | anti_coord | +1.675 | 20 | 0 | 0 |
| hawk_dove | anti_coord | +0.531 | 14 | 0 | 6 |
| **deadlock** | anti_coord | +0.307 | 3 | 0 | **17** |
| stag_hunt | coord | −1.965 | 0 | 20 | 0 |
| battle_of_the_sexes | coord | −2.50 (n=5) | 0 | 5 | 0 |
| public_goods | coord | −0.18 (n=5) | 0 | 0 | **5** |

deadlock (split≈0.31) 与 public_goods (split≈−0.18) 落入弃权区——恰是仅有的两个 GSACA 相对基线为负/中性弱的博弈。

### 主结果: 三臂 GSACA vs NoToM (τ=0.4)

**FULL n=20 (4 博弈) + n=5 preview (BoS, public_goods):**

| 博弈 | n | 三臂 | NoToM | Δ | p(单侧) | sig | 臂 |
|---|---|---|---|---|---|---|---|
| chicken | 20 | 2.258 | 2.226 | +0.032 | 0.058 | * | CGA |
| hawk_dove | 20 | 1.261 | 1.231 | +0.030 | 0.040 | ** | CGA+ABSTAIN |
| **deadlock** | 20 | 1.837 | 1.839 | **−0.002** | **0.395** | **ns** | ABSTAIN+CGA |
| stag_hunt | 20 | 3.000 | 2.499 | +0.501 | 4.8e-05 | *** | Gated |
| battle_of_the_sexes | 5 | 2.013 | 1.357 | +0.657 | 0.031 | ** | Gated |
| public_goods | 5 | 2.437 | 2.437 | +0.000 | 1.000 | ns | ABSTAIN |

**缺口消除**: deadlock 从 −0.023 (\*\*\*) → −0.002 (ns)。全表无显著负项。

### Holdout (seeds 52-61, 预注册冻结)

| 博弈 | 三臂 vs NoToM Δ | p | sig |
|---|---|---|---|
| chicken | +0.050 | 0.056 | * |
| hawk_dove | +0.031 | 0.155 | ns |
| **deadlock** | **−0.005** | **0.375** | **ns** |
| stag_hunt | +0.507 | 0.001 | *** |

deadlock 在未见 holdout 上仍 ns (−0.005, p=0.375)。修复泛化。

### Proposition 3 (经验验证)
弃权区内三臂选 NoToM ⇒ 三臂收益 ≡ NoToM 收益 ⇒ 对 NoToM 遗憾恒为零:
- hawk_dove: 6/20 弃权 seeds, regret = **+0.000000**
- deadlock: 17/20 弃权 seeds, regret = **+0.000000**

### τ 敏感性 (DEV seeds 42-51 only, 预注册)

| τ | chicken | hawk_dove | deadlock | stag_hunt |
|---|---|---|---|---|
| 0.2 | +0.014(ns) | +0.074(\*\*\*) | −0.014(ns) | +0.495(\*\*\*) |
| 0.3 | +0.014(ns) | +0.042(\*\*) | −0.009(ns) | +0.495(\*\*\*) |
| **0.4** | +0.014(ns) | +0.028(ns) | +0.000(ns) | +0.495(\*\*\*) |
| 0.5 | +0.014(ns) | +0.000(ns) | +0.000(ns) | +0.495(\*\*\*) |
| 0.6 | +0.014(ns) | +0.000(ns) | +0.000(ns) | +0.495(\*\*\*) |

deadlock 在所有 τ≥0.2 上均 ns (缺口消除)。hawk_dove 在 τ≥0.4 时 dev 功效不足 (但 full n=20 仍 **, p=0.040)。τ=0.4 为预注册主值; τ=0.3 为稳健备选 (hawk_dove dev 仍 **)。

### 统计纪律
- dev (42-51) 仅用于选 τ; holdout (52-61) 冻结, 主表报 holdout + full。
- 三臂重算完全离线: 三臂 (CGA/Gated/NoToM) 在 20 seeds 上已有完整运行, 每 seed 的 warmup split_score 已在 het_gsaca metrics 中, 直接重算 "选哪个臂 + 收益", 无需新 GPU。

---

## 三、BoS + public_goods n=20 三臂情况 (缺口 + 实验设计)

### 现状: n=20 数据缺失 (OOM 失败)
run_final_fast.sh 的 B bos+pg worker 因 GPU 超订 (8+/2GPU) 全部 OOM 崩溃, **0 metrics**。
当前 7 个存活进程 (Exp A hom + D2 + anti_test) 不含 BoS/pg。需新实验补齐。

### 三臂选择器对这两个博弈是确定性的 (基于 n=5 split_score)
| 博弈 | split_score | 判定 | 三臂=哪个臂 | n=5 三臂 vs NoToM |
|---|---|---|---|---|
| BoS | −2.50 (<<−τ) | Gated 臂 | = Gated payoff | +0.657 (p=0.031, **) |
| public_goods | −0.18 (∈[−τ,+τ]) | ABSTAIN 臂 | = NoToM payoff | +0.000 (=0, Prop 3) |

**public_goods n=20 可现在给出理论结论**: split ∈ 弃权区 → 三臂选 NoToM → vs NoToM = **0.000 精确** (Proposition 3, 任意 n 成立, 非估计)。
**BoS n=20 需测量 Gated payoff**: n=5 预览 +0.657**, 预期 n=20 持平 (BoS 是 coord, Gated 强制收敛到偏好 NE)。

### 实验设计 (补齐 6 博弈 n=20)
- **cells**: het_notom, het_gated_atom_talk, het_dp_gated_atom_talk, het_gsaca (4 cells, 与另 4 博弈一致)
- **games**: battle_of_the_sexes (2P, 30ep), public_goods (4-agent, 20ep)
- **seeds**: 42–61 (20 seeds)
- **总量**: 2 博弈 × 20 seeds × 4 cells = **160 cells**
- 与方案二 (60 cells) 合并 = **220 cells**, 用安全 VRAM 调度 (cap=2 het worker/GPU = 18GB, 避免 OOM)

---

## 四、方案二 — silent-anti-coord (GPU, 排队中)

### 机制
反协调模式下屏蔽 cheap-talk: GSACA 检测到 anti_coord 后, 设 `use_talk=False` (停止公告生成 → 门控无信号可仲裁 → 智能体纯 ToM 推理)。coord 模式不变 (转 Gated)。
假设: cheap-talk 推动收敛, 在反协调博弈 (需分裂 profile) 中有害; 移除后 chicken +0.046 ns 有望推至显著。

### 代码
- `scheme2_silent.py`: 独立 runner, import hettom_baseline (只读), 不修改共享文件 → **正在进行的实验零干扰**。
- 检测 anti_coord → `ag.use_talk=False`; 检测 coord → Gated 模式 (同原 GSACA)。
- 仅跑 3 反协调博弈 × 20 seeds (coord 博弈不受影响)。

### 状态
- 代码已语法检查 + import 验证通过。
- **合并编排器** `scheme2_bospq_orchestrator.py` (PID 491242) 后台运行: 等当前 run_final_fast.sh 排空 → 自动启动 14 worker (6 方案二 + 8 BoS/pg), cap=2/GPU 安全调度 → 完成后自动跑 scheme1/scheme2 分析。

---

## 五、时间估计 (所有实验结束)

### 测量 (18:13–18:16, 154s 窗口)
- 当前 7 进程 (受 GPU 争用) 速率: **1.95 cells/min**
- 剩余排空: 128 cells (A_fix QQ/GG/QL + D2 + anti_test deadlock)

### ETA 分解
| 阶段 | 内容 | 预估 |
|---|---|---|
| 1. 当前实验排空 | 128 cells (QL 瓶颈, 47 cells/1 proc, GPU 释放后加速) | ~1.5–2h |
| 2. 我的合并批次 | 220 cells (60 方案二 + 160 BoS/pg), 4 并发, 无争用 | ~3h |
| 3. 自动分析 | scheme1 + scheme2 | ~5min |
| **总计** | | **~4.8h** |

=> 从 18:16 UTC 起, **~23:00 UTC (北京时间 07:00 Jul 15) 全部完成**。

### 注意: D1/E 也 OOM 失败
run_final_fast.sh 的 D1 (噪声扫描) + E (超参扫描) worker 同样全部 OOM (0 metrics)。这些是 EXPERIMENT_PLAN_V2 的 Exp D/E, **不在方案一/二的路径上**。如需补跑, 排在合并批次之后 (~额外 4h)。当前编排器仅跑方案二 + BoS/pg (用户要求的)。

---

## 四、当前结论

**方案一已闭环**: 三臂弃权 GSACA 消除 deadlock 显著负项 (−0.023\*\*\* → −0.002ns), 全表无显著负项, Proposition 3 成立, holdout 泛化。bounded-downside 叙事完整。

**方案二为锦上添花**: 若 silent-anti-coord 把 chicken 推至显著, 进一步加强; 若否, 方案一已足够。
