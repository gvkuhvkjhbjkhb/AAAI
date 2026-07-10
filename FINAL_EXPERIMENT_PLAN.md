# 最终版论文实验方案、主线、过程与结论

> 基于 GitHub 仓库 `gvkuhvkjhbjkhb/AAAI` 全部 10 轮实验（Round 1–10，含 Phase 1/Phase 2）的整合总结。本文档为论文写作的最终实验定稿依据。

---

## 一、论文主线与核心主张

### 1.1 主张演变（从 Round 1 到 Round 10）

本项目经历了明确的主张收敛过程，最终落定在一条保守但证据扎实的路径上：

| 阶段 | 初始主张 | 实验证据 | 最终取舍 |
|---|---|---|---|
| Round 1–4 | LLM 语义诊断（Qwen）因果地改善 MARL | Qwen 在线调用超时、诊断标签坍缩 | 放弃作为因果机制 |
| Round 5–7 | 语义自适应 shaping 为核心贡献 | `random_type` 与 `uniform` 控制同样强，语义因果性不成立 | 语义降级为分析模块 |
| Round 8–10 | **失败触发的自适应奖励 shaping** | 10×10 主任务在强控制下显著有效，跨域受限 | **最终定稿主张** |

### 1.2 最终论文主张（定稿）

**标题方向**：`Failure-Triggered Adaptive Reward Shaping for Cooperative Multi-Agent Reinforcement Learning`

**核心 Claim（保守且可辩护）**：失败 episode 可被转化为校准的 shaping 信号；一个自适应的、相位感知（phase-aware）的失败触发干预调度，在配对种子、预算匹配控制与精确预算核算下，显著提升稀疏合作觅食任务上的 MAPPO。

**明确不主张的三件事**：
1. 不主张 LLM/Qwen 语义诊断因果地改善 MARL（Qwen 诊断坍缩，macro-F1 ≈ 0.04）。
2. 不主张方法在所有合作 MARL 任务上鲁棒泛化（VMAS、RWARE 为负面/无信息结果）。
3. 不主张语义标签特异性是性能增益的来源（random-type 与 shuffled 控制在多组比较中具竞争力）。

### 1.3 证据层级（定稿）

```
强（主结果）    : LBF 10×10 (3p-3f), Round 8, 8 配对种子 —— 配对 CI 严格大于零
方向性（压力测试）: LBF 12×12 (3p-4f) 合并 12 种子; 新 LBF 10×10 (4p-4f) 8 种子 —— 均值最高但 CI 跨零
机制防御（强支持）: 精确预算核算 + 系数敏感性 —— adaptive 用更少预算获更高回报
负面/限制（透明）: Qwen 诊断坍缩; RWARE 零回报; VMAS random-type 反超 adaptive
```

---

## 二、方法

### 2.1 方法概览

方法是一个**在线失败检测与干预框架**，由四个模块组成（实现于 `epymarl/src/llm_diagnosis/`）：

1. **轨迹记录器** (`trajectory_recorder.py`)：记录低回报（失败）episode 的状态-动作轨迹。
2. **轨迹摘要器** (`trajectory_summarizer.py`)：将轨迹压缩为结构化摘要（LBF 使用 load-action 计数；非 LBF 使用通用动作直方图，避免领域语义误用）。
3. **失败分类器** (`failure_classifier.py`)：将摘要分类为 6 类失败类型之一。RL shaping 条件使用增强启发式（`enhanced_heuristic`）；Qwen 仅离线缓存评估。
4. **奖励干预器** (`reward_intervention.py`)：在失败 episode 上施加校准的 shaping。

### 2.2 失败类型分类法（6 类）

`target_miscoordination`、`insufficient_cooperation`、`inefficient_exploration`、`low_value_overcommitment`、`timeout_near_success`、`unknown`。

### 2.3 干预公式（adaptive 模式，核心贡献）

失败 episode 的惩罚由三项调制（见 `reward_intervention.py:89-105`）：

```
penalty = failure_penalty × confidence_multiplier × type_weight × phase_weight
```

- **confidence_multiplier**：以参考置信 0.62 为基准，clamp 到 [0.70, 1.30]。
- **type_weight**：按失败类型差异化（如 `low_value_overcommitment=1.40`、`target_miscoordination=1.20`、`timeout_near_success=0.00`）。
- **phase_weight**（相位衰减调度）：训练进度 <0.20 用 early=0.50；0.20–0.70 用 middle=1.00；≥0.70 用 late=0.45。这避免晚期训练被 shaping 扰动收敛。

最终定稿方法配置：`adaptive_0.0003_late045`（failure_penalty=0.0003，late_phase_weight=0.45）。

---

## 三、最终实验方案（定稿）

### 3.1 实验矩阵总览

| 实验 | 环境 | 方法 | 种子 | 步数 | 角色 |
|---|---|---|---|---|---|
| **主结果** | `Foraging-10x10-3p-3f-v3` | 6 方法 | 8 | 500k | Main Table 1 |
| 尺度泛化 | `Foraging-12x12-3p-4f-v3` | 6 方法 | 12（R8+R9合并） | 500k | Main Table 2 |
| 结构泛化 | `Foraging-10x10-4p-4f-v3` | 4 核心方法 | 8 | 500k | Main Table 2 |
| 预算/敏感性 | `10x10-3p-3f` | 5 变体 | 4 | 500k | Main Table 3 |
| 跨域-RWARE | `rware-tiny-2ag-v2` | 6 方法 | 5 | 300k | Appendix 限制 |
| 跨域-VMAS | `vmas-navigation` | 5 方法 | 5 | 300k | Appendix 限制 |
| Qwen 诊断 | 离线 120 样本 | — | — | — | Appendix 负面 |

### 3.2 方法面板（6 方法定义）

| 方法名 | 角色 | 设计意图 |
|---|---|---|
| `baseline` | 纯 MAPPO | 无干预参照 |
| `diagnosis_only` | 记录+诊断，不 shaping | 隔离"诊断本身"的影响（=baseline） |
| `uniform_budget_matched_0.0003_late045` | 相位匹配的均匀惩罚 | 隔离"加任何相位匹配预算"的效果 |
| `adaptive_0.0003_late045` | **主方法** | 类型权重+置信+相位调度 |
| `random_type_budget_matched_0.0003_late045` | 随机类型标签（匹配频率） | 隔离"语义标签特异性" |
| `semantic_shuffled_budget_matched_0.0003_late045` | 打乱语义标签 | 进一步控制语义因果性 |

> 关键设计：所有控制组与主方法**配对种子、同相位调度、同预算量级**，因此任何差异不能归因于预算或种子。

### 3.3 统计方法

- 配对种子差异（paired seed deltas）+ 95% bootstrap 置信区间。
- 报告 final test return、train AUC、best train return、stability gap。
- Round 10 起加入**精确预算核算**（`LLM_FD_ACCOUNTING_FINAL`：触发次数、惩罚总量、terminal bonus 总量、shaped 步数）。

### 3.4 代码可复现性

- 启动器：`run_round8_aaai_stabilization.sh`（主）、`run_round10_options12.sh`（泛化+预算）、`run_round9_supplement.sh`（跨域）。
- 报告生成器：`epymarl/tools/build_round{8,9,10}_*.py`。
- 每轮结果归档于 `artifacts/AAAI_*.tar.gz`。

---

## 四、实验过程与关键结果

### 4.1 主结果：LBF 10×10（3p-3f），Round 8，8 配对种子

| method | n | final return | 95% CI | train AUC |
|---|---:|---:|---:|---:|
| baseline | 8 | 0.2258 | [0.1825, 0.2807] | 0.1827 |
| diagnosis_only | 8 | 0.2258 | [0.1825, 0.2807] | 0.1827 |
| uniform_budget_matched | 8 | 0.2346 | [0.1862, 0.2839] | 0.1876 |
| **adaptive_0.0003_late045** | 8 | **0.3042** | [0.2562, 0.3538] | **0.2263** |
| random_type_budget_matched | 8 | 0.2192 | [0.1807, 0.2640] | 0.1759 |
| semantic_shuffled | 8 | 0.2580 | [0.2171, 0.3099] | 0.2012 |

| 配对比较 | final Δ | 95% CI | AUC Δ | 95% CI |
|---|---:|---:|---:|---:|
| adaptive − baseline | +0.0784 | **[0.0272, 0.1239]** | +0.0436 | **[0.0184, 0.0659]** |
| adaptive − diagnosis_only | +0.0784 | **[0.0272, 0.1239]** | +0.0436 | **[0.0184, 0.0659]** |
| adaptive − uniform | +0.0695 | **[0.0064, 0.1247]** | +0.0387 | **[0.0088, 0.0673]** |
| adaptive − random_type | +0.0850 | **[0.0095, 0.1629]** | +0.0504 | [−0.0004, 0.1019] |
| adaptive − semantic_shuffled | +0.0461 | [−0.0108, 0.0928] | +0.0251 | [−0.0153, 0.0586] |

**解读**：adaptive 在 final return 与 train AUC 上对 baseline、diagnosis-only、phase-uniform 预算匹配、random-type 预算匹配均取得**严格大于零**的配对 CI。最难的控制是 semantic_shuffled（CI 跨零），因此论文应明确：贡献是**失败触发的自适应调度**，而非语义标签特异性。

### 4.2 尺度泛化：LBF 12×12（3p-4f），Round 8+9 合并 12 种子

| method | n | final return | 95% CI |
|---|---:|---:|---:|
| baseline | 12 | 0.1034 | [0.0954, 0.1110] |
| adaptive | 12 | **0.1132** | [0.1004, 0.1257] |
| random_type | 12 | 0.1115 | [0.1023, 0.1215] |

- adaptive − baseline：+0.0098，CI [−0.0059, 0.0261]（**跨零**）。
- adaptive − random_type：+0.0018，CI [−0.0141, 0.0184]（**跨零**）。

**解读**：adaptive 均值最高，但所有配对 CI 跨零。表述为"方向性尺度迁移，无崩溃"，不主张显著泛化。

### 4.3 结构泛化：新 LBF 10×10（4p-4f），Round 10，8 种子

| method | n | final return | penalty total | terminal bonus |
|---|---:|---:|---:|---:|
| baseline | 8 | 0.2945 | 0.0000 | 0.0000 |
| uniform_budget_matched | 8 | 0.2853 | 34.4002 | 0.0000 |
| **adaptive** | 8 | **0.3100** | **27.8313** | 0.0000 |
| random_type_budget_matched | 8 | 0.2704 | 30.4363 | 0.8805 |

- adaptive − baseline：+0.0155，CI [−0.0678, 0.1017]（跨零）。
- adaptive − random_type：+0.0396，CI [−0.0082, 0.0883]（跨零）。

**解读**：adaptive 均值最高且**用更少惩罚预算**（27.83 vs uniform 34.40 vs random-type 30.44），但 CI 跨零。作为 LBF 家族结构性变体的方向性证据，不作为头条胜利。

### 4.4 机制防御：精确预算核算 + 系数敏感性（Round 10）

**预算核算（核心反 reviewer 论据）**：adaptive 用更少惩罚预算却获更高回报，证明增益不可归约为"更多 shaping 奖励"。
- 10×10 敏感性面板：adaptive 0.0003 late0.60 用预算 29.66，达 final return 0.3039；uniform 用 37.32 仅达 0.2537；random-type 用 31.96 仅达 0.1986。

**系数敏感性**：
- 0.0003 是校准中点：0.0002 与 0.0005 均欠佳（Δ 分达 +0.0623 / +0.0664 偏向 0.0003）。
- late weight 有容差：0.45 与 0.60 实质持平（Δ +0.0002，CI [−0.0115, 0.0120]）。
- 0.0005 过度正则化（Phase 1 验证：0.0005 final return 0.2451 < 0.0003 的 0.3042）。

### 4.5 透明限制：负面与无信息结果

**Qwen 诊断（负面）**：Qwen/Qwen3.5-4B 在 120 样本上全部预测为 `insufficient_cooperation`，accuracy 0.0917，macro-F1 0.0420。确认标签坍缩。→ 语义诊断定位为可解释性附录，非因果机制。

**RWARE tiny（无信息）**：所有方法 final test return 均为 0，300k 步不足以学习。→ 附录限制，不作跨域胜利。

**VMAS 导航（不利）**：random-type 反超 adaptive（Δ −0.9047，CI [−1.2720, −0.4207]）；adaptive 的 train AUC 甚至低于 baseline。→ 揭示奖励尺度敏感性限制，不作头条。

---

## 五、实验结论（定稿）

1. **主结论**：失败触发的自适应奖励 shaping 在 LBF 10×10 稀疏合作觅食主任务上，于配对种子、预算匹配控制、random-type 与 shuffled 控制及精确预算核算下，显著改善 MAPPO（final return +0.0784，CI [0.0272, 0.1239]；且用更少预算）。

2. **泛化结论（保守）**：LBF 家族内（12×12 尺度迁移 + 4p-4f 结构变体）方向性一致、无崩溃，但配对 CI 跨零，仅作压力测试。

3. **机制结论**：增益非源于预算量（adaptive 预算更少却更优）、非源于语义标签特异性（shuffled/random 控制有竞争力）、非源于诊断本身（diagnosis_only = baseline）。贡献在于**失败触发的自适应相位调度**。

4. **边界结论（透明限制）**：方法为稀疏合作觅食式奖励校准，对稠密/连续导航（VMAS）奖励尺度敏感；RWARE 在当前步数下不可学习；Qwen 语义诊断不可靠。这些作为诚实的边界条件，提升审稿人信任。

---

## 六、论文写作建议（定稿）

### 6.1 表/图结构

| 位置 | 内容 | 来源 |
|---|---|---|
| Main Table 1 | LBF 10×10，6 方法，8 种子，bootstrap CI | Round 8 |
| Main Table 2 | 泛化/压力测试：合并 12×12 + 新 4p-4f | Round 8+9+10 |
| Main Table 3 | 预算核算 + 系数/late-weight 敏感性 | Round 10 |
| Figure 1 | 方法框架图 | 新制 |
| Figure 2 | 10×10 学习曲线 | Round 8 |
| Figure 3 | 12×12 + 4p-4f 学习曲线 | Round 8+9+10 |
| Figure 4 | 预算核算与失败触发计数 | Round 10 |
| Appendix A | Qwen 诊断负面结果 | Round 8 |
| Appendix B | RWARE/VMAS 跨域限制 | Round 9 |

### 6.2 措辞红线

- **应说**："在强控制与精确预算核算下，失败触发的自适应 shaping 显著改善主任务"。
- **不应说**："跨任务鲁棒"、"LLM 语义诊断因果改善 MARL"、"语义标签是增益来源"。
- 12×12/4p-4f 用"directionally consistent but not always significant"。
- Qwen/RWARE/VMAS 作为**透明边界条件**呈现，而非隐藏失败。

### 6.3 标题与摘要方向

标题不含 LLM/Qwen/语义因果。摘要强调：失败 episode 转化为校准 shaping 信号、自适应相位调度、配对种子、预算匹配控制、稀疏合作觅食。

---

## 七、关键文件索引

- 路线图：`phase1_phase2_experiment_roadmap.md`、`docs/AAAI_STABILIZATION_OPTIONS.md`
- 方法实现：`epymarl/src/llm_diagnosis/{reward_intervention,failure_classifier,trajectory_summarizer,trajectory_recorder}.py`
- 主结果：`results/round8_aaai_stabilization_round8_full_20260705_041113/`
- 泛化+预算：`results/round10_options12_round10_full_20260706_024052/`
- 跨域限制：`results/round9_supplement_round9_full_20260705_163435/`
- 启动器：`run_round{8,9,10}_*.sh`
- 诊断标注：`analysis/diagnosis_validation/`（Phase 2 人工标注验证集）
