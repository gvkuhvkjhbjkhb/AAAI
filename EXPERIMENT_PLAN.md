# HetToM 实验方案：验证推理陷阱与异质+ToM 的破局效应

> 生成时间：2026-07-09
> 关联代码：`hettom_baseline.py`（Layer 1）、`analyze_layer1.py`、`layer2_marl_bridge.py`（Layer 2）
> 算力：1× RTX 5090（32GB）
> 周期：4 周

---

## 0  总体目标

验证三个核心命题，产出 AAAI 论文：

1. **推理陷阱在合作博弈中存在**：同质 LLM 智能体的视角多样性受信息论边界约束（Shin 2026 推广）。
2. **异质性打破陷阱**：异质配置（不同模型/温度）的视角多样性显著高于同质。
3. **ToM 深化协作**：心智理论推理提升协作绩效与均衡收敛，与异质性互补。

任一命题被证伪也是有价值的负面结果，论文仍可成文（见下文 fallback）。

---

## 1  分层架构

```
Layer 1（核心，必完成）          Layer 2（增强，时间允许时）
LLM-as-Agent 矩阵博弈            迁移到 LBF + MAPPO
├── hettom_baseline.py           ├── layer2_marl_bridge.py
├── 4 格 × 多种子 × 多博弈        ├── 离线 ToM 意图注入
├── analyze_layer1.py            ├── mappo_hettom.yaml
└── 产出：陷阱是否被打破           └── 产出：效应是否迁移到真 MARL
```

**依赖关系**：Layer 1 先跑，产出"哪个配置赢"。Layer 2 只迁移赢家配置。
**Fallback**：若 Layer 2 时间不足，仅 Layer 1 即可成文（定位为"LLM 多智能体协作的认知多样性研究"）。

---

## 2  Layer 1 实验方案（第 1-2 周）

### 2.1 实验矩阵

4 格 2×2 设计，每个 cell 在每个种子上独立运行：

| Cell | 同质性 | ToM | 验证 |
|---|---|---|---|
| `hom_notom` | 同模型同温度 | 无 | 基线（复现推理陷阱） |
| `hom_tom` | 同模型同温度 | 有 | 钥匙2（ToM 单独） |
| `het_notom` | 异模型异温度 | 无 | 钥匙1（异质单独） |
| `het_tom` | 异模型异温度 | 有 | 完整方法 |

### 2.2 自变量与水平

| 因素 | 水平 | 说明 |
|---|---|---|
| 博弈 | stag_hunt, battle_of_the_sexes, chicken, public_goods, coordination | 5 个博弈，覆盖纯合作/冲突/协调/公共物品 |
| 种子 | 1, 2, 3, 4, 5 | 5 种子（最小可信统计） |
| ToM 阶数 | 1, 2, 3 | 递归深度消融 |
| 异质性 | 同质(Qwen2.5-7B ×n) vs 异质(Qwen/Llama/Mistral) | 模型级异质 |
| 温度（异质时） | 0.5, 0.8, 1.0 | 增加先验多样性 |
| episode 数 | 50 | 每 cell 每 seed |
| horizon | 5 轮 | 短博弈，单 GPU 可行 |

### 2.3 因变量（4 个核心指标）

| 指标 | 定义 | 验证命题 |
|---|---|---|
| 视角多样性 | 智能体间动作分布的成对 KL 散度均值 | 命题 1,2（陷阱存在/异质打破） |
| 协作绩效 | 每 episode 平均团队回报 | 命题 3（ToM 提升协作） |
| 均衡收敛 | 前后半段动作分布 TV 距离的补 | 命题 3（收敛稳定性） |
| ToM 预测准确率 | 预测队友动作 vs 实际的命中率 | ToM 模块有效性 |

每个指标附 bootstrap 95% CI（2000 次重采样）。

### 2.4 主实验（必做）

```bash
# 5 博弈 × 5 种子 × 4 格，1 阶 ToM
for game in stag_hunt battle_of_the_sexes chicken public_goods coordination; do
  python3 hettom_baseline.py --matrix --game $game \
    --seeds 1 2 3 4 5 --episodes 50 --horizon 5 --tom_order 1 \
    --out_dir results/hettom_layer1/$game
done

# 汇总分析
python3 analyze_layer1.py --results_dir results/hettom_layer1 \
  --out_dir results/hettom_layer1
```

**预估算力**：5 博弈 × 5 种子 × 4 格 = 100 runs。每 run ≈ 50 episode × 5 轮 × 2-4 智能体 × (1-2 LLM 推理/轮) ≈ 250-500 次推理。单 5090 约 0.5-1s/推理，每 run ~5-10 min，100 runs ~10-15 GPU 小时。**1-2 天可完成。**

### 2.5 消融实验（必做）

```bash
# ToM 阶数消融（在 stag_hunt 上）
for order in 1 2 3; do
  python3 hettom_baseline.py --matrix --game stag_hunt \
    --seeds 1 2 3 --episodes 50 --horizon 5 --tom_order $order \
    --out_dir results/hettom_layer1/ablation_tom_order_$order
done
```

**消融维度**：
- ToM 阶数（1/2/3）：验证递归推理的边际收益是否递减
- 异质性程度（2 模型 vs 3 模型）：用 `--models_het` 控制
- 温度差异幅度：用 `--temps_het` 控制

预估 3-5 GPU 小时。

### 2.6 预注册决策规则

分析器 `analyze_layer1.py` 内置以下**预注册**判定（在看到数据前定下）：

**陷阱被打破（正面结果）**，当且仅当：
- `het_tom` 在**视角多样性** AND **协作绩效**上均 > `hom_notom`
- 协作绩效的 Mann-Whitney p < 0.05
- 差值的 bootstrap CI 排除 0

**部分成立**：仅异质性 OR 仅 ToM 有效。

**证伪**：两者均无效。

### 2.7 论文 framing（按结果分支）

| Layer 1 结果 | 论文定位 | AAAI 概率 |
|---|---|---|
| 陷阱被打破（正面） | 强正面方法论文：HetToM 打破推理陷阱 | 40-50% |
| 部分成立（异质有效/ToM无效或反之） | 方法+分析论文 | 30-40% |
| 证伪 | 高价值负面结果+理论分析 | 25-35% |

---

## 3  Layer 2 实验方案（第 3-4 周，增强层）

### 3.1 前提

Layer 1 产出"赢家配置"（预期是 `het_tom`）。Layer 2 把赢家迁移到真 MARL 基准。

### 3.2 流程

```bash
# 1. 用 EPyMARL 跑 LBF 收集训练轨迹（baseline MAPPO，200k 步）
#    （复用仓库已有 run_qs_quick_test.sh 模式）

# 2. 离线 ToM 意图注入（每 50 episode 重新提取）
python3 layer2_marl_bridge.py inject \
  --layer1_dir results/hettom_layer1 \
  --out results/hettom_layer2/intent_features.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct

# 3. 用意图特征增强的 MAPPO 训练
python3 layer2_marl_bridge.py launch \
  --config mappo_hettom --seeds 1 2 3 4 5 \
  --intent_path results/hettom_layer2/intent_features.jsonl \
  --env lbforaging:Foraging-10x10-3p-3f-v3 --t_max 500000
```

### 3.3 对比组

| 配置 | 说明 |
|---|---|
| baseline MAPPO | 无意图注入（仓库已有） |
| MAPPO + random intent | 随机意图（控制组，排除"额外维度"本身的效果） |
| MAPPO + hom intent | 同质 LLM 意图（对应 Layer 1 hom_tom） |
| MAPPO + het intent | 异质 LLM 意图（对应 Layer 1 het_tom，赢家配置） |

### 3.4 指标

| 指标 | 来源 |
|---|---|
| Final Test Return | EPyMARL 标准输出 |
| Best Train Return | EPyMARL 标准输出 |
| Train AUC | EPyMARL 标准输出 |
| 收敛稳定性 | best-last gap |

5 种子，500k 步（仓库已验证 5090 可跑）。预估每 run ~30 min，4 配置 × 5 种子 = 20 runs ~10 GPU 小时。

### 3.5 关键设计：为什么离线注入而非在线

仓库的 LLM 管线是**离线**的（`offline_relabel.py`），无 per-step 在线推理基础设施（`Plan_Assessment.txt` 约束 1）。离线注入：
- 复用仓库已验证的离线 LLM idiom ✓
- 不改 MAPPO 训练算法，只加观测维度 ✓
- 不触碰已证伪的"自适应干预"死路（意图是观测侧，非奖励侧）✓
- 保持 PBRS 策略不变性（不改变奖励）✓

---

## 4  时间表

| 周 | 任务 | 产出 |
|---|---|---|
| 1 | Layer 1 主实验（5 博弈 × 5 种子）+ 分析 | 主结果表 + 决策判定 |
| 2 | Layer 1 消融（ToM 阶数/异质程度）+ 初稿理论部分 | 消融表 + 理论草稿 |
| 3 | Layer 2 迁移实验（LBF + MAPPO） | 迁移结果表 |
| 4 | 写论文 + 复现检查 + 投稿 | 完整论文 |

---

## 5  风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| Layer 1 证伪（异质+ToM 无效） | 30% | 仍可写高价值负面结果（25-35% AAAI） |
| ToM 意图坍缩（类似已失败的语义诊断） | 中 | 先做 100 样本人工一致性检查；坍缩则降级为"异质性-only"故事 |
| Layer 2 迁移失败 | 中 | Layer 1 独立成文，Layer 2 是增量 |
| 5090 显存不够 3×8B 异质 | 低 | 用 2 模型或同模型不同温度替代 |
| HuggingFace 模型下载受限 | 低 | 预先下载到本地缓存 |

---

## 6  代码清单

| 文件 | 作用 | 状态 |
|---|---|---|
| `hettom_baseline.py` | Layer 1：4 格矩阵博弈 + 递归 ToM + bootstrap CI + 多种子 | ✅ mock 验证通过 |
| `analyze_layer1.py` | 跨种子聚合 + Mann-Whitney/Wilcoxon + 决策判定 | ✅ 合成数据验证通过 |
| `layer2_marl_bridge.py` | Layer 2：离线 ToM 意图注入 + EPyMARL 配置 + 启动器 | ✅ mock/dry-run 验证通过 |
| `epymarl/src/config/algs/mappo_hettom.yaml` | Layer 2 的 MAPPO 配置 | ✅ 由 layer2 脚本生成 |
| `Plan_Assessment.txt` | 5 方案的故事性/可行性评估 | ✅ 已提交 |
| `LLM_MARL_Survey.txt` | 综述全文 | ✅ 已提交 |
| `EXPERIMENT_PLAN.md` | 本文件 | 本文件 |

---

## 7  最 pivot 的预期结论

无论实验结果如何，本方案都将给出**可证伪的、有理论根基的、单 GPU 可验证的**答案：

> **异质性 + 心智理论是否能打破 LLM 多智能体的推理陷阱，催生真正的协作涌现？**

这是 2026 年 LLM-MARL 领域最 pivot 的开放问题之一（Shin 2026 提出陷阱；Yao 2026 发现 LLM 偏离博弈论；Wang 2025 早期尝试 ToM）。我们的方案直接回答它，且无论正面/负面均有学术价值。
