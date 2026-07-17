# HetToM Round-2 实验方案：Cheap-talk、自适应 ToM 与泛化验证

> 生成时间：2026-07-09
> 基于 Round-1 结果（`RESULTS.md`）：推理陷阱复现，异质性打破多样性但摧毁协作，ToM 在异质团队中部分恢复协作（+123%，r=+1.0）但未超越基线。
> 关联代码：`hettom_baseline.py`（`--extend` 扩展矩阵）、`analyze_layer1.py`、`run_round2.sh`

---

## 0  Round-1 暴露的三个问题

| 问题 | Round-1 数据 | 根因 |
|---|---|---|
| A. 异质摧毁协作 | het_notom payoff 0.580 vs baseline 2.244 | 异质 agent 无对齐通道，多样性=混乱 |
| B. ToM 准确率低、增益不充分 | het_tom tom_acc 0.54，payoff 1.296 < baseline | 固定 1 阶 ToM 与异质模型推理深度失配 |
| C. 统计功效不足 | n=3，p 卡 0.10（Mann-Whitney 最小可达值） | 种子数太少 |

## 1  文献调研结论与三个对症方案

### 方案 A：Cheap-talk 通道（廉价磋商）
- **文献依据**：Madmoun & Lahlou 2025《Communication Enables Cooperation in LLM Agents》(EACL 2026) [1]。在 4 人 Stag Hunt 中，一个单词的 cheap-talk 通道把合作率从 **0% 提升到 96.7%**。
- **机制**：每轮动作前，各 agent 先输出一个意图信号（如 "Stag"/"Hare"），再据此选动作。新增 `*_talk` 实验格。
- **对症**：直接攻击问题 A——异质团队的对齐缺失。cheap-talk 与异质性正交（不改模型、不改 ToM），可干净归因。

### 方案 B：自适应 ToM（A-ToM）
- **文献依据**：Mu et al. 2026《Adaptive Theory of Mind for LLM-based Multi-Agent Coordination》(AAAI 2026) [2]。核心发现：**ToM 阶数失配（misaligned ToM orders）损害协调**。提出 A-ToM：基于历史交互估计队友 ToM 阶数并动态对齐。
- **机制**：每个 agent 维护对队友"ToM 阶数"的估计（从历史预测命中率推断），动态调整自身推理深度去对齐（命中率<0.4 则加深，>0.75 则降低，范围 1-3）。新增 `*_atom` 实验格。
- **对症**：直接攻击问题 B——不是 ToM 无效，是阶数没对齐。这是对 Round-1"ToM 增益以异质性为前提"发现的机制级深化。

### 方案 C：扩种子 + 补博弈
- **文献依据**：Sun et al. 2025《Game Theory Meets LLMs》综述 [3] 倡导跨博弈元分析；r=±1.0 保证方向 100% 一致，扩 n 即可达 p<0.05。
- **机制**：stag_hunt 扩到 6 种子（1-3 已有，新增 4-6）；新增 battle_of_the_sexes（6 种子，BoS 是 ToM 最强测试——需推断对方偏好均衡）。
- **对症**：攻击问题 C（统计功效）+ 验证泛化性。

## 2  扩展实验矩阵（11 格）

`--extend` 标志在基础 4 格之上增加 7 格：

| Cell | 同质性 | ToM | Cheap-talk | 验证 |
|---|---|---|---|---|
| hom_notom | 同 | 无 | 无 | 基线（复现陷阱）|
| hom_tom | 同 | 固定1阶 | 无 | ToM 单独 |
| het_notom | 异 | 无 | 无 | 异质单独（问题A）|
| het_tom | 异 | 固定1阶 | 无 | Round-1 方法 |
| hom_notom_talk | 同 | 无 | 有 | cheap-talk 单独（同质）|
| hom_tom_talk | 同 | 固定1阶 | 有 | cheap-talk+ToM（同质）|
| het_notom_talk | 异 | 无 | 有 | **cheap-talk 救异质协作**（方案A核心）|
| het_tom_talk | 异 | 固定1阶 | 有 | cheap-talk+ToM（异质）|
| hom_atom | 同 | 自适应 | 无 | A-ToM 单独（同质）|
| het_atom | 异 | 自适应 | 无 | **A-ToM 救异质 ToM**（方案B核心）|
| het_atom_talk | 异 | 自适应 | 有 | **完整方法**（方案A+B组合）|

## 3  预注册判定规则

**陷阱被打破（POSITIVE）**，当且仅当完整方法 `het_atom_talk` 满足：
- 在视角多样性 AND 协作回报上均 > `hom_notom`
- 协作回报 Mann-Whitney p < 0.05
- 差值 bootstrap CI 排除 0

**机制级子判定**（即使完整方法未达 POSITIVE，逐个验证）：
- **cheap-talk 有效**：`het_notom_talk` payoff > `het_notom`（p<0.05）
- **A-ToM 有效**：`het_atom` tom_acc > `het_tom` tom_acc，且 `het_atom` payoff > `het_tom`（p<0.05）
- **两机制互补**：`het_atom_talk` payoff > max(`het_atom`, `het_tom_talk`)

## 4  实验配置

- **模型**（与 Round-1 一致以便对比）：Qwen2.5-3B-Instruct（同质）/ Qwen2.5-3B + Qwen2.5-1.5B（异质）
- **博弈**：stag_hunt（seeds 4,5,6，与 Round-1 的 1-3 合并成 6 种子）、battle_of_the_sexes（seeds 1-6）
- **参数**：30 episode/cell，horizon 5，memory 2，tom_order 1（atom 动态 1-3）
- **算力**：2× RTX 5090 数据并行（seed 进程绑 GPU）
- **预估**：11 格 × 30 ep，talk 格多 1 轮 LLM 调用，atom 格多递归深度。单 seed ≈ 8-12 min，9 jobs（2 并行）≈ 40-55 min。

## 5  代码改动清单

| 文件 | 改动 |
|---|---|
| `hettom_baseline.py` | LLMAgent 加 `use_talk`/`adaptive_tom` 标志；`announce()` cheap-talk 方法；`_decide_tom_orders()` A-ToM 阶数估计；`update_tom_history()` 命中率追踪；`run_episode()` 两阶段（talk→act）；`make_matrix_configs()` `--extend` 生成 11 格 |
| `analyze_layer1.py` | CELLS 扩展到 11 格；comparisons 增加 cheap-talk/A-ToM/combined 对比；DECISION 增加 round-2 判定 |
| `run_round2.sh` | 2-GPU 波次运行器：stag_hunt(4,5,6) + battle_of_the_sexes(1-6) |

## 6  参考文献

[1] Madmoun, H. & Lahlou, S. (2025). Communication Enables Cooperation in LLM Agents. EACL 2026. arXiv:2510.05748.
[2] Mu, C. et al. (2026). Adaptive Theory of Mind for LLM-based Multi-Agent Coordination. AAAI 2026. arXiv:2603.16264.
[3] Sun, H. et al. (2025). Game Theory Meets Large Language Models: A Systematic Survey. arXiv:2502.09053.
