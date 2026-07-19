# 投稿前补充实验：冻结详细方案

## 1. 研究目标与优先级

当前论文最容易被拒的可修复问题不是 P3 的结果方向，而是：缺少同网格 payoff-in-prompt 外部基线；对 action-label 表面是否影响结论缺少控制；旧 active probe 的指标字段容易与 `team_mean_payoff` 混淆。因此只补下列三块，不再扩展新游戏、调 Safe-SCA 阈值或筛选矩阵。

| 优先级 | 实验 | 回答的问题 | 新 cell | 投稿作用 |
|---|---|---|---:|---|
| P0 | 同网格 payoff-in-prompt | 明示收益表是否足以恢复效用？ | 320 | 必做；直接补 W5/“decisive control” |
| P1 | action-label 交换 | P3 路由/策略效应是否依赖标签表面？ | 240 | 限定或加强 P3 的因果包装 |
| P2-P3 | 当前端点在线探测 | 少量在线收益能否选择正确干预臂？ | 80 | 提高证书的操作意义 |
| P2-source | S2 源网格在线探测 | 结果能否在原游戏族复现？ | 120 | 次要，可选；不能替代前三块 |

P0、P1、P2 必须使用与 S2/P3 相同的模型、revision、vLLM flags、温度、`top_p`、episode horizon 和生成 seed 规则。P3 的 8 个收益张量、10 个种子和原始结果保持不变。

## 2. 全局冻结条件

- 模型：`Qwen/Qwen2.5-7B-Instruct` 与 `THUDM/GLM-4-9B-0414`。
- revision：分别为 `a09a...bc28` 与 `645b...fcaf`，完整值在 JSON 协议中。
- serving：vLLM 0.25.1、PyTorch 2.11.0+cu128、Transformers 5.14.1；`bfloat16`、`--enforce-eager`、GPU memory utilization 0.85、max model length 2048、API key `dummy`。
- 智能体温度：Qwen 0.5、GLM 0.8；`top_p=0.9`。
- 每回合 5 step，memory=2；P3 每 cell 30 episode。
- P3 seed：102–111；请求 seed 基值 1000。
- 主终点：所有 episode、step 和智能体的平均收益 `team_mean_payoff`。
- `cooperation_payoff` 是旧实现中只取 Agent 1 的兼容字段，不得用于选择、检验或论文头条。
- 运行顺序：每个 matrix-seed 内按 seed-keyed Latin rotation 平衡策略臂顺序。
- 任何失败、重试、缺格和负结果均保留；不得用 `--force` 覆盖。

## 3. P0：同网格 payoff-in-prompt 单因素实验

### 3.1 设计

在 P3 的 8 个矩阵 × 10 个种子上，每个 context 同时运行四臂：

1. `het_notom`：NoAlign，不通信、不 ToM、不展示收益表；
2. `het_gated_atom_talk`：Always-Gated；
3. `het_safe_sca`：原冻结 Safe-SCA，同批重跑以便与外部 baseline 直接比较；
4. `het_payoff_prompt`：与 NoAlign 完全相同，仅增加 `payoff_in_prompt=True`，在 action prompt 中展示完整 2×2 收益矩阵。

收益表对每个智能体采用 agent-relative 方向：row 是自己的 action，column 是队友 action，每格按 `(self payoff, teammate payoff)` 排序。原代码中该 dormant baseline 分支对 Agent 2 没有转置；S1/S2/P3 从未启用该分支，既有结果不受影响。本包已在 P0 冻结前修正，并用非对称收益矩阵单测验证 Agent 2 的行列和收益顺序。

核心单因素对比为：

\[
\Delta_{\mathrm{payoff}} = Y(\text{NoAlign + payoff table})-Y(\text{NoAlign}).
\]

两臂具有相同模型、温度、seed、episode 数、无 ToM、无通信；被操纵的唯一算法因素是完整收益表是否进入 prompt。因此这里可以写“payoff visibility 的 context 内因果效应”，但不能外推为“收益张量是 S2→P3 差异的唯一原因”。

### 3.2 为什么同时重跑固定臂

P0 不把旧 P3 的 NoAlign/Gated 作为主对照，而是在同一批服务器、同一编排中重跑两臂，避免 R0 已观察到的 vLLM trajectory 非 bitwise 抖动成为时间混杂。旧 P3 只用于报告 temporal replay sensitivity，不替代并发对照。

### 3.3 假设与 estimand

- 主假设：P3 八矩阵总体的层级平均 `PayoffPrompt − NoAlign`。
- 次要：逐矩阵效应；`PayoffPrompt − Gated`；相对每个 matrix-seed 中较优固定臂的 regret。
- 分析单位：matrix-seed cell，n=80；每个 cell 的 endpoint 是 30×5 个 step 的团队均值。
- 总体 CI：两级 bootstrap，先重采样矩阵、再在矩阵内重采样 seed，20,000 次。
- 逐矩阵 CI：seed 配对 bootstrap，20,000 次；八个次要检验用 Holm 校正。
- 不以“至少几个矩阵显著”作为通过条件；完整报告八矩阵。

### 3.4 结果解释

- 若总体 CI 全部大于 0：收益可见性在该网格上有正效应；仍需比较 regret，不能直接声称超过选择性干预。
- 若均值接近 0/CI 跨 0：缺少收益文字不是 P3 utility failure 的充分解释。
- 若显著为负：明示收益表反而损伤团队收益，是有价值的负机制结果；不得删除该基线。
- 无论方向如何，P0 都进入主文基线表，而不是只放附录。

## 4. P1：P3 action-label permutation 控制

### 4.1 唯一改动

对每个 P3 矩阵，把 action index 0/1 所附的不透明文字标签互换。例如原始 `0=glyph-ivory, 1=glyph-slate` 变为 `0=glyph-slate, 1=glyph-ivory`。**payoff tensor 的 index 含义、seed、模型、所有 prompt 模板、控制器输入均不变**。代码测试逐 profile 验证原始/交换条件的 payoff 完全相等。

运行三策略：NoAlign、Always-Gated、Safe-SCA，共 8×10×3=240 cells。Safe-SCA 仍只看到实现的 action index 与 reward history，看不到 matrix ID、分析类别或 payoff table。

P1 的原标签参照取 P0 中同批重跑的 NoAlign、Always-Gated 和 Safe-SCA，不取旧 P3 作为主参照；因此标签 interaction 不会把旧 P3 与新实验之间的 replay drift 混入主估计。旧 P3 仍完整保留，仅用于既有结论和时序敏感性审计。

### 4.2 estimand

对 Gated 和 Safe-SCA 分别计算相对 NoAlign 的 difference-in-differences：

\[
I_A=[Y_A^{swap}-Y_N^{swap}]-[Y_A^{orig}-Y_N^{orig}].
\]

由于各矩阵收益尺度不同，主等价判定用 normalized team-payoff range：每矩阵所有 joint profile 的团队均值最大值减最小值。等价 margin 冻结为 ±0.10 normalized units。

同时比较 80 个 Safe-SCA post-warmup route 是否一致。

### 4.3 冻结判定

- route agreement 下限：0.90；报告 Wilson 95% CI。
- 每个矩阵、每个策略的 normalized interaction 95% CI 必须完全落在 [−0.10, +0.10]，才称“对本次 label swap 等价”。
- 完整 label-robustness gate 要求 route 条件和所有逐矩阵效应条件同时满足。

若不通过，论文必须保留“action-label surface 是未解决 moderator”；不能把失败包装为 payoff-only 因果结论。

## 5. P2：基于 team mean 的在线 active probe

### 5.1 算法

每个 context-seed：

1. 探测 NoAlign 5 episodes；
2. 探测 Gated 5 episodes；
3. 奇偶 seed 交换两臂先后顺序；
4. 分别计算每个探测 episode 中所有 step/agent 的 team mean；
5. 选择探测均值更高的臂，平局保守选择 NoAlign；
6. 用全新 agent objects 在剩余 20 episodes（P3）提交所选臂。

P3 共 80 cells。可选 source block 使用 S2 的六游戏 × 20 seeds；二人游戏 30 total episodes，Public Goods 沿用 20 total episodes，因此后者 commit 10 episodes。

### 5.2 两个必须分开的结果

- `bandit_commit_team_mean_payoff`：只看 commit，诊断选择后的臂质量。
- `bandit_online_total_team_mean_payoff`：把两臂共 10 个 probe episodes 和 commit 全部计入；这是部署主结果。

选择准确率的参照是同 context-seed 下并发/已有固定臂 `team_mean_payoff` 较高者；平局同样定义为 NoAlign。报告准确率 + Wilson CI、reference selection regret、online total 对 NoAlign 的增益和对较优固定臂的 regret。

### 5.3 论文意义

P2 若成功，能把“证书是否辨别”转化为更实际的结论：小规模在线收益探测可以弥补仅靠结构证书的效用外推不足。P2 若失败，则支持论文当前主张：安全证书不等于效用选择器；不能隐藏失败。

## 6. 停止条件

以下任一情况发生即停止新任务、保留已生成文件并检查：

- strict preflight 不通过或使用 version override；
- 任一模型 endpoint 身份、revision 或端口异常；
- OOM、server restart、某任务超过两次额外重试；
- 协议 hash、P3 registry hash 或 label registry hash 不匹配；
- 提议在看到部分结果后修改矩阵、seed、K、margin、Safe-SCA 阈值或删除 cell。

普通 SSH/编排中断不需要删除结果；重复同一命令会跳过完整 metrics，只恢复缺失任务。

## 7. 对论文因果表述的边界

允许：

- “Within each fixed matrix–seed context, policy arm is the manipulated factor.”
- “P0 identifies the effect of adding the payoff table to the otherwise identical NoAlign prompt.”
- P1 通过时：“The arm effects are robust to the preregistered label swap.”

不允许：

- “P3 is a payoff-matrix-only causal experiment.” P3 跨 source/P3 还伴随 seeds 与 action surfaces 的改变。
- “R0 payoff replay passed at tolerance 0.20.” 原始严格失败必须保留。
- 用 `cooperation_payoff` 替换 `team_mean_payoff`，或用 commit-only P2 payoff 代替含 probe 成本的主结果。
