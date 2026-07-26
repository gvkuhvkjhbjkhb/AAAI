# P6R Mistral–Phi 重跑包（2 × RTX 4090 + vLLM）

**目的不是"再跑一次"，而是回答一个二选一的问题：**

> P6R 那一格 `commit_target_action_rate = 0.041`，是这对模型在这批矩阵上的**行为**，
> 还是当时那次运行的**环境产物**？

论文现在把它报告为"未解释的 campaign 级效应"。这个包把它变成一个有预注册判据的
可判定实验。

---

## 0. 为什么这个实验值得做

同一对 revision-pinned 的 Mistral-7B-Instruct-v0.3 / Phi-3.5-mini-instruct：

| campaign | registry 生成器 | 实测 fidelity |
|---|---|---|
| P8 校准 | `p6_registry` | **1.000** |
| P5 实现 | 同上 | **1.000** |
| P5c 因子实验（用 P6R **原样** prompt+parser） | 同上 | **1.000** |
| **P6R 迁移** | 同上（`p6r_registry.py` 只改 id 前缀） | **0.041** |

矩阵、prompt、parser 三者都已被 P5c 排除（parser 对 fidelity 的效应 0.000
[0,0]，prompt −0.002 [−0.004,−0.001]）。剩下的只有 **serving stack**，而当时没有
记录任何足以事后识别它的指纹。

**这个包做两件事**：

1. 补上那个指纹（`fingerprint.py`），使今后任何一次运行都可被事后归因；
2. 在**当前** serving config 下按 P6R 的冻结协议重放，并用预注册判据判定。

---

## 1. 预注册判据（跑之前必须先冻结）

在 anti-safe 家族（4 矩阵 × 10 seeds = 40 个 cell）上，以 `Gated` 臂的
`commit_target_action_rate` 的 **矩阵聚类均值** $\hat f$ 及其 95% bootstrap 区间为准：

| 结果 | 判定 | 论文怎么改 |
|---|---|---|
| $\hat f \ge 0.90$ 且 CI 下界 $> 0.80$ | **A. 环境产物** | 撤回 −0.098 与 0.041，正文改为"该 cell 已被定位为 serving 产物并重测"，报告重放值 |
| $\hat f \le 0.20$ 且 CI 上界 $< 0.30$ | **B. 可复现** | 0.041 是真实的 pair × campaign 交互；这反而成为一个**更强**的结果——同一对模型、同一生成器的矩阵，fidelity 可以在 0.04 与 1.00 之间跳变 |
| 其余 | **C. 不确定** | 维持现有"未解释的 campaign 级效应"表述，并把重放区间一并报告 |

**同时冻结**（避免事后挑选）：

- 主端点：anti-safe `Gated` 臂的 fidelity。
- 次要端点：anti-tradeoff `Gated` 臂 fidelity；`ActionSafeFixed` 与
  `LegacyGateFixed` 的 route rate（应复现 40/40 与 0/40，见 §6）。
- 重复次数：anti-safe 家族跑 **3 次独立重复**（不同 `generation_seed_base` 偏移），
  用于估计连续批处理下的 run-to-run 方差。**判定用三次的合并均值**。
- 分析脚本 `analyze_replay.py` 在跑之前提交，跑完不得修改（其 SHA-256 记入
  `RUN_MANIFEST.json`）。

> 判据写成"$\hat f$ 落在中间区间 = 不确定"是刻意的：这个实验有权返回"不知道"。

---

## 2. 硬件与服务拓扑

两台机器，每台一张 RTX 4090（24 GB）。**一个角色一张卡**，不做张量并行——
7B/3.8B 在单卡上放得下，TP 只会增加通信开销并引入额外的不确定性来源。

```
┌─ node-a (4090) ────────────┐      ┌─ node-b (4090) ────────────┐
│ vLLM :8002                 │      │ vLLM :8003                 │
│ mistralai/Mistral-7B-      │      │ microsoft/Phi-3.5-mini-    │
│   Instruct-v0.3  (role 0)  │      │   instruct       (role 1)  │
│ BF16 ≈ 14.5 GB             │      │ BF16 ≈ 7.6 GB              │
├────────────────────────────┤      └────────────────────────────┘
│ orchestrator (replay_runner)│  ──── HTTP ────▶ 两个端点
└────────────────────────────┘
```

编排器放在 node-a（它对 Mistral 的调用走 loopback，对 Phi 走局域网）。
若两台机器网络往返 > 2 ms，把编排器放在中间任意一台都可以——它是纯 I/O 绑定的
asyncio 进程，本身几乎不吃 CPU。

**显存预算**：Mistral-7B BF16 权重 14.5 GB，`--gpu-memory-utilization 0.90`
留出 ≈ 7 GB KV cache；prompt ≈ 260 token、输出 12 token，`--max-model-len 2048`
下单序列 KV ≈ 0.5 MB，足够 `--max-num-seqs 256`。Phi-3.5-mini 更宽裕。

---

## 3. 请求预算与时间估计

冻结协议（`protocols/p6r_cross_pair_frozen.json`）：

- registry seed 260902，`n_per_family=4` → **16 矩阵**（4 家族 × 4）
- seeds 700–709 → **10 seeds** → 160 个 (matrix, seed) cell
- horizon 5，warmup 10 episodes，total 30 episodes → 每 cell 每臂
  30 × 5 = 150 步 × 2 角色 = **300 次请求**
- **物理臂只有 2 个**（`NoAlign`、`Gated`）；`LegacyGateFixed` 与
  `ActionSafeFixed` 从同一条 `Gated` 轨迹**派生**，不额外调用模型
  （与 P5/P6 的 common-random-number 做法一致）

| 阶段 | 请求数 | 估计墙钟 |
|---|---|---|
| Phase 1 指纹 + canary | ~200 | 2 min |
| Phase 2 全量重放（16 矩阵 × 10 seed × 2 臂） | **96,000** | 1.5–3 h |
| Phase 2b anti-safe 重复 ×2（再跑 2 遍 40 cell × 2 臂） | 48,000 | 45–90 min |
| 合计 | ~144,000 | **3–5 h** |

（每个端点各承担一半。以 batch 64、单请求 0.3–0.6 s 计。）

---

## 4. 一次完整执行

### 4.1 两台机器都装依赖

```bash
python -m pip install -U "vllm>=0.6.3" aiohttp numpy
# 编排器额外需要（只在 node-a）：
python -m pip install -U orjson
```

### 4.2 固定 revision（**必须**，这是原始运行没做够的地方）

```bash
# 在任一台上查出确切 commit，写进 protocols/p6r_replay_frozen.json
huggingface-cli scan-cache            # 或
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for r in ["mistralai/Mistral-7B-Instruct-v0.3", "microsoft/Phi-3.5-mini-instruct"]:
    print(r, api.model_info(r).sha)
PY
```

把两个 SHA 填进 `protocols/p6r_replay_frozen.json` 的 `revisions`，
launch 脚本会读它并传给 `--revision`。**revision 为空时启动脚本会拒绝启动。**

### 4.3 起服务

```bash
# node-a
bash serving/launch_mistral_node_a.sh
# node-b
bash serving/launch_phi_node_b.sh
# 任一台
bash serving/healthcheck.sh 127.0.0.1:8002 <node-b-ip>:8003
```

### 4.4 Phase 0（**零 GPU**，先跑，可能直接给出答案）

```bash
python src/forensics_reparse.py \
    --archive /path/to/P6R/results_live \
    --out out/PHASE0_FORENSICS.json
```

见 §5。

### 4.5 Phase 1 指纹

```bash
python src/fingerprint.py \
    --protocol protocols/p6r_replay_frozen.json \
    --out out/FINGERPRINT_$(date +%Y%m%dT%H%M%S).json
```

### 4.6 Phase 2 重放

```bash
python src/replay_runner.py \
    --protocol protocols/p6r_replay_frozen.json \
    --out out/replay_r0 --workers 64 --repeat 0

# anti-safe 的两次额外重复
python src/replay_runner.py --protocol protocols/p6r_replay_frozen.json \
    --out out/replay_r1 --workers 64 --repeat 1 --families anti_safe
python src/replay_runner.py --protocol protocols/p6r_replay_frozen.json \
    --out out/replay_r2 --workers 64 --repeat 2 --families anti_safe
```

分片到多台编排器（可选）：`--shard 0/2` 与 `--shard 1/2`。
中断后重跑同一条命令会**跳过已完成的 cell**（每个 cell 写完才落盘）。

### 4.7 Phase 3 分析与判定

```bash
python src/analyze_replay.py \
    --runs out/replay_r0 out/replay_r1 out/replay_r2 \
    --fingerprint out/FINGERPRINT_*.json \
    --original-fidelity 0.041 \
    --out out/REPLAY_VERDICT.json
```

输出直接给出 A / B / C 判定与该写进论文的句子。

---

## 5. Phase 0：零 GPU 的鉴别诊断（**先做这个**）

原始 P6R 只归档了 `actions`，没有原始 completion。这**限制**了能查的东西，但
恰好还留下了一个高信息量的证据：**那 4000 步到底打的是哪个 profile**。

`forensics_reparse.py` 从归档轨迹重算每个 anti-safe cell 的联合 profile 直方图，
并对三个互斥假设各给一个判别量：

| 假设 | 预测的 profile 直方图 | 判别 |
|---|---|---|
| **H1 角色/转置约定错位** | 质量集中在 $(a^\star_2, a^\star_1)$，即 target 的转置 | `transpose_mass` 高 |
| **H2 target 索引错位** | 质量集中在某个固定的非 target 格 | `single_offtarget_mass` 高 |
| **H3 真实不依从** | 质量分散，或集中在 warm-up 的低收益对角 basin $(0,0)$ | `diag_mass` 高 / 熵高 |

anti-safe 家族的 target 是 $(0,1)$、payoff ≈ (4.35, 4.05)，对角 basin ≈ (0.45,0.40)。
如果 4000 步里 pair 大量停在 $(1,0)$（≈ (1.30,1.25)），H1 几乎可以断定；
如果停在 $(0,0)$，那是真实不依从，H3。

**H1/H2 若成立，整个 GPU 重放就不必做了**——那是一个可以在 CPU 上修正并重算的
分析 bug，论文该做的是修正而不是重测。所以这一步必须先跑。

---

## 6. 派生 gate 策略与"重放是否忠实"的自检

`replay_runner.py` 在每个 cell 上：

1. 跑 `NoAlign` 10 个 warm-up episode → 得到 warm-up 记录 $W$；
2. 用 $W$ 计算 `legacy_certificate` 与 `action_certificate`（两者的实现直接
   复用代码补充包里的 `p7_runner.py` 同名函数，参数取自冻结协议的
   `legacy_safety` / `action_safety` 段）；
3. 跑 `Gated` 20 个 commitment episode；
4. `ActionSafeFixed` / `LegacyGateFixed` 的 commitment 轨迹 = 证书放行时取
   `Gated` 轨迹、否决时取 `NoAlign` 轨迹——**同一批已采样轨迹**，零额外调用。

**忠实性自检**（`analyze_replay.py` 自动执行，任一项失败即判 run 无效）：

- `ActionSafeFixed` 在 anti-safe 上放行 **40/40**，在 anti-tradeoff 上 **0/40**；
- `LegacyGateFixed` 在 anti-safe 上放行 **0/40**；
- 零缺失 cell、零 parse 失败、零 test-time probe；
- 指纹里的 canary 哈希在三次重复间完全一致（否则服务在跑的过程中变过）。

前两条是论文里对这一对模型的**授权层**结论。它们只依赖 payoff 表和 warm-up，
与 fidelity 无关，所以无论 fidelity 复现与否都**必须**复现。
**如果它们不复现，说明重放本身不忠实，fidelity 的结果不可用。**

---

## 7. 关于确定性的诚实说明

vLLM 的连续批处理下，同一请求的输出会随同批次的其他请求而微变，因此
**逐 token 的比特级复现不可得**，即使固定 `seed`。这个包不假装能做到：

- `--enable-prefix-caching` **关闭**（它会让输出依赖到达顺序）；
- 每请求带确定性 `seed = stable_seed(cell, episode, step, role, arm)`；
- 真正的对策是**重复三次并报告方差**，而不是声称确定性。

canary（Phase 1）用 `temperature=0` 且**逐条串行**发送，那部分是确定性的，
所以它可以作为服务栈的指纹；campaign 本身不是。

---

## 8. 文件清单

```
protocols/p6r_replay_frozen.json   冻结协议（含 revision 锁与预注册判据）
serving/launch_mistral_node_a.sh   node-a vLLM 启动（revision 强制）
serving/launch_phi_node_b.sh       node-b vLLM 启动
serving/healthcheck.sh             两端点连通性 + 模型身份核对
src/p6r_prompt.py                  P6R 的 prompt / parser（可挂载归档原件）
src/fingerprint.py                 Phase 1：服务指纹 + greedy canary + logprob 指纹
src/forensics_reparse.py           Phase 0：零 GPU 鉴别诊断
src/replay_runner.py               Phase 2：asyncio 多 worker 重放，可分片、可断点续跑
src/analyze_replay.py              Phase 3：矩阵聚类 bootstrap + 预注册判定
Makefile                           一键串起 phase0 / fingerprint / replay / analyze
```
