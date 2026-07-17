# 四大实验类型总览 (Scheme 1-4)

> 日期: 2026-07-14 | 代码: `code/` | 数据: Lab `/data/lab/results/v2/` (979 metrics)
> 硬件: 2× NVIDIA RTX 5090 (32GB), 4-bit 本地推理 Qwen2.5-7B + GLM-4-9B

---

## 核心问题与定位

旧 GSACA (两臂选择器: split>0→CGA, split≤0→Gated) 在 deadlock 上对 NoToM 显著为负 (−0.023, p=0.003, n=20),破坏 "bounded-downside 选择器" 叙事。四个方案瞄准**消除显著负项**而非追加正项。

---

## 方案一: 三臂弃权 GSACA (零 GPU, 离线重算) — 已完成

### 机制
两臂 → 三臂: `split>+τ → CGA, split<−τ → Gated, |split|≤τ → NoToM` (弃权)。
τ=0.4 预注册主值。dev=seeds42-51 选 τ, holdout=52-61 冻结。

### 代码
- `code/scheme1_offline.py`: 纯离线重算,读现有 metrics + per-seed split_score。
- Lab GPU 版本: `run_experiment_local.py` 的 `het_3arm` cell + `run_3arm.sh` (τ∈{0,0.2,0.4,0.6} × 4 games × 5 seeds)。

### 结果 (τ=0.4, n=20)
| 博弈 | 旧GSACA vs NoToM | 三臂 vs NoToM | 臂 |
|---|---|---|---|
| chicken | +0.046 ns | +0.032 * | CGA |
| hawk_dove | +0.072 ** | +0.030 ** | CGA+ABSTAIN |
| **deadlock** | **−0.023 \*\*\*** | **−0.002 ns** | ABSTAIN+CGA |
| stag_hunt | +0.414 *** | +0.501 *** | Gated |
| BoS (n=5) | — | +0.657 ** | Gated |
| public_goods (n=5) | — | +0.000 | ABSTAIN |

**缺口消除**, 全表无显著负项。Proposition 3 验证: 弃权区 regret=0.000000。

---

## 方案二: silent-anti-coord (GPU) — 代码就绪

### 机制
反协调模式下屏蔽 cheap-talk: GSACA 检测 anti_coord 后设 `use_talk=False`。
假设: cheap-talk 推动收敛,在反协调博弈 (需分裂 profile) 中有害。

### 代码
- `code/scheme2_silent.py`: 独立 runner,不修改共享文件 → 零干扰。
- `code/scheme2_analyze.py`: 分析脚本 (silent vs NoToM/Gated/CGA/old-GSACA)。
- 跑 3 反协调博弈 × 20 seeds = 60 cells。

---

## 方案三: BoS + public_goods 20-seed (补缺口)

### 现状
BoS + public_goods n=20 数据因 run_final_fast.sh GPU 超订 OOM 全部失败 (0 metrics)。
需补齐以完成 6 博弈 n=20 完整表。

### 三臂理论预测 (基于 n=5 split_score)
- BoS: split=−2.50 << −τ → Gated 臂 → n=5 +0.657**; n=20 预期持平
- public_goods: split=−0.18 ∈ [−τ,+τ] → ABSTAIN 臂 → vs NoToM = **0.000 精确** (Prop 3, 任意 n)

### 代码
- `code/unified_orchestrator.py`: 合并编排器,含 BoS+pg (4 cells × 20 seeds × 2 games = 160 cells)。

---

## 方案四: V2 实验 (Exp A/D/E) 恢复

### 现状
- Exp A (同质对照): 305/400 metrics (~76%)
- Exp D (噪声+MP): 59 metrics
- Exp E (超参扫描): 0 (全 OOM)
- Exp B (20-seed): 320 (4 博弈完整, BoS+pg 缺)

### 代码
- `run_final_fast.sh`, `run_v2_orchestrator.py`, `run_v2_fix.py`, `run_v2_fix_fast.py`
- `code/unified_orchestrator.py` 含 D1 (噪声) + E (θ/α/W 扫描) 恢复。

---

## 合并编排器 `unified_orchestrator.py`

一个脚本跑全部 4 类型, 安全 VRAM 调度 (cap=2 het worker/GPU=18GB, 避免 OOM):
- TYPE 1: 3-arm, 4τ × 6 games × 5 seeds = 120 cells
- TYPE 2: silent, 3 games × 20 seeds = 60 cells
- TYPE 3: BoS+pg, 2 games × 20 seeds × 4 cells = 160 cells
- TYPE 4: D1+E+anti, 285 cells
- **总计 ~625 cells, ~7.8h (4 并发)**

---

## 已有数据 (Lab `/data/lab/results/v2/`, 979 metrics)
```
exp_a_fix: 305    exp_b_20seed: 320   exp_d_stress: 50
exp_c_payoff_prompt: 30   exp_anti_test: 55   exp_d_fix: 9
exp_e_hyperparam: 10
```

## Lab 代码 (未在 repo, 在 `/data/lab/gsaca/`)
- `run_experiment_local.py` (含 het_3arm cell + abstain_tau flag)
- `run_3arm.sh` (3-arm GPU launcher)
- `hettom_baseline.py` (核心库)
- `run_final_fast.sh` (V2 原始批次)
