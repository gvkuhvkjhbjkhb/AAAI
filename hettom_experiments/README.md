# HetToM Experiments

本文件夹存放**新实验**的全部代码与方案，验证"推理陷阱 + 异质+ToM 破局"命题。

> 与仓库根目录的旧实验（FA-PBS / DG-PBS / FME 等，已证伪）完全独立。
> 详见 `Plan_Assessment.txt`（根目录）对旧方案的故事性/可行性评估。

## 文件清单

| 文件 | 作用 | 层 |
|---|---|---|
| `hettom_baseline.py` | Layer 1 核心：4 格矩阵博弈 + 递归 ToM + bootstrap CI + 多种子 | 1 |
| `analyze_layer1.py` | 跨种子聚合 + Mann-Whitney/Wilcoxon + 预注册决策判定 | 1 |
| `layer2_marl_bridge.py` | Layer 2：离线 ToM 意图注入 + EPyMARL 配置 + 启动器 | 2 |
| `mappo_hettom.yaml` | Layer 2 的 MAPPO 配置（由 layer2 脚本复制到 EPyMARL config 目录） | 2 |
| `EXPERIMENT_PLAN.md` | 完整 4 周实验方案 | — |

## 运行方式

所有脚本自动定位仓库根目录（本文件夹在 `<repo>/hettom_experiments/`），
因此可从任意目录运行，路径会自动锚定到仓库根。

### Layer 1（核心，第 1-2 周）

```bash
# 快速验证 pipeline（无 GPU/模型，CPU 秒级）
python3 hettom_experiments/hettom_baseline.py --matrix --mock --seeds 42 \
  --episodes 8 --horizon 3 --tom_order 2 --game stag_hunt

# 真实主实验（需 GPU + 模型权重）
for game in stag_hunt battle_of_the_sexes chicken public_goods coordination; do
  python3 hettom_experiments/hettom_baseline.py --matrix --game $game \
    --seeds 1 2 3 4 5 --episodes 50 --horizon 5 --tom_order 1 \
    --out_dir results/hettom_layer1/$game
done

# 汇总分析
python3 hettom_experiments/analyze_layer1.py \
  --results_dir results/hettom_layer1 --out_dir results/hettom_layer1
```

### Layer 2（增强，第 3-4 周）

```bash
# 离线 ToM 意图注入
python3 hettom_experiments/layer2_marl_bridge.py inject \
  --layer1_dir results/hettom_layer1 \
  --out results/hettom_layer2/intent_features.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct

# 启动 MAPPO 训练（自动写入 EPyMARL config + 调用 src/main.py）
python3 hettom_experiments/layer2_marl_bridge.py launch \
  --config mappo_hettom --seeds 1 2 3 4 5 \
  --intent_path results/hettom_layer2/intent_features.jsonl \
  --env lbforaging:Foraging-10x10-3p-3f-v3 --t_max 500000
```

## 设计要点

1. **离线 LLM，非 per-step 在线**：复用仓库 `offline_relabel.py` 的 HF idiom，
   避开单 GPU 上 per-step 在线推理的不可行性。
2. **短矩阵博弈**（1-5 轮/episode）：单 5090 可跑数千 LLM episode。
3. **不触碰已证伪死路**：不做自适应奖励干预（FA-PBS/DG-PBS/FME 已 3 次失败）；
   Layer 2 意图注入在观测侧，不改奖励，保持策略不变性。
4. **预注册决策**：`analyze_layer1.py` 内置"陷阱是否被打破"判定规则。

详见 `EXPERIMENT_PLAN.md`。
