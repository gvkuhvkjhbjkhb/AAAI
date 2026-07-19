# 新机器完整运行手册

以下命令面向 Linux 双 RTX 5090 实验机。不要在 Windows 规划机上启动生成实验。

## 0. 目录与变量

把本包、原 P3 结果和（若运行 source P2）S2 结果解压到独立目录：

```text
/data/aaai/supplement/
├── AAAI27_SUPPLEMENTAL_EXPERIMENTS_v1/
├── reference/
│   ├── exp_p3_transfer_test/
│   └── s2_results/                 # 仅 source P2 需要
└── results_supplement_v1/          # 必须是新目录
```

```bash
export WORK_ROOT=/data/aaai/supplement
export SUPPLEMENT_ROOT="$WORK_ROOT/AAAI27_SUPPLEMENTAL_EXPERIMENTS_v1"
export RESULT_ROOT="$WORK_ROOT/results_supplement_v1"
export PROTOCOL="$SUPPLEMENT_ROOT/protocols/supplement_frozen_protocol.json"
export P3_REFERENCE="$WORK_ROOT/reference/exp_p3_transfer_test"
export S2_REFERENCE="$WORK_ROOT/reference/s2_results"
export PYTHON_BIN=/data/venvs/safe-sca/bin/python
```

确认 `RESULT_ROOT` 不是原 S1/S2/P3 目录。运行器不会修改 `P3_REFERENCE` 或 `S2_REFERENCE`。

## 1. Python/GPU 环境

首选直接使用完成 S2/P3 的同一虚拟环境或容器。若新建环境，必须得到以下最终版本，而不是只满足宽松依赖：

```bash
"$PYTHON_BIN" - <<'PY'
import importlib.metadata as m
for name in ["vllm", "torch", "transformers", "openai", "numpy"]:
    print(name, m.version(name))
PY
```

必须至少匹配：vLLM 0.25.1、torch 2.11.0+cu128、transformers 5.14.1。`requirements-gpu-frozen.txt` 记录 Python 包要求；CUDA-specific torch 应从实验室与 S2/P3 相同的 wheel/channel 或镜像安装。不要为了让 preflight 通过而修改 `preflight_s1.py`。

## 2. 离线代码检查

```bash
test -f "$PROTOCOL"
test -f "$SUPPLEMENT_ROOT/code/run_experiment_local.py"
test -f "$P3_REFERENCE/P3_CAMPAIGN_SNAPSHOT.json"

"$PYTHON_BIN" -m py_compile "$SUPPLEMENT_ROOT"/code/*.py
PYTHONPATH="$SUPPLEMENT_ROOT/code" "$PYTHON_BIN" -m unittest discover \
  -s "$SUPPLEMENT_ROOT/code/tests" -v

"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/run_supplement_campaign.py" --help
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/validate_supplement_results.py" --help
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/analyze_supplement_results.py" --help
```

预期：10 个单元/合成集成测试全部通过；其中包含 Agent 2 收益表方向测试；测试不加载模型、不请求 endpoint。

## 3. 启动 revision-pinned vLLM

先确认 8000/8001 没有未知服务。启动脚本若发现端口已占用会主动退出，不会 kill 进程。

```bash
chmod +x "$SUPPLEMENT_ROOT/server_scripts/start_vllm_supplement.sh"
SUPPLEMENT_ROOT="$SUPPLEMENT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
  "$SUPPLEMENT_ROOT/server_scripts/start_vllm_supplement.sh"

curl --silent --fail http://localhost:8000/v1/models
curl --silent --fail http://localhost:8001/v1/models
cat "$SUPPLEMENT_ROOT/logs/vllm_supplement/pids_and_revisions.env"
nvidia-smi
```

脚本固定：两个 revision、`bfloat16`、`--enforce-eager`、0.85 GPU memory utilization、2048 max length、API key enforcement 和 `VLLM_USE_FLASHINFER_SAMPLER=0`。

## 4. strict preflight

```bash
mkdir -p "$RESULT_ROOT"
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/preflight_s1.py" \
  --out-dir "$RESULT_ROOT"
```

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import json, os
p=os.path.join(os.environ["RESULT_ROOT"],"ENVIRONMENT_MANIFEST_S1.json")
x=json.load(open(p))
print("passed=",x["preflight_passed"],"override=",x["allow_version_mismatch"])
print(x["package_versions"])
print(x["gpu"])
print(x["vllm_endpoints"])
assert x["preflight_passed"] and not x["allow_version_mismatch"]
PY
```

不要使用 `--allow-version-mismatch` 运行 confirmatory block。

## 5. dry-run 与冻结数量

```bash
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/run_supplement_campaign.py" \
  --protocol "$PROTOCOL" --results-root "$RESULT_ROOT" --dry-run
```

主块预期：640 cells、240 matrix-seed-experiment tasks、32 workers。若数字或 protocol hash 与日志记录不一致，停止。

## 6. 推荐分阶段执行

### 6.1 先运行 P0（最高优先级，320 cells）

```bash
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/run_supplement_campaign.py" \
  --protocol "$PROTOCOL" --results-root "$RESULT_ROOT" \
  --experiments p0
```

P0 完成后可以先检查完整性，但不要根据结果修改 P1/P2：

```bash
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/validate_supplement_results.py" \
  --protocol "$PROTOCOL" --results-root "$RESULT_ROOT" \
  --experiments p0
```

### 6.2 运行 P1（240 cells）

```bash
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/run_supplement_campaign.py" \
  --protocol "$PROTOCOL" --results-root "$RESULT_ROOT" \
  --experiments p1
```

### 6.3 运行 P2-P3（80 cells）

```bash
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/run_supplement_campaign.py" \
  --protocol "$PROTOCOL" --results-root "$RESULT_ROOT" \
  --experiments p2
```

如果从一开始就连续运行主块：

```bash
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/run_supplement_campaign.py" \
  --protocol "$PROTOCOL" --results-root "$RESULT_ROOT" \
  --experiments p0 p1 p2
```

可选 source P2：

```bash
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/run_supplement_campaign.py" \
  --protocol "$PROTOCOL" --results-root "$RESULT_ROOT" \
  --experiments p2 --include-source-p2
```

分阶段命令各自写独立的 immutable snapshot 和 execution report，不会覆盖其他阶段。

## 7. 监控与恢复

```bash
find "$RESULT_ROOT" -path '*/metrics.json' | wc -l
find "$RESULT_ROOT" -path '*/decision.json' | wc -l
tail -f "$RESULT_ROOT"/logs_campaign/*.log
nvidia-smi
```

普通中断：重新运行完全相同命令。已有 `metrics.json` 的 cell 自动跳过。不要删除部分结果，不要加 `--force`（脚本没有提供该参数）。

## 8. 最终完整性校验

主块：

```bash
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/validate_supplement_results.py" \
  --protocol "$PROTOCOL" --results-root "$RESULT_ROOT" \
  --experiments p0 p1 p2
```

预期 `SUPPLEMENT_INTEGRITY_REPORT.json`：640 expected/checked metrics、0 missing、0 errors、`ready_for_analysis=true`。

若包括 source P2：

```bash
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/validate_supplement_results.py" \
  --protocol "$PROTOCOL" --results-root "$RESULT_ROOT" \
  --experiments p0 p1 p2 --include-source-p2
```

预期 760 metrics。

## 9. 冻结统计分析

主块：

```bash
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/analyze_supplement_results.py" \
  --protocol "$PROTOCOL" --results-root "$RESULT_ROOT" \
  --p3-reference-root "$P3_REFERENCE"
```

包括 source P2：

```bash
"$PYTHON_BIN" "$SUPPLEMENT_ROOT/code/analyze_supplement_results.py" \
  --protocol "$PROTOCOL" --results-root "$RESULT_ROOT" \
  --p3-reference-root "$P3_REFERENCE" \
  --source-reference-root "$S2_REFERENCE" --include-source-p2
```

输出：

- `SUPPLEMENT_ANALYSIS.json`：所有冻结统计量；
- `SUPPLEMENT_ANALYSIS.md`：可直接核对的矩阵表；
- `SUPPLEMENT_CELL_LEVEL.csv`：cell-level 审计表。

分析脚本在完整性校验未通过时拒绝运行。

## 10. 归档

```bash
cd "$WORK_ROOT"
zip -r AAAI27_SUPPLEMENTAL_RESULTS_v1.zip \
  "$(basename "$RESULT_ROOT")" \
  "$(basename "$SUPPLEMENT_ROOT")"
sha256sum AAAI27_SUPPLEMENTAL_RESULTS_v1.zip \
  > AAAI27_SUPPLEMENTAL_RESULTS_v1.zip.sha256
```

归档必须包含：环境 manifest、所有 immutable snapshots、execution reports、server revision 日志、每个 metrics/trajectory/decision、完整性报告、分析 JSON/MD/CSV 和本代码包。
