#!/usr/bin/env bash
# Install the frozen S2/P3 serving stack into /data/venvs/safe-sca.
# Target versions: vllm==0.25.1, torch==2.11.0+cu128, transformers==5.14.1.
set -euo pipefail

VENV=/data/venvs/safe-sca
LOG=/data/aaai/supplement/install_env.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== install start $(date -u +%FT%TZ) ==="

# 1. venv (without --copies; --copies broke ensurepip on this host)
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools

# 2. torch 2.11.0+cu128 (explicit CUDA 12.8 wheel from pytorch index).
"$VENV/bin/python" -m pip install --no-cache-dir \
  torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 3. vllm 0.25.1.  Let it pull its own deps but keep the torch we just installed.
"$VENV/bin/python" -m pip install --no-cache-dir vllm==0.25.1

# 4. transformers pinned to 5.14.1 (vllm may have pulled a different version).
"$VENV/bin/python" -m pip install --no-cache-dir --force-reinstall --no-deps \
  transformers==5.14.1

# 5. analysis-side deps.
"$VENV/bin/python" -m pip install --no-cache-dir "openai>=1.0" "numpy>=1.26,<3"

echo "=== version check ==="
"$VENV/bin/python" - <<'PY'
import importlib.metadata as m
for n in ["vllm","torch","transformers","openai","numpy"]:
    print(n, m.version(n))
PY

echo "=== install end $(date -u +%FT%TZ) ==="
