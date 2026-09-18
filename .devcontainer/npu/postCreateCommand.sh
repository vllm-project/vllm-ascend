#!/usr/bin/env bash
# 复现 .github/workflows/pr_test.yaml 门禁中的 NPU 阶段（run-selected-tests 上卡）。
# 对应真实运行 35211545031 里 A3 双卡 runner 执行的 job：
#   - 'Check NPU availability'                      -> npu-smi info
#   - 'Install vllm-project/vllm-ascend with device' -> 编译自定义内核
#   - 'Run selected tests with device'               -> run_selected_tests.sh ... with-device
#
# 镜像：quay.io/ascend/vllm-ascend:nightly-main-a3（A3 芯片，需宿主挂载 NPU 卡）。
set -euo pipefail

PROJECT_DIR="/workspace/vllm-ascend"
cd "${PROJECT_DIR}"

echo "[1/6] 标记 git safe.directory"
git config --global --add safe.directory "${PROJECT_DIR}"

echo "[2/6] 检查 NPU 可用性 (门禁: 'Check NPU availability')"
npu-smi info

echo "[3/6] 激活 CANN 环境并安装开发依赖"
. /usr/local/Ascend/ascend-toolkit/set_env.sh
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install uv uc-manager
export UV_SYSTEM_PYTHON=1
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install -r requirements-dev.txt

echo "[4/6] 安装 triton-ascend (门禁: 'Install ... with device')"
uv pip install --force-reinstall --no-deps triton-ascend==3.2.2

echo "[5/6] 带设备编译安装 vllm-ascend（编译 NPU 自定义内核）"
export MAX_JOBS=46
uv pip install -e .

echo "[6/6] 运行 A3 双卡测试 (门禁: 'Run selected tests with device', a3-2 分区)"
VLLM_WORKER_MULTIPROC_METHOD=spawn \
  .github/workflows/scripts/run_selected_tests.sh a3 2 with-device tests/e2e/pull_request/two_card

echo "门禁 NPU(A3) 阶段复现完成"