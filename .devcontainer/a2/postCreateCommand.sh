#!/usr/bin/env bash
# 复现 .github/workflows/pr_test.yaml 门禁中的 NPU 阶段（A2 单卡 910B）。
# 对应 _selected_tests.yaml 里 a2 分区（npu_type=a2, num_npus=1）在
# linux-aarch64-a2b3-1 runner 上执行的 job：
#   - 'Check NPU availability'                         -> npu-smi info
#   - 'Install vllm-project/vllm from source'          -> checkout pin + VLLM_TARGET_DEVICE=empty uv pip install .
#   - 'Install vllm-project/vllm-ascend with device'   -> triton-ascend + 编译自定义内核
#   - 'Run selected tests with device'                 -> run_selected_tests.sh a2 1 with-device tests/e2e/pull_request/one_card
#
# 镜像：quay.io/ascend/vllm-ascend:nightly-main（A2 910B 芯片，需宿主挂载 NPU 卡）。
# nightly-main 是 all-in-one 镜像，内置的 vllm 是 v0.28.0 tag，其 API 与门禁 pin
# （.github/vllm-main-verified.commit）不一致，因此与 cpu 脚本一致先重装 vllm pin。
set -euo pipefail

PROJECT_DIR="/workspace/vllm-ascend"
cd "${PROJECT_DIR}"

echo "[1/6] 标记 git safe.directory"
git config --global --add safe.directory "${PROJECT_DIR}"

echo "[2/6] 检查 NPU 可用性 (门禁: 'Check NPU availability')"
npu-smi info

echo "[3/6] 激活 CANN 环境并安装开发依赖 (门禁: 'Install packages')"
# shellcheck disable=SC1091
. /usr/local/Ascend/ascend-toolkit/set_env.sh
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install uv uc-manager
export UV_SYSTEM_PYTHON=1
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

echo "[4/6] 从门禁 pin 提交重装 vllm (门禁: 'Install vllm-project/vllm from source')"
# 与 cpu 脚本一致：把镜像内置的 /vllm-workspace/vllm（editable）切到 pin 提交并
# 以 empty 设备重装，生成与门禁一致的 version 元数据，避免 v0.28.0 tag 的 API 差异。
VLLM_PIN="$(tr -d '[:space:]' < "${PROJECT_DIR}/.github/vllm-main-verified.commit")"
VLLM_SRC="/vllm-workspace/vllm"
git config --global --add safe.directory "${VLLM_SRC}"
git config --global --remove-section 'url.https://gh-proxy.test.osinfra.cn/https://github.com/' 2>/dev/null || true
git -C "${VLLM_SRC}" fetch --depth 1 https://github.com/vllm-project/vllm.git "${VLLM_PIN}"
git -C "${VLLM_SRC}" checkout -f FETCH_HEAD
( cd "${VLLM_SRC}" && VLLM_TARGET_DEVICE=empty uv pip install . --force-reinstall --no-deps --no-build-isolation )
pip uninstall -y triton

echo "[5/6] 安装 triton-ascend 并带设备编译安装 vllm-ascend (门禁: 'Install ... with device')"
cd "${PROJECT_DIR}"
uv pip install -r requirements-dev.txt
uv pip install --force-reinstall --no-deps triton-ascend==3.2.2
export MAX_JOBS=23
uv pip install -e . --no-build-isolation

echo "[6/6] 运行 A2 单卡测试 (门禁: 'Run selected tests with device', a2-1 分区)"
VLLM_WORKER_MULTIPROC_METHOD=spawn \
  .github/workflows/scripts/run_selected_tests.sh a2 1 with-device tests/e2e/pull_request/one_card

echo "门禁 NPU(A2) 阶段复现完成"