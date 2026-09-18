#!/usr/bin/env bash
# 复现 .github/workflows/pr_test.yaml 门禁中的 cpu-ut 阶段。
# 对应 _selected_tests.yaml 里 cpu 分区的步骤：
#   - 'Install vllm-project/vllm-ascend no device'
#   - 'Run selected tests without device'  -> run_selected_tests.sh cpu 0 without-device tests/ut
#
# 镜像：quay.io/ascend/vllm-ascend:nightly-main（all-in-one，已内置 vllm + vllm-ascend + CANN）。
# 门禁的 pre-commit/mypy 跑在独立的 quay.io/ascend-ci/vllm-ascend:lint(amd64) 镜像里，
# 与本容器无关，故此处不复现。
set -euo pipefail

PROJECT_DIR="/workspace/vllm-ascend"
cd "${PROJECT_DIR}"

# 网络兜底：本地 WSL2 容器直连外网被墙，需走宿主机 squid 代理 + 清华 pip 镜像。
# GitHub runner 上的容器可直连外网，无需代理：传 USE_HOST_PROXY=0 关闭此兜底。
if [ "${USE_HOST_PROXY:-1}" = "1" ]; then
    if [ -z "${HTTP_PROXY:-}" ]; then
        export HTTP_PROXY="http://host.docker.internal:3128"
    fi
    if [ -z "${HTTPS_PROXY:-}" ]; then
        export HTTPS_PROXY="http://host.docker.internal:3128"
    fi
fi
export http_proxy="${http_proxy:-${HTTP_PROXY:-}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY:-}}"

echo "[1/5] 标记 git safe.directory"
git config --global --add safe.directory "${PROJECT_DIR}"

echo "[2/5] 激活 CANN 环境（all-in-one 镜像需 devlib 路径才能 import torch_npu）"
# shellcheck disable=SC1091
. /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "[3/5] 安装 uv 与开发依赖 (cpu-ut: 'Install vllm-ascend no device')"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install uv uc-manager
export UV_SYSTEM_PYTHON=1
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install -r requirements-dev.txt

echo "[4/5] CPU 模式安装 vllm-ascend（跳过 NPU 自定义内核编译）"
# --no-build-isolation：复用镜像内置的 triton-ascend==3.2.2（该包仅在华为内网源上，
# 默认构建隔离会尝试从远程重新解析而失败）
SOC_VERSION=ascend910b1 COMPILE_CUSTOM_KERNELS=0 uv pip install -e . --no-build-isolation

echo "[5/5] CPU 单元测试 (cpu-ut: 'Run selected tests without device')"
# all-in-one 镜像内置了 /vllm-workspace/vllm（editable 安装），而门禁 cpu-ut 的基础
# 镜像里没有该目录。test_version_compat.py:192 的 skipif 依赖这个差异来跳过真实
# git fetch 测试；此处无该差异，故显式 deselect 这个联网测试（其会 git fetch origin
# v0.25.1 并 --unshallow 卡死），其余 11 个单测仍正常运行。
PYTEST_ADDOPTS="--deselect=tests/ut/tools/bisect/test_version_compat.py::test_adapter_switches_real_vllm_release_and_restores_source" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  .github/workflows/scripts/run_selected_tests.sh cpu 0 without-device tests/ut

echo "门禁 cpu-ut 本地复现完成"