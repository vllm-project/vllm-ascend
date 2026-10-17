#!/usr/bin/env bash
# 复现 .github/workflows/pr_test.yaml 门禁中的 cpu-ut 阶段（与门禁配置对齐）。
# 对应 _selected_tests.yaml 里 cpu 分区（npu_type=cpu）的完整步骤：
#   - 'Install vllm-project/vllm from source'   -> checkout pin 提交 + VLLM_TARGET_DEVICE=empty uv pip install .
#   - 'Install vllm-ascend no device'           -> pip install uc-manager + uv pip install -r requirements-dev.txt + uv pip install -e .
#   - 'Run selected tests without device'       -> run_selected_tests.sh cpu 0 without-device tests/ut
#
# 镜像：quay.io/ascend/vllm-ascend:nightly-main（all-in-one，已内置 vllm + vllm-ascend + CANN）。
# 门禁的 pre-commit/mypy 跑在独立的 quay.io/ascend-ci/vllm-ascend:lint(amd64) 镜像里，
# 与本容器无关，故此处不复现。
#
# 注意：nightly-main 镜像内置的 vllm 是 v0.28.0 tag，其 kv_cache_interface / attn_utils
# API 与门禁 pin（.github/vllm-main-verified.commit 指向的 main 提交）不一致：
#   - KVCacheTensor 字段在 tag 上是 shared_by，在 main pin 上是 layers/layer_stride
#   - allocate_kv_cache 在 tag 上已改名为 _allocate_kv_cache
# 这会让 tests/ut/worker 下 kvpp/attention 相关 7 个单测失败。因此此处与门禁对齐：
# 先 checkout pin 提交并以 empty 设备从源码重装 vllm，再安装 vllm-ascend。
set -euo pipefail

PROJECT_DIR="/workspace/vllm-ascend"
cd "${PROJECT_DIR}"

echo "[1/6] 标记 git safe.directory"
git config --global --add safe.directory "${PROJECT_DIR}"

echo "[2/6] 激活 CANN 环境（all-in-one 镜像需 devlib 路径才能 import torch_npu）"
# shellcheck disable=SC1091
. /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "[3/6] 安装 uv (门禁: 'Install packages')"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install uv uc-manager
export UV_SYSTEM_PYTHON=1
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

echo "[4/6] 从门禁 pin 提交重装 vllm (门禁: 'Install vllm-project/vllm from source')"
# 门禁 cpu-ut 在 base 镜像里 checkout .github/vllm-main-verified.commit 指向的 main 提交，
# 并以 VLLM_TARGET_DEVICE=empty 重新 build vllm。nightly-main 镜像内置的是 v0.28.0 tag，
# 其 _version.py（untracked）与 kv_cache_interface / attn_utils API 都和门禁 pin 不一致，
# 直接跑 tests/ut 会有 7 个 kvpp/attention 单测失败。此处复刻门禁：把镜像内置的
# /vllm-workspace/vllm（editable）切到 pin 提交并重装，生成与门禁一致的 version 元数据。
VLLM_PIN="$(tr -d '[:space:]' < "${PROJECT_DIR}/.github/vllm-main-verified.commit")"
VLLM_SRC="/vllm-workspace/vllm"
git config --global --add safe.directory "${VLLM_SRC}"
# amd64 镜像在构建时执行过 buildkite 脚本，往 /root/.gitconfig 写入了
#   url."https://gh-proxy.test.osinfra.cn/https://github.com/".insteadOf "https://github.com/"
# 会把任何 github.com URL 重写成内网代理 gh-proxy.test.osinfra.cn（GitHub runner
# 直连不到，报 418）。fetch 前先移除该 url section，aarch64 镜像无此配置时忽略。
git config --global --remove-section 'url.https://gh-proxy.test.osinfra.cn/https://github.com/' 2>/dev/null || true
git -C "${VLLM_SRC}" fetch --depth 1 https://github.com/vllm-project/vllm.git "${VLLM_PIN}"
git -C "${VLLM_SRC}" checkout -f FETCH_HEAD
( cd "${VLLM_SRC}" && VLLM_TARGET_DEVICE=empty uv pip install . --force-reinstall --no-deps --no-build-isolation )
pip uninstall -y triton

echo "[5/6] CPU 模式安装 vllm-ascend (门禁: 'Install vllm-ascend no device')"
# --no-build-isolation：复用镜像内置的 triton-ascend==3.2.2（该包仅在华为内网源上，
# 默认构建隔离会尝试从远程重新解析而失败）
cd "${PROJECT_DIR}"
uv pip install -r requirements-dev.txt
SOC_VERSION=ascend910b1 COMPILE_CUSTOM_KERNELS=0 uv pip install -e . --no-build-isolation

echo "[6/6] CPU 单元测试 (cpu-ut: 'Run selected tests without device')"
# all-in-one 镜像内置了 /vllm-workspace/vllm（editable 安装），而门禁 cpu-ut 的基础
# 镜像里没有该目录。test_version_compat.py:192 的 skipif 依赖这个差异来跳过真实
# git fetch 测试；此处无该差异，故显式 deselect 这个联网测试（其会 git fetch origin
# v0.25.1 并 --unshallow 卡死），其余 11 个单测仍正常运行。
PYTEST_ADDOPTS="--deselect=tests/ut/tools/bisect/test_version_compat.py::test_adapter_switches_real_vllm_release_and_restores_source" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  .github/workflows/scripts/run_selected_tests.sh cpu 0 without-device tests/ut

echo "门禁 cpu-ut 本地复现完成"