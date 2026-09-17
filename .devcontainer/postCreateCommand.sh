#!/usr/bin/env bash
# 对齐 .github/workflows/pr_test.yaml 门禁中 pre-commit 与 cpu-ut 两个 job 的
# 本地可复现步骤（参照真实运行记录 PR #16724 @ run 35106574759）。
#
# 说明：
# - 本 devcontainer 使用官方 all-in-one 镜像（已内置 vllm + vllm-ascend + CANN），
#   因此跳过 cpu-ut 中 "checkout vllm 主仓库并 VLLM_TARGET_DEVICE=empty 重装" 的步骤。
# - gitleaks 扫描由 pre-commit 的 gitleaks-offline-scan hook 执行（与门禁一致：
#   门禁中独立的 gitleaks.sh step 在 CI 下实际 skip，真正扫描在 pre-commit hook 内）。
set -euo pipefail

PROJECT_DIR="/workspace/vllm-ascend"
cd "${PROJECT_DIR}"

echo "[1/6] 标记 git safe.directory"
git config --global --add safe.directory "${PROJECT_DIR}"

echo "[2/6] 安装 uv 与开发依赖 (cpu-ut: 'Install vllm-ascend no device')"
pip install uv uc-manager
uv pip install -r requirements-dev.txt

echo "[3/6] CPU 模式安装 vllm-ascend（跳过 NPU 自定义内核编译）"
SOC_VERSION=ascend910b1 COMPILE_CUSTOM_KERNELS=0 uv pip install -e .

echo "[4/6] pre-commit 全量检查 (pre-commit: 'Run pre-commit')"
pre-commit run --all-files --hook-stage manual --show-diff-on-failure

echo "[5/6] mypy 类型检查 (pre-commit: 'Run mypy'; CI 跑 3.10/3.11/3.12，本地按当前解释器)"
tools/mypy.sh 1

echo "[6/6] CPU 单元测试 (cpu-ut: 'Run selected tests without device')"
VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  .github/workflows/scripts/run_selected_tests.sh cpu 0 without-device tests/ut

echo "门禁本地复现完成"