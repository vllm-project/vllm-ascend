#!/bin/bash
set -eo pipefail
set -x

BIN_NAME="./gitleaks"
CONFIG_FILE="./.gitleaks.toml"
BASE_BRANCH="${GITHUB_BASE_REF:-main}"

if [ -x "${BIN_NAME}" ]; then
    echo "gitleaks binary exists, skip download"
else
    wget --no-host-directories -c --no-check-certificate https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta-codecheck/gitleaks
    chmod +x gitleaks
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "::error::Missing config file: ${CONFIG_FILE}"
    exit 1
fi

echo "Base branch: ${BASE_BRANCH}"
# 拉取基线
git fetch origin "${BASE_BRANCH}"

# 1. 检出基线代码到临时目录
TMP_BASE="./tmp-base"
rm -rf "${TMP_BASE}"
mkdir -p "${TMP_BASE}"
git archive "origin/${BASE_BRANCH}" | tar -x -C "${TMP_BASE}"

# 2. 扫描基线，生成基线报告（存量告警）
BASE_REPORT="./baseline.json"
./gitleaks detect \
    --source="${TMP_BASE}" \
    --config="${CONFIG_FILE}" \
    --no-git \
    --exclude-path=vllm-empty \
    --report-format=json \
    --report-path="${BASE_REPORT}"

# 3. 扫描当前代码，基于基线过滤，只输出新增泄漏
./gitleaks detect \
    --source=. \
    --config="${CONFIG_FILE}" \
    --no-git \
    --exclude-path=vllm-empty \
    --baseline-path="${BASE_REPORT}" \
    --verbose \
    --redact

# 清理临时文件
rm -rf "${TMP_BASE}" "${BASE_REPORT}"
