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
git fetch origin "${BASE_BRANCH}"

TMP_BASE="./tmp-base"
rm -rf "${TMP_BASE}"
mkdir -p "${TMP_BASE}"
git archive "origin/${BASE_BRANCH}" | tar -x -C "${TMP_BASE}"

BASE_REPORT="./baseline.json"
# 基线扫描，移除 --exclude-path
./gitleaks detect \
    --source="${TMP_BASE}" \
    --config="${CONFIG_FILE}" \
    --no-git \
    --report-format=json \
    --report-path="${BASE_REPORT}"

# 当前代码扫描，基于基线过滤新增泄漏
./gitleaks detect \
    --source=. \
    --config="${CONFIG_FILE}" \
    --no-git \
    --baseline-path="${BASE_REPORT}" \
    --verbose \
    --redact

rm -rf "${TMP_BASE}" "${BASE_REPORT}"
