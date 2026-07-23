#!/bin/bash
set -eo pipefail
set -x

BIN_NAME="./gitleaks"
CONFIG_FILE="./.gitleaks.toml"
BASE_BRANCH="${GITHUB_BASE_REF:-main}"

# 下载二进制
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

# 获取基线分支 & 拉取基线
echo "Base branch: ${BASE_BRANCH}"
git fetch origin "${BASE_BRANCH}"

# 获取本次PR新增/修改的文件列表（排除删除的文件）
CHANGED_FILES=$(git diff --name-only --diff-filter=ACM "origin/${BASE_BRANCH}...HEAD")
echo "==== Changed files to scan ===="
echo "${CHANGED_FILES}"

# 如果没有变更文件，直接退出
if [ -z "${CHANGED_FILES}" ]; then
    echo "No changed files, skip scan"
    exit 0
fi

# 遍历变更文件逐个扫描
EXIT_CODE=0
while IFS= read -r file; do
    if [ -z "${file}" ]; then
        continue
    fi
    if [ ! -f "${file}" ]; then
        echo "Skip deleted file: ${file}"
        continue
    fi
    echo "Scan file: ${file}"
    ./gitleaks detect \
        --source="${file}" \
        --config="${CONFIG_FILE}" \
        --no-git \
        --verbose \
        --redact || EXIT_CODE=$?
done <<< "${CHANGED_FILES}"

if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "::error::Secret leaks found in changed files!"
    exit "${EXIT_CODE}"
fi
echo "Scan finished, no secrets found in modified files."
