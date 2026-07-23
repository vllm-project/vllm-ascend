#!/bin/bash
set -eo pipefail
set -x

BIN_NAME="./gitleaks"
CONFIG_FILE="./.gitleaks.toml"

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

BASE_BRANCH="${GITHUB_BASE_REF:-main}"
echo "Base branch: ${BASE_BRANCH}"

git fetch origin "${BASE_BRANCH}"
git diff "origin/${BASE_BRANCH}..."HEAD | ./gitleaks protect \
    --verbose \
    --redact \
    --config="${CONFIG_FILE}" \
    --stdin
