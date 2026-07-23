#!/bin/bash
set -eo pipefail
set -x

BIN_NAME="./gitleaks"

if [ -x "${BIN_NAME}" ]; then
    echo "gitleaks binary exists, skip download"
else
    wget --no-host-directories -c --no-check-certificate https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta-codecheck/gitleaks
    chmod +x gitleaks
fi

CONFIG_FILE="./.gitleaks.toml"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "::error::Missing config file: ${CONFIG_FILE}"
    exit 1
fi

./gitleaks detect \
    --verbose \
    --redact \
    --config="${CONFIG_FILE}"
