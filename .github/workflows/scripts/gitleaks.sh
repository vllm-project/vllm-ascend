#!/bin/bash
set -eo pipefail
set -x

BIN_LOCAL="./gitleaks"
CONFIG_FILE="./.gitleaks.toml"

if command -v gitleaks &> /dev/null; then
    echo "Found gitleaks in system PATH, use system binary"
    BIN_CMD="gitleaks"
else
    echo "System gitleaks not found, download from OBS"
    if [ ! -x "${BIN_LOCAL}" ]; then
        wget --no-host-directories -c --no-check-certificate https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta-codecheck/gitleaks
        chmod +x "${BIN_LOCAL}"
    fi
    BIN_CMD="${BIN_LOCAL}"
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "::error::Missing config file: ${CONFIG_FILE}"
    exit 1
fi

if [[ -n "${GITHUB_BASE_REF}" ]]; then
    echo "==== CI Mode: skip incremental changed file scan (disabled) ===="
    echo "Secret scan moved to local pre-commit only"
else
    echo "==== Local pre-commit Mode: scan staged changes ===="
    ${BIN_CMD} protect \
        --verbose \
        --redact \
        --config="${CONFIG_FILE}" \
        --staged
fi

exit 0
