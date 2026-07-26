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
    echo "==== CI Mode: scan changed files ===="
    BASE_BRANCH="${GITHUB_BASE_REF}"
    git fetch origin "${BASE_BRANCH}"
    CHANGED_FILES=$(git diff --name-only --diff-filter=ACM "origin/${BASE_BRANCH}...HEAD")
    echo "Changed files:"
    echo "${CHANGED_FILES}"

    if [ -z "${CHANGED_FILES}" ]; then
        echo "No changed files, skip scan"
        exit 0
    fi

    EXIT_CODE=0
    while IFS= read -r file; do
        [ -z "${file}" ] && continue
        [ ! -f "${file}" ] && echo "Skip deleted ${file}" && continue
        echo "Scan ${file}"
        ${BIN_CMD} detect \
            --source="${file}" \
            --config="${CONFIG_FILE}" \
            --no-git \
            --verbose \
            --redact || EXIT_CODE=$?
    done <<< "${CHANGED_FILES}"

    if [ "${EXIT_CODE}" -ne 0 ]; then
        echo "::error::Leaks found in changed files!"
        exit "${EXIT_CODE}"
    fi
else
    echo "==== Local pre-commit Mode: scan staged changes ===="
    ${BIN_CMD} protect \
        --verbose \
        --redact \
        --config="${CONFIG_FILE}" \
        --staged
fi
