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
    # CI mode (pull_request runs): scan exactly what the PR changes relative
    # to the base branch. The gitleaks binary only supports `protect --staged`
    # (index vs HEAD), so point HEAD at the base branch and the index at the
    # PR head, scan, then restore the PR head for the steps that follow.
    echo "==== CI Mode: scan PR changes against ${GITHUB_BASE_REF} ===="
    PR_SHA="$(git rev-parse HEAD)"
    git fetch --quiet origin "${GITHUB_BASE_REF}"
    git checkout --quiet "origin/${GITHUB_BASE_REF}"
    git read-tree --reset "${PR_SHA}"
    # A leak exits non-zero here and aborts the job before the restore below,
    # which is the desired behavior (the PR must not pass the secret scan).
    ${BIN_CMD} protect \
        --verbose \
        --redact \
        --config="${CONFIG_FILE}" \
        --staged
    # Scan passed: restore the PR head so later steps see the PR worktree.
    if [[ -n "${GITHUB_HEAD_REF}" ]]; then
        git checkout --quiet "${GITHUB_HEAD_REF}"
    else
        git checkout --quiet "${PR_SHA}"
    fi
    git reset --quiet --hard
else
    echo "==== Local pre-commit Mode: scan staged changes ===="
    ${BIN_CMD} protect \
        --verbose \
        --redact \
        --config="${CONFIG_FILE}" \
        --staged
fi

exit 0
