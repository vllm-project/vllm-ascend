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
    echo "==== CI Mode: scan the pull request diff ===="
    # A CI checkout has a clean index, so the staged-diff scan below finds
    # nothing to inspect. Scan the commits that the pull request adds to the
    # base branch instead, so a leaked credential in the diff fails the job.
    BASE_REF="refs/remotes/origin/${GITHUB_BASE_REF}"
    if ! git rev-parse --verify --quiet "${BASE_REF}" > /dev/null; then
        echo "::error::Cannot resolve the base branch ref '${BASE_REF}'."
        echo "::error::The secret scan cannot determine which commits to scan."
        exit 1
    fi
    BASE_SHA="$(git merge-base "${BASE_REF}" HEAD)"
    if [[ -z "${BASE_SHA}" ]]; then
        echo "::error::Cannot compute the merge base of '${BASE_REF}' and HEAD."
        exit 1
    fi
    SCAN_RANGE="${BASE_SHA}..HEAD"
    echo "Scanning the commits in '${SCAN_RANGE}'"
    ${BIN_CMD} detect \
        --verbose \
        --redact \
        --config="${CONFIG_FILE}" \
        --log-opts="${SCAN_RANGE}"
else
    echo "==== Local pre-commit Mode: scan staged changes ===="
    ${BIN_CMD} protect \
        --verbose \
        --redact \
        --config="${CONFIG_FILE}" \
        --staged
fi

exit 0
