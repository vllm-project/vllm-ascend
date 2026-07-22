#!/bin/bash
set -eo pipefail
set -x

GITLEAKS_VERSION="v8.21.0"
BIN_NAME="./gitleaks"

if [ -x "${BIN_NAME}" ]; then
    echo "gitleaks binary exists, skip download"
else
    ARCH=$(uname -m)
    if [[ "${ARCH}" == "x86_64" ]]; then
        PKG="gitleaks_${GITLEAKS_VERSION#v}_linux_amd64.tar.gz"
    elif [[ "${ARCH}" == "aarch64" ]]; then
        PKG="gitleaks_${GITLEAKS_VERSION#v}_linux_arm64.tar.gz"
    else
        echo "::error::Unsupported arch: ${ARCH}"
        exit 1
    fi

    URL="https://github.com/gitleaks/gitleaks/releases/download/${GITLEAKS_VERSION}/${PKG}"
    wget -q "${URL}"
    tar -xf "${PKG}" gitleaks
    chmod +x ./gitleaks
fi

./gitleaks detect \
    --verbose \
    --redact \
    --config=pre-commit/.gitleaks.toml

rm -f gitleaks_*.tar.gz
