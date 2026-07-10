#!/bin/bash

set -euo pipefail

# arguments:
# $1: SOC_ARG (ascend910b, ascend910_93, ascend950)

SOC_ARG="${1:-}"

log() {
    echo "[install_batch_invariant] $*"
}

# validate arguments
if [[ -z "${SOC_ARG}" ]]; then
    log "ERROR: SOC_ARG is required as first argument"
    exit 1
fi

log "Starting batch_invariant installation..."
log "SOC_ARG=${SOC_ARG}"

# determine device type from SOC_ARG
case "${SOC_ARG}" in
    ascend910b)
        BATCH_INVARIANT_DEVICE="910b"
        ;;
    ascend910_93)
        BATCH_INVARIANT_DEVICE="A3"
        ;;
    *)
        log "Warning: batch_invariant not available for SOC_ARG=${SOC_ARG}; skipping"
        exit 0
        ;;
esac

# detect system architecture
ARCH_INFO=$(uname -m)
case "${ARCH_INFO}" in
    aarch64)
        ARCH_SUFFIX="aarch64"
        ;;
    x86_64)
        ARCH_SUFFIX="x86_64"
        ;;
    *)
        log "Warning: unknown architecture ${ARCH_INFO}; cannot determine batch_invariant package"
        exit 0
        ;;
esac

# download and install run package
BATCH_INVARIANT_RUN_URL="https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/cann-ops-batch_invariant-${BATCH_INVARIANT_DEVICE}-1.0.0-linux.${ARCH_SUFFIX}.run"
BATCH_INVARIANT_RUN_FILE="cann-ops-batch_invariant-${BATCH_INVARIANT_DEVICE}-1.0.0-linux.${ARCH_SUFFIX}.run"

log "Downloading batch_invariant run package..."
unset ASCEND_CUSTOM_OPP_PATH
if curl --max-time 60 -sS -k -O "${BATCH_INVARIANT_RUN_URL}" && [[ -f "${BATCH_INVARIANT_RUN_FILE}" ]]; then
    chmod +x "${BATCH_INVARIANT_RUN_FILE}"
    log "Running installer: ${BATCH_INVARIANT_RUN_FILE}"
    if "./${BATCH_INVARIANT_RUN_FILE}"; then
        log "batch_invariant run package installed successfully"
    else
        log "Failed to install batch_invariant run package"
    fi
else
    log "Failed to download batch_invariant run package: ${BATCH_INVARIANT_RUN_URL}"
fi
# clean up downloaded run file (always clean, regardless of success/failure)
rm -f "${BATCH_INVARIANT_RUN_FILE}"

# download and install whl package
BATCH_INVARIANT_WHL_URL="https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/batch_invariant-torch_ops_extension-1.0.0.zip"
BATCH_INVARIANT_WHL_FILE="batch_invariant-torch_ops_extension-1.0.0.zip"

log "Downloading batch_invariant whl package..."
if curl --max-time 3 -sS -k -O "${BATCH_INVARIANT_WHL_URL}" >/dev/null 2>&1 && [[ -f "${BATCH_INVARIANT_WHL_FILE}" ]]; then
    if unzip -o "${BATCH_INVARIANT_WHL_FILE}" >/dev/null 2>&1; then
        if [[ -d "torch_ops_extension/batch_invariant_ops" ]]; then
            cd torch_ops_extension/batch_invariant_ops
            log "Building and installing batch_invariant whl package..."
            if bash build_and_install.sh; then
                log "batch_invariant whl package installed successfully"
            else
                log "Failed to build and install batch_invariant whl package"
            fi
            cd -
        else
            log "batch_invariant_ops directory not found in zip"
        fi
    else
        log "Failed to unzip batch_invariant whl package"
    fi
else
    log "Failed to download batch_invariant whl package: ${BATCH_INVARIANT_WHL_URL}"
fi
# clean up downloaded files (always clean, regardless of success/failure)
rm -rf "${BATCH_INVARIANT_WHL_FILE}" torch_ops_extension

log "batch_invariant_ops build completed"
