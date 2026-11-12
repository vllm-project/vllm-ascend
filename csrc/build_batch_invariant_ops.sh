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
    ascend950*)
        BATCH_INVARIANT_DEVICE="950"
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

# Use a private working directory so failed installs leave no stale downloads.
INSTALL_DIR=$(mktemp -d)
trap 'rm -rf -- "${INSTALL_DIR}"' EXIT
cd "${INSTALL_DIR}"

# Install the device package before building its matching PyTorch extension.
BATCH_INVARIANT_RUN_FILE="cann-ops-batch_invariant-${BATCH_INVARIANT_DEVICE}-2.0.0-linux.${ARCH_SUFFIX}.run"
BATCH_INVARIANT_RUN_URL="https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/${BATCH_INVARIANT_RUN_FILE}"
log "Downloading batch_invariant run package: ${BATCH_INVARIANT_RUN_URL}"
unset ASCEND_CUSTOM_OPP_PATH
curl --fail --location --retry 3 --max-time 120 -sS -o "${BATCH_INVARIANT_RUN_FILE}" "${BATCH_INVARIANT_RUN_URL}"
chmod +x "${BATCH_INVARIANT_RUN_FILE}"
"./${BATCH_INVARIANT_RUN_FILE}"
log "batch_invariant run package installed successfully"

BATCH_INVARIANT_WHL_FILE="batch_invariant-torch_ops_extension-2.0.0.zip"
BATCH_INVARIANT_WHL_URL="https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/${BATCH_INVARIANT_WHL_FILE}"
log "Downloading batch_invariant extension: ${BATCH_INVARIANT_WHL_URL}"
curl --fail --location --retry 3 --max-time 120 -sS -o "${BATCH_INVARIANT_WHL_FILE}" "${BATCH_INVARIANT_WHL_URL}"
python -m zipfile -e "${BATCH_INVARIANT_WHL_FILE}" .
cd torch_ops_extension/batch_invariant_ops
bash build_and_install.sh
log "batch_invariant_ops installation completed"
