#!/bin/bash
set -x
set -euo pipefail

# Install the packages that are not available from the official PyPI index.
# Used by every full-variant Dockerfile (A2/A3/A5, Ubuntu/openEuler), and
# installed for both release and nightly builds. Versions come from the pinned
# ARGs in the Dockerfile (the single source of truth).
#
# Required env vars:
#   MEMFABRIC_VERSION/DATE, MEMCACHE_VERSION/DATE,
#   TRITON_ASCEND_VERSION/PACKAGE_VERSION, TORCH_NPU_VERSION/DATE

: "${MEMFABRIC_VERSION:?MEMFABRIC_VERSION is required}"
: "${MEMFABRIC_DATE:?MEMFABRIC_DATE is required}"
: "${MEMCACHE_VERSION:?MEMCACHE_VERSION is required}"
: "${MEMCACHE_DATE:?MEMCACHE_DATE is required}"
: "${TRITON_ASCEND_VERSION:?TRITON_ASCEND_VERSION is required}"
: "${TRITON_ASCEND_PACKAGE_VERSION:?TRITON_ASCEND_PACKAGE_VERSION is required}"
: "${TORCH_NPU_VERSION:?TORCH_NPU_VERSION is required}"
: "${TORCH_NPU_DATE:?TORCH_NPU_DATE is required}"

ARCH=$(uname -m)

# ---- memfabric_hybrid ----
echo "Install memfabric_hybrid based on architecture..."
if [ "$ARCH" = "x86_64" ]; then
    MEMFABRIC_URL="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/mf/release/1.3/${MEMFABRIC_DATE}/memfabric_hybrid-${MEMFABRIC_VERSION}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
else
    MEMFABRIC_URL="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/mf/release/1.3/${MEMFABRIC_DATE}/memfabric_hybrid-${MEMFABRIC_VERSION}-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl"
fi
python3 -m pip install "$MEMFABRIC_URL" --force-reinstall --no-deps

# ---- memcache_hybrid ----
echo "Install memcache_hybrid based on architecture..."
if [ "$ARCH" = "x86_64" ]; then
    MEMCACHE_URL="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/memcache/release/1.3/${MEMCACHE_DATE}/memcache_hybrid-${MEMCACHE_VERSION}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
else
    MEMCACHE_URL="https://obs-memfabric-hybrid.obs.cn-north-4.myhuaweicloud.com/memcache/release/1.3/${MEMCACHE_DATE}/memcache_hybrid-${MEMCACHE_VERSION}-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl"
fi
python3 -m pip install "$MEMCACHE_URL" --force-reinstall --no-deps

# ---- mfcli kernel install ----
# Install the memfabric kernel for the target SoC. Only A5 and A3 require it.
# Disable torch device backend autoload so `import torch` does not pull in
# torch_npu (unusable on the CPU build host), which breaks mfcli's torch ABI
# check on A3/A5.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
case "${SOC_VERSION:-}" in
    ascend950dt_9582)
        echo "Install memfabric kernel for A5..."
        mfcli kernel install --soc-version A5
        ;;
    ascend910_9391)
        echo "Install memfabric kernel for A3..."
        mfcli kernel install --soc-version A3
        ;;
    *)
        echo "Skip memfabric kernel install (SOC_VERSION=${SOC_VERSION:-})."
        ;;
esac

# ---- triton-ascend ----
echo "Install triton-ascend..."
TRITON_ASCEND_URL="https://ascend-triton-open.obs.cn-north-4.myhuaweicloud.com/triton-ascend/${TRITON_ASCEND_VERSION}/triton_ascend-${TRITON_ASCEND_PACKAGE_VERSION}-cp312-cp312-manylinux_2_27_${ARCH}.manylinux_2_28_${ARCH}.whl"
python3 -m pip install "$TRITON_ASCEND_URL"

# ---- torch-npu ----
echo "Download, extract and install torch-npu..."
mkdir -p /tmp/torch_npu
wget -q --retry-connrefused --tries=5 --timeout=30 --waitretry=10 \
    -O /tmp/torch_npu/torch_npu.tar.gz \
    "https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.10.0-${TORCH_NPU_VERSION}/${TORCH_NPU_DATE}/pytorch_v2.10.0-${TORCH_NPU_VERSION}_py312.tar.gz"
tar -xzf /tmp/torch_npu/torch_npu.tar.gz -C /tmp/torch_npu
python3 -m pip install /tmp/torch_npu/torch_npu-2.10.0*_"${ARCH}".whl --extra-index-url https://download.pytorch.org/whl/cpu/
echo "Clean up temporary files..."
rm -rf /tmp/torch_npu

echo "Pinned package installation complete."
